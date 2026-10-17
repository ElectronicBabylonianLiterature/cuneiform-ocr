"""Run python test.py here: evaluate with COCO, then save predictions drawn on every test image."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data import DetectionDataset, PROCESSOR, collate_fn, move_batch
from model import build_model


DATASET_ROOT = "~/erc-work-data/data-of-cuneiform-ocr-data/processed-data-exp-1/"
VARIANT = "base-coco"
DEBUG_SUBSET = True  # True applies the subset rule defined in data.py.
DEVICE = "cuda"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CHECKPOINT = OUTPUT_DIR / "best.pt"
TEST_CONFIG = {
    "batch_size": 2,
    "num_workers": 16,
    "max_detections": 300,
    "amp_dtype": torch.float16,  # Supported by both V100 and H100.
    "visualization_threshold": 0.5,  # Used only for visualization, not evaluation filtering.
    "tensorboard_images": True,
}


@torch.inference_mode()
def evaluate(model, loader, dataset, max_detections=300, *, test_config=None):
    """Shared validation/test evaluation: map labels to COCO IDs and boxes to original image coordinates."""
    model.eval()
    config = TEST_CONFIG if test_config is None else test_config
    device = next(model.parameters()).device
    predictions = []
    for batch in tqdm(loader, desc="Evaluate"):
        batch = move_batch(batch, device)
        with torch.autocast(device.type, dtype=config["amp_dtype"], enabled=device.type == "cuda"):
            outputs = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"])
        sizes = torch.stack([target["orig_size"] for target in batch["labels"]])
        results = PROCESSOR.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=0.0, top_k=max_detections,
        )
        for target, result in zip(batch["labels"], results):
            for box, score, label in zip(result["boxes"].cpu().tolist(),
                                         result["scores"].cpu().tolist(), result["labels"].cpu().tolist()):
                x1, y1, x2, y2 = box
                predictions.append({
                    "image_id": int(target["image_id"].item()),
                    "category_id": dataset.label_to_category[label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score,
                })
    return coco_metrics(dataset, predictions, max_detections), predictions


def coco_metrics(dataset, predictions, max_detections=300):
    # Use official matching, crowd/ignore handling, area groups, and precision/recall computation.
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    ground_truth = dataset.coco
    if predictions:
        detected = ground_truth.loadRes(predictions)
    else:
        # Official loadRes does not accept an empty list; let COCOeval handle empty predictions.
        detected = COCO()
        detected.dataset = {"images": ground_truth.dataset["images"],
                            "categories": ground_truth.dataset["categories"], "annotations": []}
        detected.createIndex()
    evaluator = COCOeval(ground_truth, detected, "bbox")
    evaluator.params.imgIds = list(dataset.ids)  # Evaluate only images from the current subset.
    evaluator.params.maxDets = [1, 10, max_detections]
    evaluator.evaluate()
    evaluator.accumulate()

    # Keep maxDets=300 from the original experiment. Official summarize() hardcodes 100 for overall AP,
    # so summarize the official arrays here; use the same detection limit when comparing metrics.
    def mean(ap=True, iou=None, area="all", max_dets=max_detections):
        area_index = evaluator.params.areaRngLbl.index(area)
        max_index = evaluator.params.maxDets.index(max_dets)
        values = evaluator.eval["precision" if ap else "recall"][..., area_index, max_index]
        if iou is not None:
            values = values[np.isclose(evaluator.params.iouThrs, iou)]
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else -1.0

    metrics = {"AP": mean(), "AP50": mean(iou=0.5), "AP75": mean(iou=0.75),
               "AR_1": mean(False, max_dets=1), "AR_10": mean(False, max_dets=10),
               "AR_max": mean(False)}
    for area in ("small", "medium", "large"):
        metrics[f"AP_{area}"] = mean(area=area)
        metrics[f"AR_{area}"] = mean(False, area=area)
    return metrics


def draw_detections(image, predictions, categories, threshold=0.5, *, ground_truth=()):
    """Overlay green GT and red predictions using original COCO coordinates/category IDs."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font_size = max(14, min(32, min(canvas.size) // 50))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    def draw_box(box, caption, color, width, label_below=False):
        x, y, w, h = box
        left, top = max(0, min(canvas.width - 1, x)), max(0, min(canvas.height - 1, y))
        right, bottom = max(left, min(canvas.width - 1, x + w)), max(top, min(canvas.height - 1, y + h))
        draw.rectangle((left, top, right, bottom), outline=color, width=width)
        text_box = draw.textbbox((0, 0), caption, font=font)
        text_width = text_box[2] - text_box[0]
        label_y = top + 4 if label_below else top - font_size - 4
        position = (min(left, max(0, canvas.width - text_width)),
                    max(0, min(canvas.height - font_size - 4, label_y)))
        draw.rectangle(draw.textbbox(position, caption, font=font), fill=color)
        draw.text(position, caption, fill="black" if color == "lime" else "white", font=font)

    # A thicker GT outline remains visible even when a prediction matches it exactly.
    for annotation in ground_truth:
        category = annotation["category_id"]
        draw_box(annotation["bbox"], f"GT {category}: {categories[category]['name']}",
                 "lime", width=4, label_below=True)
    for prediction in predictions:
        if prediction["score"] < threshold:
            continue
        category = prediction["category_id"]
        caption = f"Pred {category}: {categories[category]['name']}  {prediction['score']:.2f}"
        draw_box(prediction["bbox"], caption, "red", width=2)
    return canvas


def visualize(dataset, predictions, output_dir=None, writer=None, step=0, max_images=None,
              *, test_config=None):
    config = TEST_CONFIG if test_config is None else test_config
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["image_id"]].append(prediction)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(dataset.ids[:max_images], desc="Visualize"):
        info = dataset.coco.imgs[image_id]
        with Image.open(Path(dataset.root) / info["file_name"]) as image:
            canvas = draw_detections(image, grouped[image_id], dataset.coco.cats,
                                     config["visualization_threshold"],
                                     ground_truth=dataset.coco.imgToAnns.get(image_id, []))
        if output_dir is not None:
            canvas.save(output_dir / f"{image_id}_{Path(info['file_name']).stem}.jpg", quality=95)
        if writer is not None:
            canvas.thumbnail((1600, 1600))  # TensorBoard thumbnail; disk images retain their original size.
            writer.add_image(f"detections/{image_id}", np.array(canvas), step, dataformats="HWC")


def main(*, model_builder=None, test_config=None, debug_subset=None,
         output_dir=None, checkpoint_path=CHECKPOINT):
    """Shared test workflow; alternate entrypoints supply a model builder and experiment settings."""
    model_builder = build_model if model_builder is None else model_builder
    config = TEST_CONFIG if test_config is None else test_config
    debug_subset = DEBUG_SUBSET if debug_subset is None else debug_subset
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("本脚本固定使用 CUDA，请在 GPU 环境运行")
    dataset = DetectionDataset(Path(DATASET_ROOT).expanduser() / VARIANT, "test", debug_subset)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"], collate_fn=collate_fn, pin_memory=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["class_names"] != dataset.class_names
            or checkpoint["category_to_label"] != dataset.category_to_label):
        raise ValueError("测试集的类别 ID/名称与训练 checkpoint 不一致")
    model = model_builder(dataset.class_names, checkpoint).to(DEVICE)
    del checkpoint
    metrics, predictions = evaluate(model, loader, dataset, config["max_detections"], test_config=config)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"num_images": len(dataset), "max_detections": config["max_detections"], **metrics}
    (output_dir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "test_predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    print(json.dumps(report, indent=2))

    # Visualize all current test images after evaluation, even if no boxes pass the threshold.
    writer = SummaryWriter(str(output_dir / "tensorboard" / "test")) if config["tensorboard_images"] else None
    try:
        visualize(dataset, predictions, output_dir / "visualizations", writer=writer, test_config=config)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
