"""Build divided-coco; run python detection_model/preprocess_dataset.py.

Each split is processed independently. Rectangle crops with contained boxes
become COCO images. Empty-target crops go to no-annotation-img/<split>/ instead.
Splitting failures warn and skip the source image; the splitter is not modified.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import cv2
from pycocotools.coco import COCO
from tqdm import tqdm

# Support both direct execution and python -m detection_model.preprocess_dataset.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.divide_photos import divide_tablet_photo


DATASET_ROOT = "~/erc-work-data/data-of-cuneiform-ocr-data/processed-data-exp-6/"
INPUT_VARIANT = "base-coco"
OUTPUT_VARIANT = "divided-coco"
SPLITS = ("train", "val", "test")
JPEG_QUALITY = 95
NUM_WORKERS = 16  # Each thread may decode a very large source photo.


def _split_image(path, image_info):
    # COCO/PIL coordinates refer to stored pixels, without EXIF rotation.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("Cannot read image")
    height, width = image.shape[:2]
    if (width, height) != (image_info["width"], image_info["height"]):
        raise ValueError(f"Image size {width}x{height} differs from COCO metadata")
    result = divide_tablet_photo(
        image, return_coordinates=True, return_masks=False,
        visualize=False, force_output_result=False, logging=False,
    )
    # No contours returns [], while an empty filtered result returns ([], []).
    if not result or not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("No valid subtablets returned")
    crops, coordinates = result
    if not crops or len(crops) != len(coordinates):
        raise ValueError("Empty or inconsistent subtablet results")
    rectangles = []
    for crop, rect in zip(crops, coordinates):
        rectangle = [int(rect[key]) for key in ("x", "y", "w", "h")]
        x, y, w, h = rectangle
        if (w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height
                or crop.shape[:2] != (h, w)):
            raise ValueError(f"Invalid crop rectangle: {rectangle}")
        rectangles.append(rectangle)
    return crops, rectangles


def _crop_annotations(annotations, rectangle):
    """Translate contained boxes; omit crossing boxes instead of clipping them."""
    x, y, width, height = rectangle
    kept, crossing_ids = [], []
    for annotation in annotations:
        bx, by, bw, bh = annotation["bbox"]
        if x <= bx and y <= by and bx + bw <= x + width and by + bh <= y + height:
            converted = {
                **annotation, "source_annotation_id": annotation["id"],
                "bbox": [bx - x, by - y, bw, bh], "area": bw * bh,
                "iscrowd": annotation.get("iscrowd", 0),
            }
            # This is a bbox-only release; do not retain stale spatial fields.
            for key in ("segmentation", "keypoints", "num_keypoints"):
                converted.pop(key, None)
            kept.append(converted)
        elif max(bx, x) < min(bx + bw, x + width) and max(by, y) < min(by + bh, y + height):
            crossing_ids.append(annotation["id"])
    return kept, crossing_ids


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _process_image(input_root, image_dir, no_annotation_dir, split, image_info, annotations):
    """Decode, split and save one source in a worker; return metadata, not pixels."""
    try:
        for annotation in annotations:
            box = annotation["bbox"]
            if len(box) != 4 or not all(math.isfinite(value) for value in box) or min(box[2:]) <= 0:
                raise ValueError(f"Invalid bbox in annotation {annotation['id']}")
        source_path = input_root / f"{split}2017" / image_info["file_name"]
        crops, rectangles = _split_image(source_path, image_info)
    except Exception as error:
        return image_info, annotations, [], str(error)

    items = []
    for crop_index, (crop, rectangle) in enumerate(zip(crops, rectangles)):
        converted, crossing_ids = _crop_annotations(annotations, rectangle)
        filename = f"{image_info['id']}__subtablet_{crop_index:03d}.jpg"
        destination = (image_dir if converted else no_annotation_dir) / filename
        if not cv2.imwrite(str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise OSError(f"Cannot write image: {destination}")
        items.append((
            {"file_name": filename, "width": rectangle[2], "height": rectangle[3]},
            converted,
            {"file_name": filename, "source_image_id": image_info["id"],
             "output_path": str(destination.relative_to(image_dir.parent)),
             "source_file_name": image_info["file_name"], "crop_index": crop_index,
             "crop_bbox": rectangle, "annotation_count": len(converted),
             "cross_boundary_annotation_ids": crossing_ids},
        ))
    return image_info, annotations, items, None


def process_split(input_root, output_root, split, num_workers=NUM_WORKERS):
    """Write one split and its source mapping, issues, and summary; return counts."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    
    input_root = Path(input_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if input_root == output_root:
        raise ValueError("Input and output directories must differ")
    source_file = input_root / "annotations" / f"instances_{split}2017.json"
    target_file = output_root / "annotations" / source_file.name
    image_dir = output_root / f"{split}2017"
    no_annotation_dir = output_root / "no-annotation-img" / split
    report_dir = output_root / "reports"
    report_paths = {name: report_dir / f"{split}_{name}.{suffix}"
                    for name, suffix in (("items", "jsonl"), ("issues", "jsonl"),
                                         ("no_annotation", "jsonl"), ("summary", "json"))}
    for path in (image_dir, no_annotation_dir, target_file, *report_paths.values()):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    coco = COCO(str(source_file))
    for key, index in (("images", coco.imgs), ("annotations", coco.anns), ("categories", coco.cats)):
        if len(coco.dataset[key]) != len(index):
            raise ValueError(f"Duplicate {key} IDs in {source_file}")
    for annotation in coco.anns.values():
        if annotation["image_id"] not in coco.imgs or annotation["category_id"] not in coco.cats:
            raise ValueError(f"Invalid image/category reference in annotation {annotation['id']}")

    output = {**coco.dataset, "images": [], "annotations": []}
    summary = {
        "split": split, "input_images": len(coco.imgs), "input_annotations": len(coco.anns),
        "output_images": 0, "output_annotations": 0, "empty_subtablets": 0,
        "failed_images": 0, "cross_boundary_boxes_omitted": 0,
    }
    retained_source_ids = set()
    image_dir.mkdir(parents=True)
    no_annotation_dir.mkdir(parents=True)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    with report_paths["items"].open("x", encoding="utf-8") as items_stream, \
            report_paths["issues"].open("x", encoding="utf-8") as issues_stream, \
            report_paths["no_annotation"].open("x", encoding="utf-8") as no_annotation_stream, \
            ThreadPoolExecutor(max_workers=num_workers) as executor:

        def issue(image_info, kind, message, **details):
            record = {"source_image_id": image_info["id"], "source_file_name": image_info["file_name"],
                      "kind": kind, "message": message, **details}
            issues_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"WARNING [{split}] {image_info['file_name']}: {message}", file=sys.stderr)

        def process_source(source_id):
            # Workers only read COCO indexes; the caller assigns IDs and merges
            # output lists, counters and JSONL reports.
            image_info = coco.loadImgs([source_id])[0]
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=[source_id]))
            return _process_image(input_root, image_dir, no_annotation_dir, split, image_info, annotations)

        # map yields in source order regardless of completion order, so IDs and
        # reports stay reproducible. Finished jobs retain only small metadata.
        results = executor.map(process_source, sorted(coco.getImgIds()))
        for image_info, annotations, items, error in tqdm(
                results, total=len(coco.imgs), desc=f"Preprocess {split}", unit="image"):
            if error is not None:
                summary["failed_images"] += 1
                issue(image_info, "split_failed", error)
                continue

            image_retained_ids = set()
            for image_record, converted, mapping in items:
                summary["cross_boundary_boxes_omitted"] += len(mapping["cross_boundary_annotation_ids"])
                if not converted:
                    summary["empty_subtablets"] += 1
                    no_annotation_stream.write(json.dumps(mapping, ensure_ascii=False) + "\n")
                    continue
                image_id = len(output["images"]) + 1
                output["images"].append({"id": image_id, **image_record})
                for annotation in converted:
                    annotation["id"] = len(output["annotations"]) + 1
                    annotation["image_id"] = image_id
                    output["annotations"].append(annotation)
                    image_retained_ids.add(annotation["source_annotation_id"])
                items_stream.write(json.dumps({"image_id": image_id, **mapping}, ensure_ascii=False) + "\n")
            retained_source_ids.update(image_retained_ids)
            dropped_ids = sorted({annotation["id"] for annotation in annotations} - image_retained_ids)
            if dropped_ids:
                issue(image_info, "unretained_annotations",
                      f"{len(dropped_ids)} annotations not fully contained in any output crop",
                      source_annotation_ids=dropped_ids)

    summary.update(
        output_images=len(output["images"]), output_annotations=len(output["annotations"]),
        retained_source_annotations=len(retained_source_ids),
        dropped_source_annotations=len(coco.anns) - len(retained_source_ids),
        duplicated_annotations=len(output["annotations"]) - len(retained_source_ids),
    )
    _write_json(target_file, output)
    _write_json(report_paths["summary"], summary)
    print(f"{split}: {summary['input_images']:,} source images -> {summary['output_images']:,} subtablets | "
          f"Boxes: {summary['output_annotations']:,} | No annotation (excluded): {summary['empty_subtablets']:,} | "
          f"Failed: {summary['failed_images']:,} | Dropped source boxes: {summary['dropped_source_annotations']:,}")
    return summary


def main(dataset_root=DATASET_ROOT, input_variant=INPUT_VARIANT, output_variant=OUTPUT_VARIANT,
         num_workers=NUM_WORKERS):
    root = Path(dataset_root).expanduser().resolve()
    input_root, output_root = root / input_variant, root / output_variant
    if input_root.resolve() == output_root.resolve():
        raise ValueError("Input and output directories must differ")
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input dataset does not exist: {input_root}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    return {split: process_split(input_root, output_root, split, num_workers=num_workers) for split in SPLITS}


if __name__ == "__main__":
    main()

