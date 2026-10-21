"""COCO -> Deformable DETR; training augmentation follows configs/detr.py.

Torchvision v2 handles images and boxes together; OpenCV supplies the remaining
affine/color operations. No MMDetection/Albu dependency. Resize interpolation,
tiny-box filtering and unspecified Albu defaults are not exactly identical.
"""

from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from transformers import AutoImageProcessor


# Fixed val/test resize: shortest edge 800, longest edge at most 1333.
# Training supplies its own sizes; all splits use PIL processing and ImageNet normalization.
# Used directly by the dataset; no factory call or external processor argument is needed.
PROCESSOR = AutoImageProcessor.from_pretrained(
    "SenseTime/deformable-detr-with-box-refine-two-stage",
    backend="pil", size={"shortest_edge": 800, "longest_edge": 1333},
)
Image.MAX_IMAGE_PIXELS = None  # The project's trusted data includes very large source images.
MAX_AUGMENT_ATTEMPTS = 100  # Bound empty-target refetching instead of looping forever.


def _uniform(low, high):
    return low + (high - low) * torch.rand(()).item()


def _randint(low, high):
    """Sample an integer with both endpoints included, using the worker's torch RNG."""
    return torch.randint(low, high + 1, ()).item()


# Keep annotation indices aligned with boxes, so all original fields follow filtering.
# v2's minimum supported size/area is 1: unlike MMDetection/our previous >0 rule,
# subpixel boxes are discarded. This uses the standard v2 detection behavior.
CLEAN_BOXES = v2.Compose([
    v2.ClampBoundingBoxes(),
    v2.SanitizeBoundingBoxes(labels_getter=lambda inputs: inputs[1]["indices"]),
])


class RandomRangeCrop(torch.nn.Module):
    """MMDetection absolute_range: sample H/W separately, then let v2 crop both."""
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, image, target):
        width, height = image.size
        size = [_randint(min(edge, self.low), min(edge, self.high)) for edge in (height, width)]
        image, target = v2.RandomCrop(size)(image, target)
        # Empty crops are allowed; Dataset handles refetching after the full pipeline.
        return CLEAN_BOXES(image, target)


def _shift_hsv(image, gains):
    # OpenCV uint8 BGR: hue wraps at 180; saturation/value clip at 255.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + gains[0]) % 180
    hsv[..., 1:] = np.clip(hsv[..., 1:] + gains[1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _affine_and_color(image, target):
    """configs/detr.py Albu: translation and additive colors without exact v2 equivalents."""
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if _uniform(0, 1) < 0.5:
        # ShiftScaleRotate: translation only, +/-6.25%; assume REFLECT_101 border.
        # RandomAffine uses a different fill policy, so retain OpenCV for this step.
        height, width = image.shape[:2]
        dx, dy = _uniform(-0.0625, 0.0625) * width, _uniform(-0.0625, 0.0625) * height
        image = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        boxes = target["boxes"]
        target = {**target, "boxes": tv_tensors.wrap(boxes + boxes.new_tensor([dx, dy] * 2), like=boxes)}

    if _uniform(0, 1) < 0.2:
        # RandomBrightnessContrast: positive [0.1, 0.3], brightness_by_max=True.
        # ColorJitter uses different formulas, so keep alpha * pixel + beta * 255.
        alpha, beta = 1 + _uniform(0.1, 0.3), _uniform(0.1, 0.3)
        image = np.clip(image.astype(np.float32) * alpha + beta * 255, 0, 255).astype(np.uint8)
    if _uniform(0, 1) < 0.1:
        # OneOf: total p=0.1, equal chance of RGBShift (+/-10) or HSV (+/-20/30/20).
        if _randint(0, 1) == 0:
            shifts = (torch.rand(3).numpy() * 2 - 1) * 10
            image = np.clip(image.astype(np.float32) + shifts, 0, 255).astype(np.uint8)
        else:
            image = _shift_hsv(image, (torch.rand(3).numpy() * 2 - 1) * [20, 30, 20])
    # Explicit BGR handling may differ from Albu's RGB assumptions and rounding.
    return CLEAN_BOXES(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), target)


def _blur_and_hsv(image, target):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if _uniform(0, 1) < 0.1:
        # Albu OneOf: ordinary/median blur, equal chance; assume 3x3 kernels.
        # v2.GaussianBlur is not an equivalent replacement for either operation.
        image = cv2.blur(image, (3, 3)) if _randint(0, 1) == 0 else cv2.medianBlur(image, 3)
    # YOLOX HSV: integer offsets +/-5/30/30, independent 50% enable per channel.
    gains = ((torch.rand(3) * 2 - 1) * torch.tensor([5, 30, 30])
             * torch.randint(0, 2, (3,))).to(torch.int16).numpy()
    image = _shift_hsv(image, gains)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), target


# configs/detr.py:255-475, in execution order. PIL/v2 resize is antialiased and
# rounds sizes differently from OpenCV; JPEG pixels also depend on the codec.
TRAIN_TRANSFORM = v2.Compose([
    CLEAN_BOXES,
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomChoice([  # 50% resize-only; 50% preliminary resize + crop.
        v2.Identity(),
        v2.Compose([v2.RandomShortestSize([400, 500, 600], max_size=4200),
                    RandomRangeCrop(384, 600)]),
    ]),
    v2.RandomShortestSize(list(range(480, 801, 32)), max_size=1333),
    v2.RandomApply([RandomRangeCrop(450, 600)], p=0.2),  # No resize after this crop.
    _affine_and_color,
    v2.RandomApply([v2.JPEG(quality=(85, 95))], p=0.2),
    v2.RandomApply([v2.RandomChannelPermutation()], p=0.1),
    _blur_and_hsv,
    # Preserve all three independent final flips from the original config.
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    CLEAN_BOXES,
])


def augment_train(image, annotations):
    """Return an augmented PIL RGB image and copied COCO xywh annotations.

    Annotation indices follow box filtering, preserving class/ID/crowd alignment.
    Bbox-only: segmentation/keypoints are dropped. Dataset refetches empty results.
    """
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32).reshape(-1, 4),
        format="XYWH", canvas_size=(image.height, image.width),
    )
    target = {"boxes": F.convert_bounding_box_format(boxes, new_format="XYXY"),
              "indices": torch.arange(len(annotations))}
    image, target = TRAIN_TRANSFORM(image.convert("RGB"), target)
    boxes = F.convert_bounding_box_format(target["boxes"], new_format="XYWH")
    result = []
    for index, box in zip(target["indices"].tolist(), boxes.tolist()):
        ann = {**annotations[index], 'bbox': box, 'area': box[2] * box[3]}
        # Do not pass stale spatial annotations to a processor after bbox-only augmentation.
        for key in ('segmentation', 'keypoints', 'num_keypoints'):
            ann.pop(key, None)
        result.append(ann)
    return image, result


class DetectionDataset(CocoDetection):
    def __init__(self, root, split, debug_subset=False):
        root = Path(root).expanduser()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.is_train = split == "train"
        super().__init__(
            root=str(root / f"{split}2017"),
            annFile=str(root / "annotations" / f"instances_{split}2017.json"),
        )
        categories = sorted(self.coco.cats.values(), key=lambda category: category["id"])
        self.category_to_label = {category["id"]: i for i, category in enumerate(categories)}
        self.label_to_category = {v: k for k, v in self.category_to_label.items()}
        self.class_names = [str(category["name"]) for category in categories]
        if debug_subset:
            # Match the notebook: take the first len(dataset) % 100 images.
            self.ids = self.ids[:len(self.ids) % 100]
            print(f"[DEBUG] using debug subset on {split}: orginial {len(self.coco.imgs)} images, subset {len(self.ids)} images")
        if not self.ids:
            raise ValueError(f"{split} is empty; disable DEBUG_SUBSET if the total count is a multiple of 100")
        self.real_ids = list(self.ids)
        self.synthetic_ids = []

    @property
    def dataset_config(self):
        return {"type": type(self).__name__}

    def get_raw_item(self, index):
        """Read an RGB image and independent COCO annotations, before augmentation."""
        image, annotations = super().__getitem__(index)
        return image, [{**ann, "bbox": list(ann["bbox"])} for ann in annotations]

    def _refetch_index(self, index):
        return _randint(0, len(self.ids) - 1)

    def preview_indices(self, per_kind=2):
        """Include the real prefix and synthetic tail, even with shuffled training."""
        real = list(range(min(per_kind, len(self.real_ids))))
        synthetic = list(range(max(len(self.real_ids), len(self.ids) - per_kind), len(self.ids)))
        return real + synthetic

    def __getitem__(self, index):
        for _ in range(MAX_AUGMENT_ATTEMPTS if self.is_train else 1):
            image, annotations = self.get_raw_item(index)
            self.item_cache_0 = (image, annotations)   # debug
            annotations = [
                {**ann, "category_id": self.category_to_label[ann["category_id"]]}
                for ann in annotations
            ]
            self.item_cache_1 = (image, annotations)   # debug
            if not self.is_train:
                break
            image, annotations = augment_train(image, annotations)
            self.item_cache_2 = (image, annotations)   # debug
            # Match Albu's skip_img_without_anno by refetching inside the current subset.
            # HF removes crowd boxes, so require at least one non-crowd target here;
            # unlike MMDetection, a crowd-only result is also retried.
            if any(not ann.get("iscrowd", 0) for ann in annotations):
                break
            index = self._refetch_index(index)
        else:
            raise RuntimeError(f"No usable training boxes after {MAX_AUGMENT_ATTEMPTS} augmentation attempts")
        self.item_index = index  # The actual sample after any empty-target retries.
        encoded = PROCESSOR(
            images=image,
            annotations={"image_id": self.ids[index], "annotations": annotations},
            do_resize=not self.is_train,  # Training already resized; val/test keep fixed resize.
            return_tensors="pt",
        )
        # Class labels are 0..C-1 with no extra background class; boxes use normalized cxcywh.
        return {"pixel_values": encoded["pixel_values"][0], "labels": encoded["labels"][0]}

    def statistics(self):
        """Print and return raw statistics for self.ids, before augmentation.

        Dimensions come from COCO metadata; real file sizes use stat(), without decoding.
        Synthetic images have no file size and are excluded from that metric.
        Percentiles use NumPy's default linear interpolation. Returned units are
        pixels and bytes; the table displays pixel counts in MP and sizes in MiB.
        All annotated boxes are counted, including crowd annotations.
        Class-count statistics include every defined class, including zeros.
        """
        box_counts, widths, heights, file_sizes = [], [], [], []
        category_counts = {category_id: 0 for category_id in sorted(self.coco.cats)}
        synthetic_counts = dict.fromkeys(category_counts, 0)
        synthetic_ids = set(self.synthetic_ids)
        for image_id in self.ids:
            image = self.coco.imgs[image_id]
            annotations = self.coco.imgToAnns.get(image_id, [])
            box_counts.append(len(annotations))
            for annotation in annotations:
                category_counts[annotation["category_id"]] += 1
                if image_id in synthetic_ids:
                    synthetic_counts[annotation["category_id"]] += 1
            widths.append(image["width"])
            heights.append(image["height"])
            if image_id not in synthetic_ids:
                file_sizes.append((Path(self.root) / image["file_name"]).stat().st_size)

        summary = {
            "images": len(self.ids), "boxes": sum(box_counts),
            "classes": sum(count > 0 for count in category_counts.values()),
            "defined_classes": len(self.coco.cats), "class_counts": category_counts,
            "real_images": len(self.real_ids), "synthetic_images": len(self.synthetic_ids),
            "real_boxes": sum(box_counts) - sum(synthetic_counts.values()),
            "synthetic_boxes": sum(synthetic_counts.values()),
            "real_class_counts": {key: count - synthetic_counts[key] for key, count in category_counts.items()},
            "synthetic_class_counts": synthetic_counts,
        }
        print(f"Dataset: {Path(self.root).name} (raw) | Images: {summary['images']:,} | "
              f"Boxes: {summary['boxes']:,} | Classes: {summary['classes']} present / "
              f"{summary['defined_classes']} defined")
        print(f"Real: {summary['real_images']:,} images / {summary['real_boxes']:,} boxes | "
              f"Synthetic: {summary['synthetic_images']:,} images / {summary['synthetic_boxes']:,} boxes")
        columns = ("p50", "p90", "p99", "max", "min", "mean")
        print(f"{'Metric':<20}" + "".join(f"{column.upper():>13}" for column in columns))
        metrics = [
            ("boxes_per_image", "Boxes/image", box_counts, 1),
            ("boxes_per_class", "Boxes/class (all)", list(category_counts.values()), 1),
            ("image_pixels", "Pixels (MP)", np.asarray(widths, dtype=np.float64) * heights, 1e6),
            ("image_width", "Width (px)", widths, 1),
            ("image_height", "Height (px)", heights, 1),
            ("file_bytes", "File size (MiB)", file_sizes, 1024 ** 2),
        ]
        for key, label, values, scale in metrics:
            values = np.asarray(values, dtype=np.float64)
            if values.size:
                stats = dict(zip(columns[:3], np.percentile(values, [50, 90, 99]).tolist()))
                stats.update(max=float(values.max()), min=float(values.min()), mean=float(values.mean()))
            else:
                stats = dict.fromkeys(columns)
            summary[key] = stats
            cells = [f"{stats[column] / scale:,.2f}" if stats[column] is not None else "N/A"
                     for column in columns]
            print(f"{label:<20}" + "".join(f"{cell:>13}" for cell in cells))
        for key, label, sign in (("most_common_classes", "Most common 5", -1),
                                 ("least_common_classes", "Least common 5", 1)):
            ranked_ids = sorted(
                category_counts, key=lambda category_id: (sign * category_counts[category_id], category_id),
            )[:5]
            summary[key] = [
                {"category_id": category_id, "name": str(self.coco.cats[category_id].get("name", category_id)),
                 "count": category_counts[category_id]}
                for category_id in ranked_ids
            ]
            entries = [f"{item['name']} (id={item['category_id']}): {item['count']:,}" for item in summary[key]]
            print(f"{label}: " + ("; ".join(entries) if entries else "N/A"))
        return summary


class SynDetectionDataset(DetectionDataset):
    """Append fixed train-only recipes; sample and paste sign crops at read time.

    root is the COCO variant, usually <dataset_root>/divided-coco. The sibling
    base-coco supplies held-out source filenames; sign_crops_index supplies JPEGs.
    No images are decoded during initialization and no synthetic files are written.
    """

    def __init__(self, root, split, debug_subset=False, *, target_count=500, seed=42):
        super().__init__(root, split, debug_subset)
        self.target_count, self.seed = target_count, seed
        self.synthetic_bases = {}
        self.crops_by_category = {}
        if not self.is_train:
            return
        if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 0:
            raise ValueError("target_count must be a non-negative integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        dataset_root = Path(root).expanduser().parent
        self.sign_crops_root = dataset_root / "sign_crops_index"
        with (self.sign_crops_root / "index.json").open(encoding="utf-8") as stream:
            crop_index = json.load(stream)
        crop_categories = {cat["id"]: str(cat["name"]) for cat in crop_index["categories"]}
        categories = {key: str(cat["name"]) for key, cat in self.coco.cats.items()}
        if crop_categories != categories:
            raise ValueError("Category IDs/names must match between sign_crops_index and the training set")

        held_out = set()
        for held_split in ("val", "test"):
            path = dataset_root / "base-coco" / "annotations" / f"instances_{held_split}2017.json"
            with path.open(encoding="utf-8") as stream:
                held_out.update(image["file_name"] for image in json.load(stream)["images"])
        counts = Counter(ann["category_id"] for image_id in self.real_ids
                         for ann in self.coco.imgToAnns.get(image_id, []))
        deficits = {key: target_count - counts[key] for key in sorted(categories) if counts[key] < target_count}
        self.crops_by_category = {key: [] for key in deficits}
        for crop in crop_index["crops"]:
            category = crop["category_id"]
            if category in deficits and crop["source_image"] not in held_out:
                self.crops_by_category[category].append(crop["file_name"])
        missing = [f"{key}: {categories[key]}" for key, crops in self.crops_by_category.items() if not crops]
        if missing:
            raise ValueError("No usable crops remain after excluding val/test sources for: " + "; ".join(missing))
        if not deficits:
            return

        # Integer paste rectangles cover the entire original box, including fractional edges.
        templates = []
        for image_id in self.real_ids:
            info = self.coco.imgs[image_id]
            boxes = []
            for ann in self.coco.imgToAnns.get(image_id, []):
                x, y, w, h = ann["bbox"]
                if not all(math.isfinite(value) for value in (x, y, w, h)) or min(w, h) <= 0:
                    raise ValueError(f"Invalid base bbox: image={image_id}, annotation={ann['id']}")
                left, top = max(0, math.floor(x)), max(0, math.floor(y))
                right, bottom = min(info["width"], math.ceil(x + w)), min(info["height"], math.ceil(y + h))
                if right <= left or bottom <= top:
                    raise ValueError(f"Base bbox is outside image {image_id}: {ann['bbox']}")
                boxes.append([left, top, right - left, bottom - top])
            if boxes:
                templates.append((len(boxes), image_id, boxes))
        templates.sort(key=lambda item: (item[0], item[1]))
        template_sizes = [item[0] for item in templates]
        rng = random.Random(seed)  # Independent of model, loader and crop-sampling RNGs.
        pending = [category for category, count in deficits.items() for _ in range(count)]
        rng.shuffle(pending)
        offset = 0
        next_image_id = max(self.coco.imgs) + 1
        next_ann_id = max(self.coco.anns, default=-1) + 1
        while offset < len(pending):
            eligible = bisect_right(template_sizes, len(pending) - offset)
            if not eligible:
                raise ValueError(f"Cannot fill all original boxes and meet the exact quota: {len(pending) - offset} boxes remain; "
                                 "provide training base images with fewer boxes (e.g., single-box tablet fragments)")
            size, base_id, boxes = templates[rng.randrange(eligible)]
            image_id = next_image_id + len(self.synthetic_ids)
            base = self.coco.imgs[base_id]
            self.synthetic_ids.append(image_id)
            self.synthetic_bases[image_id] = base_id
            self.coco.dataset["images"].append({
                "id": image_id, "file_name": f"synthetic_{image_id}.jpg",
                "width": base["width"], "height": base["height"],
                "base_image_id": base_id, "base_file_name": base["file_name"],
            })
            for category, box in zip(pending[offset:offset + size], boxes):
                self.coco.dataset["annotations"].append({
                    "id": next_ann_id, "image_id": image_id, "category_id": category,
                    "bbox": box, "area": box[2] * box[3], "iscrowd": 0,
                })
                next_ann_id += 1
            offset += size
        self.ids.extend(self.synthetic_ids)
        self.coco.createIndex()
        print(f"Synthetic train: {len(self.synthetic_ids):,} images / {len(pending):,} boxes | "
              f"{len(deficits)} classes topped up to {target_count} | seed={seed}")

    @property
    def dataset_config(self):
        return {**super().dataset_config, "seed": self.seed, "target_count": self.target_count}

    def _load_image(self, image_id):
        if image_id not in self.synthetic_bases:
            return super()._load_image(image_id)
        image = super()._load_image(self.synthetic_bases[image_id])
        for ann in self.coco.imgToAnns[image_id]:
            crops = self.crops_by_category[ann["category_id"]]
            path = self.sign_crops_root / crops[_randint(0, len(crops) - 1)]
            x, y, width, height = ann["bbox"]
            with Image.open(path) as crop:
                image.paste(crop.convert("RGB").resize((width, height), Image.Resampling.BILINEAR), (x, y))
        return image

    def _refetch_index(self, index):
        if self.ids[index] in self.synthetic_bases:
            return _randint(len(self.real_ids), len(self.ids) - 1)
        return _randint(0, len(self.real_ids) - 1)


def preview_samples(dataset, *, per_kind=2, seed=42):
    """Return real/synthetic before/after examples without advancing training RNGs.

    Both stages come from the same read (including any empty-target refetch).
    COCO category IDs are restored for drawing the augmented annotations.
    """
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    samples = []
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            for index in dataset.preview_indices(per_kind):
                dataset[index]
                image_id = dataset.ids[dataset.item_index]
                info = dataset.coco.imgs[image_id]
                image, annotations = dataset.item_cache_2 if dataset.is_train else dataset.item_cache_1
                annotations = [{**ann, "category_id": dataset.label_to_category[ann["category_id"]]}
                               for ann in annotations]
                samples.append({
                    "image_id": image_id, "is_synthetic": "base_image_id" in info,
                    "base_file_name": info.get("base_file_name", info["file_name"]),
                    "original": dataset.item_cache_0, "augmented": (image, annotations),
                })
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    return samples


def collate_fn(samples):
    """Pad images to a common size; keep labels as a list for varying object counts."""
    images = [sample["pixel_values"] for sample in samples]
    height = max(image.shape[-2] for image in images)
    width = max(image.shape[-1] for image in images)
    pixels = images[0].new_zeros(len(images), 3, height, width)
    mask = torch.zeros(len(images), height, width, dtype=torch.bool)
    for index, image in enumerate(images):
        h, w = image.shape[-2:]
        pixels[index, :, :h, :w] = image
        mask[index, :h, :w] = True
    # Keep box normalization relative to each image's unpadded size.
    return {"pixel_values": pixels, "pixel_mask": mask,
            "labels": [sample["labels"] for sample in samples]}


def move_batch(batch, device):
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "pixel_mask": batch["pixel_mask"].to(device, non_blocking=True),
        "labels": [{k: v.to(device, non_blocking=True) for k, v in target.items()}
                   for target in batch["labels"]],
    }
