"""COCO -> Deformable DETR; training augmentation follows configs/detr.py.

Torchvision v2 handles images and boxes together; OpenCV supplies the remaining
affine/color operations. No MMDetection/Albu dependency. Resize interpolation,
tiny-box filtering and unspecified Albu defaults are not exactly identical.
"""

from pathlib import Path

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
            # Match the notebook: take the first len(dataset) % 1000 images.
            self.ids = self.ids[:len(self.ids) % 1000]
            print(f"[DEBUG] using debug subset on {split}: orginial {len(self.coco.imgs)} images, subset {len(self.ids)} images")
        if not self.ids:
            raise ValueError(f"{split} 为空；若总数恰为 100 的倍数，请关闭 DEBUG_SUBSET")

    def __getitem__(self, index):
        for _ in range(MAX_AUGMENT_ATTEMPTS if self.is_train else 1):
            image, annotations = super().__getitem__(index)
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
            index = _randint(0, len(self.ids) - 1)
        else:
            raise RuntimeError(f"No usable training boxes after {MAX_AUGMENT_ATTEMPTS} augmentation attempts")
        encoded = PROCESSOR(
            images=image,
            annotations={"image_id": self.ids[index], "annotations": annotations},
            do_resize=not self.is_train,  # Training already resized; val/test keep fixed resize.
            return_tensors="pt",
        )
        # Class labels are 0..C-1 with no extra background class; boxes use normalized cxcywh.
        return {"pixel_values": encoded["pixel_values"][0], "labels": encoded["labels"][0]}


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
