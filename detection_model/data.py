"""Fixed COCO -> Deformable DETR preprocessing, independent of training_new."""

from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import CocoDetection
from transformers import AutoImageProcessor


# Fixed preprocessing: PIL, shortest edge 800, longest edge at most 1333, ImageNet normalization.
# Used directly by the dataset; no factory call or external processor argument is needed.
PROCESSOR = AutoImageProcessor.from_pretrained(
    "SenseTime/deformable-detr-with-box-refine-two-stage",
    backend="pil", size={"shortest_edge": 800, "longest_edge": 1333},
)
Image.MAX_IMAGE_PIXELS = None  # The project's trusted data includes very large source images.


class DetectionDataset(CocoDetection):
    def __init__(self, root, split, debug_subset=False):
        root = Path(root).expanduser()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
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
        image, annotations = super().__getitem__(index)
        annotations = [
            {**ann, "category_id": self.category_to_label[ann["category_id"]]}
            for ann in annotations
        ]
        encoded = PROCESSOR(
            images=image,
            annotations={"image_id": self.ids[index], "annotations": annotations},
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
