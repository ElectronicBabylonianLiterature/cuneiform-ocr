from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from .sign import Sign, SignResolver


@dataclass(init=False)
class Box:
    """Unified sign box.

    Coordinates are stored as corners, while center/size fields are exposed as
    mutable properties for the alignment code.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    sign: Sign
    subtablet: Optional[Any]

    def __init__(
        self,
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        x2: Optional[float] = None,
        y2: Optional[float] = None,
        score: float = 1.0,
        sign: Optional[Sign] = None,
        *,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        subtablet: Optional[Any] = None,
    ):
        if sign is None:
            raise ValueError("Box requires a Sign")

        has_corners = None not in (x1, y1, x2, y2)
        has_center = None not in (cx, cy, width, height)
        if has_center and not has_corners:
            x1 = float(cx) - float(width) / 2
            y1 = float(cy) - float(height) / 2
            x2 = float(cx) + float(width) / 2
            y2 = float(cy) + float(height) / 2
        elif not has_corners:
            raise ValueError("Box requires either x1/y1/x2/y2 or cx/cy/width/height")

        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.score = float(score)
        self.sign = sign
        self.subtablet = subtablet

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @width.setter
    def width(self, value: float) -> None:
        cx = self.cx
        half = float(value) / 2
        self.x1 = cx - half
        self.x2 = cx + half

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @height.setter
    def height(self, value: float) -> None:
        cy = self.cy
        half = float(value) / 2
        self.y1 = cy - half
        self.y2 = cy + half

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @cx.setter
    def cx(self, value: float) -> None:
        dx = float(value) - self.cx
        self.x1 += dx
        self.x2 += dx

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @cy.setter
    def cy(self, value: float) -> None:
        dy = float(value) - self.cy
        self.y1 += dy
        self.y2 += dy

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.cx, self.cy

    @property
    def bbox(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def sign_name(self) -> str:
        return self.sign.name

    @property
    def abz_name(self) -> str:
        return self.sign.abz

    def copy(self, subtablet: Optional[Any] = None) -> "Box":
        return Box(
            x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
            score=self.score,
            sign=self.sign,
            subtablet=self.subtablet if subtablet is None else subtablet,
        )

    def translate(self, dx: float, dy: float) -> "Box":
        box = self.copy()
        box.x1 += dx
        box.x2 += dx
        box.y1 += dy
        box.y2 += dy
        return box

    def crop_bounds(
        self,
        image_shape: tuple[int, int] | tuple[int, int, int],
        padding_ratio: float = 0.0,
    ) -> tuple[int, int, int, int]:
        h, w = image_shape[:2]
        pad_x = self.width * padding_ratio
        pad_y = self.height * padding_ratio
        x1 = int(max(0, np.floor(self.x1 - pad_x)))
        y1 = int(max(0, np.floor(self.y1 - pad_y)))
        x2 = int(min(w, np.ceil(self.x2 + pad_x)))
        y2 = int(min(h, np.ceil(self.y2 + pad_y)))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return x1, y1, x2, y2

    def crop_image(
        self,
        image: Optional[np.ndarray] = None,
        padding_ratio: float = 0.0,
    ) -> np.ndarray:
        if image is None:
            if self.subtablet is None or self.subtablet.img is None:
                raise ValueError("Box has no subtablet image")
            image = self.subtablet.img
        x1, y1, x2, y2 = self.crop_bounds(image.shape, padding_ratio)
        return image[y1:y2, x1:x2].copy()


class Boxes(list):
    """List of Box objects with local size statistics."""

    def __init__(self, boxes: Optional[Iterable[Box]] = None, subtablet: Optional[Any] = None):
        if subtablet is None and isinstance(boxes, Boxes):
            subtablet = boxes.subtablet
        self.subtablet = subtablet
        super().__init__()
        self.extend(boxes or [])

    def _attach(self, box: Box) -> Box:
        if self.subtablet is not None:
            box.subtablet = self.subtablet
        return box

    def append(self, box: Box) -> None:
        super().append(self._attach(box))

    def extend(self, boxes: Iterable[Box]) -> None:
        for box in boxes:
            self.append(box)

    def copy(self, subtablet: Optional[Any] = None) -> "Boxes":
        target = self.subtablet if subtablet is None else subtablet
        return Boxes((box.copy(subtablet=target) for box in self), subtablet=target)

    @classmethod
    def from_text_lines(
        cls,
        text_lines: list[list[str]],
        avg_width: float,
        avg_height: float,
        margin: Optional[float] = None,
        target_boxes: Optional[Iterable[Box]] = None,
        align_to_detection_centroid: bool = False,
        subtablet: Optional[Any] = None,
    ) -> "Boxes":
        margin = max(avg_width, avg_height) if margin is None else margin
        boxes = cls(subtablet=subtablet)
        for row_idx, line in enumerate(text_lines):
            for col_idx, sign_name in enumerate(line):
                sign = SignResolver.resolve(sign_name, expected_type="SIGN")
                boxes.append(Box(
                    cx=margin + col_idx * avg_width + avg_width / 2,
                    cy=margin + row_idx * avg_height + avg_height / 2,
                    width=avg_width,
                    height=avg_height,
                    sign=sign,
                    subtablet=subtablet,
                ))

        target_boxes = list(target_boxes or [])
        if align_to_detection_centroid and boxes and target_boxes:
            dx = float(np.mean([b.cx for b in target_boxes]) - np.mean([b.cx for b in boxes]))
            dy = float(np.mean([b.cy for b in target_boxes]) - np.mean([b.cy for b in boxes]))
            for box in boxes:
                box.cx += dx
                box.cy += dy
        return boxes

    @property
    def avg_width(self) -> float:
        return float(np.mean([box.width for box in self])) if self else 80.0

    @property
    def avg_height(self) -> float:
        return float(np.mean([box.height for box in self])) if self else 80.0

    @property
    def avg_size(self) -> float:
        return (self.avg_width + self.avg_height) / 2

    def info(self, name: str = "boxes", image: Optional[np.ndarray] = None) -> str:
        img_shape = image.shape if image is not None else None
        return (
            f"Boxes '{name}':\n"
            f"  {len(self)} signs,\n"
            f"  image shape: {img_shape}"
            f"\n  avg_width: {self.avg_width:.2f}, avg_height: {self.avg_height:.2f}"
        )


def boxes_in_crop(boxes: Iterable[Box], crop_info: dict, subtablet: Optional[Any] = None) -> Boxes:
    crop_x = crop_info["x"]
    crop_y = crop_info["y"]
    crop_w = crop_info["w"]
    crop_h = crop_info["h"]

    transformed = Boxes(subtablet=subtablet)
    for box in boxes or []:
        if not (crop_x <= box.cx < crop_x + crop_w and crop_y <= box.cy < crop_y + crop_h):
            continue

        x1 = max(box.x1, crop_x)
        y1 = max(box.y1, crop_y)
        x2 = min(box.x2, crop_x + crop_w)
        y2 = min(box.y2, crop_y + crop_h)
        if x1 >= x2 or y1 >= y2:
            continue

        transformed.append(Box(
            x1=x1 - crop_x,
            y1=y1 - crop_y,
            x2=x2 - crop_x,
            y2=y2 - crop_y,
            score=box.score,
            sign=box.sign,
            subtablet=subtablet,
        ))
    return transformed
