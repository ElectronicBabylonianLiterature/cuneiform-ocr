from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .sign import Sign, SignResolver
from .tablet import SubTablet, Tablet


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
    tablet: Tablet

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        sign: Sign,
        tablet: Tablet,
        score: float = 1.0,
    ):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.score = float(score)
        self.sign = sign
        self.tablet = tablet

    @classmethod
    def from_center(
        cls,
        cx: float,
        cy: float,
        width: float,
        height: float,
        sign: Sign,
        tablet: Tablet,
        score: float = 1.0,
    ) -> "Box":
        half_w = float(width) / 2
        half_h = float(height) / 2
        return cls(
            x1=float(cx) - half_w,
            y1=float(cy) - half_h,
            x2=float(cx) + half_w,
            y2=float(cy) + half_h,
            sign=sign,
            tablet=tablet,
            score=score,
        )

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

    def copy(self) -> "Box":
        return Box(
            x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
            score=self.score,
            sign=self.sign,
            tablet=self.tablet,
        )

    def translate(self, dx: float, dy: float) -> "Box":
        box = self.copy()
        box.x1 += dx
        box.x2 += dx
        box.y1 += dy
        box.y2 += dy
        return box

    def to_tablet(self, tablet: Tablet) -> "Box":
        if self.tablet is tablet:
            return self.copy()

        if isinstance(self.tablet, SubTablet):
            x1, y1 = self.tablet.to_root(self.x1, self.y1)
            x2, y2 = self.tablet.to_root(self.x2, self.y2)
        else:
            x1, y1 = self.x1, self.y1
            x2, y2 = self.x2, self.y2

        if isinstance(tablet, SubTablet):
            x1, y1 = tablet.from_root(x1, y1)
            x2, y2 = tablet.from_root(x2, y2)

        return Box(
            x1=x1, y1=y1, x2=x2, y2=y2,
            sign=self.sign,
            tablet=tablet,
            score=self.score,
        )

    def crop_bounds(
        self,
        padding_ratio: float = 0.0,
    ) -> tuple[int, int, int, int]:
        h, w = self.tablet.shape
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

    def crop_image(self, padding_ratio: float = 0.0) -> np.ndarray:
        image = self.tablet.img
        x1, y1, x2, y2 = self.crop_bounds(padding_ratio)
        return image[y1:y2, x1:x2].copy()


class Boxes(list):
    """List of Box objects with local size statistics."""

    def __init__(self, boxes: Iterable[Box] = (), *, tablet: Tablet):
        self.tablet = tablet
        super().__init__()
        self.extend(boxes)

    def append(self, box: Box) -> None:
        if box.tablet is not self.tablet:
            raise ValueError("Box belongs to a different tablet")
        super().append(box)

    def extend(self, boxes: Iterable[Box]) -> None:
        for box in boxes:
            self.append(box)

    def copy(self) -> "Boxes":
        return Boxes((box.copy() for box in self), tablet=self.tablet)

    def to_tablet(self, tablet: Tablet) -> "Boxes":
        return Boxes((box.to_tablet(tablet) for box in self), tablet=tablet)

    @classmethod
    def from_text_lines(
        cls,
        text_lines: list[list[str]],
        avg_width: float,
        avg_height: float,
        tablet: Tablet,
        margin: Optional[float] = None,
        target_boxes: Optional[Iterable[Box]] = None,
        align_to_detection_centroid: bool = False,
    ) -> "Boxes":
        margin = max(avg_width, avg_height) if margin is None else margin
        boxes = cls(tablet=tablet)
        for row_idx, line in enumerate(text_lines):
            for col_idx, sign_name in enumerate(line):
                sign = SignResolver.resolve(sign_name, expected_type="SIGN")
                boxes.append(Box.from_center(
                    cx=margin + col_idx * avg_width + avg_width / 2,
                    cy=margin + row_idx * avg_height + avg_height / 2,
                    width=avg_width,
                    height=avg_height,
                    sign=sign,
                    tablet=tablet,
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

    def info(self, name: str = "boxes") -> str:
        return (
            f"Boxes '{name}':\n"
            f"  {len(self)} signs,\n"
            f"  image shape: {self.tablet.img.shape}"
            f"\n  avg_width: {self.avg_width:.2f}, avg_height: {self.avg_height:.2f}"
        )


def boxes_in_crop(boxes: Iterable[Box], crop_tablet: SubTablet) -> Boxes:
    crop_h, crop_w = crop_tablet.shape

    transformed = Boxes(tablet=crop_tablet)
    for box in boxes or []:
        crop_box = box.to_tablet(crop_tablet)
        if not (0 <= crop_box.cx < crop_w and 0 <= crop_box.cy < crop_h):
            continue

        x1 = max(crop_box.x1, 0)
        y1 = max(crop_box.y1, 0)
        x2 = min(crop_box.x2, crop_w)
        y2 = min(crop_box.y2, crop_h)
        if x1 >= x2 or y1 >= y2:
            continue

        transformed.append(Box(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=box.score,
            sign=box.sign,
            tablet=crop_tablet,
        ))
    return transformed
