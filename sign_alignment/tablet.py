from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(kw_only=True)
class Tablet:
    img: np.ndarray
    name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.img.shape[:2]

    @property
    def info(self) -> str:
        return f"Tablet '{self.name}': image shape={self.img.shape}"


@dataclass(kw_only=True)
class SubTablet(Tablet):
    parent: Tablet
    offset_in_parent: tuple[float, float]
    mask: Optional[np.ndarray] = None

    @property
    def offset_in_root(self) -> tuple[float, float]:
        offset_x, offset_y = self.offset_in_parent
        if isinstance(self.parent, SubTablet):
            parent_x, parent_y = self.parent.offset_in_root
            return parent_x + offset_x, parent_y + offset_y
        return offset_x, offset_y

    def to_root(self, x: float, y: float) -> tuple[float, float]:
        offset_x, offset_y = self.offset_in_root
        return x + offset_x, y + offset_y

    def from_root(self, x: float, y: float) -> tuple[float, float]:
        offset_x, offset_y = self.offset_in_root
        return x - offset_x, y - offset_y

    @property
    def info(self) -> str:
        mask = "available" if self.mask is not None else "not set"
        return (
            f"SubTablet '{self.name}': image shape={self.img.shape}, "
            f"offset_in_parent={self.offset_in_parent}, mask={mask}"
        )
