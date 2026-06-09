from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SubTablet:
    img: np.ndarray
    mask: Optional[np.ndarray] = None
    name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.img.shape[:2]

    @property
    def info(self) -> str:
        mask = "available" if self.mask is not None else "not set"
        return f"SubTablet '{self.name}': image shape={self.img.shape}, mask={mask}"
