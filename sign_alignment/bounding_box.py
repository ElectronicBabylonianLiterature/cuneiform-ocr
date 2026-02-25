from dataclasses import dataclass
from typing import TypeAlias, List


from .sign import Sign

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    score: float
    sign: Sign

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height


Detection: TypeAlias = List[BoundingBox]
GroundTruths: TypeAlias = List[BoundingBox]
