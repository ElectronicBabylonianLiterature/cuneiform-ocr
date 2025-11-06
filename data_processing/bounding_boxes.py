from pathlib import Path
from typing import Sequence, Optional, List

import attr
import pandas as pd

CONTOUR_TYPES = ["SURFACE_AT_LINE", "STRUCT", "COLUMN_AT_LINE", "REVERSE", "OBVERSE"]
HAS_NO_SIGN = [
    *CONTOUR_TYPES,
    "BLANK",
    "RULING_DOLLAR_LINE",
    "BOTTOM",
    "x",
    "blank",
    "unknownSign",
    "unknown",
    "UnclearSign",
    "unclear",
]


class BoundingBox:
    top_left_x: float
    top_left_y: float
    width: float
    height: float
    sign: str = ""

    def _validate(self):
        MINIMUM_BOUNDING_BOX_SIZE = 7
        if not (
            self.width >= 0
            and self.height >= 0
            and self.top_left_x >= 0
            and self.top_left_y >= 0
            and self.width > MINIMUM_BOUNDING_BOX_SIZE
            and self.height > MINIMUM_BOUNDING_BOX_SIZE
        ):
            raise ValueError("Invalid Bounding Box")

    def __init__(
        self,
        top_left_x: float,
        top_left_y: float,
        width: float,
        height: float,
        sign: str = "",
    ):
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.width = width
        self.height = height
        self.sign = sign
        self._validate()

    def __str__(self):
        return f"(top left x:{self.top_left_x}, top left y:{self.top_left_y}, width:{self.width}, height:{self.height}, sign:{self.sign})"

    @property
    def is_partially_broken(self):
        return "?" in self.sign

    @property
    def clean_sign(self):
        return self.sign.replace("?", "")

    @property
    def has_sign(self):
        return self.sign not in HAS_NO_SIGN

    @property
    def is_contour_type(self):
        return self.sign in CONTOUR_TYPES

    @property
    def bottom_right_x(self):
        return self.top_left_x + self.width

    @property
    def bottom_left_x(self):
        return self.top_left_x

    @property
    def bottom_left_y(self):
        return self.top_left_y + self.height

    @property
    def bottom_right_y(self):
        return self.top_left_y + self.height

    @property
    def as_clockwise_coordinates(self):
        return [
            (self.top_left_x, self.top_left_y),
            (self.top_left_x + self.width, self.top_left_y),
            (self.bottom_right_x, self.bottom_right_y),
            (self.top_left_x, self.bottom_right_y),
        ]

    def recalculate(self, coordinates: "BoundingBox"):
        top_left_x = self.top_left_x - coordinates.top_left_x
        top_left_y = self.top_left_y - coordinates.top_left_y

        bottom_right_x = self.bottom_right_x - coordinates.top_left_x
        bottom_right_y = self.bottom_right_y - coordinates.top_left_y
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y
        return BoundingBox(top_left_x, top_left_y, width, height, self.sign)

    def contains_bbox(self, bbox: "BoundingBox"):
        result = []
        result.append(True) if bbox.top_left_x >= self.top_left_x else result.append(
            False
        )
        result.append(True) if bbox.top_left_y >= self.top_left_y else result.append(
            False
        )
        result.append(
            True
        ) if bbox.bottom_right_x <= self.bottom_right_x else result.append(False)
        result.append(
            True
        ) if bbox.bottom_right_y <= self.bottom_right_y else result.append(False)
        return all(result)

    @classmethod
    def from_two_vertices(
        cls,
        vertices,
        sign="",
    ) -> "BoundingBox":
        """
        from format: [xmin, ymin, xmax, ymax] as used in heidelberg ground truth
        """
        xmin, ymin, xmax, ymax = vertices
        return cls(xmin, ymin, xmax - xmin, ymax - ymin, sign)

    def to_list(self) -> List[float]:
        return [self.top_left_x, self.top_left_y, self.width, self.height]


@attr.s(auto_attribs=True, frozen=True)
class BoundingBoxesContainer:
    image_id: str
    bounding_boxes: Sequence[BoundingBox] = []

    @property
    def contours(self):
        return list(
            filter(lambda bbox: bbox.sign.upper() in CONTOUR_TYPES, self.bounding_boxes)
        )

    def delete_within_bbox(self, within_bbox: BoundingBox):
        return BoundingBoxesContainer(
            self.image_id,
            [
                bbox
                for bbox in self.bounding_boxes
                if not within_bbox.contains_bbox(bbox)
            ],
        )

    def shift_boxes(self, shift_x, shift_y):
        return BoundingBoxesContainer(
            self.image_id,
            [
                BoundingBox(
                    bbox.top_left_x + shift_x,
                    bbox.top_left_y + shift_y,
                    bbox.width,
                    bbox.height,
                    bbox.sign,
                )
                for bbox in self.bounding_boxes
            ],
        )

    @classmethod
    def from_list_of_lists(cls, image_id, lists):
        bboxes = []
        for elem in lists:
            if len(elem) != 5:
                raise ValueError(f"Image id: {image_id} Invalid Bounding Box: {elem}")
            if not isinstance(elem[4], str):
                raise ValueError(
                    f"Image id: {image_id} Invalid Bounding Box: {elem}, Sign should be 'str'"
                )
            assert isinstance(elem[4], str)
            try:
                bounding_box = BoundingBox(*elem)
                bboxes.append(bounding_box)
            except ValueError as e:
                print(f"Image id: {image_id} Invalid Bounding Box: {elem}, Error: {e}")
                continue
        return BoundingBoxesContainer(image_id, bboxes)

    @classmethod
    def from_file(
        cls,
        path,
    ) -> "BoundingBoxesContainer":
        try:
            bboxes = pd.read_csv(
                path, header=None, keep_default_na=False
            ).values.tolist()
        except pd.errors.ParserError as e:
            raise ValueError(f"Could not parse {path}") from e
        return cls.from_list_of_lists(path.stem, bboxes)

    def create_ground_truth_txt_from_file(self, path):
        bounding_boxes_str = []
        for bounding_box in self.bounding_boxes:
            bounding_box_list = list(map(lambda x: str(int(x)), bounding_box.to_list()))
            bounding_boxes_str.append(
                f'{",".join(bounding_box_list)},{bounding_box.sign}'
            )

        with open(path, "w") as f:
            f.write("\n".join(bounding_boxes_str))

    def create_ground_truth_txt(self, path: Optional[Path] = None) -> None:
        gt_filename = Path(f"gt_{self.image_id}.txt")
        path = path / gt_filename if path else gt_filename
        self.create_ground_truth_txt_from_file(path)
