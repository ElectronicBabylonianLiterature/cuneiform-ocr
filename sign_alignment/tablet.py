"""
SubTablet data structures for sign alignment.

Provides SignBox and SubTablet classes for organizing sign bounding boxes
and performing coordinate transformations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .sign import Sign, SignResolver, CLASSES_ABZ
from .bounding_box import BoundingBox, Detection
from .heatmap import compute_avg_dimensions


@dataclass
class SignBox:
    """
    Represents a single sign bounding box with center-based representation.
    Stored as (cx, cy, w, h) with conversion to [x1, y1, x2, y2].
    """
    sign: Sign
    score: float = 1.0
    cx: float = 0.0
    cy: float = 0.0
    width: float = 0.0
    height: float = 0.0
    row_idx: int = -1
    col_idx: int = -1
    
    @property
    def sign_name(self) -> str:
        return self.sign.name
    
    @property
    def abz_name(self) -> str:
        return self.sign.abz

    @classmethod
    def from_bbox(cls, bbox: List[float], sign: Sign,
                  score: float = 1.0, row_idx: int = -1, col_idx: int = -1) -> 'SignBox':
        """Create from [x1, y1, x2, y2] format."""
        x1, y1, x2, y2 = bbox
        return cls(
            sign=sign, score=score,
            cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
            width=x2 - x1, height=y2 - y1,
            row_idx=row_idx, col_idx=col_idx
        )
    
    @classmethod
    def from_center(cls, cx: float, cy: float, width: float, height: float,
                    sign: Sign, score: float = 1.0,
                    row_idx: int = -1, col_idx: int = -1) -> 'SignBox':
        """Create from center format."""
        return cls(sign=sign, score=score, cx=cx, cy=cy,
                   width=width, height=height,
                   row_idx=row_idx, col_idx=col_idx)
    
    @classmethod
    def from_detection(cls, det: BoundingBox, row_idx: int = -1, col_idx: int = -1) -> 'SignBox':
        """Create from a BoundingBox detection."""
        return cls(
            sign=det.sign, score=det.score,
            cx=(det.x1 + det.x2) / 2, cy=(det.y1 + det.y2) / 2,
            width=det.x2 - det.x1, height=det.y2 - det.y1,
            row_idx=row_idx, col_idx=col_idx
        )
    
    @property
    def x1(self) -> float:
        return self.cx - self.width / 2
    
    @property
    def y1(self) -> float:
        return self.cy - self.height / 2
    
    @property
    def x2(self) -> float:
        return self.cx + self.width / 2
    
    @property
    def y2(self) -> float:
        return self.cy + self.height / 2
    
    @property
    def bbox(self) -> List[float]:
        """Return [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    @property
    def center(self) -> tuple:
        return (self.cx, self.cy)
    
    def to_bounding_box(self) -> BoundingBox:
        """Convert to BoundingBox for visualizer compatibility."""
        return BoundingBox(
            x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
            score=self.score, sign=self.sign
        )
    
    def translate(self, dx: float, dy: float) -> 'SignBox':
        """Return translated copy."""
        return SignBox(
            sign=self.sign, score=self.score,
            cx=self.cx + dx, cy=self.cy + dy,
            width=self.width, height=self.height,
            row_idx=self.row_idx, col_idx=self.col_idx
        )
    
    def copy(self) -> 'SignBox':
        """Return a deep copy."""
        return SignBox(
            sign=self.sign, score=self.score,
            cx=self.cx, cy=self.cy,
            width=self.width, height=self.height,
            row_idx=self.row_idx, col_idx=self.col_idx
        )


@dataclass
class SubTablet:
    """
    Represents a sub-tablet region with image and sign boxes.
    """
    img: Optional[np.ndarray] = None
    sign_boxes: List[SignBox] = field(default_factory=list)
    
    name: str = ""
    scale_factor: int = 10
    avg_width: float = 80.0
    avg_height: float = 80.0
    margin: float = 0.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    
    def __len__(self):
        return len(self.sign_boxes)
    
    @property
    def shape(self) -> tuple:
        if self.img is not None:
            return self.img.shape[:2]
        elif self.sign_boxes:
            max_x = max(sb.cx + sb.width / 2 for sb in self.sign_boxes)
            max_y = max(sb.cy + sb.height / 2 for sb in self.sign_boxes)
            return (int(max_y + self.margin), int(max_x + self.margin))
        return (0, 0)
    
    @classmethod
    def from_detections(cls, img: np.ndarray, detections: Detection,
                        name: str = "detection",
                        avg_width: float = None, avg_height: float = None) -> 'SubTablet':
        """Create from image and BoundingBox detections."""
        sign_boxes = [SignBox.from_detection(det) for det in detections]
        
        if avg_width is None or avg_height is None:
            computed_w, computed_h = compute_avg_dimensions(detections)
            avg_width = avg_width or computed_w
            avg_height = avg_height or computed_h
        
        return cls(img=img, sign_boxes=sign_boxes, name=name,
                   avg_width=avg_width, avg_height=avg_height)
    
    @classmethod
    def from_text_lines(cls, text_lines: List[List[str]],
                        avg_width: float, avg_height: float,
                        margin: float = None,
                        img: Optional[np.ndarray] = None,
                        target_detections: Optional[Detection] = None,
                        align_to_detection_centroid: bool = False,
                        name: str = "full_text") -> 'SubTablet':
        """
        Create from text lines with uniform grid layout.

        Optionally translates the generated grid so its centroid matches
        the centroid of the provided detections.
        """
        if margin is None:
            margin = max(avg_width, avg_height)
        
        sign_boxes = []
        for row_idx, line in enumerate(text_lines):
            for col_idx, sign_name in enumerate(line):
                cx = margin + col_idx * avg_width + avg_width / 2
                cy = margin + row_idx * avg_height + avg_height / 2
                
                sign = SignResolver.resolve(sign_name, expected_type='SIGN')
                
                sb = SignBox.from_center(
                    cx=cx, cy=cy, width=avg_width, height=avg_height,
                    sign=sign, row_idx=row_idx, col_idx=col_idx
                )
                sign_boxes.append(sb)

        if align_to_detection_centroid and sign_boxes and target_detections:
            text_cx = float(np.mean([sb.cx for sb in sign_boxes]))
            text_cy = float(np.mean([sb.cy for sb in sign_boxes]))
            det_cx = float(np.mean([(d.x1 + d.x2) / 2 for d in target_detections]))
            det_cy = float(np.mean([(d.y1 + d.y2) / 2 for d in target_detections]))

            dx = det_cx - text_cx
            dy = det_cy - text_cy
            for sb in sign_boxes:
                sb.cx += dx
                sb.cy += dy

        return cls(img=img, sign_boxes=sign_boxes, name=name,
                   avg_width=avg_width, avg_height=avg_height, margin=margin)
    
    def to_detection_list(self) -> Detection:
        """Convert sign_boxes to BoundingBox list for visualizer compatibility."""
        return [sb.to_bounding_box() for sb in self.sign_boxes]
    
    def get_sign_boxes_in_bounds(self, width: float, height: float,
                                  offset_x: float = 0, offset_y: float = 0) -> List[SignBox]:
        """Get sign boxes whose centers fall within specified bounds."""
        result = []
        for sb in self.sign_boxes:
            rel_cx = sb.cx - offset_x
            rel_cy = sb.cy - offset_y
            if 0 <= rel_cx < width and 0 <= rel_cy < height:
                translated = sb.translate(-offset_x, -offset_y)
                result.append(translated)
        return result
    
    def extract_sub_region(self, offset_x: float, offset_y: float,
                           width: float, height: float,
                           img: np.ndarray = None,
                           name: str = "sub_region") -> 'SubTablet':
        """Extract a sub-region with coordinate transformation."""
        sub_sign_boxes = self.get_sign_boxes_in_bounds(width, height, offset_x, offset_y)
        
        return SubTablet(
            img=img, sign_boxes=sub_sign_boxes, name=name,
            avg_width=self.avg_width, avg_height=self.avg_height,
            margin=self.margin, origin_x=offset_x, origin_y=offset_y
        )
    
    def copy(self) -> 'SubTablet':
        """Return a deep copy."""
        return SubTablet(
            img=self.img.copy() if self.img is not None else None,
            sign_boxes=[sb.copy() for sb in self.sign_boxes],
            name=self.name, scale_factor=self.scale_factor,
            avg_width=self.avg_width, avg_height=self.avg_height,
            margin=self.margin, origin_x=self.origin_x, origin_y=self.origin_y
        )
    
    def detect_rows(self, eps: float = 0.6, min_samples: int = 1, 
                    lambda_weight: float = 0.05) -> int:
        """
        Detect rows using DBSCAN and update row_idx for all sign_boxes.
        
        Distance is normalized by average sign size (avg_width, avg_height).
        
        Args:
            eps: Maximum distance threshold as multiple of average sign size (default 0.6)
                 e.g., eps=0.6 means boxes within 0.6*avg_size are considered neighbors
            min_samples: Minimum number of samples in a neighborhood
            lambda_weight: Weight for x-coordinate in distance (default 0.05, emphasizes y)
            
        Returns:
            Number of detected rows (excluding noise)
        """
        from data_processing.line_process import detect_rows_dbscan
        
        if not self.sign_boxes:
            return 0
        
        # Detect rows with normalized distance using avg_width and avg_height
        row_labels, num_rows = detect_rows_dbscan(
            boxes=self.sign_boxes,
            eps=eps,
            min_samples=min_samples,
            lambda_weight=lambda_weight,
            avg_width=self.avg_width,
            avg_height=self.avg_height
        )
        
        # Update row_idx for all sign_boxes
        for sb, row_idx in zip(self.sign_boxes, row_labels):
            sb.row_idx = row_idx
        
        # Assign col_idx based on x-sorted position within each row
        rows_dict: dict = {}
        for sb in self.sign_boxes:
            rows_dict.setdefault(sb.row_idx, []).append(sb)
        for row_boxes in rows_dict.values():
            row_boxes.sort(key=lambda sb: sb.cx)
            for col_idx, sb in enumerate(row_boxes):
                sb.col_idx = col_idx
        
        return num_rows
    
    def get_rows(self) -> List[List[SignBox]]:
        """Group sign boxes by row_idx, return as list of lists."""
        if not self.sign_boxes:
            return []
        
        rows_dict = {}
        for sb in self.sign_boxes:
            rows_dict.setdefault(sb.row_idx, []).append(sb)
        
        for row_idx in rows_dict:
            rows_dict[row_idx].sort(key=lambda sb: sb.col_idx)
        
        return [rows_dict[k] for k in sorted(rows_dict.keys())]
    
    def get_rows_dict(self) -> dict:
        """
        Group sign boxes by row_idx, return as dictionary.
        
        Returns:
            Dictionary mapping row_idx to list of SignBox objects
        """
        if not self.sign_boxes:
            return {}
        
        rows_dict = {}
        for sb in self.sign_boxes:
            rows_dict.setdefault(sb.row_idx, []).append(sb)
        
        for row_idx in rows_dict:
            rows_dict[row_idx].sort(key=lambda sb: sb.col_idx)
        
        return rows_dict
    
    def get_row_sign_sequences(self) -> List[List[str]]:
        """
        Get sign name sequences for each row (sorted by row_idx).
        
        Returns:
            List of rows, each row is a list of sign names
        """
        rows = self.get_rows()
        return [[sb.sign_name for sb in row] for row in rows]

    @property
    def info(self) -> str:
        """Return a string summary of the SubTablet."""
        num_signs = len(self.sign_boxes)
        num_rows = len(self.get_rows())
        img_shape = self.img.shape if self.img is not None else None
        return (f"SubTablet '{self.name}':\n"
                f"  {num_signs} signs, {num_rows} rows, \n"
                f"  image shape: {img_shape}"
                f"\n  avg_width: {self.avg_width:.2f}, avg_height: {self.avg_height:.2f}")
