"""
Geometry utilities for sign alignment.
"""

import numpy as np
from typing import List, Tuple

from .bounding_box import BoundingBox, Detection, GroundTruths


def compute_avg_dimensions(detections: Detection) -> Tuple[float, float]:
    """
    Compute average width and height of detected signs.
    
    Args:
        detections: List of BoundingBox objects
    
    Returns:
        (avg_width, avg_height)
    """
    if not detections:
        return 80.0, 80.0
    
    widths = [det.width for det in detections]
    heights = [det.height for det in detections]
    
    return float(np.mean(widths)), float(np.mean(heights))


def transform_gt_to_cropped_region(gt_boxes: GroundTruths, 
                                    crop_info: dict) -> GroundTruths:
    """
    Transform ground truth boxes from full image to cropped region coordinates.
    
    Args:
        gt_boxes: List of BoundingBox objects in full image coordinates
        crop_info: Dict with 'x', 'y', 'w', 'h' keys for the crop region
    
    Returns:
        List of BoundingBox objects in cropped region coordinates
    """
    from .bounding_box import GroundTruths
    
    if not gt_boxes:
        return []
    
    crop_x = crop_info['x']
    crop_y = crop_info['y']
    crop_w = crop_info['w']
    crop_h = crop_info['h']
    
    transformed = []
    for box in gt_boxes:
        # Check if center falls within crop region
        box_cx = (box.x1 + box.x2) / 2
        box_cy = (box.y1 + box.y2) / 2
        
        if not (crop_x <= box_cx < crop_x + crop_w and crop_y <= box_cy < crop_y + crop_h):
            continue
        
        # Check intersection
        inter_x1 = max(box.x1, crop_x)
        inter_y1 = max(box.y1, crop_y)
        inter_x2 = min(box.x2, crop_x + crop_w)
        inter_y2 = min(box.y2, crop_y + crop_h)
        
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            continue
        
        # Transform to cropped coordinates and clip
        new_x1 = max(0, box.x1 - crop_x)
        new_y1 = max(0, box.y1 - crop_y)
        new_x2 = min(crop_w, box.x2 - crop_x)
        new_y2 = min(crop_h, box.y2 - crop_y)
        
        transformed.append(BoundingBox(
            x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2,
            score=box.score,
            sign=box.sign
        ))
    
    return transformed
