"""
Heatmap generation and matching utilities for sign alignment.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

from .sign import SignResolver, CLASSES_ABZ
from .bounding_box import BoundingBox, Detection, GroundTruths


def create_2d_gaussian(center_x: float, center_y: float, 
                       width: int, height: int, 
                       sigma_x: float, sigma_y: float,
                       avg_width: float = None, avg_height: float = None, 
                       scale_coefficient: float = None) -> np.ndarray:
    """
    Create a normalized 2D Gaussian centered at (center_x, center_y).
    The peak value decreases with larger sigma to prevent large signs from dominating.
    
    Args:
        center_x, center_y: Center position
        width, height: Heatmap dimensions
        sigma_x, sigma_y: Gaussian standard deviations
        avg_width, avg_height: Average sign dimensions (for auto scaling)
        scale_coefficient: Manual scaling coefficient (overrides auto scaling)
    """
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    
    base_normalization = 1.0 / (2 * np.pi * sigma_x * sigma_y)
    
    if scale_coefficient is None:
        if avg_width is not None and avg_height is not None:
            scale_up = np.pi * avg_width * avg_height * 4
            scale_down = 4.0
            scale_coefficient = scale_up / scale_down
        else:
            scale_coefficient = 1.0
    
    normalization_factor = base_normalization * scale_coefficient
    
    gaussian = normalization_factor * np.exp(-((xx - center_x)**2 / (2 * sigma_x**2) + 
                                                (yy - center_y)**2 / (2 * sigma_y**2)))
    return gaussian


def create_2d_rectangle_blur(center_x: float, center_y: float, 
                             width: int, height: int, 
                             bbox_width: float, bbox_height: float,
                             sigma_blur: float = 2.0) -> np.ndarray:
    """
    Create a rectangular region for a sign, then apply Gaussian blur.
    
    Args:
        center_x, center_y: Center position in heatmap coordinates
        width, height: Heatmap dimensions
        bbox_width, bbox_height: Bounding box dimensions in heatmap coordinates
        sigma_blur: Sigma for Gaussian blur
    """
    heatmap_channel = np.zeros((int(height), int(width)), dtype=np.float32)
    
    x1 = max(0, int(center_x - bbox_width / 2))
    y1 = max(0, int(center_y - bbox_height / 2))
    x2 = min(int(width), int(center_x + bbox_width / 2))
    y2 = min(int(height), int(center_y + bbox_height / 2))
    
    if x1 < x2 and y1 < y2:
        heatmap_channel[y1:y2, x1:x2] = 1.0
    
    kernel_size = int(sigma_blur * 6) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    heatmap_channel = cv2.GaussianBlur(heatmap_channel, (kernel_size, kernel_size), sigma_blur)
    return heatmap_channel


def create_detection_heatmap(detections: Detection, img_shape: tuple, 
                             classes_abz: List[str] = None,
                             scale_factor: int = 10,
                             avg_width: float = None, avg_height: float = None, 
                             method: str = 'gaussian') -> Tuple[np.ndarray, float, float]:
    """
    Create heatmap from BoundingBox detections.
    
    Args:
        detections: List of BoundingBox objects
        img_shape: Original image shape
        classes_abz: List of ABZ class names (defaults to CLASSES_ABZ)
        scale_factor: Scale factor for heatmap
        avg_width, avg_height: Average sign dimensions
        method: 'gaussian' or 'rectangle_blur'
    
    Returns:
        (heatmap, influence_radius, sigma)
    """
    if classes_abz is None:
        classes_abz = CLASSES_ABZ
    
    img_height, img_width = img_shape[:2]
    num_classes = len(classes_abz)
    
    heatmap_height = img_height // scale_factor
    heatmap_width = img_width // scale_factor
    heatmap = np.zeros((heatmap_height, heatmap_width, num_classes), dtype=np.float32)
    
    if avg_width is None or avg_height is None:
        avg_width, avg_height = compute_avg_dimensions(detections)
    
    influence_radius = (avg_width + avg_height) / 2 * 1.5 / scale_factor
    sigma_or_blur = 0
    
    for det in detections:
        # Work with BoundingBox objects
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        abz_name = det.sign.abz
        
        if abz_name not in classes_abz:
            continue
        class_id = classes_abz.index(abz_name)
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        center_x = (x1 + x2) / 2 / scale_factor
        center_y = (y1 + y2) / 2 / scale_factor
        bbox_width_scaled = bbox_width / scale_factor
        bbox_height_scaled = bbox_height / scale_factor
        
        if method == 'gaussian':
            sigma_x = bbox_width * 4 / scale_factor / 3
            sigma_y = bbox_height * 4 / scale_factor / 3
            sigma_or_blur = sigma_x
            response = create_2d_gaussian(center_x, center_y, heatmap_width, heatmap_height,
                                         sigma_x, sigma_y,
                                         avg_width=avg_width / scale_factor,
                                         avg_height=avg_height / scale_factor)
        elif method == 'rectangle_blur':
            sigma_blur = (bbox_width_scaled + bbox_height_scaled) / 6
            sigma_or_blur = sigma_blur
            response = create_2d_rectangle_blur(center_x, center_y,
                                                heatmap_width, heatmap_height,
                                                bbox_width_scaled, bbox_height_scaled,
                                                sigma_blur=sigma_blur)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        heatmap[:, :, class_id] = np.maximum(heatmap[:, :, class_id], response)
    
    return heatmap, influence_radius, sigma_or_blur


def create_text_heatmap(text_lines: List[List[str]], 
                        avg_width: float, avg_height: float,
                        classes_abz: List[str] = None,
                        scale_factor: int = 10, 
                        method: str = 'gaussian') -> Tuple[np.ndarray, float, float, float]:
    """
    Create heatmap from text lines.
    
    Args:
        text_lines: List of text lines (each line is a list of sign names)
        avg_width, avg_height: Average sign dimensions
        classes_abz: List of ABZ class names (defaults to CLASSES_ABZ)
        scale_factor: Scale factor for heatmap
        method: 'gaussian' or 'rectangle_blur'
    
    Returns:
        (heatmap_text, margin, influence_radius, sigma)
    """
    if classes_abz is None:
        classes_abz = CLASSES_ABZ
    
    num_classes = len(classes_abz)
    max_row_length = max(len(line) for line in text_lines) if text_lines else 1
    num_rows = len(text_lines)
    
    margin = max(avg_width, avg_height)
    heatmap_width_text = int(max_row_length * avg_width + 2 * margin)
    heatmap_height_text = int(num_rows * avg_height + 2 * margin)
    
    heatmap_width_scaled = heatmap_width_text // scale_factor
    heatmap_height_scaled = heatmap_height_text // scale_factor
    heatmap_text = np.zeros((heatmap_height_scaled, heatmap_width_scaled, num_classes), dtype=np.float32)
    
    influence_radius = (avg_width + avg_height) / 2 * 4 / scale_factor
    
    if method == 'gaussian':
        sigma_x = avg_width * 4 / scale_factor / 3
        sigma_y = avg_height * 4 / scale_factor / 3
        sigma_or_blur = sigma_x
    elif method == 'rectangle_blur':
        sigma_blur = (avg_width + avg_height) / 4 / scale_factor
        sigma_or_blur = sigma_blur
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Build sign_name to class_id lookup
    sign_name_to_class_id = {}
    for i, abz_name in enumerate(classes_abz):
        sign = SignResolver.from_abz(abz_name)
        sign_name_to_class_id[sign.name] = i
    
    for row_idx, line in enumerate(text_lines):
        for col_idx, sign_name in enumerate(line):
            class_id = sign_name_to_class_id.get(sign_name)
            if class_id is None:
                continue
            
            center_y_orig = margin + row_idx * avg_height + avg_height / 2
            center_x_orig = margin + col_idx * avg_width + avg_width / 2
            center_x_scaled = center_x_orig / scale_factor
            center_y_scaled = center_y_orig / scale_factor
            
            if method == 'gaussian':
                response = create_2d_gaussian(center_x_scaled, center_y_scaled,
                                              heatmap_width_scaled, heatmap_height_scaled,
                                              sigma_x, sigma_y,
                                              avg_width=avg_width / scale_factor,
                                              avg_height=avg_height / scale_factor)
            elif method == 'rectangle_blur':
                bbox_w_scaled = avg_width / scale_factor
                bbox_h_scaled = avg_height / scale_factor
                response = create_2d_rectangle_blur(center_x_scaled, center_y_scaled,
                                                    heatmap_width_scaled, heatmap_height_scaled,
                                                    bbox_w_scaled, bbox_h_scaled,
                                                    sigma_blur=sigma_blur)
            
            heatmap_text[:, :, class_id] = np.maximum(heatmap_text[:, :, class_id], response)
    
    return heatmap_text, margin, influence_radius, sigma_or_blur


def match_heatmaps_ncc(detection_heatmap: np.ndarray, text_heatmap: np.ndarray,
                       scale_factor: int = 10) -> Tuple[tuple, float, tuple]:
    """
    Use normalized cross-correlation to find best match position.
    
    Args:
        detection_heatmap: Heatmap from detections (H, W, C)
        text_heatmap: Heatmap from text (H, W, C)
        scale_factor: Scale factor used
    
    Returns:
        (top_left_scaled, max_correlation, top_left_original)
    """
    num_classes = detection_heatmap.shape[2]
    
    # If detection heatmap is larger than text, return center with low confidence
    if (detection_heatmap.shape[0] > text_heatmap.shape[0] or 
        detection_heatmap.shape[1] > text_heatmap.shape[1]):
        center_y = text_heatmap.shape[0] // 2
        center_x = text_heatmap.shape[1] // 2
        top_left_scaled = (max(0, center_x - detection_heatmap.shape[1] // 2),
                          max(0, center_y - detection_heatmap.shape[0] // 2))
        top_left_original = (top_left_scaled[0] * scale_factor, top_left_scaled[1] * scale_factor)
        return top_left_scaled, 0.1, top_left_original
    
    combined_result = None
    valid_channels = 0
    
    for class_idx in range(num_classes):
        template_channel = detection_heatmap[:, :, class_idx]
        target_channel = text_heatmap[:, :, class_idx]
        
        if template_channel.max() == 0 and target_channel.max() == 0:
            continue
        
        template_norm = cv2.normalize(template_channel, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        target_norm = cv2.normalize(target_channel, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        
        result_channel = cv2.matchTemplate(target_norm, template_norm, cv2.TM_CCORR_NORMED)
        
        if combined_result is None:
            combined_result = result_channel
        else:
            combined_result += result_channel
        valid_channels += 1
    
    if combined_result is None or valid_channels == 0:
        center_y = text_heatmap.shape[0] // 2
        center_x = text_heatmap.shape[1] // 2
        top_left_scaled = (max(0, center_x - detection_heatmap.shape[1] // 2),
                          max(0, center_y - detection_heatmap.shape[0] // 2))
        top_left_original = (top_left_scaled[0] * scale_factor, top_left_scaled[1] * scale_factor)
        return top_left_scaled, 0.1, top_left_original
    
    combined_result = combined_result / valid_channels
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(combined_result)
    top_left_scaled = max_loc
    top_left_original = (top_left_scaled[0] * scale_factor, top_left_scaled[1] * scale_factor)
    
    return top_left_scaled, max_val, top_left_original


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


def group_detections_into_lines(detections: Detection, 
                                 y_threshold: float = 35) -> List[Detection]:
    """
    Group detections into lines based on y-coordinate proximity.
    
    Args:
        detections: List of BoundingBox objects
        y_threshold: Maximum y-distance for same line
    
    Returns:
        List of lines, each containing BoundingBox objects sorted by x
    """
    if not detections:
        return []
    
    sorted_dets = sorted(detections, key=lambda d: (d.y1 + d.y2) / 2)
    
    lines = []
    current_line = [sorted_dets[0]]
    
    for det in sorted_dets[1:]:
        prev_y = (current_line[-1].y1 + current_line[-1].y2) / 2
        curr_y = (det.y1 + det.y2) / 2
        
        if curr_y - prev_y < y_threshold:
            current_line.append(det)
        else:
            current_line = sorted(current_line, key=lambda d: d.x1)
            lines.append(current_line)
            current_line = [det]
    
    if current_line:
        current_line = sorted(current_line, key=lambda d: d.x1)
        lines.append(current_line)
    
    return lines


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


def create_text_based_detections(
    text_lines: list,
    match_position_x: float,
    match_position_y: float,
    margin: float,
    avg_width: float,
    avg_height: float,
    image_bounds: tuple
) -> Detection:
    """
    Create BoundingBox detections from text lines positioned via heatmap matching.
    
    Places signs from text_lines onto the image coordinate system using the
    match position from NCC template matching.
    
    Args:
        text_lines: List of text lines (each line is a list of sign names)
        match_position_x: X of match position in text heatmap (original coords)
        match_position_y: Y of match position in text heatmap (original coords)
        margin: Margin used in text heatmap
        avg_width: Average sign width
        avg_height: Average sign height
        image_bounds: (img_width, img_height)
    
    Returns:
        Detection (List[BoundingBox]) of text-based sign positions  
    """
    img_width, img_height = image_bounds
    detections = []
    
    for row_idx, line in enumerate(text_lines):
        for col_idx, sign_name in enumerate(line):
            # Position in text heatmap coordinates
            center_x_text = margin + col_idx * avg_width + avg_width / 2
            center_y_text = margin + row_idx * avg_height + avg_height / 2
            
            # Transform to image coordinates
            cx = center_x_text - match_position_x
            cy = center_y_text - match_position_y
            
            if not (0 <= cx < img_width and 0 <= cy < img_height):
                continue
            
            sign = SignResolver.resolve(sign_name, expected_type='SIGN')
            if sign is None:
                continue
            
            detections.append(BoundingBox(
                x1=cx - avg_width / 2,
                y1=cy - avg_height / 2,
                x2=cx + avg_width / 2,
                y2=cy + avg_height / 2,
                score=1.0,
                sign=sign
            ))
    
    return detections
