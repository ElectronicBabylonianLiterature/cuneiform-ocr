from typing import List, Union
from pathlib import Path
import cv2
import numpy as np
from IPython.display import display
from PIL import Image
    

def merge_nearby_boxes(boxes: List[dict], distance_threshold: float, img_width: int = None, img_height: int = None) -> List[dict]:
    """
    Merge nearby bounding boxes
    Only merges small boxes to large boxes or small boxes with each other.
    Large boxes will not merge with each other even if they are close.
    
    Args:
        boxes: List of bounding boxes, each containing x, y, w, h information
        distance_threshold: Distance threshold, boxes closer than this will be merged
        img_width: Image width for determining box size (required)
        img_height: Image height for determining box size (required)
    
    Returns:
        List of merged bounding boxes
    """
    if not boxes:
        return []
    
    if img_width is None or img_height is None:
        raise ValueError("img_width and img_height are required for size-based merging")
    
    # Define thresholds for "large" boxes (1/10 of image dimensions)
    large_width_threshold = img_width / 10
    large_height_threshold = img_height / 10
    
    def is_large_box(box):
        """Check if a box is considered 'large' based on its dimensions"""
        return False    
        return box['w'] > large_width_threshold or box['h'] > large_height_threshold
    
    # Create copy to avoid modifying original data
    boxes = [box.copy() for box in boxes]
    merged = []
    used = set()
    
    for i, box1 in enumerate(boxes):
        if i in used:
            continue
        
        # Check if box1 is large
        box1_is_large = is_large_box(box1)
        
        # Initialize merged box boundaries
        x1_min = box1['x']
        y1_min = box1['y']
        x1_max = box1['x'] + box1['w']
        y1_max = box1['y'] + box1['h']
        
        # Find boxes to merge
        merged_indices = [i]
        
        for j, box2 in enumerate(boxes):
            if j <= i or j in used:
                continue
            
            # Check if box2 is large
            box2_is_large = is_large_box(box2)
            
            # Skip merging if both boxes are large
            if box1_is_large and box2_is_large:
                continue
            
            x2_min = box2['x']
            y2_min = box2['y']
            x2_max = box2['x'] + box2['w']
            y2_max = box2['y'] + box2['h']
            
            # Calculate minimum distance between two boxes
            # Check horizontal and vertical spacing
            horizontal_distance = max(0, max(x1_min - x2_max, x2_min - x1_max))
            vertical_distance = max(0, max(y1_min - y2_max, y2_min - y1_max))
            
            # Use Euclidean distance or Manhattan distance
            distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
            
            # If distance is below threshold, merge boxes
            if distance < distance_threshold:
                x1_min = min(x1_min, x2_min)
                y1_min = min(y1_min, y2_min)
                x1_max = max(x1_max, x2_max)
                y1_max = max(y1_max, y2_max)
                merged_indices.append(j)
        
        # Mark used boxes
        for idx in merged_indices:
            used.add(idx)
        
        # Create merged box
        merged_box = {
            'x': x1_min,
            'y': y1_min,
            'w': x1_max - x1_min,
            'h': y1_max - y1_min,
            'area': (x1_max - x1_min) * (y1_max - y1_min),
            'center_x': (x1_min + x1_max) // 2,
            'center_y': (y1_min + y1_max) // 2,
            'merged_count': len(merged_indices)
        }
        
        merged.append(merged_box)
    
    return merged

def merge_overlapping_boxes(boxes: List[dict]) -> List[dict]:
    """
    Merge overlapping bounding boxes
    
    Args:
        boxes: List of bounding boxes, each containing x, y, w, h information
    
    Returns:
        List of merged bounding boxes with no overlaps
    """
    if not boxes:
        return []
    
    def boxes_overlap(box1, box2):
        """Check if two boxes overlap"""
        x1_min, y1_min = box1['x'], box1['y']
        x1_max, y1_max = box1['x'] + box1['w'], box1['y'] + box1['h']
        x2_min, y2_min = box2['x'], box2['y']
        x2_max, y2_max = box2['x'] + box2['w'], box2['y'] + box2['h']
        
        # Check if boxes overlap
        return not (x1_max <= x2_min or x2_max <= x1_min or 
                   y1_max <= y2_min or y2_max <= y1_min)
    
    # Create copy to avoid modifying original data
    boxes = [box.copy() for box in boxes]
    
    # Keep merging until no more overlaps found
    changed = True
    while changed:
        changed = False
        merged = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            
            # Initialize merged box boundaries
            x1_min = box1['x']
            y1_min = box1['y']
            x1_max = box1['x'] + box1['w']
            y1_max = box1['y'] + box1['h']
            
            merged_indices = [i]
            
            for j, box2 in enumerate(boxes):
                if j <= i or j in used:
                    continue
                
                # Check if boxes overlap
                if boxes_overlap(box1, box2):
                    x2_min = box2['x']
                    y2_min = box2['y']
                    x2_max = box2['x'] + box2['w']
                    y2_max = box2['y'] + box2['h']
                    
                    # Merge boxes
                    x1_min = min(x1_min, x2_min)
                    y1_min = min(y1_min, y2_min)
                    x1_max = max(x1_max, x2_max)
                    y1_max = max(y1_max, y2_max)
                    merged_indices.append(j)
                    changed = True
            
            # Mark used boxes
            for idx in merged_indices:
                used.add(idx)
            
            # Create merged box
            merged_box = {
                'x': x1_min,
                'y': y1_min,
                'w': x1_max - x1_min,
                'h': y1_max - y1_min,
                'area': (x1_max - x1_min) * (y1_max - y1_min),
                'center_x': (x1_min + x1_max) // 2,
                'center_y': (y1_min + y1_max) // 2,
                'merged_count': box1.get('merged_count', 1) + len(merged_indices) - 1
            }
            
            merged.append(merged_box)
        
        boxes = merged
    
    return boxes

def visualize_contours(img: np.ndarray, contours: list, output_path: str = 'visualizations/contours_visualization.jpg'):
    """
    Visualize contours and their bounding rectangles
    
    Args:
        img: Input image
        contours: List of contours to visualize
        output_path: Path to save visualization (default: 'visualizations/contours_visualization.jpg')
    """
    vis_img = img.copy()
    # Draw original contours in green
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
    
    # Draw bounding rectangles in red
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Save visualization
    cv2.imwrite(output_path, vis_img)

def divide_tablet_photo(
    img_or_path: Union[np.ndarray, str, Path],
    min_area_ratio: float = 0.01,
    max_objects: int = 20,
    visualize: bool = False,
    output_path: str = None,
    force_output_result: bool = False,
    logging: bool = True,
    return_coordinates: bool = False
) -> Union[List[np.ndarray], tuple]:
    """
    Divide a photo containing multiple tablet fragments into separate images, sorted top-to-bottom, left-to-right
    
    Args:
        img_or_path: Input image (numpy array) or image path
        min_area_ratio: Minimum contour area ratio relative to the largest contour (default: 0.01)
        max_objects: Maximum number of objects to detect (default: 20)
        visualize: Whether to visualize detection results (default: False)
        output_path: Path to save visualization results (default: None)
        return_coordinates: Whether to return coordinates of cropped regions (default: False)
    
    Returns:
        If return_coordinates is False:
            List[np.ndarray]: List of cropped images sorted from top-to-bottom, left-to-right
        If return_coordinates is True:
            tuple: (List[np.ndarray], List[dict]) where dict contains 'x', 'y', 'w', 'h' of each cropped region
    """
    # Read image
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path))
    elif isinstance(img_or_path, np.ndarray):
        img = img_or_path.copy()
    else:
        raise TypeError('"img_or_path" must be a numpy array, str, or pathlib.Path object')
    
    if img is None:
        raise ValueError(f"Failed to load image from {img_or_path}")
    
    img_height, img_width = img.shape[:2]
    img_area = img_height * img_width
    

    # convert to grayscale and threshold
    merge_threshold = 30

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect background color by sampling top-left corner pixels
    sample_size = 3
    corner_samples = im_gray[:sample_size, :sample_size].flatten()
    max_val = corner_samples.max()
    min_val = corner_samples.min()
    
    # Check if all samples are dark (< 20) - black background
    if max_val <= 20:
        _, bw_img = cv2.threshold(im_gray, 20, 255, cv2.THRESH_BINARY)
    # Check if all samples are bright (within 20 of max value) - white background
    elif min_val >= max_val - 20:
        _, bw_img = cv2.threshold(im_gray, max_val - 20, 255, cv2.THRESH_BINARY_INV)
    # Otherwise use automatic thresholding
    else:
        _, bw_img = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if visualize:
        visualize_contours(img, contours, 'visualizations/contours_visualization.jpg')

    # Calculate area and bounding box for each contour
    contour_data = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        contour_data.append({
            'index': i,
            'contour': contour,
            'area': area,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'center_x': x + w // 2,
            'center_y': y + h // 2
        })

    # Sort by area to find the largest contour
    contour_data.sort(key=lambda x: x['area'], reverse=True)
    
    if not contour_data:
        return []

    # Filter contours: keep those with area larger than threshold
    largest_area = contour_data[0]['area']
    min_area_threshold = img_area * 0.00005  # 0.005% of total image area
    max_area_threshold = img_area * 0.9    # Exclude contours that may be the entire image
    
    filtered_contours = []
    for data in contour_data[:max_objects]:
        if min_area_threshold < data['area'] < max_area_threshold:
            # Filter out overly flat or elongated objects
            aspect_ratio = data['w'] / max(data['h'], 1)
            if 0.01 < aspect_ratio < 100:  # Reasonable aspect ratio
                filtered_contours.append(data)
    
    # also visualize filtered contours
    if visualize:
        filtered_contour_list = [data['contour'] for data in filtered_contours]
        visualize_contours(img, filtered_contour_list, 'visualizations/filtered_contours_visualization.jpg')

    # Merge nearby bounding boxes
    if logging:
        print(f"before: {len(filtered_contours)} boxes, merge threshold: {merge_threshold:.1f} pixels")
    merged_contours = merge_nearby_boxes(filtered_contours, merge_threshold, img_width, img_height)
    if logging:
        print(f"after nearby merge: {len(merged_contours)} boxes")

    # Merge overlapping boxes
    merged_contours = merge_overlapping_boxes(merged_contours)
    if logging:
        print(f"after overlap merge: {len(merged_contours)} boxes")

    # Keep only the largest 12 boxes
    merged_contours.sort(key=lambda x: x['area'], reverse=True)
    merged_contours = merged_contours[:12]
    if logging:
        print(f"after keeping top 12 largest: {len(merged_contours)} boxes")

    # Sort by top-to-bottom, left-to-right order
    # first by y (top to bottom), then by x (left to right)
    # use integer division to group by rows
    tolerance = img_height * 0.1  # 10% of height tolerance

    merged_contours.sort(key=lambda x: (x['center_y'] // int(tolerance), x['center_x']))

    # Crop images
    cropped_images = []
    crop_coordinates = []  # Store coordinates of each crop
    padding = 10  # Padding in pixels

    for data in merged_contours:
        x, y, w, h = data['x'], data['y'], data['w'], data['h']

        # Add padding and ensure it doesn't exceed image boundaries
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_width, x + w + padding)
        y2 = min(img_height, y + h + padding)
        
        cropped_img = img[y1:y2, x1:x2].copy()
        cropped_images.append(cropped_img)
        
        # Store the actual coordinates used for cropping
        crop_coordinates.append({
            'x': x1,
            'y': y1,
            'w': x2 - x1,
            'h': y2 - y1
        })
    
    # Visualize results
    if visualize or force_output_result:
        vis_img = img.copy()
        for i, data in enumerate(merged_contours):
            x, y, w, h = data['x'], data['y'], data['w'], data['h']
            # Draw bounding box
            color = (0, 255, 0) if data.get('merged_count', 1) == 1 else (255, 0, 255)  # Green=single, Magenta=merged
            thickness = 3 if data.get('merged_count', 1) == 1 else 5
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, thickness)
            # Add label
            label = str(i + 1)
            if data.get('merged_count', 1) > 1:
                label += f" ({data['merged_count']})"
            cv2.putText(vis_img, label, (x + 10, y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        if output_path:
            cv2.imwrite(output_path, vis_img)
            print(f"Visualized results saved to: {output_path}")

        # visualize cropped images
    if visualize:
        for i, cropped_img in enumerate(cropped_images):
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            print(f"Cropped Image {i + 1}:")
            display(pil_img)
        

    if return_coordinates:
        return cropped_images, crop_coordinates
    return cropped_images