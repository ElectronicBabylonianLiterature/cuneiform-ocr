"""
Line/row detection and processing utilities for sign alignment.
"""

import numpy as np
from typing import List, Tuple


def detect_rows_dbscan(
    boxes: List,
    eps: float = 0.6,
    min_samples: int = 1,
    lambda_weight: float = 0.05,
    avg_width: float = None,
    avg_height: float = None
) -> Tuple[List[int], int]:
    """
    Detect rows using DBSCAN clustering with custom distance metric.
    
    The custom distance metric emphasizes y-coordinate proximity:
    Dist(A, B) = sqrt(lambda * (x_A - x_B)^2 + (y_A - y_B)^2)
    
    Distance is normalized by average sign size to handle different image scales.
    
    Args:
        boxes: List of boxes (BoundingBox, SignBox, or objects with cx, cy attributes/center property)
        eps: Maximum distance threshold as multiple of average sign height (default 0.6)
             e.g., eps=0.6 means boxes within 0.6*avg_height are considered neighbors
        min_samples: Minimum number of samples in a neighborhood for a point to be core
        lambda_weight: Weight for x-coordinate in distance (default 0.05, emphasizes y)
        avg_width: Average sign width for normalization (computed from boxes if not provided)
        avg_height: Average sign height for normalization (computed from boxes if not provided)
        
    Returns:
        Tuple of (row_labels, num_rows) where:
            - row_labels: List of row indices for each box (-1 for noise)
            - num_rows: Number of detected rows (excluding noise)
    """
    from sklearn.cluster import DBSCAN
    
    if not boxes:
        return [], 0
    
    # Extract centers and dimensions from boxes
    centers = []
    widths = []
    heights = []
    
    for box in boxes:
        if hasattr(box, 'cx') and hasattr(box, 'cy'):
            # SignBox or similar
            centers.append([box.cx, box.cy])
            if hasattr(box, 'width') and hasattr(box, 'height'):
                widths.append(box.width)
                heights.append(box.height)
        elif hasattr(box, 'center'):
            # Has center property
            cx, cy = box.center
            centers.append([cx, cy])
            if hasattr(box, 'x1') and hasattr(box, 'x2') and hasattr(box, 'y1') and hasattr(box, 'y2'):
                widths.append(box.x2 - box.x1)
                heights.append(box.y2 - box.y1)
        elif hasattr(box, 'x1') and hasattr(box, 'y1') and hasattr(box, 'x2') and hasattr(box, 'y2'):
            # BoundingBox
            cx = (box.x1 + box.x2) / 2
            cy = (box.y1 + box.y2) / 2
            centers.append([cx, cy])
            widths.append(box.x2 - box.x1)
            heights.append(box.y2 - box.y1)
        else:
            raise ValueError(f"Box type {type(box)} not supported. Must have cx/cy, center, or x1/y1/x2/y2 attributes.")
    
    centers = np.array(centers)
    
    # Compute average dimensions if not provided
    if avg_width is None and widths:
        avg_width = np.mean(widths)
    if avg_height is None and heights:
        avg_height = np.mean(heights)
    
    # Use average of width and height for normalization if both available
    if avg_width is not None and avg_height is not None:
        avg_size = (avg_width + avg_height) / 2
    elif avg_height is not None:
        avg_size = avg_height
    elif avg_width is not None:
        avg_size = avg_width
    else:
        avg_size = 1.0  # Fallback if no size info available
    
    # Normalize coordinates by average size
    normalized_centers = centers / avg_size
    
    # Custom distance matrix with lambda weighting
    # Scale x-coordinates by sqrt(lambda) to get desired distance metric
    scaled_centers = normalized_centers.copy()
    scaled_centers[:, 0] *= np.sqrt(lambda_weight)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(scaled_centers)
    
    # Count number of rows (excluding noise label -1)
    num_rows = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Sort rows by average y-coordinate
    if num_rows > 0:
        row_y_means = {}
        for label in set(labels):
            if label == -1:
                continue
            mask = labels == label
            row_y_means[label] = centers[mask, 1].mean()
        
        # Create mapping from old label to new sorted label
        sorted_labels = sorted(row_y_means.keys(), key=lambda l: row_y_means[l])
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
        label_mapping[-1] = -1  # Keep noise as -1
        
        # Remap labels
        labels = np.array([label_mapping[l] for l in labels])
    
    return labels.tolist(), num_rows
