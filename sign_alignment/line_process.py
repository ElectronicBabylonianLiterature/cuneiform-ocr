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


def compute_row_similarity(row1_signs: List[str], row2_signs: List[str], 
                           method: str = 'lcs') -> float:
    """
    Compute similarity between two rows based on sign sequences.
    
    Args:
        row1_signs: List of sign names in row 1
        row2_signs: List of sign names in row 2
        method: 'lcs' for Longest Common Subsequence or 'jaccard' for Jaccard similarity
        
    Returns:
        Similarity score (higher is more similar)
    """
    if not row1_signs or not row2_signs:
        return 0.0
    
    if method == 'lcs':
        # LCS-based similarity
        m, n = len(row1_signs), len(row2_signs)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if row1_signs[i-1] == row2_signs[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        # Normalize by average length
        avg_length = (m + n) / 2
        return lcs_length / avg_length if avg_length > 0 else 0.0
    
    elif method == 'jaccard':
        # Jaccard similarity
        set1 = set(row1_signs)
        set2 = set(row2_signs)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def match_rows_dp(detection_rows: List[List[str]], 
                  text_rows: List[List[str]],
                  skip_text_penalty: float = 0.5,
                  skip_det_penalty: float = 2.0,
                  skip_small_det_penalty: float = 0.1,
                  small_det_threshold: int = 2,
                  similarity_method: str = 'lcs') -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Match detection rows to text rows using DP with free start and free end.
    
    Text sequence (source) has length N, Detection sequence (target) has length M.
    - DP[i][0] = 0 (can skip text rows at start for free - handles top damage)
    - DP[0][j] = j * skip_det_penalty (detected but no text = noise, must penalize)
    
    Transition:
    - DP[i][j] = min(
        DP[i-1][j-1] + cost(i, j),  # Normal match
        DP[i-1][j] + skip_text_penalty,  # Skip text row (middle damage)
        DP[i][j-1] + skip_det_penalty   # Skip detection row (noise)
      )
    
    Small detection rows (with <= small_det_threshold signs) are treated as likely noise
    and use a much lower skip penalty (skip_small_det_penalty) to make them easier to ignore.
    
    Args:
        detection_rows: List of detection rows, each row is list of sign names
        text_rows: List of text rows, each row is list of sign names
        skip_text_penalty: Penalty for skipping a text row (should be low)
        skip_det_penalty: Penalty for skipping a detection row (should be high)
        skip_small_det_penalty: Penalty for skipping detection rows with few signs (very low)
        small_det_threshold: Max number of signs for a row to be considered "small" (default 2)
        similarity_method: Method to compute similarity ('lcs' or 'jaccard')
        
    Returns:
        Tuple of (matches, dp_matrix) where:
            - matches: List of (text_row_idx, det_row_idx) pairs
            - dp_matrix: The full DP matrix for debugging
    """
    N = len(text_rows)  # Text (source)
    M = len(detection_rows)  # Detection (target)
    
    # DP matrix: dp[i][j] = cost to match text[0:i] with det[0:j]
    dp = np.full((N + 1, M + 1), np.inf)
    backtrack = {}  # (i, j) -> (prev_i, prev_j, action)
    
    # Initialize
    dp[0, 0] = 0.0
    
    # Free start: can skip text rows at beginning for free
    for i in range(1, N + 1):
        dp[i, 0] = 0.0
        backtrack[(i, 0)] = (i - 1, 0, 'skip_text')
    
    # Must penalize detection rows with no match (likely noise)
    # Use different penalties based on row size
    for j in range(1, M + 1):
        det_row_size = len(detection_rows[j - 1])
        penalty = skip_small_det_penalty if det_row_size <= small_det_threshold else skip_det_penalty
        dp[0, j] = dp[0, j - 1] + penalty
        backtrack[(0, j)] = (0, j - 1, 'skip_det')
    
    # Fill DP table
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Compute match cost (1 - similarity)
            similarity = compute_row_similarity(
                text_rows[i-1], 
                detection_rows[j-1], 
                method=similarity_method
            )
            match_cost = 1.0 - similarity
            
            # Determine skip detection penalty based on row size
            det_row_size = len(detection_rows[j - 1])
            current_skip_det_penalty = skip_small_det_penalty if det_row_size <= small_det_threshold else skip_det_penalty
            
            # Three options
            options = [
                (dp[i-1, j-1] + match_cost, (i-1, j-1, 'match')),
                (dp[i-1, j] + skip_text_penalty, (i-1, j, 'skip_text')),
                (dp[i, j-1] + current_skip_det_penalty, (i, j-1, 'skip_det'))
            ]
            
            best_cost, best_prev = min(options, key=lambda x: x[0])
            dp[i, j] = best_cost
            backtrack[(i, j)] = best_prev
    
    # Free end: find minimum in last column
    min_cost = np.inf
    best_i = N
    for i in range(N + 1):
        if dp[i, M] < min_cost:
            min_cost = dp[i, M]
            best_i = i
    
    # Backtrack to get matches
    matches = []
    i, j = best_i, M
    
    while j > 0:
        if (i, j) not in backtrack:
            break
        prev_i, prev_j, action = backtrack[(i, j)]
        
        if action == 'match':
            # Match text row (i-1) with detection row (j-1)
            matches.append((i - 1, j - 1))
        
        i, j = prev_i, prev_j
    
    # Reverse to get forward order
    matches.reverse()
    
    return matches, dp


def create_row_mapping(matches: List[Tuple[int, int]], 
                       num_text_rows: int, 
                       num_det_rows: int) -> Tuple[dict, dict]:
    """
    Create bidirectional row mapping from match results.
    
    Args:
        matches: List of (text_row_idx, det_row_idx) pairs
        num_text_rows: Total number of text rows
        num_det_rows: Total number of detection rows
        
    Returns:
        Tuple of (text_to_det, det_to_text) where:
            - text_to_det: {text_row_idx: det_row_idx}
            - det_to_text: {det_row_idx: text_row_idx}
    """
    text_to_det = {}
    det_to_text = {}
    
    for text_idx, det_idx in matches:
        text_to_det[text_idx] = det_idx
        det_to_text[det_idx] = text_idx
    
    return text_to_det, det_to_text
