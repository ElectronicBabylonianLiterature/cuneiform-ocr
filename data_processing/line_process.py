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
        boxes: List of boxes with cx/cy or x1/y1/x2/y2 attributes
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
            # Box or similar
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
            # Corner-based box
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


class _BoxWrapper:
    """Simple wrapper to provide cx/cy/width/height from [x1, y1, x2, y2] tensors or arrays."""

    def __init__(self, bbox):
        self.x1 = float(bbox[0])
        self.y1 = float(bbox[1])
        self.x2 = float(bbox[2])
        self.y2 = float(bbox[3])
        self.cx = (self.x1 + self.x2) / 2
        self.cy = (self.y1 + self.y2) / 2
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1


def line_signs(
    bboxes,
    labels,
    classes: List[str],
    scores=None,
    eps: float = 0.4,
    min_samples: int = 1,
    lambda_weight: float = 0.007,
    return_bboxes: bool = False,
):
    """Group sign detections into text lines using DBSCAN row detection.

    Bounding boxes are clustered into rows by their y-centres (DBSCAN with a
    custom distance that strongly de-emphasises the x-axis via *lambda_weight*).
    Within each row signs are sorted left-to-right by cx.

    Args:
        bboxes: tensor/array of shape (N, 4) with [x1, y1, x2, y2] per bbox.
        labels: label index for each detection (length N).
        classes: list of class name strings indexed by label.
        scores: optional score tensor for each detection (for debug output).
        eps: DBSCAN distance threshold as a multiple of the average sign size.
        min_samples: DBSCAN min_samples parameter.
        lambda_weight: weight for the x-coordinate in the custom distance
            (small value → clustering is almost purely based on y position).
        return_bboxes: if True, also return bounding boxes in the same order as
            the sign tokens in the result string (as a list of [x1,y1,x2,y2] lists).

    Returns:
        str: multi-line string of sign names ordered top-to-bottom, left-to-right.
        If *return_bboxes* is True, returns a tuple (str, list[list]) where the
        second element is the ordered bounding boxes.
    """
    if len(bboxes) == 0:
        return ("", []) if return_bboxes else ""

    boxes = [_BoxWrapper(b) for b in bboxes]
    row_labels, _ = detect_rows_dbscan(
        boxes,
        eps=eps,
        min_samples=min_samples,
        lambda_weight=lambda_weight,
    )

    # Group indices by row label
    rows: dict = {}
    for i, label in enumerate(row_labels):
        rows.setdefault(label, []).append(i)

    # Sort rows top-to-bottom by mean cy; noise row (-1) goes last
    sorted_keys = sorted(
        [k for k in rows if k != -1],
        key=lambda k: float(np.mean([boxes[i].cy for i in rows[k]])),
    )
    if -1 in rows:
        sorted_keys.append(-1)

    result = ""
    ordered_bboxes = []
    for key in sorted_keys:
        row_indices = sorted(rows[key], key=lambda i: boxes[i].cx)
        for i in row_indices:
            if scores is not None:
                result += f"{classes[labels[i]]} {float(scores[i]):.2f} "
            else:
                result += classes[labels[i]] + " "
            if return_bboxes:
                b = boxes[i]
                ordered_bboxes.append([b.x1, b.y1, b.x2, b.y2])
        result += "\n"

    if return_bboxes:
        return result, ordered_bboxes
    return result


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


def match_signs_in_row_dp(
    detection_signs: List[str],
    text_signs: List[str],
    skip_text_penalty: float = 0.5,
    skip_det_penalty: float = 2.0,
    mismatch_cost: float = 0.9
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Match signs within a row using DP with free start and free end.
    
    This is similar to match_rows_dp but operates on individual signs within a row.
    Uses simple class matching: same class = 0.0 cost, different class = mismatch_cost.
    
    Args:
        detection_signs: List of sign names in detection row
        text_signs: List of sign names in text row
        skip_text_penalty: Penalty for skipping a text sign (middle damage)
        skip_det_penalty: Penalty for skipping a detection sign (noise)
        mismatch_cost: Cost when signs don't match exactly (default 0.9, near 1.0)
        
    Returns:
        Tuple of (matches, dp_matrix) where:
            - matches: List of (text_sign_idx, det_sign_idx) pairs
            - dp_matrix: The full DP matrix for debugging
    """
    N = len(text_signs)  # Text (source)
    M = len(detection_signs)  # Detection (target)
    
    # DP matrix: dp[i][j] = cost to match text[0:i] with det[0:j]
    dp = np.full((N + 1, M + 1), np.inf)
    backtrack = {}  # (i, j) -> (prev_i, prev_j, action)
    
    # Initialize
    dp[0, 0] = 0.0
    
    # Free start: can skip text signs at beginning for free
    for i in range(1, N + 1):
        dp[i, 0] = 0.0
        backtrack[(i, 0)] = (i - 1, 0, 'skip_text')
    
    # Must penalize detection signs with no match (likely noise)
    for j in range(1, M + 1):
        dp[0, j] = dp[0, j - 1] + skip_det_penalty
        backtrack[(0, j)] = (0, j - 1, 'skip_det')
    
    # Fill DP table
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Compute match cost: 0.0 if same sign, mismatch_cost otherwise
            match_cost = 0.0 if text_signs[i-1] == detection_signs[j-1] else mismatch_cost
            
            # Three options
            options = [
                (dp[i-1, j-1] + match_cost, (i-1, j-1, 'match')),
                (dp[i-1, j] + skip_text_penalty, (i-1, j, 'skip_text')),
                (dp[i, j-1] + skip_det_penalty, (i, j-1, 'skip_det'))
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
            # Match text sign (i-1) with detection sign (j-1)
            matches.append((i - 1, j - 1))
        
        i, j = prev_i, prev_j
    
    # Reverse to get forward order
    matches.reverse()
    
    return matches, dp


def align_text_row_to_detection(
    text_boxes: List,
    det_boxes: List,
    matches: List[Tuple[int, int]],
    avg_width: float,
    avg_height: float,
    min_width_ratio: float = 2/3,
    max_width_ratio: float = 4/3
) -> List:
    """
    Align text boxes to detection boxes based on matches.
    
    Strategy:
    1. Filter matches to only same-label pairs (anchors)
    2. Anchored signs: use detection box position and size directly
    3. Non-anchored signs: interpolate between neighboring anchors,
       or extrapolate from nearest anchor using avg_width spacing
    
    Args:
        text_boxes: List of text Box objects (source)
        det_boxes: List of detection Box objects (target)
        matches: List of (text_idx, det_idx) pairs from match_signs_in_row_dp
        avg_width: Average sign width
        avg_height: Average sign height
        min_width_ratio: Minimum width as ratio of avg_width (default 2/3)
        max_width_ratio: Maximum width as ratio of avg_width (default 4/3)
        
    Returns:
        List of aligned Box objects with updated positions
    """
    from sign_alignment.box import Box
    
    if not det_boxes:
        return text_boxes
    
    num_text = len(text_boxes)
    
    # === Step 1: Filter to same-label matches only (anchors) ===
    anchors = []  # list of (text_idx, det_idx) where labels match
    for text_idx, det_idx in matches:
        if text_boxes[text_idx].sign_name == det_boxes[det_idx].sign_name:
            anchors.append((text_idx, det_idx))
    
    # Build anchor lookup: text_idx → det_box
    anchor_map = {}  # text_idx → det_box
    for text_idx, det_idx in anchors:
        anchor_map[text_idx] = det_boxes[det_idx]
    
    # Sorted anchor text indices for interpolation
    anchor_text_indices = sorted(anchor_map.keys())
    
    # === Step 2: Compute baseline from anchor detection boxes ===
    if len(anchor_text_indices) >= 2:
        anchor_cx = np.array([anchor_map[ti].cx for ti in anchor_text_indices])
        anchor_cy = np.array([anchor_map[ti].cy for ti in anchor_text_indices])
        slope, intercept = np.polyfit(anchor_cx, anchor_cy, 1)
    elif len(anchor_text_indices) == 1:
        slope = 0.0
        intercept = anchor_map[anchor_text_indices[0]].cy
    else:
        # No anchors: fallback to all detection signs baseline
        det_cx = np.array([box.cx for box in det_boxes])
        det_cy = np.array([box.cy for box in det_boxes])
        if len(det_cx) >= 2:
            slope, intercept = np.polyfit(det_cx, det_cy, 1)
        else:
            slope = 0.0
            intercept = det_cy[0] if len(det_cy) > 0 else 0.0
    
    def baseline_y(x: float) -> float:
        return slope * x + intercept
    
    # === Step 3: Assign position to each text sign ===
    aligned_boxes = []
    
    for text_idx in range(num_text):
        text_box = text_boxes[text_idx]
        
        if text_idx in anchor_map:
            # --- Anchored: use detection box directly ---
            det_box = anchor_map[text_idx]
            new_cx = det_box.cx
            new_cy = det_box.cy
            new_width = det_box.width
            new_height = det_box.height
        else:
            # --- Non-anchored: interpolate or extrapolate ---
            # Find left and right nearest anchors
            left_anchor = None   # largest anchor index < text_idx
            right_anchor = None  # smallest anchor index > text_idx
            
            for ai in anchor_text_indices:
                if ai < text_idx:
                    left_anchor = ai
                elif ai > text_idx and right_anchor is None:
                    right_anchor = ai
            
            if left_anchor is not None and right_anchor is not None:
                # Interpolate between two anchors
                left_det = anchor_map[left_anchor]
                right_det = anchor_map[right_anchor]
                t = (text_idx - left_anchor) / (right_anchor - left_anchor)
                new_cx = left_det.cx + t * (right_det.cx - left_det.cx)
                new_cy = left_det.cy + t * (right_det.cy - left_det.cy)
            elif left_anchor is not None:
                # Extrapolate rightward from left anchor
                left_det = anchor_map[left_anchor]
                offset = (text_idx - left_anchor) * avg_width
                new_cx = left_det.cx + offset
                new_cy = baseline_y(new_cx)
            elif right_anchor is not None:
                # Extrapolate leftward from right anchor
                right_det = anchor_map[right_anchor]
                offset = (text_idx - right_anchor) * avg_width  # negative
                new_cx = right_det.cx + offset
                new_cy = baseline_y(new_cx)
            else:
                # No anchors at all: use detection centroid + offset
                det_centroid_x = np.mean([b.cx for b in det_boxes])
                text_centroid_idx = (num_text - 1) / 2.0
                offset = (text_idx - text_centroid_idx) * avg_width
                new_cx = det_centroid_x + offset
                new_cy = baseline_y(new_cx)
            
            new_width = avg_width
            new_height = avg_height
        
        aligned_box = Box.from_center(
            sign=text_box.sign,
            score=text_box.score,
            cx=new_cx,
            cy=new_cy,
            width=new_width,
            height=new_height,
            tablet=text_box.tablet,
        )
        aligned_boxes.append(aligned_box)
    
    return aligned_boxes


def align_text_to_detection_rows(
    det_rows: dict,
    text_rows: dict,
    text_to_det: dict,
    row_sign_matches: dict,
    avg_width: float,
    avg_height: float,
    min_width_ratio: float = 2/3,
    max_width_ratio: float = 4/3
) -> List:
    """
    Align all text rows to their matched detection rows using baseline with slope.
    
    This is a high-level function that processes all matched row pairs and returns
    aligned sign boxes. Only text rows that have matching detection rows are processed.
    
    Args:
        det_rows: Dictionary mapping detection row_idx -> List[Box]
        text_rows: Dictionary mapping text row_idx -> List[Box]
        text_to_det: Dictionary mapping text row_idx -> detection row_idx
        row_sign_matches: Dictionary mapping text row_idx -> List[(text_sign_idx, det_sign_idx)]
        avg_width: Average sign width
        avg_height: Average sign height
        min_width_ratio: Minimum width scaling ratio (default 2/3)
        max_width_ratio: Maximum width scaling ratio (default 4/3)
    
    Returns:
        List of aligned Box objects for all matched rows
    """
    aligned_text_boxes = []
    
    # Process each matched text row
    for text_row_idx in sorted(row_sign_matches.keys()):
        if text_row_idx not in text_to_det:
            continue
        
        det_row_idx = text_to_det[text_row_idx]
        sign_matches = row_sign_matches[text_row_idx]
        
        # Get sign boxes for this row
        text_row_boxes = text_rows.get(text_row_idx, [])
        det_row_boxes = det_rows.get(det_row_idx, [])
        
        if not text_row_boxes or not det_row_boxes:
            continue
        
        # Align this row using baseline with slope
        aligned_row_boxes = align_text_row_to_detection(
            text_boxes=text_row_boxes,
            det_boxes=det_row_boxes,
            matches=sign_matches,
            avg_width=avg_width,
            avg_height=avg_height,
            min_width_ratio=min_width_ratio,
            max_width_ratio=max_width_ratio
        )
        
        aligned_text_boxes.extend(aligned_row_boxes)
    
    return aligned_text_boxes
