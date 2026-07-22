"""
Evaluation script for Cuneiform Signs Alignment (PSR method).

Computes object-detection-style metrics (mAP, IoU, Precision, Recall)
by comparing PSR-optimized alignment bounding boxes against ground-truth annotations.

Also includes a fast coordinate-wise hyperparameter sweep for the
PointSetRegistrationOptimizer.

Uses the step functions from sign_alignment/pipeline.py for the alignment
pipeline.
"""

import json
import os
import numpy as np
import cv2
from enum import Enum
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from PIL import Image as PILImage, ImageDraw, ImageFont
from dotenv import load_dotenv

from sign_alignment import (
    LocalTestDataSource,
    EBLAPISource,
    ModelConfig, TabletImageDetector,
    Box, Boxes,
    hyperparameter_search,
)
from sign_alignment.visualizer import ColorConfig
from sign_alignment.pipeline import (
    CropContext, Runner, SampleState, Step, VisOptions,
    align_text_rows,
    build_sign_match_info,
    create_box_sets,
    create_psr_optimizer,
    create_result_without_optimization,
    detect_rows,
    detect_signs,
    load_data,
    match_rows,
    match_signs_in_rows,
    optimize_psr,
    transform_gt_to_crop,
    vis_aligned_rows,
    vis_box_sets,
    vis_crop_ground_truth,
    vis_detected_rows_info,
    vis_detection_statistics,
    vis_detections,
    vis_loaded_data,
    vis_optimization,
    vis_psr_optimizer,
    vis_row_matches,
    vis_sign_match_info,
    vis_sign_matches,
)

load_dotenv()

# ============ Configuration ============
COCO_TEST_DIR = os.path.expanduser("~/erc-work-data/ready-for-training/coco-recognition-2025-09/data/coco")
CONFIG_FILE = "configs/detr.py"
CHECKPOINT_FILE = os.path.expanduser("~/erc-work-data/retrained_models/detr-173/epoch_1000.pth")
SCORE_THRESHOLD = 0.1
EVAL_OUTPUT_DIR = "evaluation_results"

# Number of fragments to evaluate / search
EVAL_SAMPLE_LIMIT = 100
SEARCH_SAMPLE_LIMIT = 5

# IoU thresholds for mAP computation
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# --- Default PSR optimizer hyperparameters (used by hyperparameter sweep) ---
DEFAULT_PSR_PARAMS = dict(
    sigma_factor=1.5,
    w_noise=0.1,
    lambda_data=2.0,
    lambda_anchor=0.01,
    lambda_seq=0.1,
    lambda_height=0.01,
    lambda_rows=5.0,
    lambda_boundary=1.0,
    rows_threshold_ratio_far=1 / 3.0,
    rows_threshold_ratio_close=2 / 3.0,
    rows_plateau_far=0.5,
    rows_plateau_close=1.0,
    num_iterations=150,
    lr=1.0,
    sigma_anneal=True,
)

# Faster settings for hyperparameter sweep
SEARCH_PSR_PARAMS = dict(DEFAULT_PSR_PARAMS)
SEARCH_PSR_PARAMS['num_iterations'] = 40


# ============ Prediction Mode ============

class PredictionMode(str, Enum):
    """Controls how bounding box predictions are produced during evaluation."""
    PSR = "psr"                      # Detection + text alignment + PSR optimization
    WITHOUT_PSR = "without_psr"      # Detection + text alignment, preserving detection geometry
    DETECTION = "detection"          # Raw detection model output only

# Default prediction mode used by run_evaluation
DEFAULT_PREDICTION_MODE = PredictionMode.WITHOUT_PSR


# ============ IoU & Metrics ============

def compute_iou(box_a: Box, box_b: Box) -> float:
    """Compute IoU between two Box objects."""
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = box_a.width * box_a.height
    area_b = box_b.width * box_b.height
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_iou_matrix(preds: Boxes, gts: Boxes) -> np.ndarray:
    """Compute IoU matrix of shape (len(preds), len(gts))."""
    n_pred = len(preds)
    n_gt = len(gts)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_matrix[i, j] = compute_iou(p, g)
    return iou_matrix


def match_predictions_to_gt(
    preds: Boxes,
    gts: Boxes,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> Dict:
    """
    Match predictions to ground truth boxes at a given IoU threshold.
    Each GT box is matched to at most one prediction (greedy, highest IoU first).

    Args:
        preds: Predicted Box list
        gts: Ground truth Box list
        iou_threshold: IoU threshold for a match
        class_agnostic: If False, pred and GT must share the same sign name

    Returns:
        dict with keys: tp, fp, fn, matched_ious, matched_pairs
    """
    if not preds or not gts:
        return {
            'tp': 0,
            'fp': len(preds),
            'fn': len(gts),
            'matched_ious': [],
            'matched_pairs': [],
        }

    iou_matrix = compute_iou_matrix(preds, gts)

    gt_matched = [False] * len(gts)
    pred_matched = [False] * len(preds)
    matched_ious = []
    matched_pairs = []

    # Greedy matching: sort all (pred, gt) pairs by IoU descending
    pairs = []
    for i in range(len(preds)):
        for j in range(len(gts)):
            pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(key=lambda x: -x[0])

    for iou_val, pi, gj in pairs:
        if iou_val < iou_threshold:
            break
        if pred_matched[pi] or gt_matched[gj]:
            continue
        # Class check
        if not class_agnostic:
            if preds[pi].sign.name != gts[gj].sign.name:
                continue
        pred_matched[pi] = True
        gt_matched[gj] = True
        matched_ious.append(iou_val)
        matched_pairs.append((pi, gj))

    tp = len(matched_pairs)
    fp = len(preds) - tp
    fn = len(gts) - sum(gt_matched)

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'matched_ious': matched_ious,
        'matched_pairs': matched_pairs,
    }


def compute_precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def _compute_coco_class_ap(
    preds_sorted: List[Tuple[str, 'Box']],
    gts_by_image: Dict[str, List['Box']],
    total_gt: int,
    iou_threshold: float,
) -> Tuple[float, int, int, int, List[float]]:
    """
    COCO-style 101-point interpolated AP for a single class at a single IoU threshold.

    Args:
        preds_sorted: [(fid, box), ...] sorted by score descending.
        gts_by_image: {fid: [gt_boxes_for_this_class]}.
        total_gt: Total number of GT boxes for this class (across all images).
        iou_threshold: IoU threshold for a match.

    Returns:
        (ap, tp, fp, fn, matched_ious)
    """
    if total_gt == 0:
        return 0.0, 0, len(preds_sorted), 0, []

    # Track which GT boxes have been matched (per image)
    gt_matched = {fid: [False] * len(gts) for fid, gts in gts_by_image.items()}

    is_tp = np.zeros(len(preds_sorted), dtype=np.float64)
    is_fp = np.zeros(len(preds_sorted), dtype=np.float64)
    matched_ious: List[float] = []

    for i, (fid, pred_box) in enumerate(preds_sorted):
        img_gts = gts_by_image.get(fid, [])
        if not img_gts:
            is_fp[i] = 1.0
            continue

        # Find best unmatched GT for this pred in the same image
        best_iou = -1.0
        best_j = -1
        for j, gt_box in enumerate(img_gts):
            if gt_matched[fid][j]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            gt_matched[fid][best_j] = True
            is_tp[i] = 1.0
            matched_ious.append(best_iou)
        else:
            is_fp[i] = 1.0

    # Build cumulative precision / recall arrays
    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(is_fp)
    precision_curve = cum_tp / (cum_tp + cum_fp)
    recall_curve = cum_tp / total_gt

    # 101-point interpolated AP (COCO standard)
    ap = 0.0
    for r_thresh in np.linspace(0.0, 1.0, 101):
        mask = recall_curve >= r_thresh
        if mask.any():
            ap += float(precision_curve[mask].max())
    ap /= 101.0

    tp = int(is_tp.sum())
    fp = int(is_fp.sum())
    fn = total_gt - tp
    return ap, tp, fp, fn, matched_ious


def compute_coco_map(
    all_results: List[Dict],
    iou_thresholds: List[float] = IOU_THRESHOLDS,
    max_dets: Optional[int] = None,
) -> Dict:
    """
    COCO-style mAP: per-class score-sorted 101-point interpolated AP,
    macro-averaged across classes, then averaged across IoU thresholds.

    Args:
        all_results: List of dicts with 'fragment_id', 'preds', 'gts'.
        iou_thresholds: IoU thresholds to evaluate at.
        max_dets: If set, cap predictions per class at this many (highest score first).
                  Default None = no cap (COCO default is 100; pass 100 to compare).

    Returns:
        Dict with mAP, AP@0.5, AP@0.75, AR, per_threshold details, and per_class stats.
    """
    # Organise data by class
    preds_by_class: Dict[str, List] = defaultdict(list)           # cls -> [(score, fid, box)]
    gts_by_class_image: Dict[str, Dict[str, List]] = defaultdict(
        lambda: defaultdict(list)
    )                                                              # cls -> {fid -> [boxes]}

    for r in all_results:
        fid = r['fragment_id']
        for box in r['gts']:
            gts_by_class_image[box.sign.name][fid].append(box)
        for box in r['preds']:
            preds_by_class[box.sign.name].append((box.score, fid, box))

    # Sort predictions by score descending; apply optional maxDets cap
    for cls in preds_by_class:
        preds_by_class[cls].sort(key=lambda x: -x[0])
        if max_dets is not None:
            preds_by_class[cls] = preds_by_class[cls][:max_dets]

    # Only classes that appear in the GT contribute to the macro average
    all_classes = sorted(gts_by_class_image.keys())

    per_threshold = []
    per_class_at_50: Dict[str, Dict] = {}

    for thresh in iou_thresholds:
        class_aps: List[float] = []
        class_recalls: List[float] = []   # per-class recall for COCO AR
        total_tp_all = 0
        total_fp_all = 0
        total_fn_all = 0
        all_matched_ious: List[float] = []

        for cls in all_classes:
            gts_by_image = dict(gts_by_class_image[cls])
            total_gt = sum(len(v) for v in gts_by_image.values())
            preds_sorted = [(fid, box) for _, fid, box in preds_by_class.get(cls, [])]

            ap, tp, fp, fn, m_ious = _compute_coco_class_ap(
                preds_sorted, gts_by_image, total_gt, thresh
            )
            class_aps.append(ap)
            # COCO AR uses per-class recall, macro-averaged (same denominator as AP)
            recall_cls = tp / total_gt if total_gt > 0 else 0.0
            class_recalls.append(recall_cls)
            total_tp_all += tp
            total_fp_all += fp
            total_fn_all += fn
            all_matched_ious.extend(m_ious)

            # Collect per-class breakdown at IoU=0.5
            if thresh == 0.5:
                p_cls, r_cls = compute_precision_recall(tp, fp, fn)
                f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls) if (p_cls + r_cls) > 0 else 0.0
                per_class_at_50[cls] = {
                    'ap': ap,
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'precision': p_cls,
                    'recall': r_cls,
                    'f1': f1_cls,
                }

        macro_ap = float(np.mean(class_aps)) if class_aps else 0.0
        macro_recall = float(np.mean(class_recalls)) if class_recalls else 0.0
        precision, recall = compute_precision_recall(total_tp_all, total_fp_all, total_fn_all)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_threshold.append({
            'iou_threshold': thresh,
            'ap': macro_ap,
            'macro_recall': macro_recall,   # per-class macro avg recall at this threshold
            'tp': total_tp_all,
            'fp': total_fp_all,
            'fn': total_fn_all,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_matched_iou': float(np.mean(all_matched_ious)) if all_matched_ious else 0.0,
        })

    aps = [r['ap'] for r in per_threshold]
    mAP = float(np.mean(aps))
    ap50 = per_threshold[0]['ap']
    ap75 = per_threshold[5]['ap'] if len(per_threshold) > 5 else 0.0
    # COCO AR: macro-averaged recall, averaged across all IoU thresholds
    AR = float(np.mean([r['macro_recall'] for r in per_threshold]))

    return {
        'mAP': mAP,
        'AP@0.5': ap50,
        'AP@0.75': ap75,
        'AR': AR,
        'per_threshold': per_threshold,
        'per_class': per_class_at_50,
        'num_classes': len(all_classes),
    }


# ============ Evaluation Visualization ============

def _get_eval_font(size: int):
    """Load font for evaluation annotations."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def visualize_evaluation_fragment(
    img: np.ndarray,
    fragment_id: str,
    preds: Boxes,
    gts: Boxes,
    iou_threshold: float = 0.5,
    class_agnostic: bool = True,
    output_dir: str = EVAL_OUTPUT_DIR,
):
    """
    Visualize evaluation results for a single fragment.

    Draws semi-transparent filled boxes:
      - Green:   matched GT and pred boxes (TP)
      - Orange:  false positive pred boxes (FP)
      - Magenta: missed GT boxes (FN)

    Annotates per-image evaluation metrics at the top of the image.
    """
    match_result = match_predictions_to_gt(
        preds, gts, iou_threshold=iou_threshold, class_agnostic=class_agnostic
    )

    tp = match_result['tp']
    fp = match_result['fp']
    fn = match_result['fn']
    precision, recall = compute_precision_recall(tp, fp, fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(match_result['matched_ious'])) if match_result['matched_ious'] else 0.0

    matched_pred_idxs = {pi for pi, gj in match_result['matched_pairs']}
    matched_gt_idxs = {gj for pi, gj in match_result['matched_pairs']}

    # Colors in BGR
    GREEN = (0, 200, 0)
    ORANGE = (0, 165, 255)
    MAGENTA = (255, 0, 255)
    alpha = 0.3

    # --- Step 1: Draw semi-transparent fills on overlay ---
    overlay = img.copy()

    for gj in matched_gt_idxs:
        box = gts[gj]
        cv2.rectangle(overlay, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), GREEN, -1)
    for pi in matched_pred_idxs:
        box = preds[pi]
        cv2.rectangle(overlay, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), GREEN, -1)
    for pi, box in enumerate(preds):
        if pi not in matched_pred_idxs:
            cv2.rectangle(overlay, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), ORANGE, -1)
    for gj, box in enumerate(gts):
        if gj not in matched_gt_idxs:
            cv2.rectangle(overlay, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), MAGENTA, -1)

    # --- Step 2: Blend ---
    vis_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # --- Step 3: Draw crisp outlines on top ---
    for gj in matched_gt_idxs:
        box = gts[gj]
        cv2.rectangle(vis_img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), GREEN, 2)
    for pi in matched_pred_idxs:
        box = preds[pi]
        cv2.rectangle(vis_img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), GREEN, 2)
    for pi, box in enumerate(preds):
        if pi not in matched_pred_idxs:
            cv2.rectangle(vis_img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), ORANGE, 2)
    for gj, box in enumerate(gts):
        if gj not in matched_gt_idxs:
            cv2.rectangle(vis_img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), MAGENTA, 2)

    # --- Step 4: Build text banner ---
    h, w = vis_img.shape[:2]
    font_scale = max(w / 1200.0, 0.8)
    font_title_size = int(22 * font_scale)
    font_text_size = int(18 * font_scale)
    font_legend_size = int(16 * font_scale)
    line_h = int(26 * font_scale)
    banner_height = line_h * 4 + 10

    banner = np.ones((banner_height, w, 3), dtype=np.uint8) * 255
    result_img = np.vstack([banner, vis_img])

    # Render text with PIL (Unicode-safe)
    result_pil = PILImage.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(result_pil)
    font_title = _get_eval_font(font_title_size)
    font_text = _get_eval_font(font_text_size)
    font_legend = _get_eval_font(font_legend_size)

    y = 5
    draw.text((10, y), f"Fragment: {fragment_id}    IoU threshold: {iou_threshold}",
              font=font_title, fill=(0, 0, 0))
    y += line_h
    draw.text((10, y),
              f"GT: {len(gts)}    Pred: {len(preds)}    TP: {tp}    FP: {fp}    FN: {fn}",
              font=font_text, fill=(0, 0, 0))
    y += line_h
    draw.text((10, y),
              f"Precision: {precision:.4f}    Recall: {recall:.4f}    "
              f"F1: {f1:.4f}    Mean IoU: {mean_iou:.4f}",
              font=font_text, fill=(0, 0, 0))
    y += line_h

    # Legend
    sq = int(14 * font_scale)
    gap = int(10 * font_scale)
    lx = 10
    draw.rectangle([lx, y, lx + sq, y + sq], fill=(0, 200, 0))
    draw.text((lx + sq + 4, y - 2), "Match (TP)", font=font_legend, fill=(0, 0, 0))
    lx += int(120 * font_scale)
    draw.rectangle([lx, y, lx + sq, y + sq], fill=(255, 165, 0))
    draw.text((lx + sq + 4, y - 2), "FP (pred)", font=font_legend, fill=(0, 0, 0))
    lx += int(120 * font_scale)
    draw.rectangle([lx, y, lx + sq, y + sq], fill=(255, 0, 255))
    draw.text((lx + sq + 4, y - 2), "Miss (GT)", font=font_legend, fill=(0, 0, 0))

    result_img = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{fragment_id}_eval_IoU{int(iou_threshold * 100)}.jpg")
    cv2.imwrite(save_path, result_img)
    print(f"  Saved evaluation visualization: {save_path}")


# ============ Prediction & Evaluation Runner ============

def _nms_predictions(
    preds: List[Box],
    iou_threshold: float = 0.5,
) -> List[Box]:
    """
    Per-class non-maximum suppression.

    Suppresses duplicate predictions (e.g. from overlapping crops) that share
    the same class and have IoU >= iou_threshold. Keeps the higher-score box.
    """
    by_class: Dict[str, List[Box]] = defaultdict(list)
    for p in preds:
        by_class[p.sign.name].append(p)

    kept: List[Box] = []
    for boxes in by_class.values():
        boxes = sorted(boxes, key=lambda b: -b.score)
        suppressed = [False] * len(boxes)
        for i, bi in enumerate(boxes):
            if suppressed[i]:
                continue
            kept.append(bi)
            for j in range(i + 1, len(boxes)):
                if not suppressed[j] and compute_iou(bi, boxes[j]) >= iou_threshold:
                    suppressed[j] = True
    return kept


def _load_and_detect_fragment(runner: Runner, context: CropContext, fid: str) -> bool:
    """
    Load a fragment and run sign detection.

    Sets up context.state in place. Returns False if the fragment should be
    skipped (missing data, empty GT/detections, etc.).
    """
    context.state = SampleState()
    context.state.fragment_id = fid

    runner.run([Step("Load data", load_data, vis_loaded_data)])

    s = context.state
    if s.tablet is None or not s.text_lines or not s.gt_boxes:
        return False

    # Filter out abnormally large GT boxes (e.g. sub-tablet region annotations)
    areas = [b.width * b.height for b in s.gt_boxes]
    mean_area = np.mean(areas)
    s.gt_boxes = Boxes((b for b, a in zip(s.gt_boxes, areas) if a <= mean_area * 5),
                       tablet=s.gt_boxes.tablet)
    if not s.gt_boxes:
        return False

    runner.run([
        Step("Detect signs", detect_signs, vis_detections),
        Step("Detection statistics", lambda _: None, vis_detection_statistics),
    ])
    if not s.detections:
        return False

    return True


def _predict_detection_crops(
    runner: Runner,
    context: CropContext,
    fid: str,
    visualize: bool = False,
    output_dir: str = EVAL_OUTPUT_DIR,
) -> List[Box]:
    """
    Collect raw detection model outputs across all crops, offset to full image coordinates.

    Requires _load_and_detect_fragment() to have been called first.
    Optionally saves a per-fragment evaluation visualization.
    """
    s = context.state
    raw_preds: List[Box] = []
    for crop_idx in range(len(context.tablet_detector.get_crop_tablets())):
        runner.choose_crop(crop_idx)
        if not s.det_boxes:
            continue
        for det in s.det_boxes:
            raw_preds.append(det.to_tablet(s.tablet))

    # Suppress cross-crop duplicates: the same sign detected in overlapping crops
    # appears multiple times after offsetting; all but the highest-score one are FP.
    preds = _nms_predictions(raw_preds, iou_threshold=0.5)

    if visualize and s.tablet is not None:
        visualize_evaluation_fragment(
            img=s.tablet.img,
            fragment_id=fid,
            preds=preds,
            gts=s.gt_boxes,
            iou_threshold=0.5,
            class_agnostic=False,
            output_dir=output_dir,
        )
    return preds


def _run_without_psr_alignment_steps(runner: Runner) -> None:
    """Run the notebook alignment flow through result_without_optimization.

    Both PSR and WITHOUT_PSR use this shared prefix so row detection and
    matching stay in sync.  In particular, ``detect_rows`` currently uses the
    Hough-based row detector from ``sign_alignment.pipeline``.
    """
    runner.run([
        Step("Transform GT to crop", transform_gt_to_crop, vis_crop_ground_truth),
        Step("Create box sets", create_box_sets, vis_box_sets),
        Step("Detect rows (Hough)", detect_rows, vis_detected_rows_info),
        Step("Match rows", match_rows, vis_row_matches),
        Step("Match signs", match_signs_in_rows, vis_sign_matches),
        Step("Align text rows", align_text_rows, vis_aligned_rows),
        Step(
            "Result without PSR",
            create_result_without_optimization,
        ),
    ])


def _predict_without_psr_crops(
    runner: Runner,
    context: CropContext,
    fid: str,
    visualize: bool = False,
    output_dir: str = EVAL_OUTPUT_DIR,
) -> List[Box]:
    """Return relabelled detections without changing their geometry or score."""
    s = context.state
    preds: List[Box] = []
    for crop_idx in range(len(context.tablet_detector.get_crop_tablets())):
        runner.choose_crop(crop_idx)
        if not s.det_boxes:
            continue

        _run_without_psr_alignment_steps(runner)
        if s.result_without_optimization_boxes is None:
            continue

        # create_result_without_optimization starts from s.det_boxes.copy(),
        # so detector confidence scores (already filtered by SCORE_THRESHOLD)
        # are preserved here.
        for box in s.result_without_optimization_boxes:
            preds.append(box.to_tablet(s.tablet))

    if visualize and s.tablet is not None:
        visualize_evaluation_fragment(
            img=s.tablet.img,
            fragment_id=fid,
            preds=preds,
            gts=s.gt_boxes,
            iou_threshold=0.5,
            class_agnostic=False,
            output_dir=output_dir,
        )
    return preds


def _predict_psr_crops(
    runner: Runner,
    context: CropContext,
    fid: str,
    visualize: bool = False,
    output_dir: str = EVAL_OUTPUT_DIR,
) -> List[Box]:
    """
    Run the full PSR alignment pipeline across all crops and return optimized boxes
    offset to full image coordinates.

    Requires _load_and_detect_fragment() to have been called first.
    Optionally saves a per-fragment evaluation visualization.
    """
    s = context.state
    preds: List[Box] = []
    for crop_idx in range(len(context.tablet_detector.get_crop_tablets())):
        runner.choose_crop(crop_idx)
        if not s.det_boxes:
            continue
        _run_without_psr_alignment_steps(runner)
        if not s.det_rows or not len(s.det_rows) or not s.matches or not s.aligned_boxes:
            continue

        runner.run([
            Step("Build sign match info", build_sign_match_info, vis_sign_match_info),
            Step("Create PSR optimizer", create_psr_optimizer, vis_psr_optimizer),
            Step("Optimize PSR", optimize_psr, vis_optimization),
        ])
        if not s.final_boxes:
            continue

        for sb in s.final_boxes:
            preds.append(sb.to_tablet(s.tablet))

    if visualize and s.tablet is not None:
        visualize_evaluation_fragment(
            img=s.tablet.img,
            fragment_id=fid,
            preds=preds,
            gts=s.gt_boxes,
            iou_threshold=0.5,
            class_agnostic=False,
            output_dir=output_dir,
        )
    return preds


def run_predictions(
    context: CropContext,
    fragment_ids: List[str],
    psr_params: Optional[dict] = None,
    verbose: bool = True,
    label: str = "",
    prediction_mode: PredictionMode = DEFAULT_PREDICTION_MODE,
    visualize: bool = False,
    output_dir: str = EVAL_OUTPUT_DIR,
) -> Tuple[List[Dict], int]:
    """
    Run predictions on a list of fragments and return raw results.

    Args:
        context: Pipeline context.
        fragment_ids: List of fragment IDs to process.
        psr_params: PSR optimizer hyperparameters (only used for PSR mode).
        verbose: Print progress messages.
        label: Label for progress output.
        prediction_mode: How to produce predictions (PSR, without PSR, or raw detection).
        visualize: Save per-fragment evaluation visualizations.
        output_dir: Directory for visualization outputs.

    Returns:
        Tuple of (all_results, skipped) where all_results is a list of dicts with
        keys 'fragment_id', 'preds' (List[Box]), 'gts' (List[Box]).
    """
    context.psr_params = psr_params  # None = step defaults
    vis = VisOptions(info=False, display=False, save=False)
    runner = Runner(context, vis=vis)
    all_results = []
    skipped = 0

    for i, fid in enumerate(fragment_ids):
        if verbose and (i % 5 == 0 or i == len(fragment_ids) - 1):
            print(f"  [{label}] Processing {i+1}/{len(fragment_ids)}: {fid}")

        if not _load_and_detect_fragment(runner, context, fid):
            skipped += 1
            continue

        s = context.state
        if prediction_mode == PredictionMode.DETECTION:
            all_preds = _predict_detection_crops(runner, context, fid, visualize, output_dir)
        elif prediction_mode == PredictionMode.WITHOUT_PSR:
            all_preds = _predict_without_psr_crops(
                runner, context, fid, visualize, output_dir
            )
        else:
            all_preds = _predict_psr_crops(runner, context, fid, visualize, output_dir)

        if verbose:
            print(f"    GT={len(s.gt_boxes)}, Pred={len(all_preds)}")
        all_results.append({
            'fragment_id': fid,
            'preds': all_preds,
            'gts': s.gt_boxes,
        })

    return all_results, skipped



def evaluate_predictions(
    all_results: List[Dict],
    label: str = "",
    skipped: int = 0,
    psr_params: Optional[dict] = None,
    prediction_mode: PredictionMode = DEFAULT_PREDICTION_MODE,
) -> Dict:
    """
    Compute COCO-style evaluation metrics from a list of prediction results.

    Args:
        all_results: List of dicts with 'fragment_id', 'preds', 'gts' keys,
                     as returned by run_predictions().
        label: Label for the evaluation run.
        skipped: Number of fragments skipped during prediction.
        psr_params: PSR params used (stored in summary for reference).
        prediction_mode: Prediction mode used (stored in summary for reference).

    Returns:
        Dict with COCO-style mAP, AP@0.5, AP@0.75, AR, per-threshold and per-class details.
    """
    if not all_results:
        print(f"  [{label}] No results to evaluate.")
        return {'error': 'no results'}

    # COCO-style class-aware per-class score-sorted 101-point interpolated AP
    coco = compute_coco_map(all_results)

    t05 = coco['per_threshold'][0]  # results at IoU=0.5

    return {
        'label': label,
        'prediction_mode': str(prediction_mode),
        'num_fragments': len(all_results),
        'skipped': skipped,
        'psr_params': dict(psr_params) if psr_params else {},
        'mAP': coco['mAP'],
        'AP@0.5': coco['AP@0.5'],
        'AP@0.75': coco['AP@0.75'],
        'AR': coco['AR'],
        'iou_stats': {
            'mean_iou': t05['mean_matched_iou'],
            'num_matched': t05['tp'],
        },
        'num_classes_evaluated': coco['num_classes'],
        'per_threshold': coco['per_threshold'],
        'per_class': coco['per_class'],
        'per_fragment': [
            {
                'fragment_id': r['fragment_id'],
                'num_gt': len(r['gts']),
                'num_pred': len(r['preds']),
            }
            for r in all_results
        ],
    }


def run_evaluation(
    context: CropContext,
    fragment_ids: List[str],
    psr_params: Optional[dict] = None,
    verbose: bool = True,
    label: str = "",
    visualize: bool = False,
    output_dir: str = EVAL_OUTPUT_DIR,
    prediction_mode: PredictionMode = DEFAULT_PREDICTION_MODE,
) -> Dict:
    """
    Run predictions then evaluate. Convenience wrapper around
    run_predictions() + evaluate_predictions().

    Args:
        context: Pipeline context.
        fragment_ids: List of fragment IDs to process.
        psr_params: PSR optimizer hyperparameters (only used for PSR mode).
        verbose: Print progress messages.
        label: Label for progress/summary output.
        visualize: Save per-fragment visualizations (PSR mode only).
        output_dir: Directory for outputs.
        prediction_mode: How to produce predictions.
            PredictionMode.PSR         – full PSR alignment optimization (default)
            PredictionMode.WITHOUT_PSR – aligned/relabelled detections without PSR
            PredictionMode.DETECTION   – raw detection model output only

    Returns:
        Dict with mAP, per-threshold results, per-class results, and per-fragment details.
    """
    all_results, skipped = run_predictions(
        context=context,
        fragment_ids=fragment_ids,
        psr_params=psr_params,
        verbose=verbose,
        label=label,
        prediction_mode=prediction_mode,
        visualize=visualize,
        output_dir=output_dir,
    )

    if not all_results:
        print(f"  [{label}] No results produced.")
        return {'error': 'no results'}

    return evaluate_predictions(
        all_results=all_results,
        label=label,
        skipped=skipped,
        psr_params=psr_params,
        prediction_mode=prediction_mode,
    )


def print_eval_summary(eval_result: Dict):
    """Pretty-print COCO-style evaluation results."""
    if 'error' in eval_result:
        print(f"  Evaluation error: {eval_result['error']}")
        return

    label = eval_result.get('label', '')
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS {f'({label})' if label else ''}")
    print(f"{'='*60}")
    print(f"Fragments evaluated: {eval_result['num_fragments']} (skipped: {eval_result['skipped']})")
    print(f"Classes evaluated:   {eval_result['num_classes_evaluated']}")
    print(f"Prediction mode:     {eval_result.get('prediction_mode', 'unknown')}")

    print(f"\n--- COCO-style Metrics (class-aware, macro avg, 101-pt interpolated AP) ---")
    print(f"  mAP [0.5:0.95]:  {eval_result['mAP']:.4f}")
    print(f"  AP@0.5:          {eval_result['AP@0.5']:.4f}")
    print(f"  AP@0.75:         {eval_result['AP@0.75']:.4f}")
    print(f"  AR [0.5:0.95]:   {eval_result['AR']:.4f}")

    # Detailed at IoU=0.5
    t05 = eval_result['per_threshold'][0]
    print(f"\n  At IoU=0.5 (aggregate counts): "
          f"TP={t05['tp']}, FP={t05['fp']}, FN={t05['fn']}")
    print(f"    Precision: {t05['precision']:.4f}")
    print(f"    Recall:    {t05['recall']:.4f}")
    print(f"    F1:        {t05['f1']:.4f}")
    print(f"    Mean matched IoU: {t05['mean_matched_iou']:.4f}")

    # Per-threshold AP table
    print(f"\n  Per-threshold AP:")
    for r in eval_result['per_threshold']:
        print(f"    IoU={r['iou_threshold']:.2f}  AP={r['ap']:.4f}  "
              f"P={r['precision']:.4f}  R={r['recall']:.4f}")

    # Top-10 classes by support
    if 'per_class' in eval_result:
        pc = eval_result['per_class']
        if pc:
            print(f"\n--- Per-class @ IoU=0.5 (top-10 by GT count) ---")
            sorted_cls = sorted(
                pc.items(),
                key=lambda x: x[1]['tp'] + x[1]['fn'],
                reverse=True,
            )
            for cls_name, s in sorted_cls[:10]:
                support = s['tp'] + s['fn']
                print(f"  {cls_name:15s}: AP={s['ap']:.3f}  "
                      f"P={s['precision']:.3f}  R={s['recall']:.3f}  "
                      f"F1={s['f1']:.3f}  "
                      f"(TP={s['tp']}, FP={s['fp']}, FN={s['fn']}, support={support})")


# ============ Hyperparameter Tuning ============

def _eval_score(
    context: CropContext,
    fragment_ids: List[str],
    params: dict,
) -> float:
    """Run evaluation and return mAP (higher is better)."""
    result = run_evaluation(
        context=context,
        fragment_ids=fragment_ids,
        psr_params=params,
        verbose=False,
        label="sweep",
    )
    if 'error' in result:
        return -1.0
    return result['mAP']


# ============ Main ============

if __name__ == "__main__":
    print("Cuneiform Signs Alignment - Evaluation & Hyperparameter Sweep (PSR)")
    print("=" * 60)

    test_source = LocalTestDataSource(COCO_TEST_DIR)
    fragments = test_source.get_available_fragments()
    print(f"Found {len(fragments)} fragments in COCO test set")

    print("Loading detection model...")
    model_config = ModelConfig(
        config_file=CONFIG_FILE,
        checkpoint_file=CHECKPOINT_FILE,
        device='auto',
    )
    tablet_detector = TabletImageDetector(
        model_config=model_config,
        default_score_threshold=SCORE_THRESHOLD,
        keep_crops=True,
        is_crop_itself=True,
    )
    print("Model loaded.")

    context = CropContext(
        tablet_detector=tablet_detector,
        local_source=test_source,
        api_source=EBLAPISource(strip_subtablet_suffix=True),
        color_config=ColorConfig,
        output_dir=EVAL_OUTPUT_DIR,
        img_idx=0,
        task_type="evaluation",
    )

    eval_fragments = fragments[:EVAL_SAMPLE_LIMIT]
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # Select prediction mode here:
    #   PredictionMode.PSR         – full PSR alignment optimization (default)
    #   PredictionMode.WITHOUT_PSR – text-aligned labels, detection boxes/scores
    #   PredictionMode.DETECTION   – raw detection model output only
    # ----------------------------------------------------------------
    PREDICTION_MODE = PredictionMode.WITHOUT_PSR  

    # --- STEP 1: Full evaluation with default params ---
    print(f"\n{'='*60}")
    print(f"STEP 1: {PREDICTION_MODE.value} evaluation ({len(eval_fragments)} fragments)")
    print(f"  prediction_mode  = {PREDICTION_MODE}")
    if PREDICTION_MODE == PredictionMode.PSR:
        print(f"  num_iterations   = {DEFAULT_PSR_PARAMS['num_iterations']}")
    print(f"{'='*60}")

    eval_result = run_evaluation(
        context=context,
        fragment_ids=eval_fragments,
        psr_params=DEFAULT_PSR_PARAMS,
        verbose=True,
        label=PREDICTION_MODE.value,
        visualize=True,
        output_dir=EVAL_OUTPUT_DIR,
        prediction_mode=PREDICTION_MODE,
    )
    print_eval_summary(eval_result)

    eval_save = {k: v for k, v in eval_result.items() if k != 'per_class'}
    eval_save['per_class'] = {k: v for k, v in eval_result.get('per_class', {}).items()}
    with open(os.path.join(EVAL_OUTPUT_DIR, "evaluation_default.json"), 'w') as f:
        json.dump(eval_save, f, indent=2)
    print(f"Saved to {EVAL_OUTPUT_DIR}/evaluation_default.json")

    ### ---
    # not to run tunning
    exit(0)  #

    ### ---

    # --- STEP 2: Fast coordinate-wise hyperparameter sweep ---
    search_fragments = eval_fragments[:SEARCH_SAMPLE_LIMIT]

    print(f"\n{'='*60}")
    print(f"STEP 2: Coordinate-wise sweep ({len(search_fragments)} fragments, "
          f"num_iterations = {SEARCH_PSR_PARAMS['num_iterations']})")
    print(f"{'='*60}")

    search_result = hyperparameter_search(
        context=context,
        fragment_ids=search_fragments,
        eval_fn=_eval_score,
        base_params=SEARCH_PSR_PARAMS,
        output_dir=EVAL_OUTPUT_DIR,
        full_num_iterations=DEFAULT_PSR_PARAMS['num_iterations'],
    )

    # --- STEP 3: Re-evaluate with best params on full set ---
    if search_result.get('best_params'):
        best_params = search_result['best_params']

        print(f"\n{'='*60}")
        print(f"STEP 3: Re-evaluation with best params ({len(eval_fragments)} fragments)")
        print(f"  num_iterations = {best_params['num_iterations']}")
        print(f"  Params: {best_params}")
        print(f"{'='*60}")

        eval_best = run_evaluation(
            context=context,
            fragment_ids=eval_fragments,
            psr_params=best_params,
            verbose=True,
            label="best_params",
        )
        print_eval_summary(eval_best)

        eval_best_save = {k: v for k, v in eval_best.items() if k != 'per_class'}
        eval_best_save['per_class'] = {k: v for k, v in eval_best.get('per_class', {}).items()}
        with open(os.path.join(EVAL_OUTPUT_DIR, "evaluation_best.json"), 'w') as f:
            json.dump(eval_best_save, f, indent=2)
        print(f"Saved to {EVAL_OUTPUT_DIR}/evaluation_best.json")

    print(f"\nAll evaluation results saved to {EVAL_OUTPUT_DIR}/")
