"""
Evaluation script for Cuneiform Signs Alignment (PSR method).

Computes object-detection-style metrics (mAP, IoU, Precision, Recall)
by comparing PSR-optimized alignment bounding boxes against ground-truth annotations.

Also includes a fast coordinate-wise hyperparameter sweep for the
PointSetRegistrationOptimizer.

Uses the step functions from sign_alignment/pipeline.py for the alignment
pipeline. No ProtoSnap.
"""

import json
import os
import time
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from PIL import Image as PILImage, ImageDraw, ImageFont
from dotenv import load_dotenv

from sign_alignment import (
    LocalDataSource,
    LocalTestDataSource,
    SubtabletEBLAPISource,
    ModelConfig, TabletImageDetector,
    BoundingBox, Detection, GroundTruths,
)
from sign_alignment.visualizer import ColorConfig
from sign_alignment.pipeline import (
    CropContext, PipelineConfig, SampleState, Runner, VisOptions,
    step_load_data,
    step_detect_signs,
    step_compute_statistics,
    step_transform_gt_to_img,
    step_create_subtablets,
    step_detect_rows,
    step_match_rows,
    step_match_signs_in_rows,
    step_align_text_rows,
    step_create_psr_optimizer,
    step_run_psr_optimization,
)

load_dotenv()

# ============ Configuration ============
ANNOTATIONS_DIR = os.path.expanduser("~/erc-work-data/data-of-cuneiform-ocr-data/filtered_annotations")
COCO_TEST_DIR = os.path.expanduser("~/erc-work-data/ready-for-training/coco-recognition-2025-09/data/coco")
CONFIG_FILE = "configs/detr.py"
CHECKPOINT_FILE = os.path.expanduser("~/erc-work-data/retrained_models/detr-173/epoch_1000.pth")
SCORE_THRESHOLD = 0.5
EVAL_OUTPUT_DIR = "evaluation_results"

# Number of fragments to evaluate / search
EVAL_SAMPLE_LIMIT = 30
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


# ============ IoU & Metrics ============

def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute IoU between two BoundingBox objects."""
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


def compute_iou_matrix(preds: Detection, gts: GroundTruths) -> np.ndarray:
    """Compute IoU matrix of shape (len(preds), len(gts))."""
    n_pred = len(preds)
    n_gt = len(gts)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_matrix[i, j] = compute_iou(p, g)
    return iou_matrix


def match_predictions_to_gt(
    preds: Detection,
    gts: GroundTruths,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> Dict:
    """
    Match predictions to ground truth boxes at a given IoU threshold.
    Each GT box is matched to at most one prediction (greedy, highest IoU first).

    Args:
        preds: Predicted BoundingBox list
        gts: Ground truth BoundingBox list
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


def compute_ap_at_threshold(
    all_fragment_results: List[Dict],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> Dict:
    """
    Compute AP (average precision) at a single IoU threshold
    across all fragments (micro-averaged).

    Each fragment_result should have:
      - 'preds': Detection  (list of BoundingBox)
      - 'gts': GroundTruths (list of BoundingBox)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    for fr in all_fragment_results:
        m = match_predictions_to_gt(
            fr['preds'], fr['gts'],
            iou_threshold=iou_threshold,
            class_agnostic=class_agnostic,
        )
        total_tp += m['tp']
        total_fp += m['fp']
        total_fn += m['fn']
        all_ious.extend(m['matched_ious'])

    precision, recall = compute_precision_recall(total_tp, total_fp, total_fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    return {
        'iou_threshold': iou_threshold,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_matched_iou': mean_iou,
    }


def compute_map(all_fragment_results: List[Dict], class_agnostic: bool = False) -> Dict:
    """
    Compute mAP across standard IoU thresholds [0.5, 0.55, ..., 0.95].
    Also returns detailed results per threshold.
    """
    results_per_threshold = []
    for thresh in IOU_THRESHOLDS:
        r = compute_ap_at_threshold(all_fragment_results, iou_threshold=thresh,
                                    class_agnostic=class_agnostic)
        results_per_threshold.append(r)

    # mAP = mean of precision across thresholds (COCO-style: uses precision at each threshold)
    precisions = [r['precision'] for r in results_per_threshold]
    mAP = float(np.mean(precisions))

    return {
        'mAP': mAP,
        'AP@0.5': results_per_threshold[0]['precision'],
        'AP@0.75': results_per_threshold[5]['precision'] if len(results_per_threshold) > 5 else 0.0,
        'per_threshold': results_per_threshold,
    }


# ============ Per-class metrics ============

def compute_per_class_metrics(
    all_fragment_results: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Dict]:
    """Compute precision / recall per sign class."""
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for fr in all_fragment_results:
        preds = fr['preds']
        gts = fr['gts']
        m = match_predictions_to_gt(preds, gts, iou_threshold=iou_threshold,
                                    class_agnostic=False)
        matched_pred_ids = set()
        matched_gt_ids = set()
        for pi, gj in m['matched_pairs']:
            cls = gts[gj].sign.name
            class_stats[cls]['tp'] += 1
            matched_pred_ids.add(pi)
            matched_gt_ids.add(gj)

        for pi, p in enumerate(preds):
            if pi not in matched_pred_ids:
                class_stats[p.sign.name]['fp'] += 1
        for gj, g in enumerate(gts):
            if gj not in matched_gt_ids:
                class_stats[g.sign.name]['fn'] += 1

    per_class = {}
    for cls, stats in sorted(class_stats.items()):
        p, r = compute_precision_recall(stats['tp'], stats['fp'], stats['fn'])
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[cls] = {**stats, 'precision': p, 'recall': r, 'f1': f1}
    return per_class


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
    preds: Detection,
    gts: GroundTruths,
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


# ============ Evaluation Runner ============

def run_evaluation(
    context: CropContext,
    fragment_ids: List[str],
    psr_params: Optional[dict] = None,
    verbose: bool = True,
    label: str = "",
    visualize: bool = False,
    output_dir: str = EVAL_OUTPUT_DIR,
) -> Dict:
    """
    Run evaluation on a list of fragments.

    Returns:
        Dict with mAP, per-threshold results, per-class results, and per-fragment details.
    """
    context.config.psr_params = psr_params  # None = step defaults
    vis = VisOptions(info=False, display=False, save=False)
    runner = Runner(context, steps=[], vis=vis)
    all_results = []
    skipped = 0

    for i, fid in enumerate(fragment_ids):
        if verbose and (i % 5 == 0 or i == len(fragment_ids) - 1):
            print(f"  [{label}] Processing {i+1}/{len(fragment_ids)}: {fid}")

        context.state = SampleState()
        context.fragment_id = fid

        try:
            runner.run_single_step(step_load_data)
        except Exception as e:
            skipped += 1
            continue

        s = context.state
        if not s.img or not s.text_lines or not s.gt_boxes:
            skipped += 1
            continue

        # Filter out abnormally large GT boxes (e.g. sub-tablet region annotations)
        areas = [b.width * b.height for b in s.gt_boxes]
        mean_area = np.mean(areas)
        s.gt_boxes = [b for b, a in zip(s.gt_boxes, areas) if a <= mean_area * 5]
        if not s.gt_boxes:
            skipped += 1
            continue

        runner.run_single_step(step_detect_signs)
        runner.run_single_step(step_compute_statistics)
        if not s.detections:
            skipped += 1
            continue

        all_optimized_full: List[BoundingBox] = []
        for crop_idx in range(len(context.tablet_detector.get_cropped_images())):
            runner.choose_crop(crop_idx)
            if not s.sub_image.detections:
                continue
            try:
                runner.run_single_step(step_transform_gt_to_img)
                runner.run_single_step(step_create_subtablets)
                runner.run_single_step(step_detect_rows)
                if not s.sub_tablet_detection.get_rows():
                    continue
                runner.run_single_step(step_match_rows)
                if not s.matches:
                    continue
                runner.run_single_step(step_match_signs_in_rows)
                runner.run_single_step(step_align_text_rows)
                if not s.sub_tablet_aligned:
                    continue
                runner.run_single_step(step_create_psr_optimizer)
                runner.run_single_step(step_run_psr_optimization)
            except Exception:
                continue

            ox, oy = s.crop_info['x'], s.crop_info['y']
            for sb in s.sub_tablet_final.sign_boxes:
                all_optimized_full.append(BoundingBox(
                    x1=sb.x1 + ox, y1=sb.y1 + oy,
                    x2=sb.x2 + ox, y2=sb.y2 + oy,
                    score=sb.score, sign=sb.sign,
                ))

        if visualize and s.img is not None:
            visualize_evaluation_fragment(
                img=s.img,
                fragment_id=fid,
                preds=all_optimized_full,
                gts=s.gt_boxes,
                iou_threshold=0.5,
                class_agnostic=False,
                output_dir=output_dir,
            )

        if verbose:
            print(f"    GT={len(s.gt_boxes)}, Pred={len(all_optimized_full)}")
        all_results.append({
            'fragment_id': fid,
            'preds': all_optimized_full,
            'gts': s.gt_boxes,
        })

    if not all_results:
        print(f"  [{label}] No results produced.")
        return {'error': 'no results'}

    # --- Class-agnostic metrics (localization only) ---
    map_agnostic = compute_map(all_results, class_agnostic=True)

    # --- Class-aware metrics ---
    map_aware = compute_map(all_results, class_agnostic=False)

    # --- Per-class at IoU=0.5 ---
    per_class = compute_per_class_metrics(all_results, iou_threshold=0.5)

    # --- Aggregate IoU stats (class-aware) ---
    all_matched_ious = []
    for fr in all_results:
        m = match_predictions_to_gt(fr['preds'], fr['gts'], iou_threshold=0.5,
                                    class_agnostic=False)
        all_matched_ious.extend(m['matched_ious'])

    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
    median_iou = float(np.median(all_matched_ious)) if all_matched_ious else 0.0

    summary = {
        'label': label,
        'num_fragments': len(all_results),
        'skipped': skipped,
        'psr_params': dict(psr_params) if psr_params else {},
        'class_agnostic': {
            'mAP': map_agnostic['mAP'],
            'AP@0.5': map_agnostic['AP@0.5'],
            'AP@0.75': map_agnostic['AP@0.75'],
            'per_threshold': map_agnostic['per_threshold'],
        },
        'class_aware': {
            'mAP': map_aware['mAP'],
            'AP@0.5': map_aware['AP@0.5'],
            'AP@0.75': map_aware['AP@0.75'],
            'per_threshold': map_aware['per_threshold'],
        },
        'iou_stats': {
            'mean_iou': mean_iou,
            'median_iou': median_iou,
            'num_matched': len(all_matched_ious),
        },
        'num_classes_evaluated': len(per_class),
    }

    # For the detailed output
    detail = {
        **summary,
        'per_class': per_class,
        'per_fragment': [
            {
                'fragment_id': r['fragment_id'],
                'num_gt': len(r['gts']),
                'num_pred': len(r['preds']),
            }
            for r in all_results
        ],
    }

    return detail


def print_eval_summary(eval_result: Dict):
    """Pretty-print evaluation results."""
    if 'error' in eval_result:
        print(f"  Evaluation error: {eval_result['error']}")
        return

    label = eval_result.get('label', '')
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS {f'({label})' if label else ''}")
    print(f"{'='*60}")
    print(f"Fragments evaluated: {eval_result['num_fragments']} (skipped: {eval_result['skipped']})")

    for mode in ['class_agnostic', 'class_aware']:
        m = eval_result[mode]
        print(f"\n--- {mode.replace('_', ' ').title()} ---")
        print(f"  mAP [0.5:0.95]:  {m['mAP']:.4f}")
        print(f"  AP@0.5:          {m['AP@0.5']:.4f}")
        print(f"  AP@0.75:         {m['AP@0.75']:.4f}")

        # Detailed per-threshold at 0.5
        t05 = m['per_threshold'][0]
        print(f"  At IoU=0.5: TP={t05['tp']}, FP={t05['fp']}, FN={t05['fn']}")
        print(f"    Precision: {t05['precision']:.4f}")
        print(f"    Recall:    {t05['recall']:.4f}")
        print(f"    F1:        {t05['f1']:.4f}")
        print(f"    Mean matched IoU: {t05['mean_matched_iou']:.4f}")

    iou_s = eval_result['iou_stats']
    print(f"\n--- IoU Statistics (class-agnostic, thresh=0.5) ---")
    print(f"  Mean IoU:   {iou_s['mean_iou']:.4f}")
    print(f"  Median IoU: {iou_s['median_iou']:.4f}")
    print(f"  Matched:    {iou_s['num_matched']}")

    # Top-10 classes by support
    if 'per_class' in eval_result:
        pc = eval_result['per_class']
        if pc:
            print(f"\n--- Per-class (IoU=0.5, top-10 by support) ---")
            sorted_cls = sorted(pc.items(), key=lambda x: x[1]['tp'] + x[1]['fn'], reverse=True)
            for cls_name, s in sorted_cls[:10]:
                support = s['tp'] + s['fn']
                print(f"  {cls_name:15s}: P={s['precision']:.3f} R={s['recall']:.3f} "
                      f"F1={s['f1']:.3f} (TP={s['tp']}, FP={s['fp']}, FN={s['fn']}, support={support})")


# ============ Hyperparameter Tuning ============

def _eval_score(
    context: CropContext,
    fragment_ids: List[str],
    params: dict,
) -> float:
    """Run evaluation and return class-agnostic mAP (higher is better)."""
    result = run_evaluation(
        context=context,
        fragment_ids=fragment_ids,
        psr_params=params,
        verbose=False,
        label="sweep",
    )
    if 'error' in result:
        return -1.0
    return result['class_agnostic']['mAP']


def hyperparameter_search(
    context: CropContext,
    fragment_ids: List[str],
    output_dir: str = EVAL_OUTPUT_DIR,
) -> Dict:
    """
    Fast coordinate-wise (one-parameter-at-a-time) hyperparameter sweep
    for the PointSetRegistrationOptimizer.

    Sweeps PSR-specific parameters: lambda_data, lambda_anchor, lambda_seq,
    lambda_height, lambda_rows, lambda_boundary, sigma_factor, w_noise.

    Runs two rounds to allow parameters to adapt to each other.
    """
    search_axes = {
        'lambda_data':     [0.5, 1.0, 2.0, 5.0, 10.0],
        'lambda_anchor':   [0.005, 0.01, 0.05, 0.1],
        'lambda_seq':      [0.01, 0.05, 0.1, 0.5],
        'lambda_height':   [0.0, 0.005, 0.01, 0.05],
        'lambda_rows':     [1.0, 2.0, 5.0, 10.0],
        'lambda_boundary': [0.0, 0.5, 1.0, 5.0],
        'sigma_factor':    [1.0, 1.5, 2.0, 2.5],
        'w_noise':         [0.05, 0.1, 0.2],
    }

    # Start from defaults with reduced iterations for speed
    best_params = dict(SEARCH_PSR_PARAMS)

    total_evals = 2 * sum(len(v) for v in search_axes.values())
    print(f"Coordinate-wise sweep: {total_evals} evaluations "
          f"(2 rounds × {sum(len(v) for v in search_axes.values())} candidates)")

    best_score = _eval_score(context, fragment_ids, best_params)
    print(f"  Baseline mAP = {best_score:.4f}")

    all_search_results = []
    eval_count = 0

    for round_idx in range(2):
        print(f"\n--- Round {round_idx + 1} ---")
        for key, candidates in search_axes.items():
            old_val = best_params[key]
            round_best_val = old_val
            round_best_score = best_score

            for val in candidates:
                if val == old_val:
                    continue  # already evaluated
                trial = dict(best_params)
                trial[key] = val
                eval_count += 1

                t0 = time.time()
                score = _eval_score(context, fragment_ids, trial)
                elapsed = time.time() - t0

                entry = {
                    'round': round_idx + 1,
                    'param': key,
                    'value': val,
                    'mAP': score,
                    'elapsed_s': elapsed,
                    'full_params': {k: v for k, v in trial.items()},
                }
                all_search_results.append(entry)

                tag = ""
                if score > round_best_score:
                    round_best_score = score
                    round_best_val = val
                    tag = " *"

                print(f"  [{eval_count:3d}] {key}={val:<10}  mAP={score:.4f} "
                      f"({elapsed:.1f}s){tag}")

            # Update best for this axis
            if round_best_score > best_score:
                best_params[key] = round_best_val
                best_score = round_best_score
                print(f"  >> {key} updated to {round_best_val} (mAP={best_score:.4f})")
            else:
                print(f"  >> {key} stays at {old_val}")

    # Restore full iteration count for final params
    best_params['num_iterations'] = DEFAULT_PSR_PARAMS['num_iterations']

    # Sort results
    all_search_results.sort(key=lambda x: -x['mAP'])

    print(f"\n{'='*60}")
    print(f"SWEEP RESULTS (top 10)")
    print(f"{'='*60}")
    for entry in all_search_results[:10]:
        print(f"  {entry['param']:15s}={entry['value']:<10}  mAP={entry['mAP']:.4f}")

    print(f"\nBest params: {best_params}")
    print(f"Best mAP:    {best_score:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    search_save = {
        'best_params': best_params,
        'best_mAP': best_score,
        'all_results': all_search_results,
    }
    with open(os.path.join(output_dir, "hyperparam_search.json"), 'w') as f:
        json.dump(search_save, f, indent=2)
    print(f"Saved to {output_dir}/hyperparam_search.json")

    return search_save


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
        score_threshold=SCORE_THRESHOLD,
        keep_crops=True,
        is_crop_itself=True,
    )
    print("Model loaded.")

    context = CropContext(
        config=PipelineConfig(
            model_config=model_config,
            tablet_detector=tablet_detector,
            local_source=test_source,
            api_source=SubtabletEBLAPISource(),
            color_config=ColorConfig,
            output_dir=EVAL_OUTPUT_DIR,
        ),
        task_type="evaluation"
    )

    eval_fragments = fragments[:EVAL_SAMPLE_LIMIT]
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # --- STEP 1: Full evaluation with default params ---
    print(f"\n{'='*60}")
    print(f"STEP 1: Evaluation with default PSR parameters ({len(eval_fragments)} fragments)")
    print(f"  num_iterations = {DEFAULT_PSR_PARAMS['num_iterations']}")
    print(f"{'='*60}")

    eval_result = run_evaluation(
        context=context,
        fragment_ids=eval_fragments,
        psr_params=DEFAULT_PSR_PARAMS,
        verbose=True,
        label="default",
        visualize=True,
        output_dir=EVAL_OUTPUT_DIR,
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
        output_dir=EVAL_OUTPUT_DIR,
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
