"""
Evaluation script for Cuneiform Signs Alignment.

Computes object-detection-style metrics (mAP, IoU, Precision, Recall)
by comparing optimized alignment bounding boxes against ground-truth annotations.

Also includes a simple grid-search hyperparameter tuning for the
ElasticChainOptimizer.
"""

import json
import os
import time
import itertools
import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dotenv import load_dotenv

from sign_alignment import (
    LocalDataSource, EBLAPISource, SignTextParser,
    CLASSES_ABZ, SignResolver,
    ModelConfig, TabletImageDetector, SingleImage,
    compute_avg_dimensions,
    match_heatmaps_ncc, transform_gt_to_cropped_region,
    SubTablet, ElasticChainOptimizer,
    BoundingBox, Detection, GroundTruths,
)

load_dotenv()

# ============ Configuration ============
ANNOTATIONS_DIR = os.path.expanduser("~/erc-work-data/data-of-cuneiform-ocr-data/filtered_annotations")
CONFIG_FILE = "configs/detr.py"
CHECKPOINT_FILE = os.path.expanduser("~/erc-work-data/retrained_models/detr-173/epoch_1000.pth")
SCORE_THRESHOLD = 0.5
SCALE_FACTOR = 10
EVAL_OUTPUT_DIR = "evaluation_results"

# Number of fragments to evaluate
EVAL_SAMPLE_LIMIT = 100

# Default optimizer hyperparameters
DEFAULT_OPTIMIZER_PARAMS = dict(
    lambda_data=10000.0,
    lambda_seq=0.05,
    lambda_smooth=0.15,
    lambda_anchor=0.05,
    num_iterations=100,
    lr=5.0,
)

HEATMAP_METHOD = 'gaussian'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# IoU thresholds for mAP computation
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


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


# ============ Alignment Pipeline (reused from signs_alignment_heatmap.py) ============

def run_alignment_pipeline(
    tablet_detector: TabletImageDetector,
    local_source: LocalDataSource,
    api_source: EBLAPISource,
    fragment_id: str,
    scale_factor: int = SCALE_FACTOR,
    method: str = HEATMAP_METHOD,
    optimizer_params: dict = None,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Run the full alignment pipeline for one fragment.
    Returns dict with 'preds' (optimized boxes in full img coords) and 'gts'.
    """
    if optimizer_params is None:
        optimizer_params = DEFAULT_OPTIMIZER_PARAMS

    img = local_source.load_image(fragment_id)
    if img is None:
        return None

    gt_boxes = local_source.load_annotation(fragment_id)
    if not gt_boxes:
        return None

    signs_text = api_source.get_signs(fragment_id)
    if signs_text is None:
        return None

    text_lines = SignTextParser.parse_api_signs(signs_text)
    if not text_lines:
        return None

    detections = tablet_detector.detect(img)
    if not detections:
        return None

    cropped_images = tablet_detector.get_cropped_images()
    crop_coordinates = tablet_detector.crop_coordinates
    avg_width, avg_height = compute_avg_dimensions(detections)
    margin = max(avg_width, avg_height)

    all_optimized_full: List[BoundingBox] = []

    for idx, crop_single in enumerate(cropped_images):
        crop_dets = crop_single.detections
        if not crop_dets:
            continue

        # Detection SubTablet
        sub_det = SubTablet.from_detections(
            img=crop_single.img, detections=crop_dets,
            name="det", avg_width=avg_width, avg_height=avg_height,
        )
        sub_det.create_heatmap(scale_factor=scale_factor, method=method)

        # Full-text SubTablet
        sub_text = SubTablet.from_text_lines(
            text_lines=text_lines, avg_width=avg_width,
            avg_height=avg_height, margin=margin, name="text",
        )
        sub_text.create_heatmap(scale_factor=scale_factor, method=method)

        # NCC
        _, match_score, top_left_original = match_heatmaps_ncc(
            sub_det.heatmap, sub_text.heatmap, scale_factor=scale_factor,
        )
        tx, ty = top_left_original
        eh, ew = crop_single.img.shape[:2]

        # Extract aligned region
        sub_aligned = sub_text.extract_sub_region(
            offset_x=tx, offset_y=ty, width=ew, height=eh,
            img=crop_single.img, name="aligned",
        )
        sub_aligned.create_heatmap(
            scale_factor=scale_factor, img_shape=crop_single.img.shape, method=method,
        )

        # Optimize
        prior_ar = avg_width / avg_height if avg_height > 0 else 1.0
        opt = ElasticChainOptimizer(
            sub_tablet_text=sub_aligned,
            detection_heatmap=sub_det.heatmap,
            scale_factor=scale_factor,
            lambda_data=optimizer_params.get('lambda_data', 10000.0),
            lambda_seq=optimizer_params.get('lambda_seq', 0.05),
            lambda_smooth=optimizer_params.get('lambda_smooth', 0.15),
            lambda_anchor=optimizer_params.get('lambda_anchor', 0.05),
            prior_aspect_ratio=prior_ar,
            device=DEVICE,
        )
        sub_opt = opt.optimize(
            num_iterations=optimizer_params.get('num_iterations', 100),
            lr=optimizer_params.get('lr', 5.0),
            verbose=False,
        )

        # Transform to full image coords
        ox = crop_coordinates[idx]['x']
        oy = crop_coordinates[idx]['y']
        for sb in sub_opt.sign_boxes:
            all_optimized_full.append(BoundingBox(
                x1=sb.x1 + ox, y1=sb.y1 + oy,
                x2=sb.x2 + ox, y2=sb.y2 + oy,
                score=sb.score, sign=sb.sign,
            ))

    if verbose:
        print(f"  {fragment_id}: GT={len(gt_boxes)}, Pred={len(all_optimized_full)}")

    return {
        'fragment_id': fragment_id,
        'preds': all_optimized_full,
        'gts': gt_boxes,
        'detections': detections,
    }


# ============ Evaluation Runner ============

def run_evaluation(
    tablet_detector: TabletImageDetector,
    local_source: LocalDataSource,
    api_source: EBLAPISource,
    fragment_ids: List[str],
    optimizer_params: dict = None,
    verbose: bool = True,
    label: str = "",
) -> Dict:
    """
    Run evaluation on a list of fragments.

    Returns:
        Dict with mAP, per-threshold results, per-class results, and per-fragment details.
    """
    if optimizer_params is None:
        optimizer_params = DEFAULT_OPTIMIZER_PARAMS

    all_results = []
    skipped = 0

    for i, fid in enumerate(fragment_ids):
        if verbose and (i % 10 == 0 or i == len(fragment_ids) - 1):
            print(f"  [{label}] Processing {i+1}/{len(fragment_ids)}: {fid}")

        result = run_alignment_pipeline(
            tablet_detector=tablet_detector,
            local_source=local_source,
            api_source=api_source,
            fragment_id=fid,
            optimizer_params=optimizer_params,
            verbose=verbose,
        )
        if result is None:
            skipped += 1
            continue
        all_results.append(result)

    if not all_results:
        print(f"  [{label}] No results produced.")
        return {'error': 'no results'}

    # --- Class-agnostic metrics (localization only) ---
    map_agnostic = compute_map(all_results, class_agnostic=True)

    # --- Class-aware metrics ---
    map_aware = compute_map(all_results, class_agnostic=False)

    # --- Per-class at IoU=0.5 ---
    per_class = compute_per_class_metrics(all_results, iou_threshold=0.5)

    # --- Aggregate IoU stats ---
    all_matched_ious = []
    for fr in all_results:
        m = match_predictions_to_gt(fr['preds'], fr['gts'], iou_threshold=0.5,
                                    class_agnostic=True)
        all_matched_ious.extend(m['matched_ious'])

    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
    median_iou = float(np.median(all_matched_ious)) if all_matched_ious else 0.0

    summary = {
        'label': label,
        'num_fragments': len(all_results),
        'skipped': skipped,
        'optimizer_params': optimizer_params,
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

def hyperparameter_search(
    tablet_detector: TabletImageDetector,
    local_source: LocalDataSource,
    api_source: EBLAPISource,
    fragment_ids: List[str],
    output_dir: str = EVAL_OUTPUT_DIR,
) -> Dict:
    """
    Simple grid search over optimizer hyperparameters.
    Uses a subset of fragments for speed, evaluates with class-agnostic mAP.
    """
    # Define search grid (keep it small for initial exploration)
    search_grid = {
        'lambda_data':   [1000.0, 10000.0, 50000.0],
        'lambda_seq':    [0.01, 0.05, 0.2],
        'lambda_smooth': [0.05, 0.15, 0.5],
        'lambda_anchor': [0.01, 0.05, 0.2],
    }

    # Fixed parameters during search
    fixed = {
        'num_iterations': 80,    # fewer iterations for speed
        'lr': 5.0,
    }

    # Generate all combinations
    keys = list(search_grid.keys())
    values = list(search_grid.values())
    combinations = list(itertools.product(*values))
    print(f"Hyperparameter search: {len(combinations)} combinations")

    best_score = -1
    best_params = None
    all_search_results = []

    for ci, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        params.update(fixed)

        print(f"\n--- Combo {ci+1}/{len(combinations)}: {params} ---")
        t0 = time.time()

        eval_result = run_evaluation(
            tablet_detector=tablet_detector,
            local_source=local_source,
            api_source=api_source,
            fragment_ids=fragment_ids,
            optimizer_params=params,
            verbose=False,
            label=f"combo_{ci+1}",
        )

        elapsed = time.time() - t0

        if 'error' in eval_result:
            print(f"  Skipped (error)")
            continue

        score = eval_result['class_agnostic']['mAP']
        ap50 = eval_result['class_agnostic']['AP@0.5']
        mean_iou = eval_result['iou_stats']['mean_iou']

        entry = {
            'combo_idx': ci + 1,
            'params': params,
            'mAP': score,
            'AP@0.5': ap50,
            'mean_iou': mean_iou,
            'elapsed_s': elapsed,
        }
        all_search_results.append(entry)

        print(f"  mAP={score:.4f}, AP@0.5={ap50:.4f}, IoU={mean_iou:.4f}, time={elapsed:.1f}s")

        if score > best_score:
            best_score = score
            best_params = params.copy()
            print(f"  ** New best! **")

    # Sort by mAP
    all_search_results.sort(key=lambda x: -x['mAP'])

    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SEARCH RESULTS (sorted by mAP)")
    print(f"{'='*60}")
    for entry in all_search_results[:10]:
        print(f"  Combo {entry['combo_idx']:3d}: mAP={entry['mAP']:.4f}, "
              f"AP@0.5={entry['AP@0.5']:.4f}, IoU={entry['mean_iou']:.4f} | {entry['params']}")

    print(f"\nBest params: {best_params}")
    print(f"Best mAP:    {best_score:.4f}")

    # Save search results
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
    print("Cuneiform Signs Alignment - Evaluation & Hyperparameter Search")
    print("=" * 60)

    # Data sources
    local_source = LocalDataSource(ANNOTATIONS_DIR)
    api_source = EBLAPISource()

    fragments = local_source.get_available_fragments()
    print(f"Found {len(fragments)} fragments with both image and annotation")

    # Detector
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
    )
    print("Model loaded.")

    eval_fragments = fragments[:EVAL_SAMPLE_LIMIT]
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # --- STEP 1: Full evaluation with default params ---
    print(f"\n{'='*60}")
    print(f"STEP 1: Evaluation with default parameters ({len(eval_fragments)} fragments)")
    print(f"{'='*60}")

    eval_result = run_evaluation(
        tablet_detector=tablet_detector,
        local_source=local_source,
        api_source=api_source,
        fragment_ids=eval_fragments,
        optimizer_params=DEFAULT_OPTIMIZER_PARAMS,
        verbose=True,
        label="default",
    )
    print_eval_summary(eval_result)

    # Save full evaluation
    # Convert per_class to JSON-serializable format
    eval_save = {k: v for k, v in eval_result.items() if k != 'per_class'}
    eval_save['per_class'] = {k: v for k, v in eval_result.get('per_class', {}).items()}
    with open(os.path.join(EVAL_OUTPUT_DIR, "evaluation_default.json"), 'w') as f:
        json.dump(eval_save, f, indent=2)
    print(f"Saved to {EVAL_OUTPUT_DIR}/evaluation_default.json")

    # --- STEP 2: Hyperparameter search (use a smaller subset for speed) ---
    SEARCH_SAMPLE_LIMIT = min(30, len(eval_fragments))
    search_fragments = eval_fragments[:SEARCH_SAMPLE_LIMIT]

    print(f"\n{'='*60}")
    print(f"STEP 2: Hyperparameter search ({len(search_fragments)} fragments)")
    print(f"{'='*60}")

    search_result = hyperparameter_search(
        tablet_detector=tablet_detector,
        local_source=local_source,
        api_source=api_source,
        fragment_ids=search_fragments,
        output_dir=EVAL_OUTPUT_DIR,
    )

    # --- STEP 3: Re-evaluate with best params on full set ---
    if search_result.get('best_params'):
        best_params = search_result['best_params']
        # Use full iteration count for final eval
        best_params['num_iterations'] = 100

        print(f"\n{'='*60}")
        print(f"STEP 3: Re-evaluation with best params ({len(eval_fragments)} fragments)")
        print(f"Params: {best_params}")
        print(f"{'='*60}")

        eval_best = run_evaluation(
            tablet_detector=tablet_detector,
            local_source=local_source,
            api_source=api_source,
            fragment_ids=eval_fragments,
            optimizer_params=best_params,
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
