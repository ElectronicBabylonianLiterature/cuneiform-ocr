"""
Cuneiform Signs Alignment using PSR (Point Set Registration) Method.

Pipeline:
  1. Data loading (local GT + EBL API text)
  2. DETR detection → sub-tablet crops
  3. Per-crop: DBSCAN row detection → DP row matching → DP sign matching
     → baseline-projection coarse alignment → PSR optimization
  4. Transform optimized boxes back to full-image coordinates
  5. Save visualizations

Replaces the old heatmap-based signs_alignment_heatmap.py.
"""

import json
import os
import numpy as np
import torch
from collections import Counter
from typing import List, Dict, Optional
from dotenv import load_dotenv

from sign_alignment import (
    # Data sources
    LocalDataSource, EBLAPISource, SignTextParser, SignAPIResolver,
    # Detection
    ModelConfig, TabletImageDetector, SingleImage,
    compute_avg_dimensions, transform_gt_to_cropped_region,
    # Row/sign matching
    match_rows_dp, create_row_mapping, match_signs_in_row_dp,
    align_text_to_detection_rows,
    # Sub-tablet / optimizer
    SubTablet, PointSetRegistrationOptimizer,
    # Visualization
    BboxVisualizer, TextVisualizer, CompositeVisualizer, build_sign_match_info,
    # Bounding box types
    BoundingBox, Detection, GroundTruths,
)

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

load_dotenv()

# ============ Configuration ============
ANNOTATIONS_DIR = os.path.expanduser("~/erc-work-data/data-of-cuneiform-ocr-data/filtered_annotations")
CONFIG_FILE = "configs/detr.py"
CHECKPOINT_FILE = os.path.expanduser("~/erc-work-data/retrained_models/detr-173/epoch_1000.pth")
SCORE_THRESHOLD = 0.5
OUTPUT_DIR = "alignment_results"
SAMPLE_LIMIT = 30

# DBSCAN row detection parameters
DBSCAN_EPS = 0.4
DBSCAN_MIN_SAMPLES = 1
DBSCAN_LAMBDA_WEIGHT = 0.007

# DP row matching parameters
ROW_MATCH_SKIP_TEXT_PENALTY = 0.5
ROW_MATCH_SKIP_DET_PENALTY = 1.0
ROW_MATCH_SKIP_SMALL_DET_PENALTY = 0.2
ROW_MATCH_SMALL_DET_THRESHOLD = 1
ROW_MATCH_SIMILARITY_METHOD = 'jaccard'

# DP sign matching parameters
SIGN_MATCH_SKIP_TEXT_PENALTY = 0.5
SIGN_MATCH_SKIP_DET_PENALTY = 2.0
SIGN_MATCH_MISMATCH_COST = 0.9

# Coarse alignment parameters
ALIGN_MIN_WIDTH_RATIO = 2 / 3
ALIGN_MAX_WIDTH_RATIO = 4 / 3

# PSR optimizer parameters
PSR_PARAMS = dict(
    sigma_factor=1.5,       # sigma = avg_width * sigma_factor
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
    num_iterations=80,
    lr=1.0,
    sigma_anneal=True,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============ Per-crop Alignment Pipeline ============

def process_single_crop(
    crop_img: np.ndarray,
    crop_detections: Detection,
    crop_mask: Optional[np.ndarray],
    text_lines: List[List[str]],
    avg_width: float,
    avg_height: float,
    psr_params: dict = PSR_PARAMS,
) -> Optional[Dict]:
    """
    Full PSR alignment pipeline for a single crop:
      1. Build detection & text SubTablets
      2. DBSCAN row detection on detection SubTablet
      3. DP row matching + sign matching
      4. Baseline-projection coarse alignment
      5. PSR optimization
    """
    if not crop_detections:
        return None

    # --- 1. Build SubTablets ---
    sub_det = SubTablet.from_detections(
        img=crop_img,
        detections=crop_detections,
        name="detection",
        avg_width=avg_width,
        avg_height=avg_height,
    )

    sub_text = SubTablet.from_text_lines(
        text_lines=text_lines,
        avg_width=avg_width,
        avg_height=avg_height,
        img=crop_img,
        target_detections=crop_detections,
        align_to_detection_centroid=True,
        name="text",
    )

    # --- 2. DBSCAN row detection ---
    num_rows_det = sub_det.detect_rows(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        lambda_weight=DBSCAN_LAMBDA_WEIGHT,
    )
    if num_rows_det == 0:
        return None

    # --- 3. Row-level DP matching ---
    det_row_seqs = sub_det.get_row_sign_sequences()
    text_row_seqs = sub_text.get_row_sign_sequences()

    matches, _ = match_rows_dp(
        detection_rows=det_row_seqs,
        text_rows=text_row_seqs,
        skip_text_penalty=ROW_MATCH_SKIP_TEXT_PENALTY,
        skip_det_penalty=ROW_MATCH_SKIP_DET_PENALTY,
        skip_small_det_penalty=ROW_MATCH_SKIP_SMALL_DET_PENALTY,
        small_det_threshold=ROW_MATCH_SMALL_DET_THRESHOLD,
        similarity_method=ROW_MATCH_SIMILARITY_METHOD,
    )
    if not matches:
        return None

    text_to_det, det_to_text = create_row_mapping(
        matches, len(text_row_seqs), len(det_row_seqs)
    )

    # --- 4. Sign-level DP matching ---
    row_sign_matches = {}
    for text_row_idx, det_row_idx in matches:
        text_signs = text_row_seqs[text_row_idx]
        det_signs = det_row_seqs[det_row_idx]
        sign_matches, _ = match_signs_in_row_dp(
            detection_signs=det_signs,
            text_signs=text_signs,
            skip_text_penalty=SIGN_MATCH_SKIP_TEXT_PENALTY,
            skip_det_penalty=SIGN_MATCH_SKIP_DET_PENALTY,
            mismatch_cost=SIGN_MATCH_MISMATCH_COST,
        )
        row_sign_matches[text_row_idx] = sign_matches

    # --- 5. Coarse alignment (baseline projection) ---
    det_rows = sub_det.get_rows_dict()
    text_rows = sub_text.get_rows_dict()

    aligned_boxes = align_text_to_detection_rows(
        det_rows=det_rows,
        text_rows=text_rows,
        text_to_det=text_to_det,
        row_sign_matches=row_sign_matches,
        avg_width=avg_width,
        avg_height=avg_height,
        min_width_ratio=ALIGN_MIN_WIDTH_RATIO,
        max_width_ratio=ALIGN_MAX_WIDTH_RATIO,
    )
    if not aligned_boxes:
        return None

    sub_optim = SubTablet(
        sign_boxes=aligned_boxes,
        img=crop_img,
        name="optim",
        avg_width=avg_width,
        avg_height=avg_height,
    )

    # --- 6. PSR optimization ---
    sigma = avg_width * psr_params.get('sigma_factor', 1.5)
    optimizer = PointSetRegistrationOptimizer(
        sub_tablet_text=sub_optim,
        target_detections=crop_detections,
        sigma=sigma,
        w_noise=psr_params.get('w_noise', 0.1),
        lambda_data=psr_params.get('lambda_data', 2.0),
        lambda_anchor=psr_params.get('lambda_anchor', 0.01),
        lambda_seq=psr_params.get('lambda_seq', 0.1),
        lambda_height=psr_params.get('lambda_height', 0.01),
        lambda_rows=psr_params.get('lambda_rows', 5.0),
        lambda_boundary=psr_params.get('lambda_boundary', 1.0),
        rows_threshold_ratio_far=psr_params.get('rows_threshold_ratio_far', 1 / 3.0),
        rows_threshold_ratio_close=psr_params.get('rows_threshold_ratio_close', 2 / 3.0),
        rows_plateau_far=psr_params.get('rows_plateau_far', 0.5),
        rows_plateau_close=psr_params.get('rows_plateau_close', 1.0),
        contour_mask=crop_mask,
        device=DEVICE,
    )

    sub_final = optimizer.optimize(
        num_iterations=psr_params.get('num_iterations', 80),
        lr=psr_params.get('lr', 1.0),
        sigma_anneal=psr_params.get('sigma_anneal', True),
        verbose=False,
    )

    return {
        'sub_det': sub_det,
        'sub_text': sub_text,
        'sub_optim': sub_optim,
        'sub_final': sub_final,
        'matches': matches,
        'text_to_det': text_to_det,
        'det_to_text': det_to_text,
        'row_sign_matches': row_sign_matches,
        'optimizer': optimizer,
    }


# ============ Fragment Processing ============

def process_fragment(
    tablet_detector: TabletImageDetector,
    local_source: LocalDataSource,
    api_source: EBLAPISource,
    sign_resolver: SignAPIResolver,
    fragment_id: str,
    psr_params: dict = PSR_PARAMS,
) -> Optional[Dict]:
    """Process a single fragment end-to-end."""
    print(f"\n{'='*60}")
    print(f"Processing fragment: {fragment_id}")

    # --- Load image & ground truth ---
    img = local_source.load_image(fragment_id)
    if img is None:
        print(f"  Image not found for {fragment_id}")
        return None

    gt_boxes = local_source.load_annotation(fragment_id)
    print(f"  Ground truth boxes: {len(gt_boxes) if gt_boxes else 0}")

    # --- Get & parse text (filtered, using text.lines) ---
    fragment_data = api_source.get_fragment_data(fragment_id)
    if fragment_data is None:
        print(f"  Could not get fragment data from API for {fragment_id}")
        return None

    text_data = fragment_data.get('text', {})
    text_lines = SignTextParser.parse_text_lines(
        text_data, filter_broken=True, sign_resolver=sign_resolver
    )
    if not text_lines:
        print(f"  No text lines parsed for {fragment_id}")
        return None

    total_text_signs = sum(len(line) for line in text_lines)
    print(f"  Text lines: {len(text_lines)}, total signs: {total_text_signs}")

    # --- Detect signs ---
    detections = tablet_detector.detect(img)
    print(f"  Detected signs: {len(detections)}")

    cropped_images = tablet_detector.get_cropped_images()
    crop_coordinates = tablet_detector.crop_coordinates
    print(f"  Sub-tablets: {len(cropped_images)}")

    avg_width, avg_height = compute_avg_dimensions(detections)

    # --- Process each crop ---
    crop_results = []
    all_optimized_full: List[BoundingBox] = []

    for idx, crop_single in enumerate(cropped_images):
        result = process_single_crop(
            crop_img=crop_single.img,
            crop_detections=crop_single.detections,
            crop_mask=crop_single.mask if hasattr(crop_single, 'mask') else None,
            text_lines=text_lines,
            avg_width=avg_width,
            avg_height=avg_height,
            psr_params=psr_params,
        )
        if result is None:
            continue
        crop_results.append(result)

        # Transform optimized bboxes to full-tablet coordinates
        ox = crop_coordinates[idx]['x']
        oy = crop_coordinates[idx]['y']
        for sb in result['sub_final'].sign_boxes:
            all_optimized_full.append(BoundingBox(
                x1=sb.x1 + ox, y1=sb.y1 + oy,
                x2=sb.x2 + ox, y2=sb.y2 + oy,
                score=sb.score, sign=sb.sign,
            ))

    print(f"  Alignment: {len(all_optimized_full)} optimized signs from {len(crop_results)} crops")

    # --- Visualize ---
    save_visualizations(
        fragment_id=fragment_id,
        img=img,
        gt_boxes=gt_boxes,
        detections=detections,
        text_lines=text_lines,
        all_optimized_full=all_optimized_full,
        crop_results=crop_results,
        crop_coordinates=crop_coordinates,
        output_dir=OUTPUT_DIR,
    )

    return {
        'fragment_id': fragment_id,
        'gt_count': len(gt_boxes) if gt_boxes else 0,
        'text_signs': total_text_signs,
        'detected': len(detections),
        'aligned': len(all_optimized_full),
        'num_crops': len(crop_results),
        'crop_results': crop_results,
        'all_optimized_full': all_optimized_full,
        'gt_boxes': gt_boxes,
        'detections': detections,
    }


# ============ Visualization ============

def save_visualizations(
    fragment_id: str,
    img: np.ndarray,
    gt_boxes: Optional[GroundTruths],
    detections: Detection,
    text_lines: List[List[str]],
    all_optimized_full: Detection,
    crop_results: List[Dict],
    crop_coordinates: List[Dict],
    output_dir: str,
):
    """Save visualization images and text files for one fragment."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Ground truth
    if gt_boxes:
        vis = BboxVisualizer(color=(0, 255, 0))
        vis.draw_boxes(img.copy(), gt_boxes)
        vis.save(os.path.join(output_dir, f"{fragment_id}_1_ground_truth.jpg"))

    # 2. Detections
    vis = BboxVisualizer(color=(255, 0, 0))
    vis.draw_boxes(img.copy(), detections)
    vis.save(os.path.join(output_dir, f"{fragment_id}_2_detections.jpg"))

    # 3. Text
    TextVisualizer.save_text(
        text_lines,
        path=os.path.join(output_dir, f"{fragment_id}_3_text.txt"),
        fragment_id=fragment_id,
    )

    # 4. Final optimized bounding boxes on full tablet
    vis = BboxVisualizer(color=(255, 255, 0))
    vis.draw_boxes(img.copy(), all_optimized_full)
    vis.save(os.path.join(output_dir, f"{fragment_id}_4_optimized.jpg"))

    # 5. Combined: detections (red) + optimized (yellow)
    vis_det = BboxVisualizer(color=(255, 0, 0))
    vis_det.draw_boxes(img.copy(), detections)
    vis_opt = BboxVisualizer(color=(255, 255, 0))
    vis_opt.draw_boxes(vis_det.result, all_optimized_full)
    vis_opt.save(os.path.join(output_dir, f"{fragment_id}_5_det_vs_optimized.jpg"))

    # 6. Combined: optimized (yellow) + GT (green)
    if gt_boxes:
        vis_gt = BboxVisualizer(color=(0, 255, 0))
        vis_gt.draw_boxes(img.copy(), gt_boxes)
        vis_opt_gt = BboxVisualizer(color=(255, 255, 0))
        vis_opt_gt.draw_boxes(vis_gt.result, all_optimized_full)
        vis_opt_gt.save(os.path.join(output_dir, f"{fragment_id}_6_optimized_vs_gt.jpg"))

    # 7. Per-crop detail: coarse vs final, detection rows, alignment diagnostic
    for ci, cr in enumerate(crop_results):
        prefix = f"{fragment_id}_crop{ci}"
        crop_img = cr['sub_det'].img

        # 7a. Coarse (cyan) vs Final (yellow) comparison
        vis_coarse = BboxVisualizer(color=(0, 255, 255))
        vis_coarse.draw_boxes(crop_img.copy(), cr['sub_optim'].to_detection_list())
        vis_final = BboxVisualizer(color=(255, 255, 0))
        vis_final.draw_boxes(crop_img.copy(), cr['sub_final'].to_detection_list())

        # 7b. Detection (red) + Final (yellow) overlay
        vis_d = BboxVisualizer(color=(255, 0, 0))
        vis_d.draw_boxes(crop_img.copy(), cr['sub_det'].to_detection_list())
        vis_df = BboxVisualizer(color=(255, 255, 0))
        vis_df.draw_boxes(vis_d.result, cr['sub_final'].to_detection_list())

        # 7c. GT (green) + Final (yellow) overlay for this crop
        crop_info = crop_coordinates[ci] if ci < len(crop_coordinates) else None
        gt_crop = transform_gt_to_cropped_region(gt_boxes, crop_info) if gt_boxes and crop_info else []
        if gt_crop:
            vis_gt_c = BboxVisualizer(color=(0, 255, 0))
            vis_gt_c.draw_boxes(crop_img.copy(), gt_crop)
            vis_gf = BboxVisualizer(color=(255, 255, 0))
            vis_gf.draw_boxes(vis_gt_c.result, cr['sub_final'].to_detection_list())
            gt_final_img = vis_gf.result
        else:
            gt_final_img = vis_final.result  # fallback

        # 2x2 comparison grid
        comp = CompositeVisualizer()
        comp.compose(
            images=[vis_coarse.result, vis_final.result, vis_df.result, gt_final_img],
            layout=(2, 2),
            titles=[
                f"Coarse Aligned ({len(cr['sub_optim'])} signs)",
                f"PSR Optimized ({len(cr['sub_final'])} signs)",
                "Det (red) + Final (yellow)",
                "GT (green) + Final (yellow)",
            ],
            figsize=(16, 12),
        )
        comp.save(os.path.join(output_dir, f"{prefix}_comparison.jpg"))

        # 7d. Detection rows visualization
        det_row_vis = BboxVisualizer(color=(255, 0, 0))
        det_row_vis.draw_rows(
            crop_img.copy(),
            cr['sub_det'].sign_boxes,
            show_labels=True,
            show_row_numbers=True,
            row_mapping=cr['det_to_text'],
            row_label_prefix="D",
            mapped_label_prefix="R",
        )
        det_row_vis.save(os.path.join(output_dir, f"{prefix}_det_rows.jpg"))

        # 7e. Text mapping + alignment diagnostic
        det_rows_dict = cr['sub_det'].get_rows_dict()
        text_sign_info, det_sign_info = build_sign_match_info(
            row_sign_matches=cr['row_sign_matches'],
            text_to_det=cr['text_to_det'],
            det_rows_dict=det_rows_dict,
            optim_sign_boxes=cr['sub_optim'].sign_boxes,
        )

        # 7e-i. Text rows mapped visualization
        text_row_vis = BboxVisualizer()
        text_row_vis.draw_text_mapping(
            img=None,  # White canvas for text
            sign_boxes=cr['sub_text'].sign_boxes,
            row_mapping=cr['text_to_det'],
            sign_match_info=text_sign_info,
            mapped_label_prefix="D",
            line_thickness=2,
            marker_size=5,
        )
        text_row_vis.save(os.path.join(output_dir, f"{prefix}_text_rows_mapped.jpg"))

        # 7e-ii. Side-by-side: Detection rows + Text mapping
        comp_rows = CompositeVisualizer()
        comp_rows.compose(
            images=[det_row_vis.result, text_row_vis.result],
            layout=(1, 2),
            titles=[
                f"Detection Rows ({len(cr['sub_det'].get_row_sign_sequences())} rows)",
                f"Text Mapping ({len(cr['sub_text'].get_row_sign_sequences())} rows, "
                f"{len(cr['matches'])} matched)",
            ],
        )
        comp_rows.save(os.path.join(output_dir, f"{prefix}_rows_side_by_side.jpg"))

        # 7e-iii. Alignment diagnostic
        diag_vis = BboxVisualizer()
        diag_vis.draw_alignment_diagnostic(
            img=crop_img.copy(),
            detection_sign_boxes=cr['sub_det'].sign_boxes,
            aligned_text_boxes=cr['sub_optim'].sign_boxes,
            det_sign_match_info=det_sign_info,
            text_sign_match_info=text_sign_info,
            det_to_text=cr['det_to_text'],
        )
        diag_vis.save(os.path.join(output_dir, f"{prefix}_alignment_diagnostic.jpg"))

    # 8. Summary info
    info_path = os.path.join(output_dir, f"{fragment_id}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Fragment: {fragment_id}\n")
        f.write(f"GT boxes: {len(gt_boxes) if gt_boxes else 0}\n")
        f.write(f"Detected: {len(detections)}\n")
        f.write(f"Optimized: {len(all_optimized_full)}\n")
        f.write(f"Sub-tablets processed: {len(crop_results)}\n")
        for ci, cr in enumerate(crop_results):
            n_matches = len(cr['matches'])
            n_optim = len(cr['sub_optim'])
            n_final = len(cr['sub_final'])
            f.write(f"  Crop {ci}: rows_matched={n_matches}, "
                    f"coarse={n_optim}, final={n_final}\n")

    print(f"  Saved visualizations to {output_dir}/{fragment_id}_*")


# ============ Main ============

if __name__ == "__main__":
    print("Cuneiform Signs Alignment (PSR - Point Set Registration)")
    print("=" * 60)

    # Data sources
    local_source = LocalDataSource(ANNOTATIONS_DIR)
    api_source = EBLAPISource()
    sign_resolver = SignAPIResolver()

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

    # Process samples
    results = []
    for i, fid in enumerate(fragments[:SAMPLE_LIMIT]):
        result = process_fragment(
            tablet_detector=tablet_detector,
            local_source=local_source,
            api_source=api_source,
            sign_resolver=sign_resolver,
            fragment_id=fid,
            psr_params=PSR_PARAMS,
        )
        if result:
            results.append({
                'fragment_id': result['fragment_id'],
                'gt_count': result['gt_count'],
                'text_signs': result['text_signs'],
                'detected': result['detected'],
                'aligned': result['aligned'],
                'num_crops': result['num_crops'],
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['fragment_id']}: GT={r['gt_count']}, Text={r['text_signs']}, "
              f"Det={r['detected']}, Aligned={r['aligned']}, Crops={r['num_crops']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "alignment_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
