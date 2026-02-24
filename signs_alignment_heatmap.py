"""
Cuneiform Signs Alignment using Heatmap-based Method
Aligns detected signs with unlocated text signs from ebl API using heatmap matching.
Processes full tablet images with sub-tablet detection.

Adapted to use the sign_alignment module (same API as signs_alignment.ipynb).
"""

import json
import os
import copy
import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

from sign_alignment import (
    # Data sources
    LocalDataSource, EBLAPISource, SignTextParser,
    # Sign utilities
    CLASSES_ABZ, SignResolver,
    # Detection
    ModelConfig, TabletImageDetector, SingleImage,
    # Heatmap / alignment
    compute_avg_dimensions, create_detection_heatmap, create_text_heatmap,
    match_heatmaps_ncc, create_text_based_detections,
    transform_gt_to_cropped_region,
    # Sub-tablet / optimizer
    SubTablet, ElasticChainOptimizer,
    # Visualization
    BboxVisualizer, TextVisualizer, HeatmapVisualizer,
    # Bounding box types
    BoundingBox, Detection, GroundTruths,
)

# Allow large image processing
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

load_dotenv()

# ============ Configuration ============
ANNOTATIONS_DIR = os.path.expanduser("~/erc-work-data/data-of-cuneiform-ocr-data/filtered_annotations")
CONFIG_FILE = "configs/detr.py"
CHECKPOINT_FILE = os.path.expanduser("~/erc-work-data/retrained_models/detr-173/epoch_1000.pth")
SCORE_THRESHOLD = 0.5
SCALE_FACTOR = 10
OUTPUT_DIR = "alignment_results_heatmap"
SAMPLE_LIMIT = 20

# Optimizer hyperparameters
OPTIMIZER_PARAMS = dict(
    lambda_data=20000.0,
    lambda_iou=5000.0,
    lambda_seq=0.30,
    lambda_smooth=0.05,
    lambda_anchor=0.1,
    lambda_size=0.1,
    num_iterations=100,
    alpha_geo=0.0,            # disable geometric term for ablation
)

HEATMAP_METHOD = 'gaussian'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============ Pipeline Functions ============

def process_single_crop(
    crop_img: np.ndarray,
    crop_detections: Detection,
    crop_info: dict,
    text_lines: List[List[str]],
    gt_boxes: Optional[GroundTruths],
    avg_width: float,
    avg_height: float,
    scale_factor: int = SCALE_FACTOR,
    method: str = HEATMAP_METHOD,
    optimizer_params: dict = OPTIMIZER_PARAMS,
) -> Dict:
    """
    Full alignment pipeline for a single crop:
      1. Build detection SubTablet + heatmap
      2. Build full-text SubTablet + heatmap
      3. NCC match -> extract sub-region
      4. Elastic chain optimization
      5. Return result dict with optimized SubTablet, alignment info, etc.
    """
    if not crop_detections:
        return None

    margin = max(avg_width, avg_height)

    # --- 1. Detection SubTablet ---
    sub_detection = SubTablet.from_detections(
        img=crop_img,
        detections=crop_detections,
        name="detection",
        avg_width=avg_width,
        avg_height=avg_height,
    )
    sub_detection.create_heatmap(scale_factor=scale_factor, method=method)

    # --- 2. Full-text SubTablet ---
    sub_full_text = SubTablet.from_text_lines(
        text_lines=text_lines,
        avg_width=avg_width,
        avg_height=avg_height,
        margin=margin,
        img=None,
        target_detections=None,
        align_to_detection_centroid=False,
        name="full_text",
    )
    sub_full_text.create_heatmap(scale_factor=scale_factor, method=method)

    # --- 3. NCC matching ---
    top_left_scaled, match_score, top_left_original = match_heatmaps_ncc(
        sub_detection.heatmap,
        sub_full_text.heatmap,
        scale_factor=scale_factor,
    )
    top_left_x_text = top_left_original[0]
    top_left_y_text = top_left_original[1]

    exp_h, exp_w = crop_img.shape[:2]

    # --- 4. Extract text-aligned sub-region ---
    sub_text_aligned = sub_full_text.extract_sub_region(
        offset_x=top_left_x_text,
        offset_y=top_left_y_text,
        width=exp_w,
        height=exp_h,
        img=crop_img,
        name="text_aligned",
    )
    sub_text_aligned.create_heatmap(
        scale_factor=scale_factor,
        img_shape=crop_img.shape,
        method=method,
    )

    # --- 5. Elastic chain optimization ---
    prior_aspect_ratio = avg_width / avg_height if avg_height > 0 else 1.0
    optimizer = ElasticChainOptimizer(
        sub_tablet_text=sub_text_aligned,
        detection_heatmap=sub_detection.heatmap,
        detection_boxes=crop_detections,
        scale_factor=scale_factor,
        lambda_data=optimizer_params.get('lambda_data', 10000.0),
        lambda_iou=optimizer_params.get('lambda_iou', 500.0),
        lambda_seq=optimizer_params.get('lambda_seq', 0.05),
        lambda_smooth=optimizer_params.get('lambda_smooth', 0.15),
        lambda_anchor=optimizer_params.get('lambda_anchor', 0.05),
        lambda_size=optimizer_params.get('lambda_size', 0.1),
        alpha_geo=optimizer_params.get('alpha_geo', 0.0),
        prior_aspect_ratio=prior_aspect_ratio,
        device=DEVICE,
    )
    sub_optimized = optimizer.optimize(
        num_iterations=optimizer_params.get('num_iterations', 100),
        lr=optimizer_params.get('lr', 5.0),
        verbose=False,
    )

    # GT boxes for this crop (if available)
    gt_boxes_crop = transform_gt_to_cropped_region(gt_boxes, crop_info) if gt_boxes else []

    return {
        'sub_detection': sub_detection,
        'sub_text_aligned': sub_text_aligned,
        'sub_optimized': sub_optimized,
        'gt_boxes_crop': gt_boxes_crop,
        'match_score': match_score,
        'optimizer': optimizer,
    }


def process_fragment(
    tablet_detector: TabletImageDetector,
    local_source: LocalDataSource,
    api_source: EBLAPISource,
    fragment_id: str,
    scale_factor: int = SCALE_FACTOR,
    method: str = HEATMAP_METHOD,
    optimizer_params: dict = OPTIMIZER_PARAMS,
) -> Optional[Dict]:
    """
    Process a single fragment end-to-end:
      1. Load image & ground truth
      2. Get sign text from API & parse
      3. Detect signs (full tablet with crops)
      4. For each crop run the alignment pipeline
      5. Collect results, save visualizations
    """
    print(f"\n{'='*60}")
    print(f"Processing fragment: {fragment_id}")

    # --- Load image & ground truth ---
    img = local_source.load_image(fragment_id)
    if img is None:
        print(f"  Image not found for {fragment_id}")
        return None

    gt_boxes = local_source.load_annotation(fragment_id)
    print(f"  Ground truth boxes: {len(gt_boxes) if gt_boxes else 0}")

    # --- Get & parse sign text ---
    signs_text = api_source.get_signs(fragment_id)
    if signs_text is None:
        print(f"  Could not get signs from API for {fragment_id}")
        return None

    text_lines = SignTextParser.parse_api_signs(signs_text)
    total_text_signs = sum(len(line) for line in text_lines)
    print(f"  Text lines: {len(text_lines)}, total signs: {total_text_signs}")

    # --- Detect signs ---
    detections = tablet_detector.detect(img)
    print(f"  Detected signs (score > {tablet_detector.score_threshold}): {len(detections)}")

    cropped_images = tablet_detector.get_cropped_images()
    crop_coordinates = tablet_detector.crop_coordinates
    print(f"  Sub-tablets: {len(cropped_images)}")

    # --- Compute average dimensions from full-tablet detections ---
    avg_width, avg_height = compute_avg_dimensions(detections)

    # --- Process each crop ---
    crop_results = []
    all_optimized_full = []  # optimized bboxes in full-tablet coords

    for idx, crop_single in enumerate(cropped_images):
        result = process_single_crop(
            crop_img=crop_single.img,
            crop_detections=crop_single.detections,
            crop_info=crop_coordinates[idx],
            text_lines=text_lines,
            gt_boxes=gt_boxes,
            avg_width=avg_width,
            avg_height=avg_height,
            scale_factor=scale_factor,
            method=method,
            optimizer_params=optimizer_params,
        )
        if result is None:
            continue
        crop_results.append(result)

        # Transform optimized bboxes to full-tablet coordinates
        ox = crop_coordinates[idx]['x']
        oy = crop_coordinates[idx]['y']
        for sb in result['sub_optimized'].sign_boxes:
            transformed = BoundingBox(
                x1=sb.x1 + ox, y1=sb.y1 + oy,
                x2=sb.x2 + ox, y2=sb.y2 + oy,
                score=sb.score, sign=sb.sign,
            )
            all_optimized_full.append(transformed)

    avg_match = np.mean([r['match_score'] for r in crop_results]) if crop_results else 0.0
    print(f"  Alignment: {len(all_optimized_full)} optimized signs, avg match score: {avg_match:.4f}")

    # --- Visualize ---
    save_visualizations(
        fragment_id=fragment_id,
        img=img,
        gt_boxes=gt_boxes,
        detections=detections,
        text_lines=text_lines,
        all_optimized_full=all_optimized_full,
        crop_results=crop_results,
        output_dir=OUTPUT_DIR,
    )

    return {
        'fragment_id': fragment_id,
        'gt_count': len(gt_boxes) if gt_boxes else 0,
        'text_signs': total_text_signs,
        'detected': len(detections),
        'aligned': len(all_optimized_full),
        'match_score': avg_match,
        'crop_results': crop_results,
        'all_optimized_full': all_optimized_full,
        'gt_boxes': gt_boxes,
        'detections': detections,
    }


def save_visualizations(
    fragment_id: str,
    img: np.ndarray,
    gt_boxes: Optional[GroundTruths],
    detections: Detection,
    text_lines: List[List[str]],
    all_optimized_full: Detection,
    crop_results: List[Dict],
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

    # 3. Text conversion
    TextVisualizer.save_text(
        text_lines,
        path=os.path.join(output_dir, f"{fragment_id}_3_text.txt"),
        fragment_id=fragment_id,
    )

    # 4. Optimized bounding boxes on full tablet
    vis = BboxVisualizer(color=(0, 255, 255))
    vis.draw_boxes(img.copy(), all_optimized_full)
    vis.save(os.path.join(output_dir, f"{fragment_id}_4_optimized.jpg"))

    # 5. Combined: detections (red) + optimized (cyan)
    vis_det = BboxVisualizer(color=(255, 0, 0))
    vis_det.draw_boxes(img.copy(), detections)
    vis_opt = BboxVisualizer(color=(0, 255, 255))
    vis_opt.draw_boxes(vis_det.result, all_optimized_full)
    vis_opt.save(os.path.join(output_dir, f"{fragment_id}_5_combined.jpg"))

    # 6. Combined: optimized (cyan) + GT (green)
    if gt_boxes:
        vis_gt = BboxVisualizer(color=(0, 255, 0))
        vis_gt.draw_boxes(img.copy(), gt_boxes)
        vis_opt_gt = BboxVisualizer(color=(0, 255, 255))
        vis_opt_gt.draw_boxes(vis_gt.result, all_optimized_full)
        vis_opt_gt.save(os.path.join(output_dir, f"{fragment_id}_6_optimized_vs_gt.jpg"))

    # 7. Per-crop detail visualizations
    for ci, cr in enumerate(crop_results):
        prefix = f"{fragment_id}_crop{ci}"
        # detection vs optimized
        vis_d = BboxVisualizer(color=(255, 0, 0))
        vis_d.draw_boxes(cr['sub_detection'].img.copy(), cr['sub_detection'].to_detection_list())
        vis_o = BboxVisualizer(color=(0, 255, 255))
        vis_o.draw_boxes(vis_d.result, cr['sub_optimized'].to_detection_list())
        vis_o.save(os.path.join(output_dir, f"{prefix}_det_vs_opt.jpg"))

    # 8. Summary info
    info_path = os.path.join(output_dir, f"{fragment_id}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Fragment: {fragment_id}\n")
        f.write(f"GT boxes: {len(gt_boxes) if gt_boxes else 0}\n")
        f.write(f"Detected: {len(detections)}\n")
        f.write(f"Optimized: {len(all_optimized_full)}\n")
        f.write(f"Sub-tablets: {len(crop_results)}\n")
        for ci, cr in enumerate(crop_results):
            f.write(f"  Crop {ci}: match_score={cr['match_score']:.4f}, "
                    f"det={len(cr['sub_detection'])}, opt={len(cr['sub_optimized'])}\n")

    print(f"  Saved visualizations to {output_dir}/{fragment_id}_*")


# ============ Main ============

if __name__ == "__main__":
    print("Cuneiform Signs Alignment (Heatmap + Elastic Chain Optimization)")
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

    # Process samples
    results = []
    for i, fid in enumerate(fragments[:SAMPLE_LIMIT]):
        result = process_fragment(
            tablet_detector=tablet_detector,
            local_source=local_source,
            api_source=api_source,
            fragment_id=fid,
            scale_factor=SCALE_FACTOR,
            method=HEATMAP_METHOD,
            optimizer_params=OPTIMIZER_PARAMS,
        )
        if result:
            # Store lightweight summary (drop heavy objects for JSON)
            results.append({
                'fragment_id': result['fragment_id'],
                'gt_count': result['gt_count'],
                'text_signs': result['text_signs'],
                'detected': result['detected'],
                'aligned': result['aligned'],
                'match_score': result['match_score'],
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['fragment_id']}: GT={r['gt_count']}, Text={r['text_signs']}, "
              f"Det={r['detected']}, Aligned={r['aligned']}, Score={r['match_score']:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "alignment_summary.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
