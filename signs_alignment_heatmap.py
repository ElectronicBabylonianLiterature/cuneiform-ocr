"""
Cuneiform Signs Alignment using Heatmap-based Method
Aligns detected signs with unlocated text signs from ebl API using heatmap matching.
Processes full tablet images with sub-tablet detection.
"""

import json
import os
import requests
import copy
import numpy as np
import cv2
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
from pymongo import MongoClient
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from data_processing.divide_photos import divide_tablet_photo

# Import from signs_alignment.py
from signs_alignment import (
    CLASSES, ABZ_TO_SIGN, abz_to_sign_name,
    load_ground_truth, load_image, get_signs_from_api, parse_api_signs,
    TabletImageDetector, SingleImage, SingleImageDetector,
    compute_avg_dimensions,
    create_detection_heatmap, create_text_heatmap, match_heatmaps_ncc, create_text_based_detections,
    BboxVisualizer, TextVisualizer, HeatmapVisualizer,
    get_available_fragments
)

# Allow large image processing
Image.MAX_IMAGE_PIXELS = None

# ============ Configuration ============
ANNOTATIONS_DIR = os.path.expanduser("~/erc-work-data/data-of-cuneiform-ocr-data/filtered_annotations")
CONFIG_FILE = "configs/detr.py"
CHECKPOINT_FILE = os.path.expanduser("~/erc-work-data/retrained_models/detr-173/epoch_1000.pth")
SCORE_THRESHOLD = 0.5
SCALE_FACTOR = 10  # for heatmap scaling
OUTPUT_DIR = "alignment_results_heatmap"
SAMPLE_LIMIT = 15  # number of samples to process


def transform_subtablet_detections_to_full(cropped_images: List[SingleImage], 
                                            crop_coordinates: List[Dict]) -> List[Dict]:
    """
    Transform detections from cropped sub-tablets to full tablet coordinates.
    
    Args:
        cropped_images: List of SingleImage objects with detections
        crop_coordinates: List of dicts with 'x', 'y' offsets for each crop
    
    Returns:
        all_detections: List of detection dicts in full tablet coordinates
    """
    all_detections = []
    
    for idx, crop_img in enumerate(cropped_images):
        piece_offset_x = crop_coordinates[idx]['x']
        piece_offset_y = crop_coordinates[idx]['y']
        
        for det in crop_img.detections:
            # Deep copy to avoid modifying original
            det_full = copy.deepcopy(det)
            bbox = det_full['bbox']
            det_full['bbox'] = [
                bbox[0] + piece_offset_x,
                bbox[1] + piece_offset_y,
                bbox[2] + piece_offset_x,
                bbox[3] + piece_offset_y
            ]
            all_detections.append(det_full)
    
    return all_detections


def align_signs_heatmap(img, detections, text_lines, classes, scale_factor=10):
    """
    Align detected signs with text signs using heatmap-based method.
    
    Args:
        img: Full tablet image
        detections: List of detection dicts with 'bbox', 'abz_name', etc.
        text_lines: List of lines, each line is list of sign names
        classes: List of class names (ABZ)
        scale_factor: Scale factor for heatmap
    
    Returns:
        aligned_detections: List of detection dicts with aligned positions
        alignment_info: Dict with alignment information (match_pos, margin, etc.)
    """
    if not detections or not text_lines:
        return [], {}
    
    # Compute average dimensions
    avg_width, avg_height = compute_avg_dimensions(detections)
    
    # Create detection heatmap
    detection_heatmap, _, _ = create_detection_heatmap(
        detections, img.shape, classes, scale_factor, avg_width, avg_height
    )
    
    # Create text heatmap
    text_heatmap, margin, _, _ = create_text_heatmap(
        text_lines, classes, avg_width, avg_height, scale_factor
    )
    
    # Match heatmaps
    top_left_scaled, match_score, top_left_original = match_heatmaps_ncc(
        detection_heatmap, text_heatmap, scale_factor
    )
    
    top_left_x_text = top_left_original[0]
    top_left_y_text = top_left_original[1]
    
    # Convert to full tablet coordinates
    top_left_x_tablet = top_left_x_text - margin
    top_left_y_tablet = top_left_y_text - margin
    
    # Create text-based detections
    img_height, img_width = img.shape[:2]
    aligned_detections = create_text_based_detections(
        text_lines, classes, top_left_x_text, top_left_y_text, 
        margin, avg_width, avg_height, (img_width, img_height)
    )
    
    alignment_info = {
        'match_score': match_score,
        'match_position_tablet': (top_left_x_tablet, top_left_y_tablet),
        'match_position_text': (top_left_x_text, top_left_y_text),
        'margin': margin,
        'avg_width': avg_width,
        'avg_height': avg_height,
        'num_detected': len(detections),
        'num_aligned': len(aligned_detections)
    }
    
    return aligned_detections, alignment_info


def visualize_heatmap_alignment(fragment_id, img, gt_boxes, detections, text_lines, 
                                  aligned_detections, alignment_info, output_dir):
    """Create and save visualization results for heatmap-based alignment"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ground truth boxes (if available)
    if gt_boxes:
        bbox_visualizer = BboxVisualizer(boxes_color=(0, 255, 0))
        vis_gt = bbox_visualizer.draw_boxes(img.copy(), gt_boxes)  # green
        cv2.imwrite(os.path.join(output_dir, f"{fragment_id}_1_ground_truth.jpg"), vis_gt)
    
    # 2. Detection results
    bbox_visualizer = BboxVisualizer(boxes_color=(255, 0, 0))
    vis_det = bbox_visualizer.draw_boxes(img.copy(), detections)  # red
    cv2.imwrite(os.path.join(output_dir, f"{fragment_id}_2_detections.jpg"), vis_det)
    
    # 3. Text conversion (ABZ -> sign names)
    text_visualizer = TextVisualizer(text_lines)
    text_filepath = os.path.join(output_dir, f"{fragment_id}_3_text_conversion.txt")
    text_visualizer.write_text_file(text_filepath, fragment_id)
    
    # 4. Aligned bounding boxes (blue for text-based)
    bbox_visualizer = BboxVisualizer(boxes_color=(0, 0, 255))
    vis_aligned = bbox_visualizer.draw_boxes(img.copy(), aligned_detections)  # blue
    cv2.imwrite(os.path.join(output_dir, f"{fragment_id}_4_aligned_heatmap.jpg"), vis_aligned)
    
    # 5. Combined visualization (detections in red, aligned in blue with transparency)
    img_combined = img.copy()
    # Draw detections in red
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img_combined, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Draw aligned in blue with semi-transparency
    overlay = img_combined.copy()
    for det in aligned_detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img_combined, 1 - alpha, 0, img_combined)
    cv2.imwrite(os.path.join(output_dir, f"{fragment_id}_5_combined.jpg"), img_combined)
    
    # 6. Save alignment info
    info_filepath = os.path.join(output_dir, f"{fragment_id}_6_alignment_info.txt")
    with open(info_filepath, 'w') as f:
        f.write(f"Fragment: {fragment_id}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Average Match Score: {alignment_info['match_score']:.4f}\n")
        f.write(f"Detected Signs: {alignment_info['num_detected']}\n")
        f.write(f"Aligned Signs: {alignment_info['num_aligned']}\n")
        f.write(f"Number of Sub-tablets: {alignment_info.get('num_sub_tablets', 1)}\n")
        
        # Write sub-tablet details if available
        if 'sub_tablet_infos' in alignment_info:
            f.write("\nSub-tablet Details:\n")
            for info in alignment_info['sub_tablet_infos']:
                f.write(f"  Sub-tablet {info['sub_tablet_idx']}: "
                       f"Score={info['match_score']:.4f}, "
                       f"Aligned={info['num_aligned']}\n")
    
    print(f"Saved visualizations to {output_dir}/{fragment_id}_*.jpg/txt")


def process_fragment_heatmap(model, fragment_id, scale_factor=10):
    """Process a single fragment using heatmap-based alignment"""
    print(f"\n{'='*60}")
    print(f"Processing fragment: {fragment_id}")
    
    # Load image
    img = load_image(fragment_id)
    if img is None:
        print(f"  Image not found for {fragment_id}")
        return None
    
    # Load ground truth
    gt_boxes = load_ground_truth(fragment_id)
    print(f"  Ground truth boxes: {len(gt_boxes) if gt_boxes else 0}")
    
    # Get signs from API
    signs_text = get_signs_from_api(fragment_id)
    if signs_text is None:
        print(f"  Could not get signs from API for {fragment_id}")
        return None
    
    print(f"  Raw API signs (first 200 chars): {signs_text[:200]}...")
    
    # Parse text signs (ABZ -> sign names)
    text_lines = parse_api_signs(signs_text)
    total_text_signs = sum(len(line) for line in text_lines)
    print(f"  Text lines: {len(text_lines)}, total signs: {total_text_signs}")
    
    # Detect signs using TabletImageDetector with crop
    # We need to keep the cropped images to process each sub-tablet separately
    detector = TabletImageDetector(model, CLASSES, SCORE_THRESHOLD, 
                                     visualize_crop=False, logging_crop=False, keep_crops=True)
    
    # Get cropped images with detections in sub-tablet coordinates
    cropped_images, crop_coordinates = divide_tablet_photo(
        img, visualize=False, logging=False, return_coordinates=True
    )
    
    # Detect on each cropped image separately
    from signs_alignment import SingleImageDetector
    single_detector = SingleImageDetector(model, CLASSES, SCORE_THRESHOLD)
    
    cropped_with_detections = []
    all_detections_full = []  # All detections in full tablet coordinates
    
    for idx, crop_img in enumerate(cropped_images):
        # Detect in sub-tablet coordinates
        crop_detections = single_detector.detect(crop_img)
        cropped_with_detections.append(SingleImage(img=crop_img, detections=crop_detections))
        
        # Transform to full tablet coordinates for visualization
        piece_offset_x = crop_coordinates[idx]['x']
        piece_offset_y = crop_coordinates[idx]['y']
        
        for det in crop_detections:
            det_full = copy.deepcopy(det)
            bbox = det_full['bbox']
            det_full['bbox'] = [
                bbox[0] + piece_offset_x,
                bbox[1] + piece_offset_y,
                bbox[2] + piece_offset_x,
                bbox[3] + piece_offset_y
            ]
            all_detections_full.append(det_full)
    
    print(f"  Detected signs (score > {SCORE_THRESHOLD}): {len(all_detections_full)}")
    print(f"  Processed {len(cropped_with_detections)} sub-tablets")
    
    # Process each sub-tablet separately with heatmap alignment
    all_aligned_detections = []
    alignment_infos = []
    
    for idx, crop_data in enumerate(cropped_with_detections):
        if len(crop_data.detections) == 0:
            continue
        
        # Align signs for this sub-tablet
        aligned_detections, alignment_info = align_signs_heatmap(
            crop_data.img, crop_data.detections, text_lines, CLASSES, scale_factor
        )
        
        # Transform aligned detections to full tablet coordinates
        piece_offset_x = crop_coordinates[idx]['x']
        piece_offset_y = crop_coordinates[idx]['y']
        
        for det in aligned_detections:
            det_full = copy.deepcopy(det)
            bbox = det_full['bbox']
            det_full['bbox'] = [
                bbox[0] + piece_offset_x,
                bbox[1] + piece_offset_y,
                bbox[2] + piece_offset_x,
                bbox[3] + piece_offset_y
            ]
            # Update center if it exists
            if 'center' in det_full:
                center = det_full['center']
                det_full['center'] = (center[0] + piece_offset_x, center[1] + piece_offset_y)
            
            all_aligned_detections.append(det_full)
        
        alignment_infos.append({
            'sub_tablet_idx': idx,
            'match_score': alignment_info['match_score'],
            'num_aligned': alignment_info['num_aligned']
        })
    
    # Aggregate alignment info
    avg_match_score = np.mean([info['match_score'] for info in alignment_infos]) if alignment_infos else 0.0
    total_aligned = len(all_aligned_detections)
    
    aggregated_info = {
        'match_score': avg_match_score,
        'num_detected': len(all_detections_full),
        'num_aligned': total_aligned,
        'num_sub_tablets': len(cropped_with_detections),
        'sub_tablet_infos': alignment_infos
    }
    
    print(f"  Alignment: {total_aligned} signs aligned across {len(alignment_infos)} sub-tablets, avg match score: {avg_match_score:.4f}")
    
    # Visualize
    visualize_heatmap_alignment(fragment_id, img, gt_boxes, all_detections_full, text_lines, 
                                 all_aligned_detections, aggregated_info, OUTPUT_DIR)
    
    return {
        'fragment_id': fragment_id,
        'gt_count': len(gt_boxes) if gt_boxes else 0,
        'text_signs': total_text_signs,
        'detected': len(all_detections_full),
        'aligned': total_aligned,
        'match_score': avg_match_score
    }


if __name__ == "__main__":
    print("Cuneiform Signs Alignment (Heatmap-based)")
    print("=" * 60)
    
    # Initialize model
    print("Loading detection model...")
    register_all_modules()
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
    print("Model loaded.")
    
    # Get available fragments
    fragments = get_available_fragments()
    print(f"Found {len(fragments)} fragments with both image and annotation")
    
    # Process samples
    results = []
    for i, fragment_id in enumerate(fragments[:SAMPLE_LIMIT]):
        result = process_fragment_heatmap(model, fragment_id, SCALE_FACTOR)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['fragment_id']}: GT={r['gt_count']}, Text={r['text_signs']}, "
              f"Det={r['detected']}, Aligned={r['aligned']}, Score={r['match_score']:.4f}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "alignment_summary_heatmap.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/")
