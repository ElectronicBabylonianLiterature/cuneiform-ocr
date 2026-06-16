"""Cuneiform Signs Alignment – Batch Processing Script (PSR Method)."""
import json
import os
from dotenv import load_dotenv

from sign_alignment import LocalDataSource, ModelConfig, TabletImageDetector, TextVisualizer
from sign_alignment.visualizer import ColorConfig
from sign_alignment.pipeline import (
    CropContext, Runner, SampleState, Step, VisOptions,
    align_text_rows,
    build_sign_match_info,
    create_box_sets,
    create_psr_optimizer,
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
    vis_results_comparison,
    vis_row_matches,
    vis_sign_match_info,
    vis_sign_matches,
)

load_dotenv()

ANNOTATIONS_DIR = os.path.expanduser("~/erc-work-data/data-of-cuneiform-ocr-data/filtered_annotations")
CONFIG_FILE = "configs/detr.py"
CHECKPOINT_FILE = os.path.expanduser("~/erc-work-data/retrained_models/detr-173/epoch_1000.pth")
SCORE_THRESHOLD = 0.5
OUTPUT_DIR = "alignment_results"
SAMPLE_LIMIT = 7

if __name__ == "__main__":
    local_source = LocalDataSource(ANNOTATIONS_DIR)
    model_config = ModelConfig(CONFIG_FILE, CHECKPOINT_FILE, device='auto')
    tablet_detector = TabletImageDetector(model_config, SCORE_THRESHOLD, keep_crops=True)
    context = CropContext(
        tablet_detector=tablet_detector,
        local_source=local_source,
        color_config=ColorConfig,
        output_dir=OUTPUT_DIR,
        task_type="samples",
    )

    vis = VisOptions(info=True, display=False, save=True)
    crop_runner = Runner(context, vis=vis)

    summary = []
    for idx in range(min(SAMPLE_LIMIT, len(crop_runner._fragments))):
        context.state = SampleState()
        crop_runner.choose_sample(idx)
        fid = context.state.fragment_id
        print(f"\n{'='*60}\n{fid}")

        crop_runner.run([
            Step("Load data", load_data, vis_loaded_data),
            Step("Detect signs", detect_signs, vis_detections),
            Step("Detection statistics", lambda _: None, vis_detection_statistics),
        ])

        s = context.state
        all_optimized_full = []
        for crop_idx in range(len(tablet_detector.get_crop_tablets())):
            crop_runner.choose_crop(crop_idx)
            if not s.det_boxes:
                continue
            crop_runner.run([
                Step("Transform GT to crop", transform_gt_to_crop, vis_crop_ground_truth),
                Step("Create box sets", create_box_sets, vis_box_sets),
                Step("Detect rows", detect_rows, vis_detected_rows_info),
                Step("Match rows", match_rows, vis_row_matches),
                Step("Match signs", match_signs_in_rows, vis_sign_matches),
                Step("Align text rows", align_text_rows, vis_aligned_rows),
                Step("Build sign match info", build_sign_match_info, vis_sign_match_info),
                Step("Create PSR optimizer", create_psr_optimizer, vis_psr_optimizer),
                Step("Optimize PSR", optimize_psr, vis_optimization),
            ])
            if not s.det_rows or not len(s.det_rows) or not s.matches or not s.aligned_boxes:
                continue
            crop_runner.run([
                Step("Results comparison", lambda _: None, vis_results_comparison)
            ])

            for sb in s.final_boxes:
                all_optimized_full.append(sb.to_tablet(s.tablet))

        TextVisualizer.save_text(
            s.text_lines,
            path=os.path.join(OUTPUT_DIR, f"{fid}_3_text.txt"),
            fragment_id=fid,
        )
        summary.append({
            'fragment_id': fid,
            'gt_count': len(s.gt_boxes or []),
            'detected': len(s.detections or []),
            'aligned': len(all_optimized_full),
        })
        print(f"  Aligned: {len(all_optimized_full)} signs")

    with open(os.path.join(OUTPUT_DIR, "alignment_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/")
