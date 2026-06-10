"""Cuneiform Signs Alignment – Batch Processing Script (PSR Method)."""
import json
import os
from dotenv import load_dotenv

from sign_alignment import LocalDataSource, ModelConfig, TabletImageDetector, TextVisualizer
from sign_alignment.visualizer import ColorConfig
from sign_alignment.pipeline import (
    CropContext, PipelineConfig, SampleState, Runner, VisOptions,
    step_load_data, step_detect_signs, step_compute_statistics,
    step_results_comparison,
    PIPELINE_STEPS_PER_CROP,
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
    context = CropContext(config=PipelineConfig(
        model_config=model_config,
        tablet_detector=tablet_detector,
        local_source=local_source,
        color_config=ColorConfig,
        output_dir=OUTPUT_DIR,
    ), 
    task_type="samples"
    )

    vis = VisOptions(info=True, display=False, save=True)
    crop_runner = Runner(context, steps=PIPELINE_STEPS_PER_CROP, vis=vis)

    summary = []
    for idx in range(min(SAMPLE_LIMIT, len(crop_runner._fragments))):
        context.state = SampleState()
        crop_runner.choose_sample(idx)
        fid = context.state.fragment_id
        print(f"\n{'='*60}\n{fid}")

        crop_runner.run_single_step(step_load_data)

        crop_runner.run_single_step(step_detect_signs)
        crop_runner.run_single_step(step_compute_statistics)

        s = context.state
        all_optimized_full = []
        for crop_idx in range(len(tablet_detector.get_crop_tablets())):
            crop_runner.choose_crop(crop_idx)
            if not s.det_boxes:
                continue
            crop_runner.run_all()
            if not s.det_rows or not len(s.det_rows) or not s.matches or not s.aligned_boxes:
                continue
            crop_runner.run_single_step(step_results_comparison)

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
