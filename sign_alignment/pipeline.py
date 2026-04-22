from collections import Counter
from dataclasses import dataclass, field
import os
from typing import Any, Optional

import numpy as np
import torch

from sign_alignment.detector import ModelConfig, SingleImage, TabletImageDetector
from sign_alignment.data_source import EBLAPISource, LocalDataSource, GroundTruths, SignAPIResolver, SignTextParser
from sign_alignment.visualizer import (
    BboxVisualizer,
    ColorConfig,
    CompositeVisualizer,
    TextVisualizer,
    build_sign_match_info,
)
from sign_alignment.heatmap import compute_avg_dimensions, transform_gt_to_cropped_region
from sign_alignment.tablet import SubTablet
from sign_alignment.psr_optimizer import PointSetRegistrationOptimizer
from data_processing.line_process import (
    align_text_to_detection_rows,
    create_row_mapping,
    match_rows_dp,
    match_signs_in_row_dp,
)


@dataclass
class PipelineConfig:
    model_config: ModelConfig
    tablet_detector: TabletImageDetector
    local_source: LocalDataSource
    color_config: ColorConfig
    output_dir: str
    exp_image_idx: int = 1


class PipelineTools:
    api_source = EBLAPISource()
    sign_resolver = SignAPIResolver()


@dataclass
class SampleState:
    fragment_id: str = None
    img: np.ndarray = None
    gt_boxes: Optional[GroundTruths] = None

    # Loaded text lines (filtered + unfiltered)
    text_lines: Optional[list] = None
    text_lines_unfiltered: Optional[list] = None

    # Detection on full image and on the chosen cropped piece
    detections: Optional[list] = None
    exp_image: Optional[SingleImage] = None
    crop_info: Optional[dict] = None
    gt_boxes_exp: Optional[list] = None

    # Statistics
    avg_width: float = None
    avg_height: float = None

    # Sub-tablets
    sub_tablet_detection: Optional[SubTablet] = None
    sub_tablet_text: Optional[SubTablet] = None
    sub_tablet_optim: Optional[SubTablet] = None
    sub_tablet_final: Optional[SubTablet] = None

    # Row matching
    det_row_sequences: Optional[list] = None
    text_row_sequences: Optional[list] = None
    matches: Optional[list] = None
    text_to_det: Optional[dict] = None
    det_to_text: Optional[dict] = None

    # Sign-level matching
    row_sign_matches: Optional[dict] = None
    aligned_text_boxes: Optional[list] = None

    # Sign match info + cached visualizer results for composition
    text_sign_match_info: Optional[dict] = None
    det_sign_match_info: Optional[dict] = None
    det_row_vis_image: Optional[np.ndarray] = None

    # PSR optimizer
    optimizer: Optional[PointSetRegistrationOptimizer] = None


# Keep CropContext as a thin wrapper so existing callers still work.
@dataclass
class CropContext:
    config: PipelineConfig
    tools: PipelineTools = field(default_factory=PipelineTools)
    state: SampleState = None

    def __post_init__(self):
        if self.state is None:
            self.state = SampleState()

    # Convenience pass-throughs used by Step functions
    @property
    def tablet_detector(self): return self.config.tablet_detector
    @property
    def local_source(self): return self.config.local_source
    @property
    def api_source(self): return self.tools.api_source
    @property
    def color_config(self): return self.config.color_config
    @property
    def output_dir(self): return self.config.output_dir
    @property
    def fragment_id(self): return self.state.fragment_id
    @fragment_id.setter
    def fragment_id(self, v): self.state.fragment_id = v
    @property
    def img(self): return self.state.img
    @img.setter
    def img(self, v): self.state.img = v
    @property
    def gt_boxes(self): return self.state.gt_boxes
    @gt_boxes.setter
    def gt_boxes(self, v): self.state.gt_boxes = v


class Step:
    def __init__(self, name: str, description: str = None, run=None, visualize=None):
        self.name = name
        self.description = description
        self.run = run
        self.visualize = visualize


class Runner:
    def __init__(self, context: CropContext, steps: list[Step]):
        self.context = context
        self.steps = steps

        fragments = context.local_source.get_available_fragments()
        self._fragments = fragments
        print(f"Found {len(fragments)} fragments with both image and annotation")

    def run_single_step(self, step: Step):
        info_message = f"Step: {step.name}"
        if step.description:
            info_message += f" - {step.description}"
        print(info_message)
        if step.run:
            step.run(self.context)
        if step.visualize:
            step.visualize(self.context)

    def choose_sample(self, idx: int):
        fragment_id = self._fragments[idx]
        print(f"Processing sample: {fragment_id}")
        self.context.fragment_id = fragment_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _out(context: CropContext, suffix: str) -> str:
    return os.path.join(context.output_dir, f"debug_{context.fragment_id}_{suffix}")


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

def _load_from_local_source(context: CropContext):
    context.img = context.local_source.load_image(context.fragment_id)
    context.gt_boxes = context.local_source.load_annotation(context.fragment_id)
    print(f"Ground truth boxes: {len(context.gt_boxes)}")


def _visualize_ground_truth(context: CropContext):
    gt_bbox_visualizer = BboxVisualizer(context.color_config.GT_COLOR.value)
    gt_bbox_visualizer.draw_boxes(context.img.copy(), context.gt_boxes)
    gt_bbox_visualizer.save(_out(context, "gt.jpg"))


def _load_sign_text_from_api(context: CropContext):
    fragment_data = context.api_source.get_fragment_data(context.fragment_id)
    if fragment_data is None:
        raise ValueError(f"No fragment data found for sample {context.fragment_id}")

    text_data = fragment_data.get('text', {})
    text_lines = SignTextParser.parse_text_lines(text_data, filter_broken=True, sign_resolver=context.tools.sign_resolver)
    total_text_signs = sum(len(line) for line in text_lines)
    print(f"  Text lines: {len(text_lines)}, total signs: {total_text_signs}")

    text_lines_unfiltered = SignTextParser.parse_text_lines(text_data, filter_broken=False, sign_resolver=context.tools.sign_resolver)
    total_unfiltered = sum(len(line) for line in text_lines_unfiltered)
    print(f"  Unfiltered: {total_unfiltered} signs, broken signs removed: {total_unfiltered - total_text_signs}")

    TextVisualizer.save_text(text_lines, path=_out(context, "text_filtered.txt"), fragment_id=context.fragment_id)
    TextVisualizer.save_text(text_lines_unfiltered, path=_out(context, "text.txt"), fragment_id=context.fragment_id)

    context.state.text_lines = text_lines
    context.state.text_lines_unfiltered = text_lines_unfiltered


def _detect_signs(context: CropContext):
    detections = context.tablet_detector.detect(context.img)
    cropped = context.tablet_detector.get_cropped_images()
    exp_idx = context.config.exp_image_idx

    context.state.detections = detections
    context.state.exp_image = cropped[exp_idx]
    context.state.crop_info = context.tablet_detector.crop_coordinates[exp_idx]


def _visualize_detections(context: CropContext):
    color = context.color_config.DET_COLOR.value
    exp_image = context.state.exp_image

    full_vis = BboxVisualizer(color=color)
    full_vis.draw_boxes(context.img.copy(), context.state.detections)
    full_vis.save(_out(context, "det.jpg"))

    exp_vis = BboxVisualizer(color=color)
    exp_vis.draw_boxes(exp_image.img.copy(), exp_image.detections)
    exp_vis.save(_out(context, "exp_image.jpg"))


def _transform_gt_to_exp(context: CropContext):
    crop_info = context.state.crop_info
    exp_idx = context.config.exp_image_idx
    print(f"Crop info for exp_image (index {exp_idx}): "
          f"x={crop_info['x']}, y={crop_info['y']}, w={crop_info['w']}, h={crop_info['h']}")

    gt_boxes_exp = transform_gt_to_cropped_region(context.gt_boxes, crop_info)
    print(f"GT boxes in full image: {len(context.gt_boxes)}")
    print(f"GT boxes in exp_image: {len(gt_boxes_exp)}")
    context.state.gt_boxes_exp = gt_boxes_exp


def _visualize_gt_on_exp(context: CropContext):
    if not context.state.gt_boxes_exp:
        return
    vis = BboxVisualizer(color=context.color_config.GT_COLOR.value)
    vis.draw_boxes(context.state.exp_image.img.copy(), context.state.gt_boxes_exp)
    vis.display_result(vis_opt="save", path=_out(context, "exp_gt.jpg"))


def _compute_statistics(context: CropContext):
    avg_width, avg_height = compute_avg_dimensions(context.state.detections)
    context.state.avg_width = avg_width
    context.state.avg_height = avg_height
    print(f"full image shape: {context.img.shape}")
    print(f"Exp image shape: {context.state.exp_image.img.shape}")
    print(f"Average detected sign width: {avg_width:.2f}, height: {avg_height:.2f}")


def _create_subtablets(context: CropContext):
    s = context.state
    sub_tablet_detection = SubTablet.from_detections(
        img=s.exp_image.img,
        detections=s.exp_image.detections,
        name="detection",
        avg_width=s.avg_width,
        avg_height=s.avg_height,
    )
    sub_tablet_text = SubTablet.from_text_lines(
        text_lines=s.text_lines,
        avg_width=s.avg_width,
        avg_height=s.avg_height,
        img=s.exp_image.img,
        target_detections=s.exp_image.detections,
        align_to_detection_centroid=True,
        name="text",
    )
    s.sub_tablet_detection = sub_tablet_detection
    s.sub_tablet_text = sub_tablet_text
    print(f"Sub-tablet detection: {sub_tablet_detection.info}")
    print(f"Sub-tablet text: {sub_tablet_text.info}")


def _detect_rows(context: CropContext):
    s = context.state
    eps = 0.4
    num_rows = s.sub_tablet_detection.detect_rows(
        eps=eps,
        min_samples=1,
        lambda_weight=0.007,
    )

    avg_size = (s.sub_tablet_detection.avg_width + s.sub_tablet_detection.avg_height) / 2
    print(f"=== Row Detection Results (Scale-Normalized) ===")
    print(f"Average sign size: {avg_size:.2f} pixels")
    print(f"eps=0.6 → actual distance threshold ≈ {0.6 * avg_size:.2f} pixels")
    print(f"Detected {num_rows} rows")
    print(f"Number of signs: {len(s.sub_tablet_detection)}")

    row_counts = Counter(sb.row_idx for sb in s.sub_tablet_detection.sign_boxes)
    print(f"\nBoxes per row:")
    for row_idx in sorted(row_counts.keys()):
        if row_idx == -1:
            print(f"  Noise: {row_counts[row_idx]} boxes")
        else:
            print(f"  Row {row_idx}: {row_counts[row_idx]} boxes")

    print(f"\nFirst 5 signs with row info:")
    for i, sb in enumerate(s.sub_tablet_detection.sign_boxes[:5]):
        print(f"  {i+1}. {sb.sign_name} (row={sb.row_idx}): center=({sb.cx:.1f}, {sb.cy:.1f})")

    # Text subtablet rows are pre-assigned; just summarize
    num_rows_text = len(s.sub_tablet_text.get_rows())
    print(f"\n=== Text SubTablet Row Info ===")
    print(f"Number of rows: {num_rows_text}")
    print(f"Number of signs: {len(s.sub_tablet_text)}")
    row_counts_text = Counter(sb.row_idx for sb in s.sub_tablet_text.sign_boxes)
    print(f"\nSigns per text row:")
    for row_idx in sorted(row_counts_text.keys()):
        if row_idx >= 0:
            print(f"  Row {row_idx}: {row_counts_text[row_idx]} signs")


def _match_rows(context: CropContext):
    s = context.state
    det_row_sequences = s.sub_tablet_detection.get_row_sign_sequences()
    text_row_sequences = s.sub_tablet_text.get_row_sign_sequences()
    print(f"=== Row Matching Setup ===")
    print(f"Detection rows: {len(det_row_sequences)}")
    print(f"Text rows: {len(text_row_sequences)}")

    matches, _ = match_rows_dp(
        detection_rows=det_row_sequences,
        text_rows=text_row_sequences,
        skip_text_penalty=0.5,
        skip_det_penalty=1,
        skip_small_det_penalty=0.2,
        small_det_threshold=1,
        similarity_method='jaccard',
    )

    print(f"\n=== Matching Results ===")
    print(f"Number of matched rows: {len(matches)}")
    print(f"\nMatches (Text row → Detection row):")
    for text_idx, det_idx in matches:
        text_signs = text_row_sequences[text_idx]
        det_signs = det_row_sequences[det_idx]
        print(f"  Text row {text_idx} ({len(text_signs)} signs) → Detection row {det_idx} ({len(det_signs)} signs)")
        print(f"    Text:  {' '.join(text_signs[:5])}{'...' if len(text_signs) > 5 else ''}")
        print(f"    Det:   {' '.join(det_signs[:5])}{'...' if len(det_signs) > 5 else ''}")

    text_to_det, det_to_text = create_row_mapping(matches, len(text_row_sequences), len(det_row_sequences))
    print(f"\n=== Row Mapping ===")
    print(f"Text to Detection: {text_to_det}")
    print(f"Detection to Text: {det_to_text}")

    s.det_row_sequences = det_row_sequences
    s.text_row_sequences = text_row_sequences
    s.matches = matches
    s.text_to_det = text_to_det
    s.det_to_text = det_to_text


def _visualize_detection_rows(context: CropContext):
    s = context.state
    det_row_vis = BboxVisualizer(color=(255, 0, 0))
    det_row_vis.draw_rows(
        s.sub_tablet_detection.img.copy(),
        s.sub_tablet_detection.sign_boxes,
        show_labels=True,
        show_row_numbers=True,
        row_mapping=s.det_to_text,
        row_label_prefix="D",
        mapped_label_prefix="R",
        line_thickness=2,
        marker_size=5,
    )
    print(f"Detection rows with row numbers (D# on left margin, D#→R# for matched)")
    det_row_vis.display_result(vis_opt="draw")
    det_row_vis.save(_out(context, "detection_rows.jpg"))
    s.det_row_vis_image = det_row_vis.result


def _match_signs_in_rows(context: CropContext):
    s = context.state
    row_sign_matches = {}
    print(f"=== Within-Row Sign Matching ===")
    print(f"Processing {len(s.matches)} matched row pairs...\n")

    for text_row_idx, det_row_idx in s.matches:
        text_signs = s.text_row_sequences[text_row_idx]
        det_signs = s.det_row_sequences[det_row_idx]

        sign_matches, _ = match_signs_in_row_dp(
            detection_signs=det_signs,
            text_signs=text_signs,
            skip_text_penalty=0.5,
            skip_det_penalty=2.0,
            mismatch_cost=0.9,
        )
        row_sign_matches[text_row_idx] = sign_matches

        print(f"Text row {text_row_idx} → Detection row {det_row_idx}:")
        print(f"  Text signs: {len(text_signs)}, Detection signs: {len(det_signs)}")
        print(f"  Matched signs: {len(sign_matches)}")
        print(f"  First 5 matches:")
        for i, (t_idx, d_idx) in enumerate(sign_matches[:5]):
            print(f"    {i+1}. Text[{t_idx}]={text_signs[t_idx]} ↔ Det[{d_idx}]={det_signs[d_idx]}")
        if len(sign_matches) > 5:
            print(f"    ... and {len(sign_matches)-5} more matches")
        print()

    print(f"=== Sign Matching Summary ===")
    total_text_signs = sum(len(s.text_row_sequences[t_idx]) for t_idx, _ in s.matches)
    total_det_signs = sum(len(s.det_row_sequences[d_idx]) for _, d_idx in s.matches)
    total_matched_signs = sum(len(sm) for sm in row_sign_matches.values())
    print(f"Total text signs in matched rows: {total_text_signs}")
    print(f"Total detection signs in matched rows: {total_det_signs}")
    print(f"Total matched sign pairs: {total_matched_signs}")

    s.row_sign_matches = row_sign_matches


def _align_text_rows(context: CropContext):
    s = context.state
    det_rows = s.sub_tablet_detection.get_rows_dict()
    text_rows = s.sub_tablet_text.get_rows_dict()

    print(f"=== Row Alignment ===")
    print(f"Aligning matched text rows to detection rows using baseline with slope...\n")

    aligned_text_boxes = align_text_to_detection_rows(
        det_rows=det_rows,
        text_rows=text_rows,
        text_to_det=s.text_to_det,
        row_sign_matches=s.row_sign_matches,
        avg_width=s.avg_width,
        avg_height=s.avg_height,
        min_width_ratio=2 / 3,
        max_width_ratio=4 / 3,
    )

    print(f"\n=== Alignment Summary ===")
    print(f"Total aligned sign boxes: {len(aligned_text_boxes)}")
    print(f"Matched text rows aligned: {len(s.row_sign_matches)}")
    print(f"(Only matched rows are processed; unmatched text rows are excluded)")

    s.aligned_text_boxes = aligned_text_boxes


def _create_optim_subtablet(context: CropContext):
    s = context.state
    sub_tablet_optim = SubTablet(
        sign_boxes=s.aligned_text_boxes,
        img=s.sub_tablet_detection.img,
        name="optim",
        avg_width=s.avg_width,
        avg_height=s.avg_height,
    )
    s.sub_tablet_optim = sub_tablet_optim
    print(f"optimized subtablet created, info: {sub_tablet_optim.info}")

    optim_row_counts = Counter(sb.row_idx for sb in sub_tablet_optim.sign_boxes)
    print(f"\n  Signs per row in optim subtablet:")
    for row_idx in sorted(optim_row_counts.keys()):
        if row_idx >= 0:
            print(f"    Row {row_idx}: {optim_row_counts[row_idx]} signs")

    print(f"\n  First 5 signs in optim subtablet:")
    for i, sb in enumerate(sub_tablet_optim.sign_boxes[:5]):
        print(f"    {i+1}. {sb.sign_name} (row={sb.row_idx}): center=({sb.cx:.1f}, {sb.cy:.1f}), size=({sb.width:.1f}, {sb.height:.1f})")


def _build_sign_match_info(context: CropContext):
    s = context.state
    det_rows = s.sub_tablet_detection.get_rows_dict()

    text_sign_match_info, det_sign_match_info = build_sign_match_info(
        row_sign_matches=s.row_sign_matches,
        text_to_det=s.text_to_det,
        det_rows_dict=det_rows,
        optim_sign_boxes=s.sub_tablet_optim.sign_boxes,
    )

    n_same = sum(1 for v in text_sign_match_info.values() if v["status"] == "same")
    n_diff = sum(1 for v in text_sign_match_info.values() if v["status"] == "diff")
    n_unmatched = sum(1 for v in text_sign_match_info.values() if v["status"] == "unmatched")
    n_det_unmatched = sum(1 for v in det_sign_match_info.values() if v["status"] == "unmatched")

    print(f"=== Sign Match Info ===")
    print(f"  Matched, same label:    {n_same}")
    print(f"  Matched, diff label:    {n_diff}")
    print(f"  Unmatched text signs:   {n_unmatched}")
    print(f"  Unmatched det signs:    {n_det_unmatched}")

    s.text_sign_match_info = text_sign_match_info
    s.det_sign_match_info = det_sign_match_info


def _visualize_sign_match_info(context: CropContext):
    s = context.state

    text_row_vis = BboxVisualizer()
    text_row_vis.draw_text_mapping(
        img=None,
        sign_boxes=s.sub_tablet_text.sign_boxes,
        row_mapping=s.text_to_det,
        sign_match_info=s.text_sign_match_info,
        mapped_label_prefix="D",
        line_thickness=2,
        marker_size=5,
    )
    text_row_vis.display_result(vis_opt="draw")
    text_row_vis.save(_out(context, "text_rows_mapped.jpg"))

    if s.det_row_vis_image is not None:
        comp = CompositeVisualizer()
        comp.compose(
            images=[s.det_row_vis_image, text_row_vis.result],
            layout=(1, 2),
            titles=[
                f"Detection Rows ({len(s.det_row_sequences)} rows)",
                f"Text Mapping ({len(s.text_row_sequences)} rows, {len(s.matches)} matched)",
            ],
            figsize=(20, 10),
        )
        comp.save(_out(context, "rows_side_by_side.jpg"))

    diag_vis = BboxVisualizer()
    diag_vis.draw_alignment_diagnostic(
        img=s.sub_tablet_detection.img.copy(),
        detection_sign_boxes=s.sub_tablet_detection.sign_boxes,
        aligned_text_boxes=s.sub_tablet_optim.sign_boxes,
        det_sign_match_info=s.det_sign_match_info,
        text_sign_match_info=s.text_sign_match_info,
        det_to_text=s.det_to_text,
        line_thickness=2,
        marker_size=5,
    )
    diag_vis.display_result(vis_opt="draw")
    diag_vis.save(_out(context, "alignment_diagnostic.jpg"))


def _offset_analysis(context: CropContext):
    s = context.state
    det_rows = s.sub_tablet_detection.get_rows_dict()

    match_pairs = {}
    for text_row_idx, sign_matches in s.row_sign_matches.items():
        det_row_idx = s.text_to_det[text_row_idx]
        for t_idx, d_idx in sign_matches:
            match_pairs[(text_row_idx, t_idx)] = (det_row_idx, d_idx)

    print(f"=== Position Offset Analysis (matched signs) ===")
    print(f"For matched signs, aligned cx = det_box.cx, but cy = baseline_y(cx) ≠ det_box.cy")
    print(f"Also: height = avg_height ({s.avg_height:.1f}), width = computed, not det_box dimensions\n")

    offsets_cx, offsets_cy, offsets_w, offsets_h = [], [], [], []
    for sb in s.sub_tablet_optim.sign_boxes:
        key = (sb.row_idx, sb.col_idx)
        if key in match_pairs:
            det_row_idx, det_sign_idx = match_pairs[key]
            det_box = det_rows[det_row_idx][det_sign_idx]
            offsets_cx.append(sb.cx - det_box.cx)
            offsets_cy.append(sb.cy - det_box.cy)
            offsets_w.append(sb.width - det_box.width)
            offsets_h.append(sb.height - det_box.height)

    offsets_cx = np.array(offsets_cx)
    offsets_cy = np.array(offsets_cy)
    offsets_w = np.array(offsets_w)
    offsets_h = np.array(offsets_h)

    print(f"  Δcx: mean={offsets_cx.mean():.2f}, std={offsets_cx.std():.2f}, |max|={np.abs(offsets_cx).max():.2f}")
    print(f"  Δcy: mean={offsets_cy.mean():.2f}, std={offsets_cy.std():.2f}, |max|={np.abs(offsets_cy).max():.2f}")
    print(f"  Δw:  mean={offsets_w.mean():.2f}, std={offsets_w.std():.2f}, |max|={np.abs(offsets_w).max():.2f}")
    print(f"  Δh:  mean={offsets_h.mean():.2f}, std={offsets_h.std():.2f}, |max|={np.abs(offsets_h).max():.2f}")
    print(f"\n  → Δcx should be ~0 (uses det_box.cx directly)")
    print(f"  → Δcy ≠ 0 because baseline_y(cx) ≠ det_box.cy (forced onto regression line)")
    print(f"  → Δw ≠ 0 because width is recomputed from spacing, not det_box.width")
    print(f"  → Δh ≠ 0 because height is always avg_height, not det_box.height")


def _create_psr_optimizer(context: CropContext):
    s = context.state
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimizer = PointSetRegistrationOptimizer(
        sub_tablet_text=s.sub_tablet_optim,
        target_detections=s.exp_image.detections,
        sigma=s.avg_width * 1.5,
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
        contour_mask=s.exp_image.mask,
        device=device,
    )
    s.optimizer = optimizer

    print(f"=== PSR Optimizer Created ===")
    print(f"  Device: {optimizer.device}")
    print(f"  Source points (M): {optimizer.M}")
    print(f"  Target points (N): {optimizer.N}")
    print(f"  Sigma: {optimizer.sigma:.1f}")
    print(f"  w_noise: {optimizer.w_noise}")
    print(f"  Lambdas: data={optimizer.lambda_data}, anchor={optimizer.lambda_anchor}, "
          f"seq={optimizer.lambda_seq}, height={optimizer.lambda_height}, rows={optimizer.lambda_rows}, "
          f"boundary={optimizer.lambda_boundary}")
    print(f"  Contour mask: {'available' if optimizer.contour_mask is not None else 'not set'}")

    optimizer.plot_loss_curves(save_dir="alignment_loss_functions", show=False)


def _run_psr_optimization(context: CropContext):
    s = context.state
    sub_tablet_final = s.optimizer.optimize(
        num_iterations=80,
        lr=1.0,
        sigma_anneal=True,
        sigma_final=None,
        verbose=True,
        log_every=20,
    )
    s.sub_tablet_final = sub_tablet_final
    print(f"\n=== Optimization Complete ===")
    print(f"Final subtablet: {len(sub_tablet_final)} signs")


def _plot_loss_history(context: CropContext):
    context.state.optimizer.plot_loss_history()


def _visualize_results_comparison(context: CropContext):
    s = context.state
    exp_image = s.exp_image

    before_vis = BboxVisualizer(color=(0, 255, 255))
    before_vis.draw_boxes(exp_image.img.copy(), s.sub_tablet_optim.to_detection_list())
    before_vis.save(_out(context, "coarse_aligned.jpg"))

    after_vis = BboxVisualizer(color=(255, 255, 0))
    after_vis.draw_boxes(exp_image.img.copy(), s.sub_tablet_final.to_detection_list())
    after_vis.save(_out(context, "final_optimized.jpg"))

    det_ov = BboxVisualizer(color=(255, 0, 0))
    det_ov.draw_boxes(exp_image.img.copy(), s.sub_tablet_detection.to_detection_list())
    opt_ov = BboxVisualizer(color=(255, 255, 0))
    opt_ov.draw_boxes(det_ov.result, s.sub_tablet_final.to_detection_list())
    opt_ov.save(_out(context, "overlay_det_final.jpg"))

    gt_ov = BboxVisualizer(color=(0, 255, 0))
    gt_ov.draw_boxes(exp_image.img.copy(), s.gt_boxes_exp or [])
    opt_gt = BboxVisualizer(color=(255, 255, 0))
    opt_gt.draw_boxes(gt_ov.result, s.sub_tablet_final.to_detection_list())
    opt_gt.save(_out(context, "overlay_gt_final.jpg"))

    comp = CompositeVisualizer()
    comp.compose(
        images=[before_vis.result, after_vis.result, opt_ov.result, opt_gt.result],
        layout=(2, 2),
        titles=[
            f"Before PSR: Coarse Aligned ({len(s.sub_tablet_optim)} signs)",
            f"After PSR: Final Optimized ({len(s.sub_tablet_final)} signs)",
            "Overlay: Detection (red) + Final (yellow)",
            "Overlay: GT (green) + Final (yellow)",
        ],
        figsize=(16, 12),
    )
    comp.display_result(vis_opt="draw")
    comp.save(_out(context, "results_comparison.jpg"))

    print("Cyan = Coarse aligned,  Yellow = Final optimized,  Red = Detection,  Green = Ground Truth")


def _analyze_param_changes(context: CropContext):
    s = context.state
    param_changes = s.optimizer.get_param_changes()

    print("=== Parameter Changes (Coarse → Final) ===")
    print(f"  Δcx: mean={param_changes[:, 0].mean():.2f}, std={param_changes[:, 0].std():.2f}, "
          f"|max|={np.abs(param_changes[:, 0]).max():.2f}")
    print(f"  Δcy: mean={param_changes[:, 1].mean():.2f}, std={param_changes[:, 1].std():.2f}, "
          f"|max|={np.abs(param_changes[:, 1]).max():.2f}")
    print(f"  Δw:  mean={param_changes[:, 2].mean():.2f}, std={param_changes[:, 2].std():.2f}")
    print(f"  Δh:  mean={param_changes[:, 3].mean():.2f}, std={param_changes[:, 3].std():.2f}")

    print("\n=== First 5 Signs: Coarse → Final ===")
    n = min(5, len(s.sub_tablet_optim.sign_boxes))
    for i in range(n):
        before = s.sub_tablet_optim.sign_boxes[i]
        after = s.sub_tablet_final.sign_boxes[i]
        print(f"  {i+1}. {before.sign_name}:")
        print(f"      Coarse: cx={before.cx:.1f}, cy={before.cy:.1f}, "
              f"w={before.width:.1f}, h={before.height:.1f}")
        print(f"      Final:  cx={after.cx:.1f}, cy={after.cy:.1f}, "
              f"w={after.width:.1f}, h={after.height:.1f}")
        print(f"      Δ:      Δcx={after.cx-before.cx:.1f}, Δcy={after.cy-before.cy:.1f}, "
              f"Δw={after.width-before.width:.1f}, Δh={after.height-before.height:.1f}")


# ---------------------------------------------------------------------------
# Step instances
# ---------------------------------------------------------------------------

step_load_data = Step(
    name="Load Data",
    run=_load_from_local_source)

step_show_ground_truth = Step(
    name="Show Ground Truth", visualize=_visualize_ground_truth)

step_load_sign_text = Step(
    name="Load Sign Text from API", run=_load_sign_text_from_api)

step_detect_signs = Step(
    name="Detect Signs",
    run=_detect_signs,
    visualize=_visualize_detections)

step_transform_gt_to_exp = Step(
    name="Transform GT to Exp Image",
    run=_transform_gt_to_exp,
    visualize=_visualize_gt_on_exp)

step_compute_statistics = Step(
    name="Compute Detection Statistics",
    run=_compute_statistics)

step_create_subtablets = Step(
    name="Create Sub-tablets",
    run=_create_subtablets)

step_detect_rows = Step(
    name="Detect Rows (DBSCAN)",
    run=_detect_rows)

step_match_rows = Step(
    name="Match Detection Rows to Text Rows",
    run=_match_rows)

step_visualize_detection_rows = Step(
    name="Visualize Detection Rows",
    visualize=_visualize_detection_rows)

step_match_signs_in_rows = Step(
    name="Within-Row Sign Matching",
    run=_match_signs_in_rows)

step_align_text_rows = Step(
    name="Align Text Rows to Detection Rows",
    run=_align_text_rows)

step_create_optim_subtablet = Step(
    name="Create Optim Sub-tablet",
    run=_create_optim_subtablet)

step_build_sign_match_info = Step(
    name="Build Sign Match Info & Visualize Mapping",
    run=_build_sign_match_info,
    visualize=_visualize_sign_match_info)

step_offset_analysis = Step(
    name="Position Offset Analysis",
    run=_offset_analysis)

step_create_psr_optimizer = Step(
    name="Create PSR Optimizer",
    run=_create_psr_optimizer)

step_run_psr_optimization = Step(
    name="Run PSR Optimization",
    run=_run_psr_optimization)

step_plot_loss_history = Step(
    name="Plot Loss History",
    visualize=_plot_loss_history)

step_results_comparison = Step(
    name="Results Comparison",
    visualize=_visualize_results_comparison)

step_param_changes = Step(
    name="Analyze Parameter Changes",
    run=_analyze_param_changes)


DEBUG_STEPS = [
    step_load_data,
    step_show_ground_truth,
    step_load_sign_text,
    step_detect_signs,
    step_transform_gt_to_exp,
    step_compute_statistics,
    step_create_subtablets,
    step_detect_rows,
    step_match_rows,
    step_visualize_detection_rows,
    step_match_signs_in_rows,
    step_align_text_rows,
    step_create_optim_subtablet,
    step_build_sign_match_info,
    step_offset_analysis,
    step_create_psr_optimizer,
    step_run_psr_optimization,
    step_plot_loss_history,
    step_results_comparison,
    step_param_changes,
]
