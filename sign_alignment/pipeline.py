from collections import Counter
from dataclasses import dataclass, field
import os
from typing import Optional

import cv2
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
from sign_alignment.protosnap import (
    CenterFreezeRefiner,
    CenterRefinement,
    ProtoSnapConfig,
    SnapStatus,
    load_dift_model,
    make_refiner,
    render_refinement_grid,
    summarize_refinements,
    collect_used_prototype_paths,
)
from data_processing.line_process import (
    align_text_to_detection_rows,
    create_row_mapping,
    match_rows_dp,
    match_signs_in_row_dp,
)


# ---------------------------------------------------------------------------
# Visualization options
# ---------------------------------------------------------------------------

@dataclass
class VisOptions:
    """Control which visualization outputs are produced."""
    info: bool = True      # Print text information to stdout
    display: bool = True   # Display images inline (e.g., matplotlib in notebook)
    save: bool = True      # Save images to disk


# ---------------------------------------------------------------------------
# Config / Tools
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    model_config: ModelConfig
    tablet_detector: TabletImageDetector
    local_source: LocalDataSource
    color_config: ColorConfig
    output_dir: str
    api_source: EBLAPISource = field(default_factory=EBLAPISource)
    img_idx: int = 1          # which cropped sub-image to use
    # ProtoSnap is opt-in. None disables the step (default behavior).
    # Pass a ProtoSnapConfig to enable; the step still no-ops gracefully if
    # assets/weights are missing.
    protosnap: Optional[ProtoSnapConfig] = None
    # Optional PSR optimizer parameter overrides (dict or None for defaults).
    # Keys: sigma_factor, w_noise, lambda_data, lambda_anchor, lambda_seq,
    #       lambda_height, lambda_rows, lambda_boundary,
    #       rows_threshold_ratio_far, rows_threshold_ratio_close,
    #       rows_plateau_far, rows_plateau_close, num_iterations, lr, sigma_anneal
    psr_params: Optional[dict] = None


class PipelineTools:
    sign_resolver = SignAPIResolver()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SampleState:
    """All intermediate results for a single fragment, including SubTablets."""

    fragments: list = None # all available fragments (populated at Runner init)
    fragment_id: str = None

    # full-image data
    img: np.ndarray = None
    gt_boxes: Optional[GroundTruths] = None

    # Text lines parsed from API
    text_lines: Optional[list] = None
    text_lines_unfiltered: Optional[list] = None

    # Full-image detections
    detections: Optional[list] = None

    # chosen sub-image (one crop of the tablet)
    sub_image: Optional[SingleImage] = None
    crop_info: Optional[dict] = None
    gt_boxes_img: Optional[list] = None         # GT boxes in sub-image coords

    # sign-size statistics
    avg_width: float = None
    avg_height: float = None

    # SubTablets (unified intermediate representations)
    sub_tablet_detection: Optional[SubTablet] = None   # detected signs in sub_image
    sub_tablet_text: Optional[SubTablet] = None        # text signs on virtual grid
    sub_tablet_aligned: Optional[SubTablet] = None     # coarse-aligned (formerly optim)
    sub_tablet_final: Optional[SubTablet] = None       # after PSR optimization

    # ProtoSnap center-freeze refiner + diagnostics.
    snap_refiner: Optional[CenterFreezeRefiner] = None
    snap_meta: Optional[dict] = None                   # period, font, repo_root, etc.
    snap_refinements: Optional[list] = None            # list[CenterRefinement] after hook fires
    snap_summary: Optional[dict] = None
    snap_used_prototype_paths: Optional[list] = None

    # row matching
    det_row_sequences: Optional[list] = None
    text_row_sequences: Optional[list] = None
    matches: Optional[list] = None
    text_to_det: Optional[dict] = None
    det_to_text: Optional[dict] = None

    # sign-level matching
    row_sign_matches: Optional[dict] = None

    # sign-match info for visualisation
    text_sign_match_info: Optional[dict] = None
    det_sign_match_info: Optional[dict] = None
    det_row_vis_image: Optional[np.ndarray] = None

    # PSR optimizer
    optimizer: Optional[PointSetRegistrationOptimizer] = None


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class CropContext:
    config: PipelineConfig
    tools: PipelineTools = field(default_factory=PipelineTools)
    state: SampleState = None
    # Cached SDFeaturizer for SD-DIFT; loaded once, reused across samples.
    _snap_dift: Optional[object] = field(default=None, repr=False, compare=False)

    task_type: str = "debug"  

    def __post_init__(self):
        if self.state is None:
            self.state = SampleState()

    @property
    def tablet_detector(self): return self.config.tablet_detector
    @property
    def local_source(self): return self.config.local_source
    @property
    def api_source(self): return self.config.api_source
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


# ---------------------------------------------------------------------------
# Step / Runner
# ---------------------------------------------------------------------------

class Step:
    def __init__(self, name: str, description: str = None, run=None, visualize=None):
        self.name = name
        self.description = description
        self.run = run
        self.visualize = visualize


class Runner:
    def __init__(self, context: CropContext, steps: list[Step],
                 vis: VisOptions = None):
        self.context = context
        self.steps = steps
        self.vis = vis or VisOptions()

        fragments = context.local_source.get_available_fragments()
        self._fragments = fragments
        context.state.fragments = fragments
        print(f"Found {len(fragments)} fragments with both image and annotation")

        if vis.save:
             os.makedirs(context.output_dir, exist_ok=True)

    def run_single_step(self, step: Step):
        if self.vis.info:
            info_message = f"Step: {step.name}"
            if step.description:
                info_message += f" - {step.description}"
            print(info_message)
        if step.run:
            step.run(self.context)
        if step.visualize:
            step.visualize(self.context, self.vis)

    def choose_sample(self, idx: int):
        fragment_id = self._fragments[idx]
        print(f"Processing sample: {fragment_id}")
        self.context.fragment_id = fragment_id

    def choose_crop(self, crop_idx: int):
        self.context.config.img_idx = crop_idx
        cropped = self.context.tablet_detector.get_cropped_images()
        if cropped:
            self.context.state.sub_image = cropped[crop_idx]
            self.context.state.crop_info = self.context.tablet_detector.crop_coordinates[crop_idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _out(context: CropContext, suffix: str) -> str:
    return os.path.join(context.output_dir, f"{context.task_type}_{context.fragment_id}_{suffix}")


# ---------------------------------------------------------------------------
# Step: Load Data  (image + GT + sign text from API)
# ---------------------------------------------------------------------------

def _load_data(context: CropContext):
    s = context.state

    s.img = context.local_source.load_image(context.fragment_id)
    s.gt_boxes = context.local_source.load_annotation(context.fragment_id)

    fragment_data = context.api_source.get_fragment_data(context.fragment_id)
    if fragment_data is None:
        raise ValueError(f"No fragment data found for sample {context.fragment_id}")

    text_data = fragment_data.get("text", {})
    s.text_lines = SignTextParser.parse_text_lines(
        text_data, filter_broken=True, sign_resolver=context.tools.sign_resolver)
    s.text_lines_unfiltered = SignTextParser.parse_text_lines(
        text_data, filter_broken=False, sign_resolver=context.tools.sign_resolver)


def _visualize_load_data(context: CropContext, vis: VisOptions):
    s = context.state

    if vis.info:
        print(f"Ground truth boxes: {len(s.gt_boxes)}")
        total_text = sum(len(l) for l in s.text_lines)
        total_unfilt = sum(len(l) for l in s.text_lines_unfiltered)
        print(f"  Text lines: {len(s.text_lines)}, total signs: {total_text}")
        print(f"  Unfiltered: {total_unfilt} signs, broken removed: {total_unfilt - total_text}")

    if vis.save:
        TextVisualizer.save_text(
            s.text_lines, path=_out(context, "text_filtered.txt"),
            fragment_id=context.fragment_id)
        TextVisualizer.save_text(
            s.text_lines_unfiltered, path=_out(context, "text.txt"),
            fragment_id=context.fragment_id)
        gt_vis = BboxVisualizer(context.color_config.GT_COLOR.value)
        gt_vis.draw_boxes(s.img.copy(), s.gt_boxes)
        gt_vis.save(_out(context, "gt.jpg"))

    if vis.display:
        gt_vis = BboxVisualizer(context.color_config.GT_COLOR.value)
        gt_vis.draw_boxes(s.img.copy(), s.gt_boxes)
        gt_vis.display_result(vis_opt="draw")


# ---------------------------------------------------------------------------
# Step: Detect Signs
# ---------------------------------------------------------------------------

def _detect_signs(context: CropContext):
    s = context.state
    s.detections = context.tablet_detector.detect(s.img)
    cropped = context.tablet_detector.get_cropped_images()
    img_idx = context.config.img_idx
    if img_idx >= len(cropped):
        img_idx = len(cropped) - 1
    s.sub_image = cropped[img_idx]
    s.crop_info = context.tablet_detector.crop_coordinates[img_idx]


def _visualize_detect_signs(context: CropContext, vis: VisOptions):
    s = context.state
    color = context.color_config.DET_COLOR.value

    if vis.info:
        print(f"Total detections (full image): {len(s.detections)}")
        print(f"Sub-image detections: {len(s.sub_image.detections)}")
        ci = s.crop_info
        print(f"Crop info (img_idx={context.config.img_idx}): "
              f"x={ci['x']}, y={ci['y']}, w={ci['w']}, h={ci['h']}")

    if vis.save:
        full_vis = BboxVisualizer(color=color)
        full_vis.draw_boxes(s.img.copy(), s.detections)
        full_vis.save(_out(context, "det.jpg"))

        img_vis = BboxVisualizer(color=color)
        img_vis.draw_boxes(s.sub_image.img.copy(), s.sub_image.detections)
        img_vis.save(_out(context, "sub_image.jpg"))

    if vis.display:
        img_vis = BboxVisualizer(color=color)
        img_vis.draw_boxes(s.sub_image.img.copy(), s.sub_image.detections)
        img_vis.display_result(vis_opt="draw")


# ---------------------------------------------------------------------------
# Step: Transform GT to sub-image coords
# ---------------------------------------------------------------------------

def _transform_gt_to_img(context: CropContext):
    s = context.state
    s.gt_boxes_img = transform_gt_to_cropped_region(s.gt_boxes, s.crop_info)


def _visualize_transform_gt(context: CropContext, vis: VisOptions):
    s = context.state

    if vis.info:
        print(f"GT boxes (full image): {len(s.gt_boxes)}")
        print(f"GT boxes (sub-image):  {len(s.gt_boxes_img)}")

    if not s.gt_boxes_img:
        return

    if vis.save:
        v = BboxVisualizer(color=context.color_config.GT_COLOR.value)
        v.draw_boxes(s.sub_image.img.copy(), s.gt_boxes_img)
        v.save(_out(context, "sub_image_gt.jpg"))

    if vis.display:
        v = BboxVisualizer(color=context.color_config.GT_COLOR.value)
        v.draw_boxes(s.sub_image.img.copy(), s.gt_boxes_img)
        v.display_result(vis_opt="draw")


# ---------------------------------------------------------------------------
# Step: Compute Statistics
# ---------------------------------------------------------------------------

def _compute_statistics(context: CropContext):
    s = context.state
    s.avg_width, s.avg_height = compute_avg_dimensions(s.detections)


def _visualize_statistics(context: CropContext, vis: VisOptions):
    s = context.state
    if vis.info:
        print(f"Full image shape: {s.img.shape}")
        print(f"Sub-image shape:  {s.sub_image.img.shape}")
        print(f"Average detected sign  width: {s.avg_width:.2f}")
        print(f"Average detected sign height: {s.avg_height:.2f}")


# ---------------------------------------------------------------------------
# Step: Create SubTablets
# ---------------------------------------------------------------------------

def _create_subtablets(context: CropContext):
    s = context.state
    s.sub_tablet_detection = SubTablet.from_detections(
        img=s.sub_image.img,
        detections=s.sub_image.detections,
        name="detection",
        avg_width=s.avg_width,
        avg_height=s.avg_height,
    )
    s.sub_tablet_text = SubTablet.from_text_lines(
        text_lines=s.text_lines,
        avg_width=s.avg_width,
        avg_height=s.avg_height,
        img=s.sub_image.img,
        target_detections=s.sub_image.detections,
        align_to_detection_centroid=True,
        name="text",
    )


def _visualize_create_subtablets(context: CropContext, vis: VisOptions):
    if vis.info:
        print(context.state.sub_tablet_detection.info)
        print(context.state.sub_tablet_text.info)


# ---------------------------------------------------------------------------
# Step: Detect Rows
# ---------------------------------------------------------------------------

def _detect_rows(context: CropContext):
    context.state.sub_tablet_detection.detect_rows(
        eps=0.4, min_samples=1, lambda_weight=0.007)


def _visualize_detect_rows(context: CropContext, vis: VisOptions):
    s = context.state
    det = s.sub_tablet_detection

    if vis.info:
        avg_size = (det.avg_width + det.avg_height) / 2
        num_rows = len(det.get_rows())
        print(f"=== Row Detection Results ===")
        print(f"Average sign size: {avg_size:.2f} px, detected {num_rows} rows, {len(det)} signs")
        row_counts = Counter(sb.row_idx for sb in det.sign_boxes)
        for row_idx in sorted(row_counts):
            label = "Noise" if row_idx == -1 else f"Row {row_idx}"
            print(f"  {label}: {row_counts[row_idx]} boxes")

        print(f"\n=== Text SubTablet Row Info ===")
        print(f"Rows: {len(s.sub_tablet_text.get_rows())}, signs: {len(s.sub_tablet_text)}")
        row_counts_text = Counter(sb.row_idx for sb in s.sub_tablet_text.sign_boxes)
        for row_idx in sorted(row_counts_text):
            if row_idx >= 0:
                print(f"  Row {row_idx}: {row_counts_text[row_idx]} signs")


# ---------------------------------------------------------------------------
# Step: Match Rows
# ---------------------------------------------------------------------------

def _match_rows(context: CropContext):
    s = context.state
    det_row_sequences = s.sub_tablet_detection.get_row_sign_sequences()
    text_row_sequences = s.sub_tablet_text.get_row_sign_sequences()

    matches, _ = match_rows_dp(
        detection_rows=det_row_sequences,
        text_rows=text_row_sequences,
        skip_text_penalty=0.5,
        skip_det_penalty=1,
        skip_small_det_penalty=0.2,
        small_det_threshold=1,
        similarity_method="jaccard",
    )
    text_to_det, det_to_text = create_row_mapping(
        matches, len(text_row_sequences), len(det_row_sequences))

    s.det_row_sequences = det_row_sequences
    s.text_row_sequences = text_row_sequences
    s.matches = matches
    s.text_to_det = text_to_det
    s.det_to_text = det_to_text


def _visualize_match_rows(context: CropContext, vis: VisOptions):
    s = context.state
    if not vis.info:
        return
    print(f"=== Row Matching ===")
    print(f"Detection rows: {len(s.det_row_sequences)}, Text rows: {len(s.text_row_sequences)}, "
          f"Matched: {len(s.matches)}")
    for text_idx, det_idx in s.matches:
        ts = s.text_row_sequences[text_idx]
        ds = s.det_row_sequences[det_idx]
        print(f"  Text row {text_idx} ({len(ts)} signs) → Det row {det_idx} ({len(ds)} signs)")
        print(f"    Text: {' '.join(ts[:5])}{'...' if len(ts) > 5 else ''}")
        print(f"    Det:  {' '.join(ds[:5])}{'...' if len(ds) > 5 else ''}")
    print(f"Text→Det: {s.text_to_det}")
    print(f"Det→Text: {s.det_to_text}")


# ---------------------------------------------------------------------------
# Step: Visualize Detection Rows
# ---------------------------------------------------------------------------

def _visualize_detection_rows(context: CropContext, vis: VisOptions):
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
    s.det_row_vis_image = det_row_vis.result

    if vis.info:
        print("Detection rows: D# on left margin, matched rows show D#->R#")
    if vis.display:
        det_row_vis.display_result(vis_opt="draw")
    if vis.save:
        det_row_vis.save(_out(context, "detection_rows.jpg"))


# ---------------------------------------------------------------------------
# Step: Match Signs Within Rows
# ---------------------------------------------------------------------------

def _match_signs_in_rows(context: CropContext):
    s = context.state
    row_sign_matches = {}
    for text_row_idx, det_row_idx in s.matches:
        sign_matches, _ = match_signs_in_row_dp(
            detection_signs=s.det_row_sequences[det_row_idx],
            text_signs=s.text_row_sequences[text_row_idx],
            skip_text_penalty=0.5,
            skip_det_penalty=2.0,
            mismatch_cost=0.9,
        )
        row_sign_matches[text_row_idx] = sign_matches
    s.row_sign_matches = row_sign_matches


def _visualize_match_signs(context: CropContext, vis: VisOptions):
    s = context.state
    if not vis.info:
        return
    print(f"=== Within-Row Sign Matching ===")
    for text_row_idx, det_row_idx in s.matches:
        ts = s.text_row_sequences[text_row_idx]
        ds = s.det_row_sequences[det_row_idx]
        sm = s.row_sign_matches[text_row_idx]
        print(f"Text row {text_row_idx} -> Det row {det_row_idx}: "
              f"{len(ts)} text, {len(ds)} det, {len(sm)} matched")
        for i, (t_idx, d_idx) in enumerate(sm[:5]):
            print(f"  {i+1}. Text[{t_idx}]={ts[t_idx]} <-> Det[{d_idx}]={ds[d_idx]}")
        if len(sm) > 5:
            print(f"  ... and {len(sm)-5} more")
    total_matched = sum(len(sm) for sm in s.row_sign_matches.values())
    print(f"Total matched sign pairs: {total_matched}")


# ---------------------------------------------------------------------------
# Step: Align Text Rows to Detection Rows
# ---------------------------------------------------------------------------

def _align_text_rows(context: CropContext):
    s = context.state
    aligned_text_boxes = align_text_to_detection_rows(
        det_rows=s.sub_tablet_detection.get_rows_dict(),
        text_rows=s.sub_tablet_text.get_rows_dict(),
        text_to_det=s.text_to_det,
        row_sign_matches=s.row_sign_matches,
        avg_width=s.avg_width,
        avg_height=s.avg_height,
        min_width_ratio=2 / 3,
        max_width_ratio=4 / 3,
    )
    s.sub_tablet_aligned = SubTablet(
        sign_boxes=aligned_text_boxes,
        img=s.sub_tablet_detection.img,
        name="aligned",
        avg_width=s.avg_width,
        avg_height=s.avg_height,
    )


def _visualize_align_text_rows(context: CropContext, vis: VisOptions):
    s = context.state
    if not vis.info:
        return
    print(f"=== Row Alignment Summary ===")
    print(f"Total aligned sign boxes: {len(s.sub_tablet_aligned)}")
    print(f"Matched text rows aligned: {len(s.row_sign_matches)}")
    row_counts = Counter(sb.row_idx for sb in s.sub_tablet_aligned.sign_boxes)
    for row_idx in sorted(row_counts):
        if row_idx >= 0:
            print(f"  Row {row_idx}: {row_counts[row_idx]} signs")


# ---------------------------------------------------------------------------
# Step: Build Sign Match Info & Diagnostic
# ---------------------------------------------------------------------------

def _build_sign_match_info(context: CropContext):
    s = context.state
    text_sign_match_info, det_sign_match_info = build_sign_match_info(
        row_sign_matches=s.row_sign_matches,
        text_to_det=s.text_to_det,
        det_rows_dict=s.sub_tablet_detection.get_rows_dict(),
        optim_sign_boxes=s.sub_tablet_aligned.sign_boxes,
    )
    s.text_sign_match_info = text_sign_match_info
    s.det_sign_match_info = det_sign_match_info


def _visualize_sign_match_info(context: CropContext, vis: VisOptions):
    s = context.state

    if vis.info:
        n_same = sum(1 for v in s.text_sign_match_info.values() if v["status"] == "same")
        n_diff = sum(1 for v in s.text_sign_match_info.values() if v["status"] == "diff")
        n_unmatched_text = sum(1 for v in s.text_sign_match_info.values() if v["status"] == "unmatched")
        n_unmatched_det = sum(1 for v in s.det_sign_match_info.values() if v["status"] == "unmatched")
        print(f"=== Sign Match Info ===")
        print(f"  Matched, same label:  {n_same}")
        print(f"  Matched, diff label:  {n_diff}")
        print(f"  Unmatched text signs: {n_unmatched_text}")
        print(f"  Unmatched det signs:  {n_unmatched_det}")

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

    diag_vis = BboxVisualizer()
    diag_vis.draw_alignment_diagnostic(
        img=s.sub_tablet_detection.img.copy(),
        detection_sign_boxes=s.sub_tablet_detection.sign_boxes,
        aligned_text_boxes=s.sub_tablet_aligned.sign_boxes,
        det_sign_match_info=s.det_sign_match_info,
        text_sign_match_info=s.text_sign_match_info,
        det_to_text=s.det_to_text,
        line_thickness=2,
        marker_size=5,
    )

    if vis.display:
        text_row_vis.display_result(vis_opt="draw")
        diag_vis.display_result(vis_opt="draw")

    if vis.save:
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

        diag_vis.save(_out(context, "alignment_diagnostic.jpg"))


# ---------------------------------------------------------------------------
# Step: Offset Analysis
# ---------------------------------------------------------------------------

def _visualize_offset_analysis(context: CropContext, vis: VisOptions):
    if not vis.info:
        return
    s = context.state
    det_rows = s.sub_tablet_detection.get_rows_dict()

    match_pairs = {}
    for text_row_idx, sign_matches in s.row_sign_matches.items():
        det_row_idx = s.text_to_det[text_row_idx]
        for t_idx, d_idx in sign_matches:
            match_pairs[(text_row_idx, t_idx)] = (det_row_idx, d_idx)

    offsets = {"cx": [], "cy": [], "w": [], "h": []}
    for sb in s.sub_tablet_aligned.sign_boxes:
        key = (sb.row_idx, sb.col_idx)
        if key in match_pairs:
            det_row_idx, det_sign_idx = match_pairs[key]
            det_box = det_rows[det_row_idx][det_sign_idx]
            offsets["cx"].append(sb.cx - det_box.cx)
            offsets["cy"].append(sb.cy - det_box.cy)
            offsets["w"].append(sb.width - det_box.width)
            offsets["h"].append(sb.height - det_box.height)

    if not offsets["cx"]:
        print("No matched pairs found for offset analysis.")
        return

    print(f"=== Position Offset Analysis (coarse-aligned vs detection) ===")
    for key, label in [("cx", "cx"), ("cy", "cy"), ("w", "w "), ("h", "h ")]:
        arr = np.array(offsets[key])
        print(f"  Delta {label}: mean={arr.mean():.2f}, std={arr.std():.2f}, |max|={np.abs(arr).max():.2f}")


# ---------------------------------------------------------------------------
# Step: ProtoSnap setup (build refiner, resolve period; no compute yet)
# ---------------------------------------------------------------------------

def _protosnap_setup(context: CropContext):
    """Build the ProtoSnap CenterFreezeRefiner if enabled.

    Resolves `period` from the EBL fragment API (script.period), maps it
    to a font, and instantiates the refiner. Does NOT run any feature
    matching here — that happens inside the PSR mid-iteration hook.

    No-op (graceful) when:
      - config.protosnap is None
      - the prototype bank cannot find its repo / fonts directory
    """
    s = context.state
    s.snap_refiner = None
    s.snap_refinements = None
    s.snap_summary = None
    s.snap_used_prototype_paths = None

    cfg = context.config.protosnap
    period = None
    fragment_data = context.api_source.get_fragment_data(context.fragment_id)
    if fragment_data is not None:
        script = fragment_data.get("script") or {}
        period = script.get("period") or None

    # Pre-load SD-DIFT model once per session (cached on context, not state).
    if cfg is not None and context._snap_dift is None:
        print("  [ProtoSnap] Loading SD-DIFT model (first time, ~30 s)...")
        context._snap_dift = load_dift_model(cfg)
        if context._snap_dift is None:
            print("  [ProtoSnap] WARNING: SD-DIFT model failed to load; step will be disabled.")

    refiner, meta = make_refiner(cfg, period=period, dift=context._snap_dift)
    s.snap_meta = meta
    s.snap_refiner = refiner


def _visualize_protosnap_setup(context: CropContext, vis: VisOptions):
    s = context.state
    if not vis.info:
        return
    meta = s.snap_meta or {}
    print("=== ProtoSnap Setup ===")
    print(f"  API period:    {meta.get('period')!r}")
    print(f"  Resolved font: {meta.get('font')!r}  ({meta.get('font_reason')})")
    print(f"  repo_root:     {meta.get('repo_root')}")
    print(f"  estimator:     {meta.get('estimator')}")
    print(f"  enabled:       {meta.get('enabled')}")
    if s.snap_refiner is not None:
        cov = s.snap_refiner.bank.coverage(s.sub_tablet_aligned.sign_boxes)
        print(f"  prototype coverage: {cov['with_prototype']}/{cov['total']} "
              f"(metadata_size={cov['metadata_size']})")
        if cov["missing_sign_top10"]:
            print(f"  top missing signs: {', '.join(cov['missing_sign_top10'])}")


# ---------------------------------------------------------------------------
# Step: ProtoSnap per-sign crop grid visualization
# ---------------------------------------------------------------------------

def _visualize_protosnap_crops(context: CropContext, vis: VisOptions):
    """Render a grid showing every sign that passed through ProtoSnap.

    Each cell shows the prototype (left) and the sign crop (right) with
    the original bbox center (cyan circle) and the ProtoSnap-estimated
    center (green cross = accepted, red cross = rejected) connected by an
    arrow.  A color-coded border and text header/footer indicate the
    acceptance status and gate reason.
    """
    s = context.state
    if not s.snap_refinements:
        if vis.info:
            print("ProtoSnap crop grid: no refinements to show.")
        return

    if vis.info:
        n_ok  = sum(1 for r in s.snap_refinements if r.accepted)
        n_all = len(s.snap_refinements)
        print(f"=== ProtoSnap Crop Grid ({n_all} signs, {n_ok} accepted) ===")
        print("  Legend: cyan circle = original center  "
              "green cross = accepted  red cross = rejected")

    grid = render_refinement_grid(s.snap_refinements, thumb=128, cols=5)

    if vis.display:
        import matplotlib.pyplot as plt
        grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        h, w = grid_rgb.shape[:2]
        fig_w = min(20.0, w / 60.0)
        fig_h = fig_w * h / max(w, 1)
        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(grid_rgb)
        plt.axis("off")
        plt.title("ProtoSnap: prototype (left) | annotated crop (right)")
        plt.tight_layout(pad=0.3)
        plt.show()

    if vis.save:
        out_path = _out(context, "protosnap_crop_grid.jpg")
        cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if vis.info:
            print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Step: Create PSR Optimizer
# ---------------------------------------------------------------------------

def _create_psr_optimizer(context: CropContext):
    s = context.state
    p = context.config.psr_params or {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s.optimizer = PointSetRegistrationOptimizer(
        sub_tablet_text=s.sub_tablet_aligned,
        target_detections=s.sub_image.detections,
        sigma=s.avg_width * p.get('sigma_factor', 1.5),
        w_noise=p.get('w_noise', 0.1),
        lambda_data=p.get('lambda_data', 2.0),
        lambda_anchor=p.get('lambda_anchor', 0.01),
        lambda_seq=p.get('lambda_seq', 0.1),
        lambda_height=p.get('lambda_height', 0.01),
        lambda_rows=p.get('lambda_rows', 5.0),
        lambda_boundary=p.get('lambda_boundary', 0.0),
        rows_threshold_ratio_far=p.get('rows_threshold_ratio_far', 1 / 3.0),
        rows_threshold_ratio_close=p.get('rows_threshold_ratio_close', 2 / 3.0),
        rows_plateau_far=p.get('rows_plateau_far', 0.5),
        rows_plateau_close=p.get('rows_plateau_close', 1.0),
        contour_mask=s.sub_image.mask,
        device=device,
    )


def _visualize_psr_optimizer(context: CropContext, vis: VisOptions):
    s = context.state
    opt = s.optimizer
    if vis.info:
        print(f"=== PSR Optimizer ===")
        print(f"  Device: {opt.device}, Source (M): {opt.M}, Target (N): {opt.N}")
        print(f"  Sigma: {opt.sigma:.1f}, w_noise: {opt.w_noise}")
        print(f"  Lambdas: data={opt.lambda_data}, anchor={opt.lambda_anchor}, "
              f"seq={opt.lambda_seq}, height={opt.lambda_height}, "
              f"rows={opt.lambda_rows}, boundary={opt.lambda_boundary}")
        print(f"  Contour mask: {'available' if opt.contour_mask is not None else 'not set'}")
    if vis.save:
        opt.plot_loss_curves(save_dir="alignment_loss_functions", show=False)


# ---------------------------------------------------------------------------
# Step: Run PSR Optimization
# ---------------------------------------------------------------------------

def _run_psr_optimization(context: CropContext):
    s = context.state
    cfg = context.config.protosnap
    refiner = s.snap_refiner
    snap_iter = cfg.snap_at_iter if cfg is not None else None
    fired = {"done": False}

    def _snap_freeze_hook(opt, it):
        if refiner is None or snap_iter is None:
            return
        if it != snap_iter or fired["done"]:
            return
        fired["done"] = True
        # Build a SubTablet view that reflects current params, so the
        # crops we feed to the refiner are correct after the first 10
        # iterations of PSR have moved things around.
        current = opt.get_optimized_subtablet()
        refinements = refiner.refine_subtablet(current)
        s.snap_refinements = refinements

        accepted = [r for r in refinements if r.accepted]
        idx = [r.sign_index for r in accepted]
        new_centers = [(r.new_cx, r.new_cy) for r in accepted]
        opt.freeze_centers(idx, new_centers)

        s.snap_summary = summarize_refinements(refinements)
        s.snap_used_prototype_paths = collect_used_prototype_paths(
            refinements, refiner.bank, opt.sign_boxes_flat)

        print(f"  [iter {it}] ProtoSnap fired: "
              f"{s.snap_summary['accepted']}/{s.snap_summary['total']} "
              f"signs frozen.")

    p = context.config.psr_params or {}
    s.sub_tablet_final = s.optimizer.optimize(
        num_iterations=p.get('num_iterations', 80),
        lr=p.get('lr', 1.0),
        sigma_anneal=p.get('sigma_anneal', True),
        sigma_final=None,
        verbose=True,
        log_every=20,
        mid_iteration_hook=_snap_freeze_hook,
    )


def _visualize_psr_optimization(context: CropContext, vis: VisOptions):
    if vis.info:
        s = context.state
        print(f"=== Optimization Complete: {len(s.sub_tablet_final)} signs ===")


# ---------------------------------------------------------------------------
# Step: Visualize ProtoSnap freeze (blue=frozen, orange=free)
# ---------------------------------------------------------------------------

def _visualize_protosnap_freeze(context: CropContext, vis: VisOptions):
    """Show the optimizer-state at the moment ProtoSnap fired.

    Frozen-center bboxes are drawn in **blue**, the rest (still being
    optimized) in **orange**. We render two snapshots side-by-side:
      (a) snapshot taken right after freeze (before further optimization);
      (b) final SubTablet after optimization completed.
    """
    s = context.state
    if s.snap_refinements is None:
        if vis.info:
            print("ProtoSnap freeze visualization: nothing to show "
                  "(refiner disabled or no signs accepted).")
        return

    if vis.info:
        summ = s.snap_summary or {}
        print("=== ProtoSnap Freeze Summary ===")
        print(f"  Total refined: {summ.get('total')}, Accepted: {summ.get('accepted')}")
        for status_value, count in sorted(summ.get("by_status", {}).items()):
            print(f"    {status_value}: {count}")
        # Top reasons for rejection (helps gate tuning)
        reasons = Counter(
            (r.info or {}).get("gate_reason", "?")
            for r in s.snap_refinements if not r.accepted
        )
        if reasons:
            print("  Top rejection reasons:")
            for reason, n in reasons.most_common(5):
                print(f"    [{n}] {reason}")
        if s.snap_used_prototype_paths:
            print(f"  Used {len(s.snap_used_prototype_paths)} unique prototype PNGs.")

    # Build a per-sign color function based on which indices are frozen.
    accepted_idx = {r.sign_index for r in s.snap_refinements if r.accepted}
    final_boxes = s.sub_tablet_final.to_detection_list()

    BLUE = (0, 102, 255)     # frozen
    ORANGE = (255, 128, 0)   # free

    def _color_func_factory(idx_to_box):
        # Identity-based lookup: BboxVisualizer iterates boxes in order,
        # so we use a counter via id(box) -> index.
        index_lookup = {id(b): i for i, b in enumerate(idx_to_box)}
        def _color(box):
            i = index_lookup.get(id(box), -1)
            return BLUE if i in accepted_idx else ORANGE
        return _color

    final_vis = BboxVisualizer()
    final_vis.color_func = _color_func_factory(final_boxes)
    final_vis.draw_boxes(s.sub_image.img.copy(), final_boxes)

    if vis.display:
        final_vis.display_result(vis_opt="draw")
    if vis.save:
        final_vis.save(_out(context, "protosnap_freeze_final.jpg"))


# ---------------------------------------------------------------------------
# Step: Plot Loss History
# ---------------------------------------------------------------------------

def _visualize_loss_history(context: CropContext, vis: VisOptions):
    if vis.display or vis.save:
        context.state.optimizer.plot_loss_history()


# ---------------------------------------------------------------------------
# Step: Results Comparison
# ---------------------------------------------------------------------------

def _visualize_results_comparison(context: CropContext, vis: VisOptions):
    s = context.state
    img = s.sub_image.img

    before_vis = BboxVisualizer(color=(0, 255, 255))
    before_vis.draw_boxes(img.copy(), s.sub_tablet_aligned.to_detection_list())

    after_vis = BboxVisualizer(color=(255, 255, 0))
    after_vis.draw_boxes(img.copy(), s.sub_tablet_final.to_detection_list())

    det_base = BboxVisualizer(color=(255, 0, 0))
    det_base.draw_boxes(img.copy(), s.sub_tablet_detection.to_detection_list())
    det_final_ov = BboxVisualizer(color=(255, 255, 0))
    det_final_ov.draw_boxes(det_base.result, s.sub_tablet_final.to_detection_list())

    gt_base = BboxVisualizer(color=(0, 255, 0))
    gt_base.draw_boxes(img.copy(), s.gt_boxes_img or [])
    gt_final_ov = BboxVisualizer(color=(255, 255, 0))
    gt_final_ov.draw_boxes(gt_base.result, s.sub_tablet_final.to_detection_list())

    comp = CompositeVisualizer()
    comp.compose(
        images=[before_vis.result, after_vis.result, det_final_ov.result, gt_final_ov.result],
        layout=(2, 2),
        titles=[
            f"Before PSR: Coarse Aligned ({len(s.sub_tablet_aligned)} signs)",
            f"After PSR: Final Optimized ({len(s.sub_tablet_final)} signs)",
            "Overlay: Detection (red) + Final (yellow)",
            "Overlay: GT (green) + Final (yellow)",
        ],
        figsize=(16, 12),
    )

    if vis.info:
        print("Cyan=Coarse aligned  Yellow=Final optimized  Red=Detection  Green=GT")
    if vis.display:
        comp.display_result(vis_opt="draw")
    if vis.save:
        before_vis.save(_out(context, "coarse_aligned.jpg"))
        after_vis.save(_out(context, "final_optimized.jpg"))
        det_final_ov.save(_out(context, "overlay_det_final.jpg"))
        gt_final_ov.save(_out(context, "overlay_gt_final.jpg"))
        comp.save(_out(context, "results_comparison.jpg"))


# ---------------------------------------------------------------------------
# Step: Analyze Parameter Changes
# ---------------------------------------------------------------------------

def _visualize_param_changes(context: CropContext, vis: VisOptions):
    if not vis.info:
        return
    s = context.state
    param_changes = s.optimizer.get_param_changes()
    print("=== Parameter Changes (Coarse -> Final) ===")
    for i, label in enumerate(["cx", "cy", "w ", "h "]):
        arr = param_changes[:, i]
        print(f"  Delta {label}: mean={arr.mean():.2f}, std={arr.std():.2f}, |max|={np.abs(arr).max():.2f}")

    n = min(5, len(s.sub_tablet_aligned.sign_boxes))
    print(f"\n=== First {n} Signs: Coarse -> Final ===")
    for i in range(n):
        before = s.sub_tablet_aligned.sign_boxes[i]
        after = s.sub_tablet_final.sign_boxes[i]
        print(f"  {i+1}. {before.sign_name}:")
        print(f"      Coarse: cx={before.cx:.1f}, cy={before.cy:.1f}, "
              f"w={before.width:.1f}, h={before.height:.1f}")
        print(f"      Final:  cx={after.cx:.1f},  cy={after.cy:.1f}, "
              f"w={after.width:.1f},  h={after.height:.1f}")
        print(f"      Delta:  cx={after.cx-before.cx:.1f}, cy={after.cy-before.cy:.1f}, "
              f"w={after.width-before.width:.1f}, h={after.height-before.height:.1f}")


# ---------------------------------------------------------------------------
# Step instances
# ---------------------------------------------------------------------------

step_load_data = Step(
    name="Load Data (image, GT, sign text)",
    run=_load_data,
    visualize=_visualize_load_data,
)

step_detect_signs = Step(
    name="Detect Signs",
    run=_detect_signs,
    visualize=_visualize_detect_signs,
)

step_transform_gt_to_img = Step(
    name="Transform GT to Sub-image Coords",
    run=_transform_gt_to_img,
    visualize=_visualize_transform_gt,
)

step_compute_statistics = Step(
    name="Compute Detection Statistics",
    run=_compute_statistics,
    visualize=_visualize_statistics,
)

step_create_subtablets = Step(
    name="Create Sub-tablets",
    run=_create_subtablets,
    visualize=_visualize_create_subtablets,
)

step_detect_rows = Step(
    name="Detect Rows (DBSCAN)",
    run=_detect_rows,
    visualize=_visualize_detect_rows,
)

step_match_rows = Step(
    name="Match Detection Rows to Text Rows",
    run=_match_rows,
    visualize=_visualize_match_rows,
)

step_visualize_detection_rows = Step(
    name="Visualize Detection Rows",
    visualize=_visualize_detection_rows,
)

step_match_signs_in_rows = Step(
    name="Within-Row Sign Matching",
    run=_match_signs_in_rows,
    visualize=_visualize_match_signs,
)

step_align_text_rows = Step(
    name="Align Text Rows to Detection Rows",
    run=_align_text_rows,
    visualize=_visualize_align_text_rows,
)

step_build_sign_match_info = Step(
    name="Build Sign Match Info & Visualize Mapping",
    run=_build_sign_match_info,
    visualize=_visualize_sign_match_info,
)

step_offset_analysis = Step(
    name="Position Offset Analysis",
    visualize=_visualize_offset_analysis,
)

# ---------------------------------------------------------------------------
# Step: Unload detector model (free GPU memory before DIFT / PSR)
# ---------------------------------------------------------------------------

def _unload_detector(context: CropContext):
    context.tablet_detector.unload_model()

step_unload_detector = Step(
    name="Unload Detector Model (free GPU memory)",
    description="Releases the sign detector from GPU so DIFT and PSR have more VRAM.",
    run=_unload_detector,
)

step_protosnap = Step(
    name="ProtoSnap Setup (resolve period, build refiner)",
    run=_protosnap_setup,
    visualize=_visualize_protosnap_setup,
)

step_protosnap_freeze_vis = Step(
    name="ProtoSnap Freeze Visualization (blue=frozen, orange=free)",
    visualize=_visualize_protosnap_freeze,
)

step_protosnap_crop_vis = Step(
    name="ProtoSnap Crop Grid (prototype | annotated crop per sign)",
    visualize=_visualize_protosnap_crops,
)

step_create_psr_optimizer = Step(
    name="Create PSR Optimizer",
    run=_create_psr_optimizer,
    visualize=_visualize_psr_optimizer,
)

step_run_psr_optimization = Step(
    name="Run PSR Optimization",
    run=_run_psr_optimization,
    visualize=_visualize_psr_optimization,
)

step_plot_loss_history = Step(
    name="Plot Loss History",
    visualize=_visualize_loss_history,
)

step_results_comparison = Step(
    name="Results Comparison",
    visualize=_visualize_results_comparison,
)

step_param_changes = Step(
    name="Analyze Parameter Changes",
    visualize=_visualize_param_changes,
)


DEBUG_STEPS = [
    step_load_data,
    step_detect_signs,
    step_transform_gt_to_img,
    step_compute_statistics,
    step_create_subtablets,
    step_detect_rows,
    step_match_rows,
    step_visualize_detection_rows,
    step_match_signs_in_rows,
    step_align_text_rows,
    step_build_sign_match_info,
    step_offset_analysis,
    step_unload_detector,
    step_create_psr_optimizer,
    step_run_psr_optimization,
    step_plot_loss_history,
    step_results_comparison,
    step_param_changes,
]


# Opt-in variant that wires ProtoSnap center-freeze into the PSR
# optimizer as a mid-iteration hook (fires at config.protosnap.snap_at_iter,
# default 10). step_protosnap only builds the refiner; the actual freeze
# happens inside step_run_psr_optimization. step_protosnap_freeze_vis
# renders blue (frozen) / orange (free) bboxes after optimization.
#
# Default DEBUG_STEPS is unchanged so existing notebooks keep working.
DEBUG_STEPS_WITH_PROTOSNAP = [
    step_load_data,
    step_detect_signs,
    step_transform_gt_to_img,
    step_compute_statistics,
    step_create_subtablets,
    step_detect_rows,
    step_match_rows,
    step_visualize_detection_rows,
    step_match_signs_in_rows,
    step_align_text_rows,
    step_build_sign_match_info,
    step_offset_analysis,
    step_unload_detector,
    step_protosnap,
    step_create_psr_optimizer,
    step_run_psr_optimization,
    step_protosnap_freeze_vis,
    step_protosnap_crop_vis,
    step_plot_loss_history,
    step_results_comparison,
    step_param_changes,
]
