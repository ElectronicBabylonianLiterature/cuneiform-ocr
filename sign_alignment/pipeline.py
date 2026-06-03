from collections import Counter
from dataclasses import dataclass, field
import os
from typing import Callable, List, Optional

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
from sign_alignment.dift_model import DiftConfig, load_dift_model, make_dift_wrapper
from data_processing.line_process import (
    align_text_to_detection_rows,
    create_row_mapping,
    match_rows_dp,
    match_signs_in_row_dp,
)
from sign_alignment.dift_align import (
    CanonicalSignSource,
    CanonicalFeatureCache, DiscoveryConfig,
    GapDetector, SignIdentifier,
    GapDiscovery,
    render_discovery_grid, summarize_discoveries,
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
    dift: Optional[DiftConfig] = None
    # Optional PSR optimizer parameter overrides (dict or None for defaults).
    # Keys: sigma_factor, w_noise, lambda_data, lambda_anchor, lambda_seq,
    #       lambda_height, lambda_rows, lambda_boundary,
    #       rows_threshold_ratio_far, rows_threshold_ratio_close,
    #       rows_plateau_far, rows_plateau_close, num_iterations, lr, sigma_anneal
    psr_params: Optional[dict] = None

    discovery: Optional["DiscoveryConfig"] = None
    canonical_source_factory: Optional[Callable] = None
    canonical_feature_dir: Optional[str] = None


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

    # Gap-driven DIFT discovery state (period-driven canonical source + cache).
    canonical_source: Optional["CanonicalSignSource"] = None
    canonical_cache: Optional["CanonicalFeatureCache"] = None
    discovery_meta: Optional[dict] = None
    # Coordinate-based gap candidates (before identification).
    gap_candidates: Optional[list] = None  # list[GapCandidate]
    # Final per-gap discoveries (gap + assigned canonical sign).
    discoveries: Optional[list] = None  # list[GapDiscovery]
    discovery_summary: Optional[dict] = None


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class CropContext:
    config: PipelineConfig
    tools: PipelineTools = field(default_factory=PipelineTools)
    state: SampleState = None
    # Cached SDFeaturizer for SD-DIFT; loaded once, reused across samples.
    _dift_model: Optional[object] = field(default=None, repr=False, compare=False)
    # CanonicalFeatureCache instances shared across samples.
    _canonical_caches: dict = field(default_factory=dict, repr=False, compare=False)

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
    """Base class for pipeline steps.

    Subclasses override *name*, and optionally *description*, *run*, and
    *visualize*.  The default implementations are no-ops so a step only
    needs to override the methods it actually uses.
    """

    name: str = ""
    description: Optional[str] = None

    def run(self, context: "CropContext") -> None:
        """Execute the computational logic of this step."""
        pass

    def visualize(self, context: "CropContext", vis: "VisOptions") -> None:
        """Produce any visual / textual output for this step."""
        pass





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _out(context: CropContext, suffix: str) -> str:
    return os.path.join(context.output_dir, f"{context.task_type}_{context.fragment_id}_{suffix}")


# ---------------------------------------------------------------------------
# Step: Load Data  (image + GT + sign text from API)
# ---------------------------------------------------------------------------

class StepLoadData(Step):
    name = "Load Data (image, GT, sign text)"

    def run(self, context: CropContext):
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

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepDetectSigns(Step):
    name = "Detect Signs"

    def run(self, context: CropContext):
        s = context.state
        s.detections = context.tablet_detector.detect(s.img)
        cropped = context.tablet_detector.get_cropped_images()
        img_idx = context.config.img_idx
        if not cropped:
            raise RuntimeError("detector produced no cropped images")
        if img_idx < 0 or img_idx >= len(cropped):
            raise IndexError(
                f"crop index {img_idx} is out of range after detection; "
                f"available crop indices are 0..{len(cropped) - 1}"
            )
        s.sub_image = cropped[img_idx]
        s.crop_info = context.tablet_detector.crop_coordinates[img_idx]

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepTransformGtToImg(Step):
    name = "Transform GT to Sub-image Coords"

    def run(self, context: CropContext):
        s = context.state
        s.gt_boxes_img = transform_gt_to_cropped_region(s.gt_boxes, s.crop_info)

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepComputeStatistics(Step):
    name = "Compute Detection Statistics"

    def run(self, context: CropContext):
        s = context.state
        s.avg_width, s.avg_height = compute_avg_dimensions(s.detections)

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if vis.info:
            print(f"Full image shape: {s.img.shape}")
            print(f"Sub-image shape:  {s.sub_image.img.shape}")
            print(f"Average detected sign  width: {s.avg_width:.2f}")
            print(f"Average detected sign height: {s.avg_height:.2f}")


# ---------------------------------------------------------------------------
# Step: Create SubTablets
# ---------------------------------------------------------------------------

class StepCreateSubTablets(Step):
    name = "Create Sub-tablets"

    def run(self, context: CropContext):
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

    def visualize(self, context: CropContext, vis: VisOptions):
        if vis.info:
            print(context.state.sub_tablet_detection.info)
            print(context.state.sub_tablet_text.info)


# ---------------------------------------------------------------------------
# Step: Detect Rows
# ---------------------------------------------------------------------------

class StepDetectRows(Step):
    name = "Detect Rows (DBSCAN)"

    def run(self, context: CropContext):
        context.state.sub_tablet_detection.detect_rows(
            eps=0.4, min_samples=1, lambda_weight=0.007)

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepMatchRows(Step):
    name = "Match Detection Rows to Text Rows"

    def run(self, context: CropContext):
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

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepVisualizeDetectionRows(Step):
    name = "Visualize Detection Rows"

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepMatchSignsInRows(Step):
    name = "Within-Row Sign Matching"

    def run(self, context: CropContext):
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

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepAlignTextRows(Step):
    name = "Align Text Rows to Detection Rows"

    def run(self, context: CropContext):
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

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepBuildSignMatchInfo(Step):
    name = "Build Sign Match Info & Visualize Mapping"

    def run(self, context: CropContext):
        s = context.state
        text_sign_match_info, det_sign_match_info = build_sign_match_info(
            row_sign_matches=s.row_sign_matches,
            text_to_det=s.text_to_det,
            det_rows_dict=s.sub_tablet_detection.get_rows_dict(),
            optim_sign_boxes=s.sub_tablet_aligned.sign_boxes,
        )
        s.text_sign_match_info = text_sign_match_info
        s.det_sign_match_info = det_sign_match_info

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepOffsetAnalysis(Step):
    name = "Position Offset Analysis"

    def visualize(self, context: CropContext, vis: VisOptions):
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
# Step: Unload detector model (free GPU memory before DIFT / PSR)
# ---------------------------------------------------------------------------

class StepUnloadDetector(Step):
    name = "Unload Detector Model (free GPU memory)"
    description = "Releases the sign detector from GPU so DIFT and PSR have more VRAM."

    def run(self, context: CropContext):
        context.tablet_detector.unload_model()


# ---------------------------------------------------------------------------
# Step: Create PSR Optimizer
# ---------------------------------------------------------------------------

class StepCreatePsrOptimizer(Step):
    name = "Create PSR Optimizer"

    def run(self, context: CropContext):
        s = context.state
        p = context.config.psr_params or {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Augment target detections with DIFT-discovered gap bboxes (if any).
        # Discoveries already carry an assigned sign label, so the GMM data
        # term naturally pulls source signs of the same class onto them.
        target_detections = list(s.sub_image.detections)
        if s.discoveries:
            extra = [d.to_bounding_box() for d in s.discoveries
                     if not d.low_confidence]
            target_detections.extend(extra)
            s.discovery_augmented_count = len(extra)
        else:
            s.discovery_augmented_count = 0

        s.optimizer = PointSetRegistrationOptimizer(
            sub_tablet_text=s.sub_tablet_aligned,
            target_detections=target_detections,
            sigma=s.avg_width * p.get('sigma_factor', 1.5),
            w_noise=p.get('w_noise', 0.1),
            lambda_data=p.get('lambda_data', 2.0),
            lambda_anchor=p.get('lambda_anchor', 0.01),
            lambda_seq=p.get('lambda_seq', 0.03),
            lambda_height=p.get('lambda_height', 0.01),
            lambda_rows=p.get('lambda_rows', 1.0),
            lambda_boundary=p.get('lambda_boundary', 0.0),
            rows_threshold_ratio_far=p.get('rows_threshold_ratio_far', 1 / 3.0),
            rows_threshold_ratio_close=p.get('rows_threshold_ratio_close', 2 / 3.0),
            rows_plateau_far=p.get('rows_plateau_far', 0.5),
            rows_plateau_close=p.get('rows_plateau_close', 1.0),
            contour_mask=s.sub_image.mask,
            device=device,
        )

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        opt = s.optimizer
        if vis.info:
            print(f"=== PSR Optimizer ===")
            print(f"  Device: {opt.device}, Source (M): {opt.M}, Target (N): {opt.N}")
            n_extra = getattr(s, "discovery_augmented_count", 0)
            if n_extra:
                print(f"    Target breakdown: {opt.N - n_extra} model detections "
                      f"+ {n_extra} DIFT-discovered gap signs")
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

class StepRunPsrOptimization(Step):
    name = "Run PSR Optimization"

    def run(self, context: CropContext):
        s = context.state
        p = context.config.psr_params or {}
        s.sub_tablet_final = s.optimizer.optimize(
            num_iterations=p.get('num_iterations', 80),
            lr=p.get('lr', 1.0),
            sigma_anneal=p.get('sigma_anneal', True),
            sigma_final=None,
            verbose=True,
            log_every=20,
        )

    def visualize(self, context: CropContext, vis: VisOptions):
        if vis.info:
            s = context.state
            print(f"=== Optimization Complete: {len(s.sub_tablet_final)} signs ===")


# ---------------------------------------------------------------------------
# Step: Plot Loss History
# ---------------------------------------------------------------------------

class StepPlotLossHistory(Step):
    name = "Plot Loss History"

    def visualize(self, context: CropContext, vis: VisOptions):
        if vis.display or vis.save:
            context.state.optimizer.plot_loss_history()


# ---------------------------------------------------------------------------
# Step: Results Comparison
# ---------------------------------------------------------------------------

class StepResultsComparison(Step):
    name = "Results Comparison"

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepParamChanges(Step):
    name = "Analyze Parameter Changes"

    def visualize(self, context: CropContext, vis: VisOptions):
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
# Step: Setup canonical-sign feature cache (period-driven, swappable source)
# ---------------------------------------------------------------------------

class StepSetupCanonicalSigns(Step):
    name = "Setup Canonical Signs (period -> source + eager DIFT-feature cache)"
    description = (
        "Builds the canonical sign source for this fragment's period and "
        "eagerly DIFT-featurises its inventory. The in-memory cache is "
        "stored on the CropContext keyed by source, and an optional disk "
        "cache can persist per-sign features across Python sessions."
    )

    def run(self, context: CropContext):
        s = context.state
        s.canonical_source = None
        s.canonical_cache = None
        s.discovery_meta = None

        fragment_data = context.api_source.get_fragment_data(context.fragment_id)
        period = fragment_data["script"]["period"]

        dift_cfg = context.config.dift
        if dift_cfg is None:
            raise ValueError("PipelineConfig.dift is required for canonical DIFT features")

        if context._dift_model is None:
            print("  [Canonical] Loading SD-DIFT model (first time, ~30 s)...")
            context._dift_model = load_dift_model(dift_cfg)

        wrapper = make_dift_wrapper(dift_cfg, context._dift_model, prompt="")

        factory = getattr(context.config, "canonical_source_factory", None)
        if factory is None:
            raise ValueError("PipelineConfig.canonical_source_factory is required")

        source = factory(period, context.config)
        source_kind = type(source).__name__

        s.canonical_source = source
        meta = {
            "enabled": source.is_ready(),
            "period": period,
            "source_kind": source_kind,
            "precompute": None,
        }
        s.discovery_meta = meta
        if not source.is_ready():
            raise RuntimeError(f"canonical source is not ready: {source.describe()}")

        cache_key = self._cache_key(source)
        cache = context._canonical_caches.get(cache_key)

        disc_cfg = context.config.discovery or DiscoveryConfig()
        if cache is None:
            cache = CanonicalFeatureCache(
                source=source,
                wrapper=wrapper,
                disk_dir=context.config.canonical_feature_dir,
            )
            print(f"  [Canonical] Eager precompute: featurising "
                  f"{len(source.list_sign_names())} canonical images "
                  f"(this takes a few minutes on first call)...")
            stats = cache.precompute_all(
                verbose=True, limit=disc_cfg.precompute_limit,
                progress_every=50,
            )
            meta["precompute"] = stats
            context._canonical_caches[cache_key] = cache
        else:
            meta["precompute"] = {"reused": True, "size": len(cache)}

        s.canonical_cache = cache

    @staticmethod
    def _cache_key(source: CanonicalSignSource) -> str:
        return source.cache_namespace()

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if not vis.info:
            return
        meta = s.discovery_meta or {}
        print("=== Canonical Signs Setup ===")
        print(f"  Enabled:      {meta.get('enabled')}")
        print(f"  API period:   {meta.get('period')!r}")
        print(f"  Source:       {meta.get('source_kind')}")
        if s.canonical_source is not None:
            print(f"  Source info:  {s.canonical_source.describe()}")
        if s.canonical_cache is not None:
            print(f"  Cache size:   {len(s.canonical_cache)} cached canonical signs")
        pre = meta.get("precompute")
        if isinstance(pre, dict):
            if pre.get("reused"):
                print(f"  Precompute:   reused cross-sample cache "
                      f"(size={pre.get('size')})")
            else:
                print(f"  Precompute:   total={pre.get('total')}  "
                      f"computed={pre.get('computed')}  cached={pre.get('cached')}  "
                      f"disk={pre.get('disk_cached')}")


# ---------------------------------------------------------------------------
# Step: Discover missing signs via DIFT on interpolated row-gap candidates
# ---------------------------------------------------------------------------

class StepIdentifyGapSigns(Step):
    name = "Identify Gap Signs (coordinate-based gap detection + DIFT ranking)"
    description = (
        "Scans detection rows for coordinate gaps where a sign is likely "
        "missing (centre-to-centre stride approximately N x avg_width). For "
        "each pending bbox, crops the sub-image and ranks the entire cached "
        "canonical inventory by DIFT best-buddy similarity; the top-scoring "
        "sign is assigned as the label. Discovered bboxes are added to "
        "target_detections in step_create_psr_optimizer."
    )

    def run(self, context: CropContext):
        s = context.state
        s.gap_candidates = None
        s.discoveries = None
        s.discovery_summary = None

        cache = s.canonical_cache
        if cache is None or len(cache) == 0:
            raise RuntimeError("canonical feature cache is empty; run step_setup_canonical_signs first")
        if s.sub_tablet_detection is None:
            raise RuntimeError("sub_tablet_detection is missing; run row setup steps first")

        cfg = context.config.discovery or DiscoveryConfig()

        # ---- 1. Coordinate-based gap detection ----------------------------
        detector = GapDetector(cfg.gap)
        gaps = detector.find_gaps(s.sub_tablet_detection, s.sub_image.img)
        s.gap_candidates = gaps
        if not gaps:
            s.discoveries = []
            s.discovery_summary = summarize_discoveries([], 0)
            return

        # ---- 2. DIFT identification per gap -------------------------------
        identifier = SignIdentifier(cache, cfg.identification)
        top_k_show = max(1, cfg.identification.top_k)
        discoveries: List[GapDiscovery] = []
        for g in gaps:
            ranked = identifier.identify(g.crop_img)
            if not ranked:
                raise RuntimeError("DIFT identification returned no candidates")
            best = ranked[0]
            disc = GapDiscovery(
                gap=g, best=best,
                top_k=ranked[:top_k_show],
                low_confidence=best.score < cfg.identification.min_score,
            )
            discoveries.append(disc)

        s.discoveries = discoveries
        s.discovery_summary = summarize_discoveries(discoveries, len(gaps))

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if s.gap_candidates is None:
            if vis.info:
                print("Discovery: skipped (no canonical cache available).")
            return

        summ = s.discovery_summary or {}
        if vis.info:
            print("=== Gap-Driven Sign Identification ===")
            print(f"  Gaps found:           {summ.get('gaps_total', 0)}")
            print(f"  Discoveries:          {summ.get('discoveries', 0)} "
                  f"(confident={summ.get('confident', 0)}, "
                  f"low_confidence={summ.get('low_confidence', 0)})")
            top_signs = summ.get("by_sign_top10", {})
            if top_signs:
                print("  Top assigned signs:")
                for name, count in top_signs.items():
                    print(f"    {count}x  {name}")
            # Per-gap detailed log
            if s.discoveries:
                print("  Per-gap top-3 assignments (score):")
                for d in s.discoveries:
                    g = d.gap
                    line = (f"    R{g.row_idx} ins{g.insert_idx+1}/{g.n_inserts} "
                            f"@({g.cx:.0f},{g.cy:.0f})  ")
                    line += " | ".join(
                        f"{r.sign_name[:10]}:{r.score:.3f}"
                        for r in d.top_k[:3]
                    )
                    print(line)

        # Per-gap grid is only useful when there is at least one discovery.
        if s.discoveries:
            grid = render_discovery_grid(s.discoveries, thumb=120,
                                         top_k_show=min(4, len(s.discoveries[0].top_k)))
        else:
            grid = None
        if grid is not None:
            if vis.display:
                import matplotlib.pyplot as plt
                grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
                h, w = grid_rgb.shape[:2]
                fig_w = min(20.0, w / 60.0)
                fig_h = max(2.0, fig_w * h / max(w, 1))
                plt.figure(figsize=(fig_w, fig_h))
                plt.imshow(grid_rgb)
                plt.axis("off")
                plt.title("Gap Identification: crop | top-K canonical (green = assigned)")
                plt.tight_layout(pad=0.3)
                plt.show()
            if vis.save:
                cv2.imwrite(_out(context, "discovery_grid.jpg"), grid,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Sub-image overlay: detection bboxes (red) + discovered bboxes (green)
        # with the assigned sign name and score annotated.
        if vis.display or vis.save:
            overlay = s.sub_image.img.copy()
            avg_size = (s.avg_width + s.avg_height) / 2.0 if s.avg_width else 60.0
            t_line = max(2, int(round(avg_size / 30.0)))
            font_scale = max(0.6, avg_size / 250.0)
            font_thick = max(2, int(round(avg_size / 60.0)))

            RED = (60, 60, 220)        # model detections
            GREEN = (80, 200, 80)      # accepted gap discoveries
            AMBER = (40, 170, 220)     # low-confidence

            # Existing model detections (for context).
            for det in s.sub_image.detections:
                cv2.rectangle(overlay,
                              (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)),
                              RED, t_line, cv2.LINE_AA)

            # Gap discoveries.
            # Outline EVERY gap candidate even if no sign was confidently
            # assigned, so the user can verify the gap-detection coverage.
            CYAN = (210, 210, 0)
            disc_by_gap = {id(d.gap): d for d in s.discoveries}
            for g in s.gap_candidates or []:
                d = disc_by_gap.get(id(g))
                if d is not None:
                    color = AMBER if d.low_confidence else GREEN
                    thick = t_line + 1
                else:
                    color = CYAN
                    thick = t_line
                cv2.rectangle(overlay,
                              (int(g.cx - g.width / 2), int(g.cy - g.height / 2)),
                              (int(g.cx + g.width / 2), int(g.cy + g.height / 2)),
                              color, thick, cv2.LINE_AA)
                if d is not None:
                    label = f"{d.sign_name[:10]} {d.score:.2f}"
                else:
                    label = f"R{g.row_idx} (no match)"
                tx = int(g.cx - g.width / 2)
                ty = int(g.cy - g.height / 2 - 6)
                cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), font_thick + 2, cv2.LINE_AA)
                cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, font_thick, cv2.LINE_AA)

            if vis.save:
                cv2.imwrite(_out(context, "discovery_overlay.jpg"), overlay)
            if vis.display:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(14, 9))
                plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.title("Discovery overlay  "
                          "(red=model detection, green=gap discovery (assigned), "
                          "amber=low confidence)")
                plt.tight_layout()
                plt.show()


# ---------------------------------------------------------------------------
# Step instances
# ---------------------------------------------------------------------------



step_load_data = StepLoadData()
step_detect_signs = StepDetectSigns()
step_transform_gt_to_img = StepTransformGtToImg()
step_compute_statistics = StepComputeStatistics()
step_create_subtablets = StepCreateSubTablets()
step_detect_rows = StepDetectRows()
step_match_rows = StepMatchRows()
step_visualize_detection_rows = StepVisualizeDetectionRows()
step_match_signs_in_rows = StepMatchSignsInRows()
step_align_text_rows = StepAlignTextRows()
step_build_sign_match_info = StepBuildSignMatchInfo()
step_offset_analysis = StepOffsetAnalysis()
step_unload_detector = StepUnloadDetector()
step_create_psr_optimizer = StepCreatePsrOptimizer()
step_run_psr_optimization = StepRunPsrOptimization()
step_plot_loss_history = StepPlotLossHistory()
step_results_comparison = StepResultsComparison()
step_param_changes = StepParamChanges()
# Gap-driven DIFT identification
step_setup_canonical_signs = StepSetupCanonicalSigns()
step_identify_gap_signs = StepIdentifyGapSigns()


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


# Detector + coordinate-based gap detection + DIFT full-inventory ranking.
# Discovered gap bboxes are appended to target_detections before PSR runs.
DEBUG_STEPS_WITH_DIFT = [
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
    step_setup_canonical_signs,
    step_identify_gap_signs,
    step_create_psr_optimizer,
    step_run_psr_optimization,
    step_plot_loss_history,
    step_results_comparison,
    step_param_changes,
]

# Per-crop sub-pipeline used in batch scripts: runs everything from GT transform
# through PSR optimization for a single crop. Skipping logic (empty rows/matches)
# must still be handled by the caller between choose_crop() and run_all().
PIPELINE_STEPS_PER_CROP = [
    step_transform_gt_to_img,
    step_create_subtablets,
    step_detect_rows,
    step_match_rows,
    step_match_signs_in_rows,
    step_align_text_rows,
    step_build_sign_match_info,
    step_create_psr_optimizer,
    step_run_psr_optimization,
]


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

        if self.vis.save:
            os.makedirs(context.output_dir, exist_ok=True)

    def run_single_step(self, step: Step):
        if self.vis.info:
            info_message = f"Step: {step.name}"
            if step.description:
                info_message += f" - {step.description}"
            print(info_message)
        step.run(self.context)
        step.visualize(self.context, self.vis)
    
    def run_all(self):
        """Run every step in self.steps in order."""
        for step in self.steps:
            self.run_single_step(step)

    def choose_sample(self, idx = 0, name=""):
        if name:
            if name in self._fragments:
                idx = self._fragments.index(name)
            else:
                raise ValueError(f"Fragment name '{name}' not found in available fragments.")
        fragment_id = self._fragments[idx]
        print(f"Processing sample: {fragment_id}")
        self.context.state = SampleState(fragments=self._fragments, fragment_id=fragment_id)

    def choose_crop(self, crop_idx: int):
        self.context.config.img_idx = crop_idx
        cropped = self.context.tablet_detector.get_cropped_images()
        if not cropped or self.context.state.detections is None:
            return
        if crop_idx < 0 or crop_idx >= len(cropped):
            raise IndexError(
                f"crop index {crop_idx} is out of range; "
                f"available crop indices are 0..{len(cropped) - 1}"
            )
        self.context.state.sub_image = cropped[crop_idx]
        self.context.state.crop_info = self.context.tablet_detector.crop_coordinates[crop_idx]
