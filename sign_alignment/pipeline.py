from dataclasses import dataclass, field
import os
from typing import Optional

import cv2
import numpy as np
import torch

from sign_alignment.detector import ModelConfig, TabletImageDetector
from sign_alignment.data_source import (
    CanonicalSignSource,
    EBLMongoCanonicalSource,
    EBLAPISource,
    LocalDataSource,
    SignAPIResolver,
    SignTextParser,
)
from sign_alignment.box import Boxes, boxes_in_crop
from sign_alignment.tablet import SubTablet, Tablet
from sign_alignment.visualizer import (
    BboxVisualizer,
    ColorConfig,
    CompositeVisualizer,
    TextVisualizer,
    build_sign_match_info,
)
from sign_alignment.psr_optimizer import PointSetRegistrationOptimizer
from sign_alignment.dift_model import DiftConfig, load_dift_model, make_dift_wrapper
from data_processing.line_process import (
    align_text_row_to_detection,
    create_row_mapping,
    detect_rows_dbscan,
    match_rows_dp,
    match_signs_in_row_dp,
)
from sign_alignment.dift_align import (
    CanonicalFeatureCache,
    DiftAlignmentConfig,
    build_dift_affine_probe,
    collect_detected_canonical_feature_rows,
    render_canonical_feature_grid,
    render_canonical_sign_overlay,
    render_dift_affine_probe,
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

    dift_alignment: Optional["DiftAlignmentConfig"] = None
    canonical_mongodb_uri: Optional[str] = None
    canonical_db_name: str = "ebl"
    canonical_form: str = "canonical1"
    canonical_require_centroid: bool = True
    canonical_feature_dir: Optional[str] = None


class PipelineTools:
    sign_resolver = SignAPIResolver()


# ---------------------------------------------------------------------------
# Row structure
# ---------------------------------------------------------------------------

@dataclass
class BoxRows:
    """Lightweight row topology for a box collection.

    rows stores box indices, not boxes themselves. Row ids are the list indices;
    columns are positions within each row.
    """

    boxes: Boxes
    rows: list[list[int]]
    noise: list[int] = field(default_factory=list)

    @classmethod
    def from_text_lines(cls, boxes: Boxes, text_lines: list[list[str]]) -> "BoxRows":
        rows = []
        offset = 0
        for line in text_lines:
            row = list(range(offset, offset + len(line)))
            rows.append(row)
            offset += len(line)
        return cls(boxes=boxes, rows=rows)

    @classmethod
    def detect(
        cls,
        boxes: Boxes,
        eps: float = 0.6,
        min_samples: int = 1,
        lambda_weight: float = 0.05,
        avg_width: Optional[float] = None,
        avg_height: Optional[float] = None,
    ) -> "BoxRows":
        labels, _ = detect_rows_dbscan(
            boxes=boxes,
            eps=eps,
            min_samples=min_samples,
            lambda_weight=lambda_weight,
            avg_width=boxes.avg_width if avg_width is None else avg_width,
            avg_height=boxes.avg_height if avg_height is None else avg_height,
        )
        grouped: dict[int, list[int]] = {}
        noise = []
        for idx, label in enumerate(labels):
            if label == -1:
                noise.append(idx)
            else:
                grouped.setdefault(label, []).append(idx)
        rows = [sorted(grouped[k], key=lambda i: boxes[i].cx) for k in sorted(grouped)]
        return cls(boxes=boxes, rows=rows, noise=sorted(noise, key=lambda i: boxes[i].cx))

    def __len__(self) -> int:
        return len(self.rows)

    def row_boxes(self, row_idx: int) -> list:
        return [self.boxes[i] for i in self.rows[row_idx]]

    def as_dict(self) -> dict[int, list]:
        return {idx: self.row_boxes(idx) for idx in range(len(self.rows))}

    def as_lists(self) -> list[list]:
        return [self.row_boxes(idx) for idx in range(len(self.rows))]

    def sign_sequences(self) -> list[list[str]]:
        return [[box.sign_name for box in row] for row in self.as_lists()]

    def counts(self) -> list[int]:
        return [len(row) for row in self.rows]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SampleState:
    """All intermediate results for a single fragment."""

    fragments: list = None # all available fragments (populated at Runner init)
    fragment_id: str = None

    # full-image data
    tablet: Optional[Tablet] = None
    gt_boxes: Optional[Boxes] = None

    # Text lines parsed from API
    text_lines: Optional[list] = None
    text_lines_unfiltered: Optional[list] = None

    # Full-image detections
    detections: Optional[Boxes] = None

    # chosen crop of the tablet
    crop_tablet: Optional[SubTablet] = None
    det_boxes: Optional[Boxes] = None
    gt_boxes_crop: Optional[Boxes] = None

    # Box collections in the selected crop coordinate frame
    text_boxes: Optional[Boxes] = None
    aligned_boxes: Optional[Boxes] = None
    final_boxes: Optional[Boxes] = None

    # row matching
    det_rows: Optional[BoxRows] = None
    text_rows: Optional[BoxRows] = None
    aligned_rows: Optional[BoxRows] = None
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

    # Canonical DIFT feature cache and diagnostics.
    canonical_source: Optional["CanonicalSignSource"] = None
    canonical_cache: Optional["CanonicalFeatureCache"] = None
    canonical_meta: Optional[dict] = None
    dift_affine_probe_iteration: Optional[int] = None
    dift_affine_probe_boxes: Optional[Boxes] = None
    dift_affine_probe_results: Optional[list] = None
    dift_affine_probe_error: Optional[str] = None
    canonical_overlay_iteration: Optional[int] = None
    canonical_overlay_image: Optional[np.ndarray] = None
    canonical_overlay_stats: Optional[dict] = None
    canonical_overlay_error: Optional[str] = None


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
    return os.path.join(
        context.config.output_dir,
        f"{context.task_type}_{context.state.fragment_id}_{suffix}",
    )


def _dift_alignment_config(config: PipelineConfig) -> DiftAlignmentConfig:
    return config.dift_alignment or DiftAlignmentConfig()


def _display_bgr(img: np.ndarray, title: str, px_per_in: float = 80.0) -> None:
    import matplotlib.pyplot as plt

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    fig_w = min(20.0, w / px_per_in)
    fig_h = max(2.0, fig_w * h / max(w, 1))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0.3)
    plt.show()


def _dift_diagnostic_error(s: SampleState) -> Optional[str]:
    if s.optimizer is None:
        return "PSR optimizer is not available"
    if s.canonical_cache is None:
        return "canonical feature cache is not available"
    if s.crop_tablet is None:
        return "crop tablet image is not available"
    return None


# ---------------------------------------------------------------------------
# Step: Load Data  (image + GT + sign text from API)
# ---------------------------------------------------------------------------

class StepLoadData(Step):
    name = "Load Data (image, GT, sign text)"

    def run(self, context: CropContext):
        s = context.state

        img = context.config.local_source.load_image(s.fragment_id)
        if img is None:
            raise ValueError(f"No image found for sample {s.fragment_id}")
        s.tablet = Tablet(img=img, name=s.fragment_id)
        s.gt_boxes = context.config.local_source.load_annotation(s.fragment_id, s.tablet)

        fragment_data = context.config.api_source.get_fragment_data(s.fragment_id)
        if fragment_data is None:
            raise ValueError(f"No fragment data found for sample {s.fragment_id}")

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
                fragment_id=s.fragment_id)
            TextVisualizer.save_text(
                s.text_lines_unfiltered, path=_out(context, "text.txt"),
                fragment_id=s.fragment_id)
            gt_vis = BboxVisualizer(context.config.color_config.GT_COLOR.value)
            gt_vis.draw_boxes(s.tablet.img.copy(), s.gt_boxes)
            gt_vis.save(_out(context, "gt.jpg"))

        if vis.display:
            gt_vis = BboxVisualizer(context.config.color_config.GT_COLOR.value)
            gt_vis.draw_boxes(s.tablet.img.copy(), s.gt_boxes)
            gt_vis.display_result(vis_opt="draw")


# ---------------------------------------------------------------------------
# Step: Detect Signs
# ---------------------------------------------------------------------------

class StepDetectSigns(Step):
    name = "Detect Signs"

    def run(self, context: CropContext):
        s = context.state
        s.detections = context.config.tablet_detector.detect(s.tablet)
        crop_tablets = context.config.tablet_detector.get_crop_tablets()
        img_idx = context.config.img_idx
        if not crop_tablets:
            raise RuntimeError("detector produced no cropped images")
        if img_idx < 0 or img_idx >= len(crop_tablets):
            raise IndexError(
                f"crop index {img_idx} is out of range after detection; "
                f"available crop indices are 0..{len(crop_tablets) - 1}"
        )
        s.crop_tablet = crop_tablets[img_idx]
        s.det_boxes = context.config.tablet_detector.get_crop_boxes()[img_idx]

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        color = context.config.color_config.DET_COLOR.value

        if vis.info:
            print(f"Total detections (full image): {len(s.detections)}")
            print(f"Sub-image detections: {len(s.det_boxes)}")
            x, y = s.crop_tablet.offset_in_parent
            h, w = s.crop_tablet.shape
            print(f"Crop info (img_idx={context.config.img_idx}): "
                  f"x={x}, y={y}, w={w}, h={h}")

        if vis.save:
            full_vis = BboxVisualizer(color=color)
            full_vis.draw_boxes(s.tablet.img.copy(), s.detections)
            full_vis.save(_out(context, "det.jpg"))

            img_vis = BboxVisualizer(color=color)
            img_vis.draw_boxes(s.crop_tablet.img.copy(), s.det_boxes)
            img_vis.save(_out(context, "sub_image.jpg"))

        if vis.display:
            img_vis = BboxVisualizer(color=color)
            img_vis.draw_boxes(s.crop_tablet.img.copy(), s.det_boxes)
            img_vis.display_result(vis_opt="draw")


# ---------------------------------------------------------------------------
# Step: Transform GT to sub-image coords
# ---------------------------------------------------------------------------

class StepTransformGtToImg(Step):
    name = "Transform GT to Sub-image Coords"

    def run(self, context: CropContext):
        s = context.state
        s.gt_boxes_crop = boxes_in_crop(s.gt_boxes, s.crop_tablet)

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state

        if vis.info:
            print(f"GT boxes (full image): {len(s.gt_boxes)}")
            print(f"GT boxes (sub-image):  {len(s.gt_boxes_crop)}")

        if not s.gt_boxes_crop:
            return

        if vis.save:
            v = BboxVisualizer(color=context.config.color_config.GT_COLOR.value)
            v.draw_boxes(s.crop_tablet.img.copy(), s.gt_boxes_crop)
            v.save(_out(context, "sub_image_gt.jpg"))

        if vis.display:
            v = BboxVisualizer(color=context.config.color_config.GT_COLOR.value)
            v.draw_boxes(s.crop_tablet.img.copy(), s.gt_boxes_crop)
            v.display_result(vis_opt="draw")


# ---------------------------------------------------------------------------
# Step: Compute Statistics
# ---------------------------------------------------------------------------

class StepComputeStatistics(Step):
    name = "Compute Detection Statistics"

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if vis.info:
            print(f"Full image shape: {s.tablet.img.shape}")
            print(f"Sub-image shape:  {s.crop_tablet.img.shape}")
            print(f"Average detected sign  width: {s.detections.avg_width:.2f}")
            print(f"Average detected sign height: {s.detections.avg_height:.2f}")


# ---------------------------------------------------------------------------
# Step: Create box sets
# ---------------------------------------------------------------------------

class StepCreateBoxSets(Step):
    name = "Create Box Sets"

    def run(self, context: CropContext):
        s = context.state
        s.text_boxes = Boxes.from_text_lines(
            text_lines=s.text_lines,
            avg_width=s.detections.avg_width,
            avg_height=s.detections.avg_height,
            target_boxes=s.det_boxes,
            align_to_detection_centroid=True,
            tablet=s.crop_tablet,
        )
        s.text_rows = BoxRows.from_text_lines(s.text_boxes, s.text_lines)

    def visualize(self, context: CropContext, vis: VisOptions):
        if vis.info:
            s = context.state
            print(s.crop_tablet.info)
            print(s.text_boxes.info("text"))
            print(f"Text rows: {len(s.text_rows)}, signs: {len(s.text_boxes)}")


# ---------------------------------------------------------------------------
# Step: Detect Rows
# ---------------------------------------------------------------------------

class StepDetectRows(Step):
    name = "Detect Rows (DBSCAN)"

    def run(self, context: CropContext):
        s = context.state
        s.det_rows = BoxRows.detect(
            s.det_boxes,
            eps=0.4, min_samples=1, lambda_weight=0.007,
            avg_width=s.detections.avg_width,
            avg_height=s.detections.avg_height,
        )

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if vis.info:
            avg_size = s.detections.avg_size
            num_rows = len(s.det_rows)
            print(f"=== Row Detection Results ===")
            print(f"Average sign size: {avg_size:.2f} px, detected {num_rows} rows, {len(s.det_boxes)} signs")
            for row_idx, count in enumerate(s.det_rows.counts()):
                print(f"  Row {row_idx}: {count} boxes")
            if s.det_rows.noise:
                print(f"  Noise: {len(s.det_rows.noise)} boxes")

            print(f"\n=== Text Box Row Info ===")
            print(f"Rows: {len(s.text_rows)}, signs: {len(s.text_boxes)}")
            for row_idx, count in enumerate(s.text_rows.counts()):
                print(f"  Row {row_idx}: {count} signs")


# ---------------------------------------------------------------------------
# Step: Match Rows
# ---------------------------------------------------------------------------

class StepMatchRows(Step):
    name = "Match Detection Rows to Text Rows"

    def run(self, context: CropContext):
        s = context.state
        det_row_sequences = s.det_rows.sign_sequences()
        text_row_sequences = s.text_rows.sign_sequences()

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
            s.crop_tablet.img.copy(),
            s.det_rows.as_lists(),
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
        det_rows = s.det_rows.as_dict()
        text_rows = s.text_rows.as_dict()
        aligned_boxes = Boxes(tablet=s.crop_tablet)
        aligned_row_indices = [[] for _ in range(len(s.text_rows))]

        for text_row_idx in sorted(s.row_sign_matches):
            if text_row_idx not in s.text_to_det:
                continue
            det_row_idx = s.text_to_det[text_row_idx]
            text_row_boxes = text_rows.get(text_row_idx, [])
            det_row_boxes = det_rows.get(det_row_idx, [])
            if not text_row_boxes or not det_row_boxes:
                continue

            row_boxes = align_text_row_to_detection(
                text_boxes=text_row_boxes,
                det_boxes=det_row_boxes,
                matches=s.row_sign_matches[text_row_idx],
                avg_width=s.detections.avg_width,
                avg_height=s.detections.avg_height,
                min_width_ratio=2 / 3,
                max_width_ratio=4 / 3,
            )
            for box in row_boxes:
                aligned_row_indices[text_row_idx].append(len(aligned_boxes))
                aligned_boxes.append(box)

        s.aligned_boxes = aligned_boxes
        s.aligned_rows = BoxRows(s.aligned_boxes, aligned_row_indices)

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if not vis.info:
            return
        print(f"=== Row Alignment Summary ===")
        print(f"Total aligned sign boxes: {len(s.aligned_boxes)}")
        print(f"Matched text rows aligned: {len(s.row_sign_matches)}")
        for row_idx, count in enumerate(s.aligned_rows.counts()):
            if count:
                print(f"  Row {row_idx}: {count} signs")


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
            det_rows=s.det_rows.as_lists(),
            aligned_rows=s.aligned_rows.as_lists(),
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
            rows=s.text_rows.as_lists(),
            row_mapping=s.text_to_det,
            sign_match_info=s.text_sign_match_info,
            mapped_label_prefix="D",
            line_thickness=2,
            marker_size=5,
        )

        diag_vis = BboxVisualizer()
        diag_vis.draw_alignment_diagnostic(
            img=s.crop_tablet.img.copy(),
            det_rows=s.det_rows.as_lists(),
            aligned_rows=s.aligned_rows.as_lists(),
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
        det_rows = s.det_rows.as_dict()

        match_pairs = {}
        for text_row_idx, sign_matches in s.row_sign_matches.items():
            det_row_idx = s.text_to_det[text_row_idx]
            for t_idx, d_idx in sign_matches:
                match_pairs[(text_row_idx, t_idx)] = (det_row_idx, d_idx)

        offsets = {"cx": [], "cy": [], "w": [], "h": []}
        for text_row_idx, row in enumerate(s.aligned_rows.as_lists()):
            for text_col_idx, sb in enumerate(row):
                key = (text_row_idx, text_col_idx)
                if key not in match_pairs:
                    continue
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
        context.config.tablet_detector.unload_model()


# ---------------------------------------------------------------------------
# Step: Create PSR Optimizer
# ---------------------------------------------------------------------------

class StepCreatePsrOptimizer(Step):
    name = "Create PSR Optimizer"

    def run(self, context: CropContext):
        s = context.state
        p = context.config.psr_params or {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        target_detections = s.det_boxes

        s.optimizer = PointSetRegistrationOptimizer(
            source_rows=s.aligned_rows.as_lists(),
            target_detections=target_detections,
            sigma=s.detections.avg_width * p.get('sigma_factor', 1.5),
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
            contour_mask=s.crop_tablet.mask,
            device=device,
        )

    def visualize(self, context: CropContext, vis: VisOptions):
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

class StepRunPsrOptimization(Step):
    name = "Run PSR Optimization"

    def __init__(
        self,
        num_iterations: Optional[int] = None,
        num_iterations_from_affine_probe: bool = False,
        remaining_after_affine_probe: bool = False,
        name: Optional[str] = None,
    ):
        self.num_iterations = num_iterations
        self.num_iterations_from_affine_probe = num_iterations_from_affine_probe
        self.remaining_after_affine_probe = remaining_after_affine_probe
        if name is not None:
            self.name = name

    def _num_iterations(self, context: CropContext) -> int:
        p = context.config.psr_params or {}
        total = int(p.get('num_iterations', 80))
        if self.num_iterations is not None:
            return max(0, int(self.num_iterations))

        cfg = _dift_alignment_config(context.config)
        probe_iter = cfg.affine_probe_iteration
        if probe_iter is None:
            probe_iter = total
        probe_iter = max(0, int(probe_iter))

        if self.num_iterations_from_affine_probe:
            return min(total, probe_iter)
        if self.remaining_after_affine_probe:
            return max(0, total - probe_iter)
        return total

    def run(self, context: CropContext):
        s = context.state
        p = context.config.psr_params or {}
        num_iterations = self._num_iterations(context)
        if num_iterations <= 0:
            s.final_boxes = s.optimizer.get_optimized_boxes()
            return

        s.final_boxes = s.optimizer.optimize(
            num_iterations=num_iterations,
            lr=p.get('lr', 1.0),
            sigma_anneal=p.get('sigma_anneal', True),
            sigma_final=None,
            verbose=True,
            log_every=20,
        )

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if vis.info:
            print(f"=== Optimization Complete: {len(s.final_boxes)} signs ===")


# ---------------------------------------------------------------------------
# Step: Paste canonical signs at current PSR boxes
# ---------------------------------------------------------------------------

class StepVisualizeCanonicalSignsAtPsrBoxes(Step):
    name = "Visualize Canonical Signs at Current Optimization Boxes"
    description = (
        "Snapshots the current PSR boxes and pastes each sign's canonical "
        "image into that box for visualization only."
    )

    def run(self, context: CropContext):
        s = context.state
        s.canonical_overlay_iteration = None
        s.canonical_overlay_image = None
        s.canonical_overlay_stats = None
        s.canonical_overlay_error = None

        s.canonical_overlay_error = _dift_diagnostic_error(s)
        if s.canonical_overlay_error:
            return

        cfg = _dift_alignment_config(context.config)
        try:
            current_boxes = s.optimizer.get_optimized_boxes()
            overlay, stats = render_canonical_sign_overlay(
                image=s.crop_tablet.img,
                boxes=current_boxes,
                cache=s.canonical_cache,
                max_boxes=cfg.canonical_overlay_max_boxes,
                draw_boxes=False,
                draw_labels=False,
            )
            s.canonical_overlay_iteration = len(s.optimizer.loss_history)
            s.canonical_overlay_image = overlay
            s.canonical_overlay_stats = stats
        except Exception as exc:
            s.canonical_overlay_error = f"{type(exc).__name__}: {exc}"

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if vis.info:
            if s.canonical_overlay_stats is not None:
                st = s.canonical_overlay_stats
                print(f"=== Canonical Sign Overlay @ iter {s.canonical_overlay_iteration}: "
                      f"{st.get('pasted', 0)}/{st.get('total', 0)} pasted ===")
                missing = st.get("missing_names") or []
                if missing:
                    shown = ", ".join(missing[:20])
                    suffix = " ..." if len(missing) > 20 else ""
                    print(f"  Missing canonical images: {shown}{suffix}")
            elif s.canonical_overlay_error:
                print(f"=== Canonical Sign Overlay skipped: {s.canonical_overlay_error} ===")

        if not (vis.display or vis.save):
            return
        if s.canonical_overlay_image is None:
            return

        if vis.save:
            iter_tag = f"iter{s.canonical_overlay_iteration}"
            cv2.imwrite(_out(context, f"canonical_sign_overlay_{iter_tag}.jpg"),
                        s.canonical_overlay_image,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        if vis.display:
            _display_bgr(
                s.canonical_overlay_image,
                f"Canonical signs pasted at PSR boxes @ iter {s.canonical_overlay_iteration}",
            )


# ---------------------------------------------------------------------------
# Step: DIFT affine probe for current PSR state
# ---------------------------------------------------------------------------

class StepDiftAffineProbe(Step):
    name = "DIFT Affine Probe"
    description = (
        "Snapshots the current PSR optimizer boxes and estimates canonical "
        "DIFT affine transforms for visualization."
    )

    def run(self, context: CropContext):
        s = context.state
        cfg = _dift_alignment_config(context.config)

        s.dift_affine_probe_iteration = None
        s.dift_affine_probe_boxes = None
        s.dift_affine_probe_results = None
        s.dift_affine_probe_error = None

        s.dift_affine_probe_error = _dift_diagnostic_error(s)
        if s.dift_affine_probe_error:
            return

        try:
            probe_boxes = s.optimizer.get_optimized_boxes()
            results = build_dift_affine_probe(
                boxes=probe_boxes,
                cache=s.canonical_cache,
                padding_ratio=cfg.affine_probe_padding_ratio,
                max_boxes=cfg.affine_probe_max_boxes,
                max_matches=cfg.affine_probe_max_matches,
                min_matches=cfg.affine_probe_min_matches,
                ransac_threshold=cfg.affine_probe_ransac_threshold,
            )
            s.dift_affine_probe_iteration = len(s.optimizer.loss_history)
            s.dift_affine_probe_boxes = probe_boxes
            s.dift_affine_probe_results = results
        except Exception as exc:
            s.dift_affine_probe_error = f"{type(exc).__name__}: {exc}"

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if vis.info:
            if s.dift_affine_probe_results is not None:
                ok = sum(1 for r in s.dift_affine_probe_results if r.affine is not None)
                print(f"=== DIFT Affine Probe @ iter {s.dift_affine_probe_iteration}: "
                      f"{ok}/{len(s.dift_affine_probe_results)} affine estimates ===")
            elif s.dift_affine_probe_error:
                print(f"=== DIFT Affine Probe skipped: {s.dift_affine_probe_error} ===")

        if not (vis.display or vis.save):
            return
        if s.dift_affine_probe_results is None or s.dift_affine_probe_boxes is None:
            return

        cfg = _dift_alignment_config(context.config)
        overlay, grid = render_dift_affine_probe(
            image=s.crop_tablet.img,
            boxes=s.dift_affine_probe_boxes,
            results=s.dift_affine_probe_results,
            iteration=s.dift_affine_probe_iteration,
            thumb=cfg.affine_probe_thumb,
        )
        if vis.save:
            iter_tag = f"iter{s.dift_affine_probe_iteration}"
            cv2.imwrite(_out(context, f"dift_affine_probe_{iter_tag}_boxes.jpg"), overlay)
            cv2.imwrite(_out(context, f"dift_affine_probe_{iter_tag}_grid.jpg"), grid,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        if vis.display:
            for title, img in [
                (f"DIFT affine probe boxes @ iter {s.dift_affine_probe_iteration}", overlay),
                (f"DIFT affine probe grid @ iter {s.dift_affine_probe_iteration}", grid),
            ]:
                _display_bgr(img, title)


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
        img = s.crop_tablet.img

        before_vis = BboxVisualizer(color=(0, 255, 255))
        before_vis.draw_boxes(img.copy(), s.aligned_boxes)

        after_vis = BboxVisualizer(color=(255, 255, 0))
        after_vis.draw_boxes(img.copy(), s.final_boxes)

        det_base = BboxVisualizer(color=(255, 0, 0))
        det_base.draw_boxes(img.copy(), s.det_boxes)
        det_final_ov = BboxVisualizer(color=(255, 255, 0))
        det_final_ov.draw_boxes(det_base.result, s.final_boxes)

        gt_base = BboxVisualizer(color=(0, 255, 0))
        gt_base.draw_boxes(img.copy(), s.gt_boxes_crop or [])
        gt_final_ov = BboxVisualizer(color=(255, 255, 0))
        gt_final_ov.draw_boxes(gt_base.result, s.final_boxes)

        comp = CompositeVisualizer()
        comp.compose(
            images=[before_vis.result, after_vis.result, det_final_ov.result, gt_final_ov.result],
            layout=(2, 2),
            titles=[
                f"Before PSR: Coarse Aligned ({len(s.aligned_boxes)} signs)",
                f"After PSR: Final Optimized ({len(s.final_boxes)} signs)",
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

        n = min(5, len(s.aligned_boxes))
        print(f"\n=== First {n} Signs: Coarse -> Final ===")
        for i in range(n):
            before = s.aligned_boxes[i]
            after = s.final_boxes[i]
            print(f"  {i+1}. {before.sign_name}:")
            print(f"      Coarse: cx={before.cx:.1f}, cy={before.cy:.1f}, "
                  f"w={before.width:.1f}, h={before.height:.1f}")
            print(f"      Final:  cx={after.cx:.1f},  cy={after.cy:.1f}, "
                  f"w={after.width:.1f},  h={after.height:.1f}")
            print(f"      Delta:  cx={after.cx-before.cx:.1f}, cy={after.cy-before.cy:.1f}, "
                  f"w={after.width-before.width:.1f}, h={after.height-before.height:.1f}")



# ---------------------------------------------------------------------------
# Step: Setup canonical-sign feature cache
# ---------------------------------------------------------------------------

class StepSetupCanonicalSigns(Step):
    name = "Setup Canonical Signs"
    description = "Build canonical sign features for this fragment's period."

    def run(self, context: CropContext):
        s = context.state
        s.canonical_source = None
        s.canonical_cache = None
        s.canonical_meta = None

        fragment_data = context.config.api_source.get_fragment_data(s.fragment_id)
        period = fragment_data["script"]["period"]

        dift_cfg = context.config.dift
        if dift_cfg is None:
            raise ValueError("PipelineConfig.dift is required for canonical DIFT features")

        if context._dift_model is None:
            print("  [Canonical] Loading SD-DIFT model (first time, ~30 s)...")
            context._dift_model = load_dift_model(dift_cfg)

        wrapper = make_dift_wrapper(dift_cfg, context._dift_model, prompt="")

        source = EBLMongoCanonicalSource(
            mongodb_uri=context.config.canonical_mongodb_uri,
            period=period,
            db_name=context.config.canonical_db_name,
            form=context.config.canonical_form,
            require_centroid=context.config.canonical_require_centroid,
        )
        s.canonical_source = source
        meta = {
            "enabled": source.is_ready(),
            "period": period,
            "source_kind": type(source).__name__,
            "precompute": None,
        }
        s.canonical_meta = meta
        if not source.is_ready():
            raise RuntimeError(f"canonical source is not ready: {source.describe()}")

        cache_key = source.cache_namespace()
        cache = context._canonical_caches.get(cache_key)

        dift_align_cfg = _dift_alignment_config(context.config)
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
                verbose=True, limit=dift_align_cfg.precompute_limit,
                progress_every=50,
            )
            meta["precompute"] = stats
            context._canonical_caches[cache_key] = cache
        else:
            meta["precompute"] = {"reused": True, "size": len(cache)}

        s.canonical_cache = cache

    def visualize(self, context: CropContext, vis: VisOptions):
        s = context.state
        if vis.info:
            meta = s.canonical_meta or {}
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

        if not (vis.display or vis.save):
            return

        cfg = _dift_alignment_config(context.config)
        rows, missing, total = collect_detected_canonical_feature_rows(
            s.det_boxes,
            s.canonical_cache,
            max_signs=cfg.feature_viz_max_signs,
        )
        if vis.info and total:
            print(f"  Feature-map viz: detected unique signs={total}, shown={len(rows)}")
            if missing:
                print("  Missing canonical features: " + ", ".join(missing[:20]))
        if not rows:
            return

        grid = render_canonical_feature_grid(rows)
        if vis.display:
            _display_bgr(grid, "Canonical DIFT feature maps", px_per_in=60.0)
        if vis.save:
            cv2.imwrite(_out(context, "canonical_feature_maps.jpg"), grid,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------------------
# Step instances
# ---------------------------------------------------------------------------



step_load_data = StepLoadData()
step_detect_signs = StepDetectSigns()
step_transform_gt_to_img = StepTransformGtToImg()
step_compute_statistics = StepComputeStatistics()
step_create_box_sets = StepCreateBoxSets()
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
step_run_psr_optimization_until_dift_probe = StepRunPsrOptimization(
    num_iterations_from_affine_probe=True,
    name="Run PSR Optimization (before DIFT affine probe)",
)
step_visualize_canonical_signs_at_psr_boxes = StepVisualizeCanonicalSignsAtPsrBoxes()
step_dift_affine_probe = StepDiftAffineProbe()
step_run_psr_optimization_after_dift_probe = StepRunPsrOptimization(
    remaining_after_affine_probe=True,
    name="Run PSR Optimization (after DIFT affine probe)",
)
step_plot_loss_history = StepPlotLossHistory()
step_results_comparison = StepResultsComparison()
step_param_changes = StepParamChanges()
step_setup_canonical_signs = StepSetupCanonicalSigns()


DEBUG_STEPS = [
    step_load_data,
    step_detect_signs,
    step_transform_gt_to_img,
    step_compute_statistics,
    step_create_box_sets,
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


# Detector + canonical DIFT feature diagnostics.
DEBUG_STEPS_WITH_DIFT = [
    step_load_data,
    step_detect_signs,
    step_transform_gt_to_img,
    step_compute_statistics,
    step_create_box_sets,
    step_detect_rows,
    step_match_rows,
    step_visualize_detection_rows,
    step_match_signs_in_rows,
    step_align_text_rows,
    step_build_sign_match_info,
    step_offset_analysis,
    step_unload_detector,
    step_setup_canonical_signs,
    step_create_psr_optimizer,
    step_run_psr_optimization_until_dift_probe,
    step_visualize_canonical_signs_at_psr_boxes,
    step_dift_affine_probe,
    step_run_psr_optimization_after_dift_probe,
    step_plot_loss_history,
    step_results_comparison,
    step_param_changes,
]

# Per-crop sub-pipeline used in batch scripts: runs everything from GT transform
# through PSR optimization for a single crop. Skipping logic (empty rows/matches)
# must still be handled by the caller between choose_crop() and run_all().
PIPELINE_STEPS_PER_CROP = [
    step_transform_gt_to_img,
    step_create_box_sets,
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

        fragments = context.config.local_source.get_available_fragments()
        self._fragments = fragments
        context.state.fragments = fragments
        print(f"Found {len(fragments)} fragments with both image and annotation")

        if self.vis.save:
            os.makedirs(context.config.output_dir, exist_ok=True)

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
        crop_tablets = self.context.config.tablet_detector.get_crop_tablets()
        if not crop_tablets or self.context.state.detections is None:
            return
        if crop_idx < 0 or crop_idx >= len(crop_tablets):
            raise IndexError(
                f"crop index {crop_idx} is out of range; "
                f"available crop indices are 0..{len(crop_tablets) - 1}"
            )
        self.context.state.crop_tablet = crop_tablets[crop_idx]
        self.context.state.det_boxes = self.context.config.tablet_detector.get_crop_boxes()[crop_idx]
