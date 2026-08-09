"""Unified sign-alignment pipeline.

The current fixed-candidate workflow comes first.  Result-without-optimization,
PSR, and DIFT/prototype steps follow as optional supplements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from sign_alignment.detector import TabletImageDetector
from sign_alignment.data_source import (
    DataSource,
    EBLAPISource,
    LocalDataSource,
    SignAPIResolver,
    SignTextParser,
)
from sign_alignment.box import Box, Boxes, SignCandidate, boxes_in_crop
from sign_alignment.tablet import SubTablet, Tablet
from sign_alignment.visualizer import (
    BboxVisualizer,
    ColorConfig,
    CompositeVisualizer,
    TextVisualizer,
    build_sign_match_info as build_sign_match_info_data,
)
from sign_alignment.psr_optimizer import PointSetRegistrationOptimizer
from data_processing.line_process import (
    align_text_row_to_detection,
    create_row_mapping,
    match_rows_dp,
    match_signs_in_row_dp,
)
from data_processing.hough_row_detection import detect_hough_rows
from sign_alignment.dift_align import (
    DiftAffineProbe,
    DiftMatchConfig,
    DiftRuntime,
    SignOverlay,
    build_dift_affine_probe,
    collect_detected_source_feature_rows,
    render_dift_affine_probe,
    render_source_feature_grid,
    render_source_sign_overlay,
    source_foreground_mask,
)


@dataclass
class VisOptions:
    info: bool = True
    display: bool = True
    save: bool = True


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

    def __repr__(self):
        boxes_repr = repr([box.sign_name for box in self.boxes])
        return (
            f"{type(self).__name__}("
            f"boxes={boxes_repr},\n"
            f"rows={self.rows!r},\n"
            f"noise={self.noise!r}"
            f")"
        )

    @classmethod
    def detect_using_hough(
        cls,
        boxes: Boxes,
        angle_range_deg: float = 15.0,
        angle_step_deg: float = 1.0,
        rho_step_factor: float = 0.04,
        rho_sigma_factor: float = 0.10,
        min_line_distance_factor: float = 0.30,
        assignment_distance_factor: float = 0.24,
        min_peak_votes: float = 1.25,
        min_row_size: int = 2,
        curve_search_angle_deg: float = 5.0,
        curve_curvature_penalty: float = 1.0,
    ) -> "BoxRows":
        """Group boxes using 2-D Hough peaks and one fitted angle per row.

        Every detection center has equal weight. Candidate baselines are local
        maxima in the full ``(theta, rho)`` parameter space, rather than peaks
        from a single global-angle slice. Candidate support points are assigned
        disjointly and each row is robustly refitted. A robust low-curvature,
        zero-average-slope angle curve is fitted through that first selection,
        then the final selection is repeated inside its local angle window.
        Both the Hough grid and every continuous refit are bounded by
        ``angle_range_deg``.
        """
        if not boxes:
            return cls(boxes=boxes, rows=[])

        centers = np.asarray(
            [[box.cx, box.cy] for box in boxes],
            dtype=np.float64,
        )
        scale = float(np.median([box.height for box in boxes]))
        detection = detect_hough_rows(
            centers=centers,
            scale=scale,
            angle_range_deg=angle_range_deg,
            angle_step_deg=angle_step_deg,
            rho_step_factor=rho_step_factor,
            rho_sigma_factor=rho_sigma_factor,
            min_line_distance_factor=min_line_distance_factor,
            assignment_distance_factor=assignment_distance_factor,
            min_peak_votes=min_peak_votes,
            min_row_size=min_row_size,
            curve_search_angle_deg=curve_search_angle_deg,
            curve_curvature_penalty=curve_curvature_penalty,
        )
        result = cls(
            boxes=boxes,
            rows=detection.rows,
            noise=detection.noise,
        )
        # hough_angle_deg is retained as a backward-compatible global
        # diagnostic; selected baselines use hough_row_angles_deg instead.
        result.hough_angle_deg = detection.global_angle_deg
        result.hough_row_angles_deg = detection.row_angles_deg
        result.hough_rhos = detection.row_rhos
        result.hough_row_x_ranges = detection.row_x_ranges
        result.hough_angles_deg = detection.angles_deg
        result.hough_rho_grid = detection.rho_grid
        result.hough_parameter_space = detection.parameter_space
        result.hough_angle_scores = detection.angle_scores
        result.hough_x_origin = detection.x_origin
        result.hough_initial_row_angles_deg = detection.initial_row_angles_deg
        result.hough_initial_row_rhos = detection.initial_row_rhos
        result.hough_angle_curve_deg = detection.angle_curve_angles_deg
        result.hough_angle_curve_inlier_mask = detection.angle_curve_inlier_mask
        result.hough_curve_search_angle_deg = detection.curve_search_angle_deg
        return result


@dataclass
class SampleState:
    """All intermediate results for a single fragment."""

    fragments: list = None
    fragment_id: str = None
    fragment_data: dict = None

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
    result_without_optimization_boxes: Optional[Boxes] = None
    result_without_optimization_relabelled: int = 0
    result_without_optimization_changed: int = 0
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

    source_period: Optional[str] = None
    dift_affine_probe: Optional[DiftAffineProbe] = None
    source_overlay: Optional[SignOverlay] = None

    # Extension pipelines keep isolated results here.  The base state has no
    # dependency on extension-specific result classes.
    extras: dict[str, object] = field(default_factory=dict)


@dataclass
class CropContext:
    tablet_detector: TabletImageDetector
    local_source: LocalDataSource
    color_config: ColorConfig
    output_dir: str
    api_source: EBLAPISource = field(default_factory=EBLAPISource)
    sign_resolver: SignAPIResolver = field(default_factory=SignAPIResolver)
    img_idx: int = 1
    psr_params: Optional[dict] = None
    dift: Optional[DiftRuntime] = None
    sign_source: Optional[DataSource] = None
    canonical_source: Optional[DataSource] = None
    state: SampleState = field(default_factory=SampleState)
    task_type: str = "debug"
    gt_visualization_excluded_prefixes: tuple[str, ...] = ("SURFACE_",)


RunFn = Callable[[CropContext], None]
VisFn = Callable[[CropContext, VisOptions], None]


@dataclass
class Step:
    name: str
    run: RunFn
    visualize: Optional[VisFn] = None


def output_path(context: CropContext, suffix: str) -> str:
    return os.path.join(
        context.output_dir,
        f"{context.task_type}_{context.state.fragment_id}_{suffix}",
    )


_out = output_path  # Backward-compatible alias for existing callers.


def source_period(context: CropContext) -> str:
    period = context.state.source_period
    if period is None:
        period = context.state.fragment_data["script"]["period"]
        context.state.source_period = period
    return period


_source_period = source_period  # Backward-compatible alias for existing callers.


def gt_boxes_for_visualization(
    context: CropContext,
    boxes: Optional[Boxes],
) -> list[Box]:
    """Return sign-level GT boxes, leaving the stored annotations unchanged."""
    excluded_prefixes = tuple(
        prefix.strip().upper()
        for prefix in context.gt_visualization_excluded_prefixes
        if prefix.strip()
    )
    if not excluded_prefixes:
        return list(boxes or [])
    return [
        box
        for box in boxes or []
        if not box.sign_name.strip().upper().startswith(excluded_prefixes)
    ]


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


# =============================================================================
# Part 1: current fixed-candidate workflow
# =============================================================================


def load_data(context: CropContext) -> None:
    s = context.state
    img = context.local_source.load_image(s.fragment_id)
    if img is None:
        raise ValueError(f"No image found for sample {s.fragment_id}")
    s.tablet = Tablet(img=img, name=s.fragment_id)
    s.gt_boxes = context.local_source.load_annotation(s.fragment_id, s.tablet)

    s.fragment_data = context.api_source.get_fragment_data(s.fragment_id)
    if s.fragment_data is None:
        raise ValueError(f"No fragment data found for sample {s.fragment_id}")
    text_data = s.fragment_data.get("text", {})
    s.text_lines = SignTextParser.parse_text_lines(
        text_data, filter_broken=True, sign_resolver=context.sign_resolver)
    s.text_lines_unfiltered = SignTextParser.parse_text_lines(
        text_data, filter_broken=False, sign_resolver=context.sign_resolver)


def vis_loaded_data(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    visual_gt_boxes = gt_boxes_for_visualization(context, s.gt_boxes)
    gt_vis = BboxVisualizer(context.color_config.GT_COLOR.value)
    gt_vis.draw_boxes(s.tablet.img.copy(), visual_gt_boxes)

    if vis.info:
        total_text = sum(map(len, s.text_lines))
        total_unfiltered = sum(map(len, s.text_lines_unfiltered))
        hidden_gt_count = len(s.gt_boxes or []) - len(visual_gt_boxes)
        print(f"Ground truth sign boxes: {len(visual_gt_boxes)}")
        if hidden_gt_count:
            print(f"  Non-sign GT boxes hidden: {hidden_gt_count}")
        print(f"  Text lines: {len(s.text_lines)}, total signs: {total_text}")
        print(
            f"  Unfiltered: {total_unfiltered} signs, "
            f"broken removed: {total_unfiltered - total_text}"
        )
    if vis.save:
        TextVisualizer.save_text(
            s.text_lines, path=_out(context, "text_filtered.txt"),
            fragment_id=s.fragment_id)
        TextVisualizer.save_text(
            s.text_lines_unfiltered, path=_out(context, "text.txt"),
            fragment_id=s.fragment_id)
        gt_vis.save(_out(context, "gt.jpg"))
    if vis.display:
        gt_vis.display_result(vis_opt="draw")


def detect_signs(context: CropContext) -> None:
    s = context.state
    s.detections = context.tablet_detector.detect(s.tablet)
    _select_crop(context, context.img_idx)


def _select_crop(context: CropContext, img_idx: int) -> None:
    crop_tablets = context.tablet_detector.get_crop_tablets()
    if not crop_tablets:
        raise RuntimeError("detector produced no cropped images")
    if not 0 <= img_idx < len(crop_tablets):
        raise IndexError(
            f"crop index {img_idx} is out of range after detection; "
            f"available crop indices are 0..{len(crop_tablets) - 1}"
        )
    context.img_idx = img_idx
    context.state.crop_tablet = crop_tablets[img_idx]
    context.state.det_boxes = context.tablet_detector.get_crop_boxes()[img_idx]


def vis_detections(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    color = context.color_config.DET_COLOR.value
    full_vis = BboxVisualizer(color=color)
    full_vis.draw_boxes(s.tablet.img.copy(), s.detections, show_scores=True)
    crop_vis = BboxVisualizer(color=color)
    crop_vis.draw_boxes(s.crop_tablet.img.copy(), s.det_boxes, show_scores=True)
    if vis.info:

        x, y = s.crop_tablet.offset_in_parent
        h, w = s.crop_tablet.shape
        print(f"Total detections (full image): {len(s.detections)}")
        print(f"Sub-image detections: {len(s.det_boxes)}")
        print(
            f"Crop info (img_idx={context.img_idx}): "
            f"x={x}, y={y}, w={w}, h={h}"
        )
    if vis.save:
        full_vis.save(_out(context, "det.jpg"))
        crop_vis.save(_out(context, "sub_image.jpg"))
    if vis.display:
        crop_vis.display_result(vis_opt="draw")


def transform_gt_to_crop(context: CropContext) -> None:
    s = context.state
    s.gt_boxes_crop = boxes_in_crop(s.gt_boxes, s.crop_tablet)


def vis_crop_ground_truth(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    visual_gt_boxes = gt_boxes_for_visualization(context, s.gt_boxes)
    visual_gt_boxes_crop = gt_boxes_for_visualization(context, s.gt_boxes_crop)
    if vis.info:
        print(f"GT sign boxes (full image): {len(visual_gt_boxes)}")
        print(f"GT sign boxes (sub-image):  {len(visual_gt_boxes_crop)}")
        hidden_gt_count = len(s.gt_boxes or []) - len(visual_gt_boxes)
        hidden_gt_crop_count = len(s.gt_boxes_crop or []) - len(visual_gt_boxes_crop)
        if hidden_gt_count or hidden_gt_crop_count:
            print(
                "Non-sign GT boxes hidden "
                f"(full/sub-image): {hidden_gt_count}/{hidden_gt_crop_count}"
            )
    if not visual_gt_boxes_crop:
        return

    gt_vis = BboxVisualizer(color=context.color_config.GT_COLOR.value)
    gt_vis.draw_boxes(s.crop_tablet.img.copy(), visual_gt_boxes_crop)
    if vis.save:
        gt_vis.save(_out(context, "sub_image_gt.jpg"))
    if vis.display:
        gt_vis.display_result(vis_opt="draw")


def vis_detection_statistics(context: CropContext, vis: VisOptions) -> None:
    if not vis.info:
        return
    s = context.state
    print(f"Full image shape: {s.tablet.img.shape}")
    print(f"Sub-image shape:  {s.crop_tablet.img.shape}")
    print(f"Average detected sign  width: {s.detections.avg_width:.2f}")
    print(f"Average detected sign height: {s.detections.avg_height:.2f}")


def create_box_sets(context: CropContext) -> None:
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


def vis_box_sets(context: CropContext, vis: VisOptions) -> None:
    if not vis.info:
        return
    s = context.state
    print(s.crop_tablet.info)
    print(s.text_boxes.info("text"))
    print(f"Text rows: {len(s.text_rows)}, signs: {len(s.text_boxes)}")


def detect_rows(context: CropContext) -> None:
    context.state.det_rows = BoxRows.detect_using_hough(context.state.det_boxes)


def vis_detected_rows_info(context: CropContext, vis: VisOptions) -> None:
    if not vis.info:
        return
    s = context.state
    print("=== Row Detection Results ===")
    print(
        f"Average sign size: {s.detections.avg_size:.2f} px, "
        f"detected {len(s.det_rows)} rows, {len(s.det_boxes)} signs"
    )
    row_angles = getattr(s.det_rows, "hough_row_angles_deg", np.asarray([]))
    if getattr(s.det_rows, "hough_angle_deg", None) is not None:
        if len(row_angles):
            initial_angles = getattr(
                s.det_rows,
                "hough_initial_row_angles_deg",
                np.asarray([]),
            )
            curve_inliers = getattr(
                s.det_rows,
                "hough_angle_curve_inlier_mask",
                np.asarray([], dtype=bool),
            )
            curve_outliers = (
                int((~curve_inliers).sum())
                if len(curve_inliers) == len(initial_angles)
                else 0
            )
            print(
                f"  Hough row angles: {row_angles.min():.1f} to "
                f"{row_angles.max():.1f} deg, global score peak: "
                f"{s.det_rows.hough_angle_deg:.1f} deg, "
                f"baselines: {len(s.det_rows.hough_rhos)}"
            )
            print(
                f"  Curve reselection: {len(initial_angles)} initial, "
                f"{curve_outliers} robust-fit outliers, "
                f"±{s.det_rows.hough_curve_search_angle_deg:.1f} deg window"
            )
        else:
            print(
                f"  Hough global score peak: "
                f"{s.det_rows.hough_angle_deg:.1f} deg, baselines: 0"
            )
    for row_idx, count in enumerate(s.det_rows.counts()):
        angle_info = (
            f", angle: {row_angles[row_idx]:.1f} deg"
            if row_idx < len(row_angles)
            else ""
        )
        print(f"  Row {row_idx}: {count} boxes{angle_info}")
    if s.det_rows.noise:
        print(f"  Noise: {len(s.det_rows.noise)} boxes")

    print("\n=== Text Box Row Info ===")
    print(f"Rows: {len(s.text_rows)}, signs: {len(s.text_boxes)}")
    for row_idx, count in enumerate(s.text_rows.counts()):
        print(f"  Row {row_idx}: {count} signs")


def match_rows(context: CropContext) -> None:
    s = context.state
    s.det_row_sequences = s.det_rows.sign_sequences()
    s.text_row_sequences = s.text_rows.sign_sequences()
    s.matches, _ = match_rows_dp(
        detection_rows=s.det_row_sequences,
        text_rows=s.text_row_sequences,
        skip_text_penalty=0.5,
        skip_det_penalty=1,
        skip_small_det_penalty=0.2,
        small_det_threshold=1,
        similarity_method="jaccard",
    )
    s.text_to_det, s.det_to_text = create_row_mapping(
        s.matches, len(s.text_row_sequences), len(s.det_row_sequences))


def vis_row_matches(context: CropContext, vis: VisOptions) -> None:
    if not vis.info:
        return
    s = context.state
    print("=== Row Matching ===")
    print(
        f"Detection rows: {len(s.det_row_sequences)}, "
        f"Text rows: {len(s.text_row_sequences)}, Matched: {len(s.matches)}"
    )
    for text_idx, det_idx in s.matches:
        text = s.text_row_sequences[text_idx]
        detected = s.det_row_sequences[det_idx]
        print(
            f"  Text row {text_idx} ({len(text)} signs) -> "
            f"Det row {det_idx} ({len(detected)} signs)"
        )
        print(f"    Text: {' '.join(text[:5])}{'...' if len(text) > 5 else ''}")
        print(
            f"    Det:  {' '.join(detected[:5])}"
            f"{'...' if len(detected) > 5 else ''}"
        )
    print(f"Text->Det: {s.text_to_det}")
    print(f"Det->Text: {s.det_to_text}")


def _render_hough_parameter_space(rows: BoxRows) -> np.ndarray:
    """Render the equal-weight Hough accumulator and selected row peaks."""
    import matplotlib.pyplot as plt

    angles = rows.hough_angles_deg
    rho_grid = rows.hough_rho_grid
    parameter_space = rows.hough_parameter_space
    angle_step = float(angles[1] - angles[0])
    rho_step = float(rho_grid[1] - rho_grid[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(
        parameter_space,
        cmap="magma",
        aspect="auto",
        interpolation="nearest",
        extent=(
            angles[0] - angle_step / 2,
            angles[-1] + angle_step / 2,
            rho_grid[-1] + rho_step / 2,
            rho_grid[0] - rho_step / 2,
        ),
    )
    ax.axvline(
        rows.hough_angle_deg,
        color="cyan",
        linewidth=1.5,
        linestyle="--",
        label=f"global score peak: {rows.hough_angle_deg:.1f} deg",
    )
    initial_angles = getattr(
        rows,
        "hough_initial_row_angles_deg",
        np.asarray([]),
    )
    initial_rhos = getattr(
        rows,
        "hough_initial_row_rhos",
        np.asarray([]),
    )
    curve_inliers = getattr(
        rows,
        "hough_angle_curve_inlier_mask",
        np.ones(len(initial_angles), dtype=bool),
    )
    if len(initial_angles):
        ax.scatter(
            initial_angles,
            initial_rhos,
            facecolors="none",
            edgecolors="deepskyblue",
            marker="o",
            s=52,
            linewidths=1.3,
            label="first-pass row baselines",
        )
        if len(curve_inliers) == len(initial_angles) and (~curve_inliers).any():
            ax.scatter(
                initial_angles[~curve_inliers],
                initial_rhos[~curve_inliers],
                edgecolors="orange",
                marker="s",
                facecolors="none",
                s=64,
                linewidths=1.5,
                label="robust curve-fit outliers",
            )
    angle_curve = getattr(rows, "hough_angle_curve_deg", np.asarray([]))
    if len(angle_curve) == len(rho_grid):
        curve_window = rows.hough_curve_search_angle_deg
        ax.fill_betweenx(
            rho_grid,
            np.maximum(angles[0], angle_curve - curve_window),
            np.minimum(angles[-1], angle_curve + curve_window),
            color="cyan",
            alpha=0.10,
            label=f"curve search window (±{curve_window:.1f} deg)",
        )
        ax.plot(
            angle_curve,
            rho_grid,
            color="white",
            linewidth=1.5,
            label="zero-mean-slope quadratic angle curve",
        )
    row_angles = getattr(
        rows,
        "hough_row_angles_deg",
        np.full(len(rows.hough_rhos), rows.hough_angle_deg),
    )
    ax.scatter(
        row_angles,
        rows.hough_rhos,
        color="lime",
        marker="x",
        s=45,
        linewidths=1.5,
        label="final curve-window reselection",
    )
    ax.set_xlabel("row angle theta (degrees)")
    x_origin = getattr(rows, "hough_x_origin", 0.0)
    ax.set_ylabel(
        f"normal coordinate rho (pixels; x centered at {x_origin:.1f})"
    )
    ax.set_title("Two-pass Hough row selection from detection-box centers")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="equal-weight Gaussian votes")
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    result = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return result


def vis_detection_rows(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    row_vis = BboxVisualizer(color=(255, 0, 0))
    row_vis.draw_rows(
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
    s.det_row_vis_image = row_vis.result
    s.hough_parameter_space_image = _render_hough_parameter_space(s.det_rows)
    if vis.info:
        print("Detection rows: D# on left margin, matched rows show D#->R#")
    if vis.display:
        row_vis.display_result(vis_opt="draw")
        _display_bgr(
            s.hough_parameter_space_image,
            "Hough parameter space",
        )
    if vis.save:
        row_vis.save(_out(context, "detection_rows.jpg"))
        hough_path = _out(context, "hough_parameter_space.jpg")
        cv2.imwrite(hough_path, s.hough_parameter_space_image)
        if vis.info:
            print(f"✓ Saved to: {os.path.abspath(hough_path)}")


def match_signs_in_rows(context: CropContext) -> None:
    s = context.state
    s.row_sign_matches = {}
    for text_row_idx, det_row_idx in s.matches:
        matches, _ = match_signs_in_row_dp(
            detection_signs=s.det_row_sequences[det_row_idx],
            text_signs=s.text_row_sequences[text_row_idx],
            skip_text_penalty=0.5,
            skip_det_penalty=2.0,
            mismatch_cost=0.9,
        )
        s.row_sign_matches[text_row_idx] = matches


def vis_sign_matches(context: CropContext, vis: VisOptions) -> None:
    if not vis.info:
        return
    s = context.state
    MAX_MATCHES_TO_DISPLAY = 20
    print("=== Within-Row Sign Matching ===")
    for text_row_idx, det_row_idx in s.matches:
        text = s.text_row_sequences[text_row_idx]
        detected = s.det_row_sequences[det_row_idx]
        matches = s.row_sign_matches[text_row_idx]
        print(
            f"Text row {text_row_idx} -> Det row {det_row_idx}: "
            f"{len(text)} text, {len(detected)} det, {len(matches)} matched"
        )
        for i, (text_idx, det_idx) in enumerate(matches[:MAX_MATCHES_TO_DISPLAY]):
            print(
                f"  {i + 1}. Text[{text_idx}]={text[text_idx]} "
                f"<-> Det[{det_idx}]={detected[det_idx]}"
            )
        if len(matches) > MAX_MATCHES_TO_DISPLAY:
            print(f"  ... and {len(matches) - MAX_MATCHES_TO_DISPLAY} more")
    print(
        "Total matched sign pairs: "
        f"{sum(map(len, s.row_sign_matches.values()))}"
    )


def align_text_rows(context: CropContext) -> None:
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
    s.aligned_rows = BoxRows(aligned_boxes, aligned_row_indices)


def vis_aligned_rows(context: CropContext, vis: VisOptions) -> None:
    if not vis.info:
        return
    s = context.state
    print("=== Row Alignment Summary ===")
    print(f"Total aligned sign boxes: {len(s.aligned_boxes)}")
    print(f"Matched text rows aligned: {len(s.row_sign_matches)}")
    for row_idx, count in enumerate(s.aligned_rows.counts()):
        if count:
            print(f"  Row {row_idx}: {count} signs")


def unload_detector(context: CropContext) -> None:
    context.tablet_detector.unload_model()


# Fixed-candidate attraction

# Experiment-wide semantic palette (RGB).  Green is reserved for GT only.
FIXED_CANDIDATE_COLOR = (255, 0, 0)
ANCHOR_COLOR = (180, 80, 255)
CANDIDATE_MATCH_COLOR = (255, 215, 0)
NULL_COLOR = (0, 210, 255)
GT_COLOR = (0, 255, 0)
MOVEMENT_COLOR = (255, 255, 255)


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[::-1]


@dataclass(frozen=True)
class CandidateAttractionConfig:
    """Configuration for position-first attraction within matched rows."""

    temperatures: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25)
    steps_per_temperature: int = 35
    learning_rate: float = 0.04
    device: str = "auto"

    # Boxes at almost the same position are one physical candidate.  Their
    # class hypotheses are pooled, but their geometry is never averaged.
    duplicate_iou: float = 0.85
    duplicate_center_ratio: float = 0.18

    # Pair cost.  Position is intentionally much stronger than class.
    along_weight: float = 1.0
    normal_weight: float = 3.0
    size_weight: float = 0.35
    objectness_bonus: float = 0.10
    class_bonus: float = 0.06
    diff_pair_bonus: float = 0.04

    # Row geometry and weak memory of the coarse starting point.
    baseline_weight: float = 0.55
    gap_weight: float = 0.12
    order_weight: float = 4.0
    min_gap: float = 0.25
    diff_prior_weight: float = 0.12
    unmatched_prior_weight: float = 0.04
    size_prior_weight: float = 0.08

    # NULL is always available.  A diff match has more evidence than a wholly
    # unmatched text sign and therefore needs stronger evidence to remain NULL.
    diff_null_cost: float = 2.2
    unmatched_null_cost: float = 1.5
    anchor_interval_margin: float = 0.65
    softassign_iterations: int = 30

    def __post_init__(self) -> None:
        if not self.temperatures or any(value <= 0 for value in self.temperatures):
            raise ValueError("temperatures must contain positive values")
        if self.steps_per_temperature < 0:
            raise ValueError("steps_per_temperature must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.softassign_iterations <= 0:
            raise ValueError("softassign_iterations must be positive")


@dataclass(frozen=True)
class PhysicalCandidate:
    """A fixed geometry with all detector class hypotheses at that location."""

    candidate_idx: int
    representative_det_idx: int
    member_det_indices: tuple[int, ...]
    box: Box
    label_scores: dict[str, float]
    objectness: float


@dataclass
class CandidateTextAssignment:
    text_row_idx: int
    det_row_idx: int
    text_idx: int
    sign_name: str
    input_status: str
    output_status: str
    candidate_idx: Optional[int]
    representative_det_idx: Optional[int]
    detector_labels: tuple[str, ...]
    class_support: float
    soft_probability: float
    null_probability: float
    initial_center: tuple[float, float]
    final_center: tuple[float, float]
    final_box: Box
    included_in_result: bool = True


@dataclass
class CandidateRowResult:
    text_row_idx: int
    det_row_idx: int
    candidates: list[PhysicalCandidate]
    assignments: list[CandidateTextAssignment]


@dataclass
class CandidateAttractionRun:
    config: CandidateAttractionConfig
    boxes: Boxes
    rows: BoxRows
    row_results: dict[int, CandidateRowResult]
    text_sign_match_info: Optional[dict] = None
    det_sign_match_info: Optional[dict] = None


CANDIDATE_RESULT_KEY = "candidate_attraction"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _smooth_l1(value: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    absolute = value.abs()
    return torch.where(
        absolute < beta,
        0.5 * absolute.square() / beta,
        absolute - 0.5 * beta,
    )


def _box_iou(first: Box, second: Box) -> float:
    ix1 = max(first.x1, second.x1)
    iy1 = max(first.y1, second.y1)
    ix2 = min(first.x2, second.x2)
    iy2 = min(first.y2, second.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = first.width * first.height + second.width * second.height - intersection
    return float(intersection / union) if union > 0 else 0.0


def _same_physical_candidate(
    first: Box,
    second: Box,
    config: CandidateAttractionConfig,
) -> bool:
    scale = max(
        1.0,
        0.5 * (first.width + second.width),
        0.5 * (first.height + second.height),
    )
    center_distance = np.hypot(first.cx - second.cx, first.cy - second.cy)
    return (
        _box_iou(first, second) >= config.duplicate_iou
        and center_distance / scale <= config.duplicate_center_ratio
    )


def _build_physical_candidates(
    det_boxes: list[Box],
    protected_det_indices: set[int],
    config: CandidateAttractionConfig,
) -> tuple[list[PhysicalCandidate], dict[int, int]]:
    """Group duplicate hypotheses but never merge two reliable anchors."""

    clusters: list[list[int]] = []
    for det_idx in sorted(range(len(det_boxes)), key=lambda i: det_boxes[i].score, reverse=True):
        for cluster in clusters:
            has_other_anchor = any(i in protected_det_indices for i in cluster)
            if det_idx in protected_det_indices and has_other_anchor:
                continue
            representative = max(cluster, key=lambda i: det_boxes[i].score)
            if _same_physical_candidate(det_boxes[det_idx], det_boxes[representative], config):
                cluster.append(det_idx)
                break
        else:
            clusters.append([det_idx])

    unsorted: list[PhysicalCandidate] = []
    for cluster in clusters:
        representative_idx = max(cluster, key=lambda i: det_boxes[i].score)
        label_scores: dict[str, float] = {}
        for member_idx in cluster:
            for hypothesis in det_boxes[member_idx].candidates:
                name = hypothesis.sign.name
                label_scores[name] = max(label_scores.get(name, 0.0), float(hypothesis.score))
        unsorted.append(PhysicalCandidate(
            candidate_idx=-1,
            representative_det_idx=representative_idx,
            member_det_indices=tuple(sorted(cluster)),
            box=det_boxes[representative_idx],
            label_scores=label_scores,
            objectness=max(label_scores.values(), default=0.0),
        ))

    unsorted.sort(key=lambda candidate: candidate.box.cx)
    candidates: list[PhysicalCandidate] = []
    det_to_candidate: dict[int, int] = {}
    for candidate_idx, candidate in enumerate(unsorted):
        candidate = PhysicalCandidate(
            candidate_idx=candidate_idx,
            representative_det_idx=candidate.representative_det_idx,
            member_det_indices=candidate.member_det_indices,
            box=candidate.box,
            label_scores=candidate.label_scores,
            objectness=candidate.objectness,
        )
        candidates.append(candidate)
        for det_idx in candidate.member_det_indices:
            det_to_candidate[det_idx] = candidate_idx
    return candidates, det_to_candidate


@dataclass(frozen=True)
class _RowBasis:
    tangent: np.ndarray
    normal: np.ndarray
    pitch: float
    height: float
    width: float


def _make_row_basis(
    det_boxes: list[Box],
    fallback_width: float,
    fallback_height: float,
) -> _RowBasis:
    centers = np.asarray([[box.cx, box.cy] for box in det_boxes], dtype=np.float64)
    slope = 0.0
    if len(centers) >= 2 and np.ptp(centers[:, 0]) > 1e-6:
        slope = float(np.polyfit(centers[:, 0], centers[:, 1], 1)[0])
    tangent = np.asarray([1.0, slope], dtype=np.float64)
    tangent /= np.linalg.norm(tangent)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)

    projected = np.sort(centers @ tangent) if len(centers) else np.asarray([])
    gaps = np.diff(projected)
    useful_gaps = gaps[gaps > max(1.0, fallback_width * 0.20)]
    pitch = float(np.median(useful_gaps)) if len(useful_gaps) else float(fallback_width)
    return _RowBasis(
        tangent=tangent,
        normal=normal,
        pitch=max(pitch, 1.0),
        width=max(float(fallback_width), 1.0),
        height=max(float(fallback_height), 1.0),
    )


def _encode_boxes(boxes: list[Box], basis: _RowBasis) -> np.ndarray:
    encoded = []
    for box in boxes:
        center = np.asarray([box.cx, box.cy], dtype=np.float64)
        encoded.append([
            float(center @ basis.tangent / basis.pitch),
            float(center @ basis.normal / basis.height),
            float(np.log(max(box.width, 1.0) / basis.width)),
            float(np.log(max(box.height, 1.0) / basis.height)),
        ])
    return np.asarray(encoded, dtype=np.float32).reshape((-1, 4))


def _decode_box(
    parameters: np.ndarray,
    template: Box,
    basis: _RowBasis,
    score: float,
) -> Box:
    center = (
        basis.tangent * float(parameters[0]) * basis.pitch
        + basis.normal * float(parameters[1]) * basis.height
    )
    # Decoding must preserve the optimized/extrapolated geometry.  Clipping the
    # center to the image rectangle collapses every off-image sign onto the same
    # border and creates the artificial edge piles seen in the attraction plot.
    cx = float(center[0])
    cy = float(center[1])
    width = max(float(np.exp(parameters[2]) * basis.width), 1.0)
    height = max(float(np.exp(parameters[3]) * basis.height), 1.0)
    return Box.from_center(
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        sign=template.sign,
        tablet=template.tablet,
        score=float(np.clip(score, 0.0, 1.0)),
    )


def _box_fully_inside_image(box: Box) -> bool:
    """Return whether the complete box lies inside its rectangular image."""

    image_height, image_width = box.tablet.shape
    coordinates = np.asarray([box.x1, box.y1, box.x2, box.y2], dtype=np.float64)
    return bool(
        np.isfinite(coordinates).all()
        and box.x1 >= 0.0
        and box.y1 >= 0.0
        and box.x2 <= float(image_width)
        and box.y2 <= float(image_height)
    )


def capped_soft_assignment(
    pair_cost: torch.Tensor,
    null_cost: torch.Tensor,
    temperature: float,
    allowed: Optional[torch.Tensor] = None,
    iterations: int = 30,
) -> torch.Tensor:
    """Entropy-soft partial assignment with real-column capacity <= 1.

    The returned matrix has one row per text sign and ``N + 1`` columns.  The
    last column is NULL and has unlimited capacity.  Each row sums to one; all
    real candidate columns sum to at most one.
    """

    text_count, candidate_count = pair_cost.shape
    if text_count == 0:
        return pair_cost.new_zeros((0, candidate_count + 1))
    if candidate_count == 0:
        return pair_cost.new_ones((text_count, 1))

    if allowed is None:
        allowed = torch.ones_like(pair_cost, dtype=torch.bool)
    all_cost = torch.cat([pair_cost, null_cost[:, None]], dim=1)
    shifted = all_cost - all_cost.min(dim=1, keepdim=True).values
    kernel = torch.exp(-shifted / max(float(temperature), 1e-6))
    kernel[:, :candidate_count] = kernel[:, :candidate_count] * allowed.to(kernel.dtype)
    probability = kernel.clamp_min(1e-30)

    for _ in range(max(1, int(iterations))):
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-30)
        real = probability[:, :candidate_count]
        column_sum = real.sum(dim=0, keepdim=True)
        real = real * torch.clamp(1.0 / column_sum.clamp_min(1e-30), max=1.0)
        probability = torch.cat([real, probability[:, candidate_count:]], dim=1)

    # End on the feasible intersection: scale overloaded columns once more and
    # put the removed probability mass in the unlimited NULL column.
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-30)
    real = probability[:, :candidate_count]
    real = real * torch.clamp(
        1.0 / real.sum(dim=0, keepdim=True).clamp_min(1e-30),
        max=1.0,
    )
    null = 1.0 - real.sum(dim=1, keepdim=True)
    return torch.cat([real, null.clamp_min(0.0)], dim=1)


def ordered_partial_assignment(
    pair_cost: np.ndarray,
    null_cost: np.ndarray,
    allowed: Optional[np.ndarray] = None,
) -> list[Optional[int]]:
    """Minimum-cost, order-preserving, one-to-one assignment with NULL."""

    pair_cost = np.asarray(pair_cost, dtype=np.float64)
    null_cost = np.asarray(null_cost, dtype=np.float64)
    text_count, candidate_count = pair_cost.shape
    if allowed is None:
        allowed = np.ones_like(pair_cost, dtype=bool)
    else:
        allowed = np.asarray(allowed, dtype=bool)

    score = np.full((text_count + 1, candidate_count + 1), -np.inf)
    action = np.full((text_count + 1, candidate_count + 1), "", dtype=object)
    score[0, :] = 0.0
    action[0, 1:] = "skip"
    for text_idx in range(1, text_count + 1):
        score[text_idx, 0] = score[text_idx - 1, 0] - null_cost[text_idx - 1]
        action[text_idx, 0] = "null"

    for text_idx in range(1, text_count + 1):
        for candidate_idx in range(1, candidate_count + 1):
            options = [
                (score[text_idx, candidate_idx - 1], "skip"),
                (score[text_idx - 1, candidate_idx] - null_cost[text_idx - 1], "null"),
            ]
            if allowed[text_idx - 1, candidate_idx - 1]:
                options.append((
                    score[text_idx - 1, candidate_idx - 1]
                    - pair_cost[text_idx - 1, candidate_idx - 1],
                    "match",
                ))
            score[text_idx, candidate_idx], action[text_idx, candidate_idx] = max(
                options, key=lambda item: item[0]
            )

    assignments: list[Optional[int]] = [None] * text_count
    text_idx, candidate_idx = text_count, candidate_count
    while text_idx > 0 or candidate_idx > 0:
        choice = action[text_idx, candidate_idx]
        if choice == "match":
            assignments[text_idx - 1] = candidate_idx - 1
            text_idx -= 1
            candidate_idx -= 1
        elif choice == "null":
            text_idx -= 1
        elif choice == "skip":
            candidate_idx -= 1
        elif candidate_idx > 0:
            candidate_idx -= 1
        else:
            text_idx -= 1
    return assignments


def _pair_cost(
    free_parameters: torch.Tensor,
    candidate_parameters: torch.Tensor,
    class_support: torch.Tensor,
    objectness: torch.Tensor,
    diff_support: torch.Tensor,
    allowed: torch.Tensor,
    config: CandidateAttractionConfig,
    use_class_and_size: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_parameters.shape[0] == 0:
        empty = free_parameters.new_zeros((free_parameters.shape[0], 0))
        return empty, empty
    delta = free_parameters[:, None, :] - candidate_parameters[None, :, :]
    geometry = (
        config.along_weight * _smooth_l1(delta[..., 0])
        + config.normal_weight * _smooth_l1(delta[..., 1])
    )
    if use_class_and_size:
        geometry = geometry + config.size_weight * (
            _smooth_l1(delta[..., 2]) + _smooth_l1(delta[..., 3])
        )
    cost = geometry
    if use_class_and_size:
        cost = (
            cost
            - config.objectness_bonus * objectness[None, :]
            - config.class_bonus * class_support
            - config.diff_pair_bonus * diff_support
        )
    cost = torch.where(allowed, cost, torch.full_like(cost, 1e6))
    return cost, geometry


def _copy_candidate_geometry(candidate: PhysicalCandidate, text_box: Box, score: float) -> Box:
    source = candidate.box
    return Box.from_center(
        cx=source.cx,
        cy=source.cy,
        width=source.width,
        height=source.height,
        sign=text_box.sign,
        tablet=text_box.tablet,
        score=float(np.clip(score, 0.0, 1.0)),
    )


def _optimize_row(
    context: CropContext,
    text_row_idx: int,
    det_row_idx: int,
    config: CandidateAttractionConfig,
    device: torch.device,
) -> CandidateRowResult:
    state = context.state
    text_boxes = list(state.aligned_rows.row_boxes(text_row_idx))
    det_boxes = list(state.det_rows.row_boxes(det_row_idx))
    sign_pairs = list((state.row_sign_matches or {}).get(text_row_idx, []))

    exact_pairs = {
        text_idx: det_idx
        for text_idx, det_idx in sign_pairs
        if text_boxes[text_idx].sign_name == det_boxes[det_idx].sign_name
    }
    protected_det_indices = set(exact_pairs.values())
    candidates, det_to_candidate = _build_physical_candidates(
        det_boxes, protected_det_indices, config
    )
    anchor_candidates = {
        text_idx: det_to_candidate[det_idx]
        for text_idx, det_idx in exact_pairs.items()
    }
    consumed_candidate_indices = set(anchor_candidates.values())
    free_candidate_indices = [
        candidate.candidate_idx
        for candidate in candidates
        if candidate.candidate_idx not in consumed_candidate_indices
    ]
    free_candidates = [candidates[idx] for idx in free_candidate_indices]
    free_text_indices = [
        text_idx for text_idx in range(len(text_boxes))
        if text_idx not in anchor_candidates
    ]

    matched_pair_by_text = {text_idx: det_idx for text_idx, det_idx in sign_pairs}
    diff_text_indices = set(matched_pair_by_text) - set(exact_pairs)
    basis = _make_row_basis(
        det_boxes,
        fallback_width=state.detections.avg_width,
        fallback_height=state.detections.avg_height,
    )
    initial_parameters_np = _encode_boxes(text_boxes, basis)
    candidate_parameters_np = _encode_boxes([c.box for c in free_candidates], basis)

    # Hard anchors use detector geometry throughout optimization.
    fixed_parameters_np = initial_parameters_np.copy()
    for text_idx, candidate_idx in anchor_candidates.items():
        fixed_parameters_np[text_idx] = _encode_boxes([candidates[candidate_idx].box], basis)[0]

    fixed_parameters = torch.as_tensor(fixed_parameters_np, device=device)
    free_parameters = torch.nn.Parameter(
        torch.as_tensor(fixed_parameters_np[free_text_indices], device=device).clone()
    )
    candidate_parameters = torch.as_tensor(candidate_parameters_np, device=device)

    free_count = len(free_text_indices)
    candidate_count = len(free_candidates)
    class_support_np = np.zeros((free_count, candidate_count), dtype=np.float32)
    diff_support_np = np.zeros_like(class_support_np)
    objectness_np = np.asarray([c.objectness for c in free_candidates], dtype=np.float32)

    candidate_global_to_free = {
        global_idx: free_idx for free_idx, global_idx in enumerate(free_candidate_indices)
    }
    for free_idx, text_idx in enumerate(free_text_indices):
        sign_name = text_boxes[text_idx].sign_name
        for candidate_idx, candidate in enumerate(free_candidates):
            class_support_np[free_idx, candidate_idx] = candidate.label_scores.get(sign_name, 0.0)
        original_det_idx = matched_pair_by_text.get(text_idx)
        if original_det_idx is not None:
            global_candidate_idx = det_to_candidate[original_det_idx]
            free_candidate_idx = candidate_global_to_free.get(global_candidate_idx)
            if free_candidate_idx is not None:
                diff_support_np[free_idx, free_candidate_idx] = 1.0

    # A candidate must remain inside the interval bounded by the nearest hard
    # anchors.  This is a broad gate, not a nearest-neighbour decision.
    allowed_np = np.ones((free_count, candidate_count), dtype=bool)
    anchor_u = {
        text_idx: fixed_parameters_np[text_idx, 0]
        for text_idx in anchor_candidates
    }
    for free_idx, text_idx in enumerate(free_text_indices):
        left = [idx for idx in anchor_u if idx < text_idx]
        right = [idx for idx in anchor_u if idx > text_idx]
        lower = anchor_u[max(left)] - config.anchor_interval_margin if left else -np.inf
        upper = anchor_u[min(right)] + config.anchor_interval_margin if right else np.inf
        if candidate_count:
            allowed_np[free_idx] = (
                (candidate_parameters_np[:, 0] >= lower)
                & (candidate_parameters_np[:, 0] <= upper)
            )

    class_support = torch.as_tensor(class_support_np, device=device)
    diff_support = torch.as_tensor(diff_support_np, device=device)
    objectness = torch.as_tensor(objectness_np, device=device)
    allowed = torch.as_tensor(allowed_np, device=device)
    null_cost_np = np.asarray([
        config.diff_null_cost if text_idx in diff_text_indices else config.unmatched_null_cost
        for text_idx in free_text_indices
    ], dtype=np.float32)
    null_cost = torch.as_tensor(null_cost_np, device=device)

    final_probability = torch.ones((free_count, candidate_count + 1), device=device)
    if free_count:
        optimizer = torch.optim.Adam([free_parameters], lr=config.learning_rate)
        initial_free = torch.as_tensor(
            fixed_parameters_np[free_text_indices], device=device
        )
        prior_weight = torch.as_tensor([
            config.diff_prior_weight
            if text_idx in diff_text_indices
            else config.unmatched_prior_weight
            for text_idx in free_text_indices
        ], device=device)
        baseline_v = (
            torch.median(torch.as_tensor(_encode_boxes(det_boxes, basis)[:, 1], device=device))
            if det_boxes else initial_free[:, 1].median()
        )

        def assemble_all() -> torch.Tensor:
            free_lookup = {text_idx: i for i, text_idx in enumerate(free_text_indices)}
            return torch.stack([
                free_parameters[free_lookup[text_idx]]
                if text_idx in free_lookup else fixed_parameters[text_idx]
                for text_idx in range(len(text_boxes))
            ])

        for stage_idx, temperature in enumerate(config.temperatures):
            use_class_and_size = stage_idx > 0
            for _ in range(config.steps_per_temperature):
                optimizer.zero_grad()
                pair_cost, geometry = _pair_cost(
                    free_parameters,
                    candidate_parameters,
                    class_support,
                    objectness,
                    diff_support,
                    allowed,
                    config,
                    use_class_and_size,
                )
                probability = capped_soft_assignment(
                    pair_cost.detach(),
                    null_cost,
                    temperature,
                    allowed=allowed,
                    iterations=config.softassign_iterations,
                )
                real_probability = probability[:, :candidate_count]
                attraction = (
                    (real_probability.detach() * geometry).sum() / max(free_count, 1)
                    if candidate_count else free_parameters.sum() * 0.0
                )
                all_parameters = assemble_all()
                baseline_loss = (free_parameters[:, 1] - baseline_v).square().mean()
                if len(text_boxes) >= 2:
                    gaps = all_parameters[1:, 0] - all_parameters[:-1, 0]
                    initial_gaps = fixed_parameters[1:, 0] - fixed_parameters[:-1, 0]
                    gap_loss = _smooth_l1(gaps - initial_gaps).mean()
                    order_loss = torch.relu(config.min_gap - gaps).square().mean()
                else:
                    gap_loss = free_parameters.sum() * 0.0
                    order_loss = free_parameters.sum() * 0.0
                center_prior = (
                    prior_weight
                    * (free_parameters[:, :2] - initial_free[:, :2])
                    .square()
                    .sum(dim=1)
                ).mean()
                size_prior = (
                    free_parameters[:, 2:] - initial_free[:, 2:]
                ).square().mean()
                loss = (
                    attraction
                    + config.baseline_weight * baseline_loss
                    + config.gap_weight * gap_loss
                    + config.order_weight * order_loss
                    + center_prior
                    + config.size_prior_weight * size_prior
                )
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    free_parameters[:, 2:].clamp_(min=-1.2, max=1.2)

        final_pair_cost, _ = _pair_cost(
            free_parameters.detach(),
            candidate_parameters,
            class_support,
            objectness,
            diff_support,
            allowed,
            config,
            use_class_and_size=True,
        )
        final_probability = capped_soft_assignment(
            final_pair_cost,
            null_cost,
            config.temperatures[-1],
            allowed=allowed,
            iterations=config.softassign_iterations,
        )
        hard_assignment = ordered_partial_assignment(
            final_pair_cost.detach().cpu().numpy(),
            null_cost_np,
            allowed_np,
        )
        optimized_free_np = free_parameters.detach().cpu().numpy()
    else:
        hard_assignment = []
        optimized_free_np = np.empty((0, 4), dtype=np.float32)

    assignments: list[CandidateTextAssignment] = []
    free_lookup = {text_idx: i for i, text_idx in enumerate(free_text_indices)}
    for text_idx, text_box in enumerate(text_boxes):
        initial_center = (float(text_box.cx), float(text_box.cy))
        if text_idx in anchor_candidates:
            candidate = candidates[anchor_candidates[text_idx]]
            final_box = _copy_candidate_geometry(candidate, text_box, score=1.0)
            assignment = CandidateTextAssignment(
                text_row_idx=text_row_idx,
                det_row_idx=det_row_idx,
                text_idx=text_idx,
                sign_name=text_box.sign_name,
                input_status="fully_matched",
                output_status="anchor",
                candidate_idx=candidate.candidate_idx,
                representative_det_idx=candidate.representative_det_idx,
                detector_labels=tuple(sorted(candidate.label_scores)),
                class_support=candidate.label_scores.get(text_box.sign_name, 0.0),
                soft_probability=1.0,
                null_probability=0.0,
                initial_center=initial_center,
                final_center=(final_box.cx, final_box.cy),
                final_box=final_box,
            )
        else:
            free_idx = free_lookup[text_idx]
            selected_free_candidate_idx = hard_assignment[free_idx]
            null_probability = float(final_probability[free_idx, -1].detach().cpu())
            input_status = "diff_matched" if text_idx in diff_text_indices else "unmatched"
            if selected_free_candidate_idx is None:
                final_box = _decode_box(
                    optimized_free_np[free_idx],
                    text_box,
                    basis,
                    score=1.0 - null_probability,
                )
                assignment = CandidateTextAssignment(
                    text_row_idx=text_row_idx,
                    det_row_idx=det_row_idx,
                    text_idx=text_idx,
                    sign_name=text_box.sign_name,
                    input_status=input_status,
                    output_status="null",
                    candidate_idx=None,
                    representative_det_idx=None,
                    detector_labels=(),
                    class_support=0.0,
                    soft_probability=0.0,
                    null_probability=null_probability,
                    initial_center=initial_center,
                    final_center=(final_box.cx, final_box.cy),
                    final_box=final_box,
                )
            else:
                candidate = free_candidates[selected_free_candidate_idx]
                probability = float(
                    final_probability[free_idx, selected_free_candidate_idx].detach().cpu()
                )
                final_box = _copy_candidate_geometry(candidate, text_box, score=probability)
                assignment = CandidateTextAssignment(
                    text_row_idx=text_row_idx,
                    det_row_idx=det_row_idx,
                    text_idx=text_idx,
                    sign_name=text_box.sign_name,
                    input_status=input_status,
                    output_status="candidate",
                    candidate_idx=candidate.candidate_idx,
                    representative_det_idx=candidate.representative_det_idx,
                    detector_labels=tuple(sorted(candidate.label_scores)),
                    class_support=candidate.label_scores.get(text_box.sign_name, 0.0),
                    soft_probability=probability,
                    null_probability=null_probability,
                    initial_center=initial_center,
                    final_center=(final_box.cx, final_box.cy),
                    final_box=final_box,
                )
        assignment.included_in_result = _box_fully_inside_image(assignment.final_box)
        assignments.append(assignment)

    return CandidateRowResult(
        text_row_idx=text_row_idx,
        det_row_idx=det_row_idx,
        candidates=candidates,
        assignments=assignments,
    )


def get_candidate_run(
    context: CropContext,
    *,
    required: bool = True,
) -> Optional[CandidateAttractionRun]:
    run = context.state.extras.get(CANDIDATE_RESULT_KEY)
    if run is None and required:
        raise RuntimeError("candidate attraction has not been run")
    return run  # type: ignore[return-value]


def run_candidate_attraction(
    context: CropContext,
    config: Optional[CandidateAttractionConfig] = None,
) -> CandidateAttractionRun:
    """Run candidate attraction without overwriting base alignment fields."""

    state = context.state
    if state.aligned_rows is None or state.det_rows is None:
        raise RuntimeError(
            "run_candidate_attraction requires the pipeline through align_text_rows"
        )
    config = config or CandidateAttractionConfig()
    device = _resolve_device(config.device)

    output_boxes = Boxes(tablet=state.crop_tablet)
    output_row_indices = [[] for _ in range(len(state.text_rows))]
    row_results: dict[int, CandidateRowResult] = {}
    for text_row_idx, det_row_idx in state.matches or []:
        if not state.aligned_rows.row_boxes(text_row_idx):
            continue
        result = _optimize_row(
            context,
            text_row_idx=text_row_idx,
            det_row_idx=det_row_idx,
            config=config,
            device=device,
        )
        row_results[text_row_idx] = result
        for assignment in result.assignments:
            if not assignment.included_in_result:
                continue
            output_row_indices[text_row_idx].append(len(output_boxes))
            output_boxes.append(assignment.final_box)

    output_rows = BoxRows(output_boxes, output_row_indices)
    run = CandidateAttractionRun(
        config=config,
        boxes=output_boxes,
        rows=output_rows,
        row_results=row_results,
    )
    state.extras[CANDIDATE_RESULT_KEY] = run
    # Keep the evaluation script's established state contract unchanged.
    state.candidate_test_boxes = run.boxes
    state.candidate_test_rows = run.rows
    state.candidate_test_run = run
    return run


def candidate_attraction_records(context: CropContext) -> list[dict]:
    """Flat diagnostics suitable for ``pandas.DataFrame`` in the notebook."""

    run = get_candidate_run(context)
    records = []
    for text_row_idx in sorted(run.row_results):
        for assignment in run.row_results[text_row_idx].assignments:
            # Build this explicitly instead of dataclasses.asdict(): Box owns a
            # Tablet/image, which should never be deep-copied for a small table.
            record = {
                "text_row_idx": assignment.text_row_idx,
                "det_row_idx": assignment.det_row_idx,
                "text_idx": assignment.text_idx,
                "sign_name": assignment.sign_name,
                "input_status": assignment.input_status,
                "output_status": assignment.output_status,
                "candidate_idx": assignment.candidate_idx,
                "representative_det_idx": assignment.representative_det_idx,
                "detector_labels": assignment.detector_labels,
                "class_support": assignment.class_support,
                "soft_probability": assignment.soft_probability,
                "null_probability": assignment.null_probability,
                "initial_center": assignment.initial_center,
                "final_center": assignment.final_center,
                "included_in_result": assignment.included_in_result,
            }
            record["movement_px"] = float(np.hypot(
                assignment.final_center[0] - assignment.initial_center[0],
                assignment.final_center[1] - assignment.initial_center[1],
            ))
            records.append(record)
    return records


def build_candidate_sign_match_info(context: CropContext) -> None:
    """Build diagnostic statuses for the experimental one-to-one assignment.

    ``same`` means that the text label occurs anywhere in the physical
    candidate's hypotheses.  ``diff`` is a deliberately accepted
    position-only match.  ``unmatched`` is the optimized NULL result.
    """

    state = context.state
    run = get_candidate_run(context)
    text_info = {}
    det_info = {
        (row_idx, col_idx): {"status": "unmatched", "text_sign_name": None}
        for row_idx, row in enumerate(state.det_rows.as_lists())
        for col_idx, _ in enumerate(row)
    }

    assignment_by_box = {
        id(assignment.final_box): (row_result, assignment)
        for row_result in run.row_results.values()
        for assignment in row_result.assignments
        if assignment.included_in_result
    }
    for text_row_idx, row in enumerate(run.rows.as_lists()):
        for output_col_idx, box in enumerate(row):
            row_result, assignment = assignment_by_box[id(box)]
            if assignment.output_status == "anchor":
                status = "same"
            elif assignment.output_status == "candidate":
                status = "same" if assignment.class_support > 0.0 else "diff"
            else:
                status = "unmatched"
            det_sign_name = None
            if assignment.candidate_idx is not None:
                candidate = row_result.candidates[assignment.candidate_idx]
                det_sign_name = candidate.box.sign_name
                for det_idx in candidate.member_det_indices:
                    det_info[(row_result.det_row_idx, det_idx)] = {
                        "status": status,
                        "text_sign_name": assignment.sign_name,
                    }
            text_info[(text_row_idx, output_col_idx)] = {
                "status": status,
                "det_sign_name": det_sign_name,
            }

    run.text_sign_match_info = text_info
    run.det_sign_match_info = det_info


def vis_candidate_alignment_diagnostic(context: CropContext, vis: VisOptions) -> None:
    """Render coarse and candidate-result alignment diagnostics side by side."""

    state = context.state
    run = get_candidate_run(context)
    if run.text_sign_match_info is None:
        build_candidate_sign_match_info(context)
    if state.text_sign_match_info is None or state.det_sign_match_info is None:
        state.text_sign_match_info, state.det_sign_match_info = build_sign_match_info_data(
            row_sign_matches=state.row_sign_matches,
            text_to_det=state.text_to_det,
            det_rows=state.det_rows.as_lists(),
            aligned_rows=state.aligned_rows.as_lists(),
        )

    coarse = BboxVisualizer()
    coarse.draw_alignment_diagnostic(
        img=state.crop_tablet.img.copy(),
        det_rows=state.det_rows.as_lists(),
        aligned_rows=state.aligned_rows.as_lists(),
        det_sign_match_info=state.det_sign_match_info,
        text_sign_match_info=state.text_sign_match_info,
        det_to_text=state.det_to_text,
    )
    final = BboxVisualizer()
    final.draw_alignment_diagnostic(
        img=state.crop_tablet.img.copy(),
        det_rows=state.det_rows.as_lists(),
        aligned_rows=run.rows.as_lists(),
        det_sign_match_info=run.det_sign_match_info,
        text_sign_match_info=run.text_sign_match_info,
        det_to_text=state.det_to_text,
    )
    comparison = CompositeVisualizer()
    comparison.compose(
        images=[coarse.result, final.result],
        layout=(1, 2),
        titles=["Before: coarse alignment diagnostic", "After: candidate alignment diagnostic"],
        figsize=(22, 10),
    )

    if vis.info:
        statuses = [
            info["status"]
            for info in run.text_sign_match_info.values()
        ]
        print("=== Candidate alignment diagnostic ===")
        print(f"  Same-label/anchor or supported: {statuses.count('same')}")
        print(f"  Accepted position-only diff-label: {statuses.count('diff')}")
        print(f"  NULL/unmatched: {statuses.count('unmatched')}")
        print("  Solid detection boxes + dashed non-same text boxes; solid/dashed row baselines.")
    if vis.save:
        os.makedirs(context.output_dir, exist_ok=True)
        final.save(output_path(context, "candidate_alignment_diagnostic.jpg"))
        comparison.save(output_path(
            context,
            "candidate_alignment_diagnostic_comparison.jpg",
        ))
    if vis.display:
        final.display_result(vis_opt="draw")
        comparison.display_result(vis_opt="draw")


def vis_candidate_results_comparison(context: CropContext, vis: VisOptions) -> None:
    """Mirror the original PSR 2x2 comparison for the candidate experiment."""

    state = context.state
    image = state.crop_tablet.img
    final_boxes = get_candidate_run(context).boxes

    before = BboxVisualizer(color=NULL_COLOR)
    before.draw_boxes(image.copy(), state.aligned_boxes)
    after = BboxVisualizer(color=CANDIDATE_MATCH_COLOR)
    after.draw_boxes(image.copy(), final_boxes)
    det_base = BboxVisualizer(color=FIXED_CANDIDATE_COLOR)
    det_base.draw_boxes(image.copy(), state.det_boxes)
    det_overlay = BboxVisualizer(color=CANDIDATE_MATCH_COLOR)
    det_overlay.draw_boxes(det_base.result, final_boxes)
    gt_base = BboxVisualizer(color=GT_COLOR)
    gt_base.draw_boxes(
        image.copy(), gt_boxes_for_visualization(context, state.gt_boxes_crop)
    )
    gt_overlay = BboxVisualizer(color=CANDIDATE_MATCH_COLOR)
    gt_overlay.draw_boxes(gt_base.result, final_boxes)

    comparison = CompositeVisualizer()
    comparison.compose(
        images=[before.result, after.result, det_overlay.result, gt_overlay.result],
        layout=(2, 2),
        titles=[
            f"Before: coarse aligned ({len(state.aligned_boxes)} signs)",
            f"After: candidate result ({len(final_boxes)} signs)",
            "Detection (red) + candidate result (yellow)",
            "GT (green) + candidate result (yellow)",
        ],
        figsize=(18, 14),
    )
    if vis.info:
        print("=== Candidate results comparison ===")
        print("  Cyan=coarse, yellow=result, red=detection, green=GT")
    if vis.save:
        os.makedirs(context.output_dir, exist_ok=True)
        before.save(output_path(context, "candidate_coarse_aligned.jpg"))
        after.save(output_path(context, "candidate_final.jpg"))
        det_overlay.save(output_path(context, "candidate_overlay_det_final.jpg"))
        gt_overlay.save(output_path(context, "candidate_overlay_gt_final.jpg"))
        comparison.save(output_path(context, "candidate_results_comparison.jpg"))
    if vis.display:
        comparison.display_result(vis_opt="draw")


def vis_candidate_attraction(context: CropContext, vis: VisOptions) -> None:
    """Overlay fixed candidates, attraction paths, and final assignments."""

    state = context.state
    run = get_candidate_run(context)
    image = state.crop_tablet.img.copy()

    # All fixed physical candidates: red, thin.
    for row_result in run.row_results.values():
        for candidate in row_result.candidates:
            box = candidate.box
            cv2.rectangle(
                image,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                _bgr(FIXED_CANDIDATE_COLOR),
                1,
            )
        for assignment in row_result.assignments:
            if not assignment.included_in_result:
                continue
            start = tuple(map(int, assignment.initial_center))
            end = tuple(map(int, assignment.final_center))
            if start != end:
                cv2.line(image, start, end, _bgr(MOVEMENT_COLOR), 1, cv2.LINE_AA)
            cv2.circle(image, start, 2, (180, 180, 180), -1)

    color_by_box = {}
    for row_result in run.row_results.values():
        for assignment in row_result.assignments:
            color_by_box[id(assignment.final_box)] = {
                "anchor": ANCHOR_COLOR,
                "candidate": CANDIDATE_MATCH_COLOR,
                "null": NULL_COLOR,
            }[assignment.output_status]
    visualizer = BboxVisualizer(color=CANDIDATE_MATCH_COLOR)
    visualizer.color_func = lambda box: color_by_box[id(box)]
    result_image = visualizer.draw_boxes(image, run.boxes, show_labels=True)

    if vis.info:
        records = candidate_attraction_records(context)
        included_records = [record for record in records if record["included_in_result"]]
        filtered_count = len(records) - len(included_records)
        counts = {
            status: sum(record["output_status"] == status for record in included_records)
            for status in ("anchor", "candidate", "null")
        }
        wrong_label_matches = sum(
            record["output_status"] == "candidate" and record["class_support"] <= 0.0
            for record in included_records
        )
        mean_movement = (
            float(np.mean([r["movement_px"] for r in included_records]))
            if included_records else 0.0
        )
        print("=== Candidate attraction ===")
        print(f"  Device: {_resolve_device(run.config.device)}")
        print(
            f"  Output: {counts['anchor']} anchors, "
            f"{counts['candidate']} candidate matches, {counts['null']} NULL"
        )
        print(f"  Off-image boxes filtered from result: {filtered_count}")
        print(f"  Position-only/wrong-label candidate matches: {wrong_label_matches}")
        print(f"  Mean center movement: {mean_movement:.1f} px")
        print("  Red=fixed candidates, purple=anchors, yellow=candidate, cyan=NULL, white=movement")
    if vis.save:
        os.makedirs(context.output_dir, exist_ok=True)
        path = output_path(context, "candidate_attraction.jpg")
        cv2.imwrite(path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if vis.info:
            print(f"  Saved: {os.path.abspath(path)}")
    if vis.display:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 10))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("Fixed-candidate attraction (position-first)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# =============================================================================
# Part 2 (supplement): result without optimization and coarse diagnostics
# =============================================================================

def create_result_without_optimization(context: CropContext) -> None:
    """Relabel copied detection boxes using the aligned text correspondences."""
    s = context.state
    result_boxes = s.det_boxes.copy()
    relabelled = 0
    changed = 0

    for text_row_idx, matches in s.row_sign_matches.items():
        if text_row_idx not in s.text_to_det:
            continue
        det_row_idx = s.text_to_det[text_row_idx]
        aligned_row = s.aligned_rows.row_boxes(text_row_idx)
        det_row_indices = s.det_rows.rows[det_row_idx]

        for text_idx, det_idx in matches:
            result_box = result_boxes[det_row_indices[det_idx]]
            aligned_box = aligned_row[text_idx]
            relabelled += 1
            if result_box.sign_name != aligned_box.sign_name:
                changed += 1
            result_box.candidates = [
                SignCandidate(sign=aligned_box.sign, score=result_box.score)
            ]

    s.result_without_optimization_boxes = result_boxes
    s.result_without_optimization_relabelled = relabelled
    s.result_without_optimization_changed = changed

def vis_result_without_optimization(
    context: CropContext,
    vis: VisOptions,
) -> None:
    s = context.state
    image = s.crop_tablet.img
    result_boxes = s.result_without_optimization_boxes
    visual_gt_boxes = gt_boxes_for_visualization(context, s.gt_boxes_crop)

    detection_vis = BboxVisualizer(color=(255, 0, 0))
    detection_vis.draw_boxes(image.copy(), s.det_boxes)
    result_vis = BboxVisualizer(color=(255, 255, 0))
    result_vis.draw_boxes(image.copy(), result_boxes)

    gt_base = BboxVisualizer(color=(0, 255, 0))
    gt_base.draw_boxes(image.copy(), visual_gt_boxes)
    gt_overlay = BboxVisualizer(color=(255, 255, 0))
    gt_overlay.draw_boxes(gt_base.result, result_boxes)

    comparison = CompositeVisualizer()
    comparison.compose(
        images=[
            detection_vis.result,
            result_vis.result,
            gt_overlay.result,
        ],
        layout=(1, 3),
        titles=[
            "Detection",
            "Result",
            "Result + GT overlay",
        ],
        figsize=(24, 8),
    )

    if vis.info:
        print("=== Result Without Optimization ===")
        print(f"  Detection boxes: {len(s.det_boxes or [])}")
        print(f"  Result boxes: {len(result_boxes or [])}")
        print(f"  GT sign boxes: {len(visual_gt_boxes)}")
        print(
            f"  Relabelled matched boxes: "
            f"{s.result_without_optimization_relabelled}"
        )
        print(
            f"  Labels actually changed: "
            f"{s.result_without_optimization_changed}"
        )
        print("  Box positions and sizes are copied from detections unchanged.")
        print("  Red=Detection  Yellow=Relabelled result  Green=GT")
    if vis.display:
        comparison.display_result(vis_opt="draw")
    if vis.save:
        comparison.save(
            _out(context, "results_without_optimization_comparison.jpg")
        )

def build_sign_match_info(context: CropContext) -> None:
    s = context.state
    s.text_sign_match_info, s.det_sign_match_info = build_sign_match_info_data(
        row_sign_matches=s.row_sign_matches,
        text_to_det=s.text_to_det,
        det_rows=s.det_rows.as_lists(),
        aligned_rows=s.aligned_rows.as_lists(),
    )

def vis_sign_match_info(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    if vis.info:
        text_statuses = [value["status"] for value in s.text_sign_match_info.values()]
        det_statuses = [value["status"] for value in s.det_sign_match_info.values()]
        print("=== Sign Match Info ===")
        print(f"  Matched, same label:  {text_statuses.count('same')}")
        print(f"  Matched, diff label:  {text_statuses.count('diff')}")
        print(f"  Unmatched text signs: {text_statuses.count('unmatched')}")
        print(f"  Unmatched det signs:  {det_statuses.count('unmatched')}")

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
    diagnostic_vis = BboxVisualizer()
    diagnostic_vis.draw_alignment_diagnostic(
        img=s.crop_tablet.img.copy(),
        det_rows=s.det_rows.as_lists(),
        aligned_rows=s.aligned_rows.as_lists(),
        det_sign_match_info=s.det_sign_match_info,
        text_sign_match_info=s.text_sign_match_info,
        det_to_text=s.det_to_text,
        line_thickness=2,
        marker_size=5,
    )

    rows_vis = CompositeVisualizer()
    if s.det_row_vis_image is not None:
        rows_vis.compose(
            images=[s.det_row_vis_image, text_row_vis.result],
            layout=(1, 2),
            titles=[
                f"Detection Rows ({len(s.det_row_sequences)} rows)",
                f"Text Mapping ({len(s.text_row_sequences)} rows, "
                f"{len(s.matches)} matched)",
            ],
            figsize=(20, 10),
        )
    if vis.display:
        text_row_vis.display_result(vis_opt="draw")
        diagnostic_vis.display_result(vis_opt="draw")
    if vis.save:
        text_row_vis.save(_out(context, "text_rows_mapped.jpg"))
        if s.det_row_vis_image is not None:
            rows_vis.save(_out(context, "rows_side_by_side.jpg"))
        diagnostic_vis.save(_out(context, "alignment_diagnostic.jpg"))

def vis_offset_analysis(context: CropContext, vis: VisOptions) -> None:
    if not vis.info:
        return
    s = context.state
    det_rows = s.det_rows.as_dict()
    match_pairs = {
        (text_row_idx, text_idx): (s.text_to_det[text_row_idx], det_idx)
        for text_row_idx, matches in s.row_sign_matches.items()
        for text_idx, det_idx in matches
    }
    offsets = {"cx": [], "cy": [], "w": [], "h": []}
    for text_row_idx, row in enumerate(s.aligned_rows.as_lists()):
        for text_col_idx, box in enumerate(row):
            match = match_pairs.get((text_row_idx, text_col_idx))
            if match is None:
                continue
            det_row_idx, det_sign_idx = match
            det_box = det_rows[det_row_idx][det_sign_idx]
            offsets["cx"].append(box.cx - det_box.cx)
            offsets["cy"].append(box.cy - det_box.cy)
            offsets["w"].append(box.width - det_box.width)
            offsets["h"].append(box.height - det_box.height)

    if not offsets["cx"]:
        print("No matched pairs found for offset analysis.")
        return
    print("=== Position Offset Analysis (coarse-aligned vs detection) ===")
    for key, label in [("cx", "cx"), ("cy", "cy"), ("w", "w "), ("h", "h ")]:
        values = np.array(offsets[key])
        print(
            f"  Delta {label}: mean={values.mean():.2f}, "
            f"std={values.std():.2f}, |max|={np.abs(values).max():.2f}"
        )

# =============================================================================
# Part 3 (supplement): PSR optimization
# =============================================================================

def create_psr_optimizer(context: CropContext) -> None:
    s = context.state
    params = context.psr_params or {}
    s.optimizer = PointSetRegistrationOptimizer(
        source_rows=s.aligned_rows.as_lists(),
        target_detections=s.det_boxes,
        sigma=s.detections.avg_width * params.get("sigma_factor", 1.5),
        w_noise=params.get("w_noise", 0.1),
        lambda_data=params.get("lambda_data", 2.0),
        lambda_anchor=params.get("lambda_anchor", 0.01),
        lambda_seq=params.get("lambda_seq", 0.03),
        lambda_height=params.get("lambda_height", 0.01),
        lambda_rows=params.get("lambda_rows", 1.0),
        lambda_boundary=params.get("lambda_boundary", 0.0),
        rows_threshold_ratio_far=params.get("rows_threshold_ratio_far", 1 / 3),
        rows_threshold_ratio_close=params.get("rows_threshold_ratio_close", 2 / 3),
        rows_plateau_far=params.get("rows_plateau_far", 0.5),
        rows_plateau_close=params.get("rows_plateau_close", 1.0),
        contour_mask=s.crop_tablet.mask,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

def vis_psr_optimizer(context: CropContext, vis: VisOptions) -> None:
    optimizer = context.state.optimizer
    if vis.info:
        print("=== PSR Optimizer ===")
        print(
            f"  Device: {optimizer.device}, Source (M): {optimizer.M}, "
            f"Target (N): {optimizer.N}"
        )
        print(f"  Sigma: {optimizer.sigma:.1f}, w_noise: {optimizer.w_noise}")
        print(
            f"  Lambdas: data={optimizer.lambda_data}, "
            f"anchor={optimizer.lambda_anchor}, seq={optimizer.lambda_seq}, "
            f"height={optimizer.lambda_height}, rows={optimizer.lambda_rows}, "
            f"boundary={optimizer.lambda_boundary}"
        )
        print(
            "  Contour mask: "
            f"{'available' if optimizer.contour_mask is not None else 'not set'}"
        )
    if vis.save:
        optimizer.plot_loss_curves(
            save_dir="alignment_loss_functions", show=False)

def _optimization_target(context: CropContext, stop_at_probe: bool) -> int:
    total = int((context.psr_params or {}).get("num_iterations", 200))
    if not stop_at_probe:
        return total
    probe = context.dift.config.affine_probe_iteration
    probe = total if probe is None else max(0, int(probe))
    return min(total, probe)

def _optimize_psr(context: CropContext, stop_at_probe: bool = False) -> None:
    s = context.state
    params = context.psr_params or {}
    target = _optimization_target(context, stop_at_probe)
    iterations = max(0, target - len(s.optimizer.loss_history))
    if iterations <= 0:
        s.final_boxes = s.optimizer.get_optimized_boxes()
        return
    s.final_boxes = s.optimizer.optimize(
        num_iterations=iterations,
        lr=params.get("lr", 1.0),
        sigma_anneal=params.get("sigma_anneal", True),
        sigma_final=None,
        verbose=True,
        log_every=20,
    )

def optimize_psr(context: CropContext) -> None:
    _optimize_psr(context)

def vis_optimization(context: CropContext, vis: VisOptions) -> None:
    if vis.info:
        print(
            f"=== Optimization Complete: "
            f"{len(context.state.final_boxes)} signs ==="
        )

def vis_loss_history(context: CropContext, vis: VisOptions) -> None:
    if vis.display or vis.save:
        context.state.optimizer.plot_loss_history()

def vis_results_comparison(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    image = s.crop_tablet.img
    before_vis = BboxVisualizer(color=(0, 255, 255))
    before_vis.draw_boxes(image.copy(), s.aligned_boxes)
    after_vis = BboxVisualizer(color=(255, 255, 0))
    after_vis.draw_boxes(image.copy(), s.final_boxes)

    det_base = BboxVisualizer(color=(255, 0, 0))
    det_base.draw_boxes(image.copy(), s.det_boxes)
    det_overlay = BboxVisualizer(color=(255, 255, 0))
    det_overlay.draw_boxes(det_base.result, s.final_boxes)

    gt_base = BboxVisualizer(color=(0, 255, 0))
    gt_base.draw_boxes(
        image.copy(),
        gt_boxes_for_visualization(context, s.gt_boxes_crop),
    )
    gt_overlay = BboxVisualizer(color=(255, 255, 0))
    gt_overlay.draw_boxes(gt_base.result, s.final_boxes)

    comparison = CompositeVisualizer()
    comparison.compose(
        images=[
            before_vis.result,
            after_vis.result,
            det_overlay.result,
            gt_overlay.result,
        ],
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
        comparison.display_result(vis_opt="draw")
    if vis.save:
        before_vis.save(_out(context, "coarse_aligned.jpg"))
        after_vis.save(_out(context, "final_optimized.jpg"))
        det_overlay.save(_out(context, "overlay_det_final.jpg"))
        gt_overlay.save(_out(context, "overlay_gt_final.jpg"))
        comparison.save(_out(context, "results_comparison.jpg"))

def vis_parameter_changes(context: CropContext, vis: VisOptions) -> None:
    if not vis.info:
        return
    s = context.state
    changes = s.optimizer.get_param_changes()
    print("=== Parameter Changes (Coarse -> Final) ===")
    for i, label in enumerate(["cx", "cy", "w ", "h "]):
        values = changes[:, i]
        print(
            f"  Delta {label}: mean={values.mean():.2f}, "
            f"std={values.std():.2f}, |max|={np.abs(values).max():.2f}"
        )

    count = min(5, len(s.aligned_boxes))
    print(f"\n=== First {count} Signs: Coarse -> Final ===")
    for i in range(count):
        before = s.aligned_boxes[i]
        after = s.final_boxes[i]
        print(f"  {i + 1}. {before.sign_name}:")
        print(
            f"      Coarse: cx={before.cx:.1f}, cy={before.cy:.1f}, "
            f"w={before.width:.1f}, h={before.height:.1f}"
        )
        print(
            f"      Final:  cx={after.cx:.1f}, cy={after.cy:.1f}, "
            f"w={after.width:.1f}, h={after.height:.1f}"
        )
        print(
            f"      Delta:  cx={after.cx - before.cx:.1f}, "
            f"cy={after.cy - before.cy:.1f}, "
            f"w={after.width - before.width:.1f}, "
            f"h={after.height - before.height:.1f}"
        )

# =============================================================================
# Part 4 (supplement): DIFT and prototype workflows
# =============================================================================

def optimize_psr_until_dift_probe(context: CropContext) -> None:
    _optimize_psr(context, stop_at_probe=True)

def optimize_psr_after_dift_probe(context: CropContext) -> None:
    _optimize_psr(context)

def setup_source_signs(context: CropContext) -> None:
    s = context.state
    s.source_period = s.fragment_data["script"]["period"]
    context.dift.source = context.sign_source or context.canonical_source
    if context.dift.source is None:
        raise ValueError("CropContext requires sign_source")

def vis_source_signs(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    source = context.dift.source
    if source is None:
        raise ValueError("DiftRuntime.source must be set")
    period = _source_period(context)
    if vis.info:
        print("=== Source Signs Setup ===")
        print(f"  API period:   {period!r}")
        print(f"  Source:       {type(source).__name__}")
        print(
            f"  Feature cache: {len(context.dift.feature_cache)} loaded; "
            "missing features are computed on demand"
        )

    rows, missing, total = collect_detected_source_feature_rows(
        s.det_boxes,
        context.dift,
        period,
        max_signs=context.dift.config.feature_viz_max_signs,
    )
    if vis.info and total:
        print(f"  Feature-map viz: detected unique signs={total}, shown={len(rows)}")
        if missing:
            print("  Missing source features: " + ", ".join(missing[:20]))
    if not rows:
        return

    grid = render_source_feature_grid(rows)
    if vis.display:
        _display_bgr(grid, "Source DIFT feature maps", px_per_in=60.0)
    if vis.save:
        cv2.imwrite(
            _out(context, "source_feature_maps.jpg"),
            grid,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

def create_source_sign_overlay(context: CropContext) -> None:
    s = context.state
    image, stats = render_source_sign_overlay(
        image=s.crop_tablet.img,
        boxes=s.optimizer.get_optimized_boxes(),
        runtime=context.dift,
        period=_source_period(context),
        max_boxes=context.dift.config.source_overlay_max_boxes,
        draw_boxes=False,
        draw_labels=False,
    )
    s.source_overlay = SignOverlay(
        iteration=len(s.optimizer.loss_history),
        image=image,
        stats=stats,
    )

def vis_source_sign_overlay(
    context: CropContext,
    vis: VisOptions,
) -> None:
    s = context.state
    result = s.source_overlay
    if result is None:
        return

    if vis.info:
        stats = result.stats
        print(
            f"=== Source Sign Overlay @ iter {result.iteration}: "
            f"{stats.get('pasted', 0)}/{stats.get('total', 0)} pasted ==="
        )
        missing = stats.get("missing_names") or []
        if missing:
            suffix = " ..." if len(missing) > 20 else ""
            print(f"  Missing source images: {', '.join(missing[:20])}{suffix}")

    if vis.save:
        cv2.imwrite(
            _out(
                context,
                f"source_sign_overlay_iter{result.iteration}.jpg",
            ),
            result.image,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    if vis.display:
        _display_bgr(
            result.image,
            f"Source signs pasted at PSR boxes @ iter {result.iteration}",
        )

def run_dift_affine_probe(context: CropContext) -> None:
    s = context.state
    boxes = s.optimizer.get_optimized_boxes()
    s.dift_affine_probe = DiftAffineProbe(
        iteration=len(s.optimizer.loss_history),
        boxes=boxes,
        results=build_dift_affine_probe(
            boxes,
            context.dift,
            _source_period(context),
            context.dift.config,
        ),
    )

def vis_dift_affine_probe(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    probe = s.dift_affine_probe
    if probe is None:
        return

    if vis.info:
        valid = sum(result.affine is not None for result in probe.results)
        print(
            f"=== DIFT Affine Probe @ iter {probe.iteration}: "
            f"{valid}/{len(probe.results)} affine estimates ==="
        )

    overlay, grid = render_dift_affine_probe(
        image=s.crop_tablet.img,
        boxes=probe.boxes,
        results=probe.results,
        iteration=probe.iteration,
        thumb=context.dift.config.affine_probe_thumb,
    )
    if vis.save:
        prefix = f"dift_affine_probe_iter{probe.iteration}"
        cv2.imwrite(_out(context, f"{prefix}_boxes.jpg"), overlay)
        cv2.imwrite(
            _out(context, f"{prefix}_grid.jpg"),
            grid,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
    if vis.display:
        _display_bgr(
            overlay,
            f"DIFT affine probe boxes @ iter {probe.iteration}",
        )
        _display_bgr(
            grid,
            f"DIFT affine probe grid @ iter {probe.iteration}",
        )

FEATURE_COARSE_RESULT_KEY = "feature_coarse"


@dataclass
class FeatureCoarseAlignmentConfig:
    step_px: float = 100.0
    search_margin_px: float = 100.0
    window_width: Optional[float] = None
    window_height: Optional[float] = None
    max_candidates_per_row: Optional[int] = None
    assignment_min_score: float = 0.0
    match: DiftMatchConfig = field(default_factory=DiftMatchConfig)

    def __post_init__(self) -> None:
        if self.step_px <= 0:
            raise ValueError("step_px must be positive")
        if self.search_margin_px < 0:
            raise ValueError("search_margin_px must be non-negative")
        for name, value in (
            ("window_width", self.window_width),
            ("window_height", self.window_height),
            ("max_candidates_per_row", self.max_candidates_per_row),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SlidingWindow:
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    text_start: int
    text_end: int


@dataclass
class FeatureSignAssignment:
    text_idx: int
    sign_name: str
    source: str
    box: Box
    score: float = 0.0
    geometry: float = 0.0
    support: float = 0.0
    inlier_score: float = 0.0
    n_matches: int = 0
    n_inliers: int = 0
    candidate_idx: Optional[int] = None


@dataclass
class FeatureScoreGrid:
    score: np.ndarray
    geometry: np.ndarray
    support: np.ndarray
    inlier_score: np.ndarray
    matches: np.ndarray
    inliers: np.ndarray

    @classmethod
    def empty(cls, rows: int, columns: int) -> "FeatureScoreGrid":
        shape = (rows, columns)
        return cls(
            score=np.zeros(shape, dtype=np.float32),
            geometry=np.zeros(shape, dtype=np.float32),
            support=np.zeros(shape, dtype=np.float32),
            inlier_score=np.zeros(shape, dtype=np.float32),
            matches=np.zeros(shape, dtype=np.int32),
            inliers=np.zeros(shape, dtype=np.int32),
        )


@dataclass
class FeatureRowAlignmentResult:
    text_row_idx: int
    det_row_idx: int
    anchors: dict[int, int]
    unmatched_text_indices: list[int]
    candidates: list[SlidingWindow]
    scores: FeatureScoreGrid
    assignments: list[FeatureSignAssignment]


@dataclass
class FeatureCoarseRun:
    config: FeatureCoarseAlignmentConfig
    rows: dict[int, FeatureRowAlignmentResult]
    window_size: tuple[int, int]

    @property
    def assignments(self) -> list[FeatureSignAssignment]:
        return [
            assignment
            for row in self.rows.values()
            for assignment in row.assignments
        ]


def get_feature_coarse_run(
    context: CropContext,
    *,
    required: bool = True,
) -> Optional[FeatureCoarseRun]:
    run = context.state.extras.get(FEATURE_COARSE_RESULT_KEY)
    if run is None and required:
        raise RuntimeError("feature coarse alignment has not been run")
    return run  # type: ignore[return-value]


def align_text_rows_with_feature_search(
    context: CropContext,
    config: Optional[FeatureCoarseAlignmentConfig] = None,
) -> FeatureCoarseRun:
    """Replace the base coarse-alignment step with DIFT window search."""

    run = _FeatureCoarseAligner(
        context,
        config or FeatureCoarseAlignmentConfig(),
    ).run()
    context.state.extras[FEATURE_COARSE_RESULT_KEY] = run
    return run


class _FeatureCoarseAligner:
    def __init__(
        self,
        context: CropContext,
        config: FeatureCoarseAlignmentConfig,
    ) -> None:
        if context.dift is None:
            raise ValueError("CropContext.dift is required")
        if context.dift.source is None:
            raise ValueError("DiftRuntime.source must be set")

        self.context = context
        self.state = context.state
        self.config = config
        self.runtime = context.dift
        self.source = context.dift.source
        self.period = source_period(context)
        self.window_width = _window_dimension(
            config.window_width,
            self.state.detections.avg_width,
        )
        self.window_height = _window_dimension(
            config.window_height,
            self.state.detections.avg_height,
        )

    def run(self) -> FeatureCoarseRun:
        state = self.state
        det_rows = state.det_rows.as_dict()
        text_rows = state.text_rows.as_dict()
        aligned_boxes = Boxes(tablet=state.crop_tablet)
        aligned_indices = [[] for _ in range(len(state.text_rows))]
        results = {}

        for text_row_idx in sorted(state.row_sign_matches):
            det_row_idx = state.text_to_det.get(text_row_idx)
            if det_row_idx is None:
                continue
            result, row_boxes = self._align_row(
                text_row_idx,
                det_row_idx,
                text_rows[text_row_idx],
                det_rows[det_row_idx],
            )
            results[text_row_idx] = result
            for box in row_boxes:
                aligned_indices[text_row_idx].append(len(aligned_boxes))
                aligned_boxes.append(box)

        state.aligned_boxes = aligned_boxes
        state.aligned_rows = BoxRows(aligned_boxes, aligned_indices)
        return FeatureCoarseRun(
            config=self.config,
            rows=results,
            window_size=(self.window_width, self.window_height),
        )

    def _align_row(
        self,
        text_row_idx: int,
        det_row_idx: int,
        text_boxes: list[Box],
        det_boxes: list[Box],
    ) -> tuple[FeatureRowAlignmentResult, list[Box]]:
        sign_matches = self.state.row_sign_matches[text_row_idx]
        anchors = _exact_anchor_map(text_boxes, det_boxes, sign_matches)
        unmatched = [idx for idx in range(len(text_boxes)) if idx not in anchors]
        expected_centers = _expected_text_centers(
            len(text_boxes),
            anchors,
            det_boxes,
            self.window_width,
        )
        candidates = _build_sliding_windows(
            unmatched_text_indices=unmatched,
            anchors=anchors,
            anchor_boxes={
                text_idx: det_boxes[det_idx]
                for text_idx, det_idx in anchors.items()
            },
            expected_centers=expected_centers,
            baseline=_fit_row_baseline(det_boxes),
            image_shape=self.state.crop_tablet.img.shape[:2],
            window_size=(self.window_width, self.window_height),
            step_px=self.config.step_px,
            margin_px=self.config.search_margin_px,
            max_candidates=self.config.max_candidates_per_row,
        )
        scores = self._score_candidates(text_boxes, unmatched, candidates)
        selected = _ordered_score_assignment(
            unmatched,
            candidates,
            scores.score,
            self.config.assignment_min_score,
        )
        fallback_boxes = align_text_row_to_detection(
            text_boxes=text_boxes,
            det_boxes=det_boxes,
            matches=sign_matches,
            avg_width=self.window_width,
            avg_height=self.window_height,
        )

        matrix_rows = {text_idx: row for row, text_idx in enumerate(unmatched)}
        assignments = []
        for text_idx, text_box in enumerate(text_boxes):
            if text_idx in anchors:
                assignment = _anchor_assignment(
                    text_idx,
                    text_box,
                    det_boxes[anchors[text_idx]],
                )
            elif text_idx in selected:
                candidate_idx = selected[text_idx]
                assignment = _feature_assignment(
                    text_idx,
                    text_box,
                    candidate_idx,
                    candidates[candidate_idx],
                    scores,
                    matrix_rows[text_idx],
                )
            else:
                assignment = FeatureSignAssignment(
                    text_idx=text_idx,
                    sign_name=text_box.sign_name,
                    source="fallback",
                    box=fallback_boxes[text_idx],
                )
            assignments.append(assignment)

        return FeatureRowAlignmentResult(
            text_row_idx=text_row_idx,
            det_row_idx=det_row_idx,
            anchors=anchors,
            unmatched_text_indices=unmatched,
            candidates=candidates,
            scores=scores,
            assignments=assignments,
        ), [assignment.box for assignment in assignments]

    def _score_candidates(
        self,
        text_boxes: list[Box],
        unmatched: list[int],
        candidates: list[SlidingWindow],
    ) -> FeatureScoreGrid:
        scores = FeatureScoreGrid.empty(len(unmatched), len(candidates))
        if not unmatched or not candidates:
            return scores

        source_by_sign = {}
        for text_idx in unmatched:
            sign = text_boxes[text_idx].sign
            if sign.name not in source_by_sign:
                source_by_sign[sign.name] = (
                    self.source.get(sign.name, self.period),
                    self.runtime.get_sign_feature(sign, self.period),
                )

        for candidate_idx, candidate in enumerate(candidates):
            crop = _window_box(
                candidate,
                text_boxes[unmatched[0]],
                self.state.crop_tablet,
            ).crop_image()
            crop_feature = self.runtime.featurize_image(crop)
            matches_by_sign = {}

            for matrix_row, text_idx in enumerate(unmatched):
                sign_name = text_boxes[text_idx].sign_name
                source_img, source_feature = source_by_sign[sign_name]
                if source_img is None or source_feature is None:
                    continue
                if sign_name not in matches_by_sign:
                    matches_by_sign[sign_name] = self.runtime.match(
                        source_feature,
                        crop_feature,
                        source_img.shape[:2],
                        crop.shape[:2],
                        self.config.match,
                        src_foreground_mask=source_foreground_mask(
                            source_img,
                            source_feature.shape[-2:],
                        ),
                    )
                result = matches_by_sign[sign_name]
                scores.score[matrix_row, candidate_idx] = result.coarse_score
                scores.geometry[matrix_row, candidate_idx] = result.geometry_score
                scores.support[matrix_row, candidate_idx] = result.support_score
                scores.inlier_score[matrix_row, candidate_idx] = result.inlier_score
                scores.matches[matrix_row, candidate_idx] = result.n_matches
                scores.inliers[matrix_row, candidate_idx] = result.n_inliers
        return scores


def _window_dimension(configured: Optional[float], fallback: float) -> int:
    return max(1, int(round(fallback if configured is None else configured)))


def _window_box(
    window: SlidingWindow,
    text_box: Box,
    tablet,
) -> Box:
    return Box(
        x1=window.x1,
        y1=window.y1,
        x2=window.x2,
        y2=window.y2,
        candidates=[SignCandidate(sign=text_box.sign, score=1.0)],
        tablet=tablet,
    )


def _anchor_assignment(
    text_idx: int,
    text_box: Box,
    det_box: Box,
) -> FeatureSignAssignment:
    box = Box(
        x1=det_box.x1,
        y1=det_box.y1,
        x2=det_box.x2,
        y2=det_box.y2,
        candidates=[SignCandidate(sign=text_box.sign, score=det_box.score)],
        tablet=text_box.tablet,
    )
    return FeatureSignAssignment(
        text_idx=text_idx,
        sign_name=text_box.sign_name,
        source="anchor",
        box=box,
        score=1.0,
    )


def _feature_assignment(
    text_idx: int,
    text_box: Box,
    candidate_idx: int,
    candidate: SlidingWindow,
    scores: FeatureScoreGrid,
    matrix_row: int,
) -> FeatureSignAssignment:
    score = float(scores.score[matrix_row, candidate_idx])
    return FeatureSignAssignment(
        text_idx=text_idx,
        sign_name=text_box.sign_name,
        source="feature",
        box=Box(
            x1=candidate.x1,
            y1=candidate.y1,
            x2=candidate.x2,
            y2=candidate.y2,
            candidates=[SignCandidate(sign=text_box.sign, score=score)],
            tablet=text_box.tablet,
        ),
        score=score,
        geometry=float(scores.geometry[matrix_row, candidate_idx]),
        support=float(scores.support[matrix_row, candidate_idx]),
        inlier_score=float(scores.inlier_score[matrix_row, candidate_idx]),
        n_matches=int(scores.matches[matrix_row, candidate_idx]),
        n_inliers=int(scores.inliers[matrix_row, candidate_idx]),
        candidate_idx=candidate_idx,
    )


def _exact_anchor_map(
    text_boxes: list[Box],
    det_boxes: list[Box],
    sign_matches: list[tuple[int, int]],
) -> dict[int, int]:
    return {
        text_idx: det_idx
        for text_idx, det_idx in sign_matches
        if text_boxes[text_idx].sign_name == det_boxes[det_idx].sign_name
    }


def _fit_row_baseline(det_boxes: list[Box]) -> tuple[float, float]:
    if len(det_boxes) >= 2:
        xs = np.asarray([box.cx for box in det_boxes], dtype=np.float64)
        ys = np.asarray([box.cy for box in det_boxes], dtype=np.float64)
        if float(np.ptp(xs)) > 1e-6:
            slope, intercept = np.polyfit(xs, ys, 1)
            return float(slope), float(intercept)
    return 0.0, float(np.mean([box.cy for box in det_boxes]))


def _expected_text_centers(
    num_text: int,
    anchors: dict[int, int],
    det_boxes: list[Box],
    avg_width: float,
) -> np.ndarray:
    indices = np.arange(num_text, dtype=np.float64)
    anchor_items = sorted(anchors.items())
    if len(anchor_items) >= 2:
        text_indices = np.asarray([item[0] for item in anchor_items])
        detection_x = np.asarray([det_boxes[item[1]].cx for item in anchor_items])
        slope, intercept = np.polyfit(text_indices, detection_x, 1)
        return slope * indices + intercept

    spacing = _estimated_detection_spacing(det_boxes, avg_width)
    if len(anchor_items) == 1:
        text_idx, det_idx = anchor_items[0]
        return det_boxes[det_idx].cx + (indices - text_idx) * spacing

    detection_center = float(np.mean([box.cx for box in det_boxes]))
    return detection_center + (indices - (num_text - 1) / 2.0) * spacing


def _estimated_detection_spacing(det_boxes: list[Box], fallback: float) -> float:
    centers = np.sort(np.asarray([box.cx for box in det_boxes], dtype=np.float64))
    gaps = np.diff(centers)
    gaps = gaps[gaps > 1e-6]
    return float(np.median(gaps)) if gaps.size else float(fallback)


def _build_sliding_windows(
    unmatched_text_indices: list[int],
    anchors: dict[int, int],
    anchor_boxes: dict[int, Box],
    expected_centers: np.ndarray,
    baseline: tuple[float, float],
    image_shape: tuple[int, int],
    window_size: tuple[int, int],
    step_px: float,
    margin_px: float,
    max_candidates: Optional[int],
) -> list[SlidingWindow]:
    if not unmatched_text_indices:
        return []

    image_h, image_w = image_shape
    window_w, window_h = window_size
    min_cx, max_cx = window_w / 2.0, image_w - window_w / 2.0
    min_cy, max_cy = window_h / 2.0, image_h - window_h / 2.0
    if max_cx < min_cx or max_cy < min_cy:
        return []

    slope, intercept = baseline
    windows = []
    for text_start, text_end in _contiguous_runs(unmatched_text_indices):
        start_x = float(expected_centers[text_start:text_end + 1].min() - margin_px)
        end_x = float(expected_centers[text_start:text_end + 1].max() + margin_px)
        if text_start - 1 in anchors:
            start_x = max(
                start_x,
                anchor_boxes[text_start - 1].cx + step_px / 2.0,
            )
        if text_end + 1 in anchors:
            end_x = min(
                end_x,
                anchor_boxes[text_end + 1].cx - step_px / 2.0,
            )

        for cx in _sample_interval(
            max(min_cx, start_x),
            min(max_cx, end_x),
            step_px,
            origin=float(expected_centers[text_start]),
        ):
            cy = float(np.clip(slope * cx + intercept, min_cy, max_cy))
            x1 = min(max(0, int(round(cx - window_w / 2.0))), image_w - window_w)
            y1 = min(max(0, int(round(cy - window_h / 2.0))), image_h - window_h)
            windows.append(SlidingWindow(
                x1=x1,
                y1=y1,
                x2=x1 + window_w,
                y2=y1 + window_h,
                cx=x1 + window_w / 2.0,
                cy=y1 + window_h / 2.0,
                text_start=text_start,
                text_end=text_end,
            ))

    unique = {
        (w.x1, w.y1, w.x2, w.y2, w.text_start, w.text_end): w
        for w in windows
    }
    windows = sorted(unique.values(), key=lambda window: (window.cx, window.cy))
    if max_candidates is not None and len(windows) > max_candidates:
        keep = np.linspace(0, len(windows) - 1, max_candidates, dtype=np.int64)
        windows = [windows[idx] for idx in keep]
    return windows


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    return runs + [(start, previous)]


def _sample_interval(
    start: float,
    end: float,
    step: float,
    origin: float,
) -> list[float]:
    if end < start:
        return []
    first = int(np.ceil((start - origin) / step))
    last = int(np.floor((end - origin) / step))
    if first > last:
        return [(start + end) / 2.0]
    return [float(origin + idx * step) for idx in range(first, last + 1)]


def _ordered_score_assignment(
    text_indices: list[int],
    candidates: list[SlidingWindow],
    score_matrix: np.ndarray,
    min_score: float,
) -> dict[int, int]:
    """Maximize assignment count, then score, from left to right."""

    text_count, candidate_count = len(text_indices), len(candidates)
    if not text_count or not candidate_count:
        return {}

    scores = np.full((text_count + 1, candidate_count + 1), -np.inf)
    counts = np.full((text_count + 1, candidate_count + 1), -1, dtype=np.int32)
    back = {}
    scores[0, 0], counts[0, 0] = 0.0, 0

    def update(source, target, score_delta, count_delta, action):
        source_i, source_j = source
        target_i, target_j = target
        score = scores[source_i, source_j] + score_delta
        count = counts[source_i, source_j] + count_delta
        if (
            count > counts[target_i, target_j]
            or (
                count == counts[target_i, target_j]
                and score > scores[target_i, target_j] + 1e-12
            )
        ):
            scores[target_i, target_j] = score
            counts[target_i, target_j] = count
            back[target] = (*source, action)

    for i in range(text_count + 1):
        for j in range(candidate_count + 1):
            if not np.isfinite(scores[i, j]):
                continue
            if i < text_count:
                update((i, j), (i + 1, j), 0.0, 0, "skip_text")
            if j < candidate_count:
                update((i, j), (i, j + 1), 0.0, 0, "skip_candidate")
            if i < text_count and j < candidate_count:
                text_idx = text_indices[i]
                candidate = candidates[j]
                score = float(score_matrix[i, j])
                if (
                    candidate.text_start <= text_idx <= candidate.text_end
                    and score > min_score
                ):
                    update((i, j), (i + 1, j + 1), score, 1, "assign")

    selected = {}
    position = (text_count, candidate_count)
    while position != (0, 0) and position in back:
        previous_i, previous_j, action = back[position]
        if action == "assign":
            selected[text_indices[previous_i]] = previous_j
        position = previous_i, previous_j
    return selected


def vis_feature_coarse_alignment(
    context: CropContext,
    vis: VisOptions,
) -> None:
    run = get_feature_coarse_run(context, required=False)
    if run is None or not run.rows:
        if vis.info:
            print("=== DIFT coarse alignment: no matched rows ===")
        return

    overlay = _render_overlay(context, run)
    if vis.info:
        counts = {source: 0 for source in ("anchor", "feature", "fallback")}
        for assignment in run.assignments:
            counts[assignment.source] += 1
        print("=== DIFT Sliding-Window Coarse Alignment ===")
        print(
            f"  window={run.window_size[0]}x{run.window_size[1]}, "
            f"step={run.config.step_px:g}px, rows={len(run.rows)}"
        )
        print(
            f"  anchors={counts['anchor']}, feature={counts['feature']}, "
            f"fallback={counts['fallback']}"
        )
    if vis.save:
        cv2.imwrite(
            output_path(context, "feature_coarse_alignment.jpg"),
            overlay,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    if vis.display:
        _display_bgr(overlay, "DIFT sliding-window coarse alignment")


def _render_overlay(
    context: CropContext,
    run: FeatureCoarseRun,
) -> np.ndarray:
    image = _to_bgr(context.state.crop_tablet.img).copy()
    for row_result in run.rows.values():
        for candidate in row_result.candidates:
            cv2.rectangle(
                image,
                (candidate.x1, candidate.y1),
                (candidate.x2, candidate.y2),
                (100, 100, 100),
                1,
                cv2.LINE_AA,
            )
        for assignment in row_result.assignments:
            color = {
                "anchor": (80, 220, 80),
                "feature": (0, 165, 255),
                "fallback": (255, 220, 0),
            }[assignment.source]
            box = assignment.box
            p1 = int(round(box.x1)), int(round(box.y1))
            p2 = int(round(box.x2)), int(round(box.y2))
            cv2.rectangle(image, p1, p2, color, 2, cv2.LINE_AA)
            label = f"{assignment.sign_name}:{assignment.source[0].upper()}"
            if assignment.source == "feature":
                label += f" {assignment.score:.2f}"
            _draw_label(image, label, p1, color)
    return image


def _draw_label(image, label, origin, color) -> None:
    x, y = max(0, origin[0]), max(12, origin[1] - 4)
    for thickness, text_color in ((2, (0, 0, 0)), (1, color)):
        cv2.putText(
            image,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    return image.astype(np.uint8)

# =============================================================================
# Shared runner
# =============================================================================

class Runner:
    def __init__(
        self,
        context: CropContext,
        vis: Optional[VisOptions] = None,
    ):
        self.context = context
        self.vis = vis or VisOptions()
        self._fragments = context.local_source.get_available_fragments()
        context.state.fragments = self._fragments
        print(f"Found {len(self._fragments)} fragments with both image and annotation")

        if self.vis.save:
            os.makedirs(context.output_dir, exist_ok=True)

    def run(self, steps: list[Step]) -> None:
        for step in steps:
            step.run(self.context)
            if step.visualize:
                step.visualize(self.context, self.vis)

    def choose_sample(self, idx: int = 0, name: str = "") -> None:
        if name:
            idx = self._fragments.index(name)
        fragment_id = self._fragments[idx]
        print(f"Processing sample: {fragment_id}")
        self.context.state = SampleState(
            fragments=self._fragments,
            fragment_id=fragment_id,
        )

    def choose_crop(self, crop_idx: int) -> None:
        self.context.img_idx = crop_idx
        if self.context.state.detections is None:
            return
        _select_crop(self.context, crop_idx)
