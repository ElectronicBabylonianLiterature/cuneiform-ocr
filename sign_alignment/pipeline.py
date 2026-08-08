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
    DiftRuntime,
    SignOverlay,
    build_dift_affine_probe,
    collect_detected_source_feature_rows,
    render_dift_affine_probe,
    render_source_feature_grid,
    render_source_sign_overlay,
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


def unload_detector(context: CropContext) -> None:
    context.tablet_detector.unload_model()


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


def optimize_psr_until_dift_probe(context: CropContext) -> None:
    _optimize_psr(context, stop_at_probe=True)


def optimize_psr_after_dift_probe(context: CropContext) -> None:
    _optimize_psr(context)


def vis_optimization(context: CropContext, vis: VisOptions) -> None:
    if vis.info:
        print(
            f"=== Optimization Complete: "
            f"{len(context.state.final_boxes)} signs ==="
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
