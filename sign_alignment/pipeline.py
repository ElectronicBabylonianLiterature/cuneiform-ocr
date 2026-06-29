from dataclasses import dataclass, field
import os
from typing import TYPE_CHECKING, Callable, Optional

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
from sign_alignment.box import Box, Boxes, boxes_in_crop
from sign_alignment.sign import SignResolver
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
    detect_rows_dbscan,
    match_rows_dp,
    match_signs_in_row_dp,
)
from sign_alignment.dift_align import (
    CanonicalFeatureSet,
    CanonicalOverlay,
    DiftAffineProbe,
    DiftRuntime,
    build_dift_affine_probe,
    collect_detected_canonical_feature_rows,
    render_canonical_feature_grid,
    render_canonical_sign_overlay,
    render_dift_crop_warp,
    render_dift_affine_probe,
    render_dift_feature_matches,
)

if TYPE_CHECKING:
    from sign_alignment.pipeline_2 import FeatureCoarseRun


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

    canonical: Optional[CanonicalFeatureSet] = None
    dift_affine_probe: Optional[DiftAffineProbe] = None
    canonical_overlay: Optional[CanonicalOverlay] = None
    feature_coarse: Optional["FeatureCoarseRun"] = None
    dift_sampling_scores: Optional[list] = None
    dift_sampling_score_image: Optional[np.ndarray] = None
    dift_sampling_semantic_image: Optional[np.ndarray] = None
    dift_sampling_global_similarity_image: Optional[np.ndarray] = None


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
    canonical_source: Optional[DataSource] = None
    state: SampleState = field(default_factory=SampleState)
    task_type: str = "debug"


RunFn = Callable[[CropContext], None]
VisFn = Callable[[CropContext, VisOptions], None]


@dataclass
class Step:
    name: str
    run: RunFn
    visualize: Optional[VisFn] = None


def _out(context: CropContext, suffix: str) -> str:
    return os.path.join(
        context.output_dir,
        f"{context.task_type}_{context.state.fragment_id}_{suffix}",
    )


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
    gt_vis = BboxVisualizer(context.color_config.GT_COLOR.value)
    gt_vis.draw_boxes(s.tablet.img.copy(), s.gt_boxes)

    if vis.info:
        total_text = sum(map(len, s.text_lines))
        total_unfiltered = sum(map(len, s.text_lines_unfiltered))
        print(f"Ground truth boxes: {len(s.gt_boxes)}")
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
    crop_tablets = context.tablet_detector.get_crop_tablets()
    img_idx = context.img_idx
    if not crop_tablets:
        raise RuntimeError("detector produced no cropped images")
    if not 0 <= img_idx < len(crop_tablets):
        raise IndexError(
            f"crop index {img_idx} is out of range after detection; "
            f"available crop indices are 0..{len(crop_tablets) - 1}"
        )
    s.crop_tablet = crop_tablets[img_idx]
    s.det_boxes = context.tablet_detector.get_crop_boxes()[img_idx]


def vis_detections(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    color = context.color_config.DET_COLOR.value
    full_vis = BboxVisualizer(color=color)
    full_vis.draw_boxes(s.tablet.img.copy(), s.detections)
    crop_vis = BboxVisualizer(color=color)
    crop_vis.draw_boxes(s.crop_tablet.img.copy(), s.det_boxes)

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
    if vis.info:
        print(f"GT boxes (full image): {len(s.gt_boxes)}")
        print(f"GT boxes (sub-image):  {len(s.gt_boxes_crop)}")
    if not s.gt_boxes_crop:
        return

    gt_vis = BboxVisualizer(color=context.color_config.GT_COLOR.value)
    gt_vis.draw_boxes(s.crop_tablet.img.copy(), s.gt_boxes_crop)
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
    s = context.state
    s.det_rows = BoxRows.detect(
        s.det_boxes,
        eps=0.4,
        min_samples=1,
        lambda_weight=0.007,
        avg_width=s.detections.avg_width,
        avg_height=s.detections.avg_height,
    )


def vis_detected_rows_info(context: CropContext, vis: VisOptions) -> None:
    if not vis.info:
        return
    s = context.state
    print("=== Row Detection Results ===")
    print(
        f"Average sign size: {s.detections.avg_size:.2f} px, "
        f"detected {len(s.det_rows)} rows, {len(s.det_boxes)} signs"
    )
    for row_idx, count in enumerate(s.det_rows.counts()):
        print(f"  Row {row_idx}: {count} boxes")
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
    if vis.info:
        print("Detection rows: D# on left margin, matched rows show D#->R#")
    if vis.display:
        row_vis.display_result(vis_opt="draw")
    if vis.save:
        row_vis.save(_out(context, "detection_rows.jpg"))


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
    print("=== Within-Row Sign Matching ===")
    for text_row_idx, det_row_idx in s.matches:
        text = s.text_row_sequences[text_row_idx]
        detected = s.det_row_sequences[det_row_idx]
        matches = s.row_sign_matches[text_row_idx]
        print(
            f"Text row {text_row_idx} -> Det row {det_row_idx}: "
            f"{len(text)} text, {len(detected)} det, {len(matches)} matched"
        )
        for i, (text_idx, det_idx) in enumerate(matches[:5]):
            print(
                f"  {i + 1}. Text[{text_idx}]={text[text_idx]} "
                f"<-> Det[{det_idx}]={detected[det_idx]}"
            )
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more")
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
    total = int((context.psr_params or {}).get("num_iterations", 80))
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


def create_canonical_sign_overlay(context: CropContext) -> None:
    s = context.state
    image, stats = render_canonical_sign_overlay(
        image=s.crop_tablet.img,
        boxes=s.optimizer.get_optimized_boxes(),
        cache=s.canonical.cache,
        max_boxes=context.dift.config.canonical_overlay_max_boxes,
        draw_boxes=False,
        draw_labels=False,
    )
    s.canonical_overlay = CanonicalOverlay(
        iteration=len(s.optimizer.loss_history),
        image=image,
        stats=stats,
    )


def vis_canonical_sign_overlay(
    context: CropContext,
    vis: VisOptions,
) -> None:
    s = context.state
    result = s.canonical_overlay
    if result is None:
        return

    if vis.info:
        stats = result.stats
        print(
            f"=== Canonical Sign Overlay @ iter {result.iteration}: "
            f"{stats.get('pasted', 0)}/{stats.get('total', 0)} pasted ==="
        )
        missing = stats.get("missing_names") or []
        if missing:
            suffix = " ..." if len(missing) > 20 else ""
            print(f"  Missing canonical images: {', '.join(missing[:20])}{suffix}")

    if vis.save:
        cv2.imwrite(
            _out(
                context,
                f"canonical_sign_overlay_iter{result.iteration}.jpg",
            ),
            result.image,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    if vis.display:
        _display_bgr(
            result.image,
            f"Canonical signs pasted at PSR boxes @ iter {result.iteration}",
        )


def run_dift_affine_probe(context: CropContext) -> None:
    s = context.state
    boxes = s.optimizer.get_optimized_boxes()
    s.dift_affine_probe = DiftAffineProbe(
        iteration=len(s.optimizer.loss_history),
        boxes=boxes,
        results=build_dift_affine_probe(
            boxes,
            s.canonical.cache,
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
    gt_base.draw_boxes(image.copy(), s.gt_boxes_crop or [])
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


def setup_canonical_signs(context: CropContext) -> None:
    s = context.state
    period = s.fragment_data["script"]["period"]
    if context.canonical_source is None:
        raise ValueError("canonical_source is required for DIFT setup")
    s.canonical = context.dift.setup(context.canonical_source, period)


def vis_canonical_signs(context: CropContext, vis: VisOptions) -> None:
    s = context.state
    canonical = s.canonical
    if vis.info:
        print("=== Canonical Signs Setup ===")
        print(f"  API period:   {canonical.period!r}")
        print(f"  Source:       {type(canonical.source).__name__}")
        print(
            f"  Feature cache: {len(canonical.cache)} loaded; "
            "missing features are computed on demand"
        )

    rows, missing, total = collect_detected_canonical_feature_rows(
        s.det_boxes,
        canonical.cache,
        max_signs=context.dift.config.feature_viz_max_signs,
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
        cv2.imwrite(
            _out(context, "canonical_feature_maps.jpg"),
            grid,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )


def vis_dift_score_on_whole_tablet(context: CropContext, vis: VisOptions) -> None:
    """Score DIFT matches on a simple crop-tablet sampling grid."""
    s = context.state
    first_cx, first_cy = 200, 200
    step_x, step_y = 100, 100
    last_cx_dist = 200
    last_cy_dist = 200
    height, width = s.crop_tablet.img.shape[:2]
    x_coords = range(first_cx, width - last_cx_dist, step_x)
    y_coords = range(first_cy, height - last_cy_dist, step_y)

    box_width = s.detections.avg_width
    box_height = s.detections.avg_height
    chosen_draw_box_ix = 0
    chosen_draw_box_iy = 0
    sign_name = "DUB"
    sign = SignResolver.from_name(sign_name)

    point_vis = s.crop_tablet.img.copy()
    for iy, cy in enumerate(y_coords):
        for ix, cx in enumerate(x_coords):
            if ix == chosen_draw_box_ix and iy == chosen_draw_box_iy:
                box = Box.from_center(cx=cx, cy=cy, width=box_width, height=box_height, sign=sign, tablet=s.crop_tablet)
                x1, y1, x2, y2 = box.crop_bounds()
                cv2.rectangle(point_vis, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(point_vis, (cx, cy), 15, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(point_vis, (cx, cy), 9, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(point_vis, f"{iy},{ix}", (cx + 16, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if vis.display:
        _display_bgr(point_vis, "DIFT sampling points and boxes on crop tablet")

    score_grid = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)
    semantic_grid = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)
    global_similarity_grid = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)
    score_rows = []
    from tqdm.auto import tqdm
    points = [(iy, ix, cy, cx) for iy, cy in enumerate(y_coords) for ix, cx in enumerate(x_coords)]
    for iy, ix, cy, cx in tqdm(points, desc="DIFT scores", disable=not vis.info):
        box = Box.from_center(cx=cx, cy=cy, width=box_width, height=box_height, sign=sign, tablet=s.crop_tablet)
        x1, y1, x2, y2 = box.crop_bounds()
        _, _, result = _compute_manual_dift_crop_match(context, bounds=(x1, x2, y1, y2), sign_name=sign_name)
        score_grid[iy, ix] = result.coarse_score
        semantic_grid[iy, ix] = result.semantic_score
        global_similarity_grid[iy, ix] = result.global_similarity_score
        score_rows.append({
            "ix": ix,
            "iy": iy,
            "center": (cx, cy),
            "bounds": (x1, x2, y1, y2),
            "score": result.score,
            "semantic": result.semantic_score,
            "global_similarity": result.global_similarity_score,
            "geometry": result.geometry_score,
            "support": result.support_score,
            "coarse": result.coarse_score,
            "n_matches": result.n_matches,
            "n_inliers": result.n_inliers,
            "message": result.message,
        })

    cell_size = 80
    axis_top, axis_left = 50, 60

    def make_heatmap(grid):
        heat = cv2.applyColorMap((np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (len(x_coords) * cell_size, len(y_coords) * cell_size), interpolation=cv2.INTER_NEAREST)
        heat_axes = np.full((heat.shape[0] + axis_top, heat.shape[1] + axis_left, 3), 255, dtype=np.uint8)
        heat_axes[axis_top:, axis_left:] = heat
        for ix in range(len(x_coords)):
            cv2.putText(heat_axes, str(ix), (axis_left + ix * cell_size + 28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        for iy in range(len(y_coords)):
            cv2.putText(heat_axes, str(iy), (15, axis_top + iy * cell_size + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        legend = np.full((heat_axes.shape[0], 120, 3), 255, dtype=np.uint8)
        bar_h = max(1, heat_axes.shape[0] - 50)
        bar = cv2.applyColorMap(np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None], cv2.COLORMAP_JET)
        legend[25:25 + bar_h, 20:50] = np.repeat(bar, 30, axis=1)
        cv2.putText(legend, "1.0", (60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
        cv2.putText(legend, "0.0", (60, 25 + bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
        return np.hstack([heat_axes, legend])

    score_vis = make_heatmap(score_grid)
    semantic_vis = make_heatmap(semantic_grid)
    global_similarity_vis = make_heatmap(global_similarity_grid)

    s.dift_sampling_scores = score_rows
    s.dift_sampling_score_image = score_vis
    s.dift_sampling_semantic_image = semantic_vis
    s.dift_sampling_global_similarity_image = global_similarity_vis

    if vis.display:
        _display_bgr(score_vis, "DIFT coarse scores (geometry * support)")
        _display_bgr(semantic_vis, "DIFT semantic scores")
        _display_bgr(global_similarity_vis, "DIFT global similarity")
    if vis.save:
        cv2.imwrite(_out(context, "dift_sampling_coarse_scores.jpg"), score_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(_out(context, "dift_sampling_semantic_scores.jpg"), semantic_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(_out(context, "dift_sampling_global_similarity.jpg"), global_similarity_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])


def vis_manual_dift_crop_match(
    context: CropContext,
    vis: VisOptions,
) -> None:
    """Create notebook controls for manual crop-to-canonical DIFT matching."""
    s = context.state
    if not vis.display:
        if vis.info:
            print("=== Manual DIFT match requires VisOptions(display=True) ===")
        return

    import ipywidgets as widgets
    from IPython.display import clear_output, display

    sign_names = sorted({
        box.sign_name
        for boxes in (s.det_boxes, s.text_boxes)
        if boxes
        for box in boxes
    })
    if not sign_names:
        print("=== Manual DIFT match skipped: no canonical signs are available ===")
        return

    default_bounds, default_sign = _manual_dift_match_defaults(
        s, sign_names
    )
    x1_default, x2_default, y1_default, y2_default = default_bounds
    width_default = x2_default - x1_default
    height_default = y2_default - y1_default
    img_h, img_w = s.crop_tablet.img.shape[:2]
    field_layout = widgets.Layout(width="155px")
    field_style = {"description_width": "48px"}
    x1_input = widgets.BoundedIntText(
        value=x1_default,
        min=0,
        max=max(0, img_w - 1),
        description="x1",
        layout=field_layout,
        style=field_style,
    )
    y1_input = widgets.BoundedIntText(
        value=y1_default,
        min=0,
        max=max(0, img_h - 1),
        description="y1",
        layout=field_layout,
        style=field_style,
    )
    width_input = widgets.BoundedIntText(
        value=width_default,
        min=1,
        max=max(1, img_w),
        description="width",
        layout=field_layout,
        style=field_style,
    )
    height_input = widgets.BoundedIntText(
        value=height_default,
        min=1,
        max=max(1, img_h),
        description="height",
        layout=field_layout,
        style=field_style,
    )
    sign_input = widgets.Combobox(
        options=sign_names,
        value=default_sign,
        description="sign",
        placeholder="canonical sign name",
        layout=widgets.Layout(width="420px"),
        style={"description_width": "42px"},
    )
    compare_button = widgets.Button(
        description="Compare crop",
        button_style="primary",
        tooltip="Compute the crop feature and compare it with the cached canonical feature",
    )
    output = widgets.Output()

    if vis.info:
        print("=== Manual DIFT Crop Match ===")
        print(
            "score = semantic * sqrt(geometry * support); "
            "this is an uncalibrated diagnostic score, not a probability."
        )
        print(
            f"Subtablet coordinate range: x=[0, {img_w}], y=[0, {img_h}]"
        )

    def compare_crop(_button) -> None:
        compare_button.disabled = True
        try:
            with output:
                clear_output(wait=True)
                x1 = x1_input.value
                y1 = y1_input.value
                x2 = x1 + width_input.value
                y2 = y1 + height_input.value
                bounds = (
                    x1,
                    x2,
                    y1,
                    y2,
                )
                record, crop, result = _compute_manual_dift_crop_match(
                    context,
                    bounds=bounds,
                    sign_name=sign_input.value,
                )
                print(
                    f"score={result.score:.4f}  "
                    f"semantic={result.semantic_score:.4f}  "
                    f"geometry={result.geometry_score:.4f}  "
                    f"support={result.support_score:.4f}"
                )
                print(
                    f"mutual matches={result.n_matches}, "
                    f"RANSAC inliers={result.n_inliers}, "
                    f"mean inlier cosine={result.mean_inlier_similarity:.4f}"
                )
                if result.message:
                    print(f"status: {result.message}")
                figure = _manual_dift_match_figure(
                    context,
                    bounds,
                    record.img,
                    crop,
                    result,
                )
                display(figure)
                import matplotlib.pyplot as plt
                plt.close(figure)
        finally:
            compare_button.disabled = False

    compare_button.on_click(compare_crop)
    controls = widgets.VBox([
        widgets.HBox([x1_input, y1_input, width_input, height_input]),
        widgets.HBox([sign_input, compare_button]),
        output,
    ])
    display(controls)
    compare_crop(None)


def _manual_dift_match_defaults(
    state: SampleState,
    sign_names: list[str],
) -> tuple[tuple[int, int, int, int], str]:
    img_h, img_w = state.crop_tablet.img.shape[:2]
    available = set(sign_names)
    if state.det_boxes:
        candidates = [
            box for box in state.det_boxes if box.sign_name in available
        ]
        if candidates:
            box = candidates[0]
            x1 = max(0, min(img_w - 1, int(np.floor(box.x1))))
            x2 = max(x1 + 1, min(img_w, int(np.ceil(box.x2))))
            y1 = max(0, min(img_h - 1, int(np.floor(box.y1))))
            y2 = max(y1 + 1, min(img_h, int(np.ceil(box.y2))))
            return (x1, x2, y1, y2), box.sign_name

    x1 = img_w // 4
    x2 = max(x1 + 1, img_w * 3 // 4)
    y1 = img_h // 4
    y2 = max(y1 + 1, img_h * 3 // 4)
    return (x1, min(x2, img_w), y1, min(y2, img_h)), sign_names[0]


def _compute_manual_dift_crop_match(
    context: CropContext,
    bounds: tuple[int, int, int, int],
    sign_name: str,
):
    s = context.state
    x1, x2, y1, y2 = map(int, bounds)
    img_h, img_w = s.crop_tablet.img.shape[:2]
    if not (0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h):
        raise ValueError(
            f"invalid bounds {(x1, x2, y1, y2)} for image "
            f"width={img_w}, height={img_h}"
        )

    sign_name = sign_name.strip()
    if not sign_name:
        raise ValueError("canonical sign name is required")
    record = s.canonical.cache.get(SignResolver.from_name(sign_name))
    if record is None:
        raise KeyError(f"no canonical image/feature for sign {sign_name!r}")

    crop = s.crop_tablet.img[y1:y2, x1:x2].copy()
    result = s.canonical.cache.match(record, crop, context.dift.config.match)
    return record, crop, result


def _manual_dift_match_figure(
    context: CropContext,
    bounds: tuple[int, int, int, int],
    canonical_img: np.ndarray,
    crop_img: np.ndarray,
    result,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    config = context.dift.config
    matches = render_dift_feature_matches(
        canonical_img,
        crop_img,
        result,
        thumb=config.manual_match_thumb,
        max_lines=config.manual_match_max_lines,
    )
    warp = render_dift_crop_warp(canonical_img, crop_img, result)

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.15))
    overview_ax = fig.add_subplot(grid[0, 0])
    canonical_ax = fig.add_subplot(grid[0, 1])
    crop_ax = fig.add_subplot(grid[0, 2])
    matches_ax = fig.add_subplot(grid[1, :2])
    warp_ax = fig.add_subplot(grid[1, 2])

    _imshow_notebook(overview_ax, context.state.crop_tablet.img)
    x1, x2, y1, y2 = bounds
    overview_ax.add_patch(Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor="yellow",
        linewidth=2,
    ))
    overview_ax.set_title(
        f"Subtablet crop: x1={x1}, y1={y1}, "
        f"w={x2 - x1}, h={y2 - y1} (x2={x2}, y2={y2})"
    )
    _imshow_notebook(canonical_ax, canonical_img)
    canonical_ax.set_title("Canonical sign")
    _imshow_notebook(crop_ax, crop_img)
    crop_ax.set_title("Selected crop")
    _imshow_notebook(matches_ax, matches)
    matches_ax.set_title("DIFT mutual matches and affine-RANSAC inliers")
    _imshow_notebook(warp_ax, warp)
    warp_ax.set_title("Warped canonical overlay")

    for axis in (
        overview_ax,
        canonical_ax,
        crop_ax,
        matches_ax,
        warp_ax,
    ):
        axis.axis("off")
    fig.suptitle(
        f"Manual DIFT match score: {result.score:.3f}",
        fontsize=15,
    )
    return fig


def _imshow_notebook(axis, image: np.ndarray) -> None:
    if image.ndim == 2:
        axis.imshow(image, cmap="gray")
    else:
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class Runner:
    def __init__(
        self,
        context: CropContext,
        vis: VisOptions = None,
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

    def choose_sample(self, idx = 0, name=""):
        if name:
            idx = self._fragments.index(name)
        fragment_id = self._fragments[idx]
        print(f"Processing sample: {fragment_id}")
        self.context.state = SampleState(fragments=self._fragments, fragment_id=fragment_id)

    def choose_crop(self, crop_idx: int):
        self.context.img_idx = crop_idx
        crop_tablets = self.context.tablet_detector.get_crop_tablets()
        if not crop_tablets or self.context.state.detections is None:
            return
        if crop_idx < 0 or crop_idx >= len(crop_tablets):
            raise IndexError(
                f"crop index {crop_idx} is out of range; "
                f"available crop indices are 0..{len(crop_tablets) - 1}"
            )
        self.context.state.crop_tablet = crop_tablets[crop_idx]
        self.context.state.det_boxes = self.context.tablet_detector.get_crop_boxes()[crop_idx]
