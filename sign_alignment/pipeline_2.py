"""DIFT sliding-window extension for the base sign-alignment pipeline.

The extension replaces only ``pipeline.align_text_rows``.  It consumes the
base row/sign matches, keeps exact same-label matches as anchors, and searches
fixed windows for the remaining text signs.  All other steps continue to come
from :mod:`sign_alignment.pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from data_processing.line_process import align_text_row_to_detection

from . import pipeline as base
from .box import Box, Boxes, SignCandidate
from .dift_align import DiftMatchConfig, source_foreground_mask


RESULT_KEY = "feature_coarse"


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
    context: base.CropContext,
    *,
    required: bool = True,
) -> Optional[FeatureCoarseRun]:
    run = context.state.extras.get(RESULT_KEY)
    if run is None and required:
        raise RuntimeError("feature coarse alignment has not been run")
    return run  # type: ignore[return-value]


def align_text_rows_with_feature_search(
    context: base.CropContext,
    config: Optional[FeatureCoarseAlignmentConfig] = None,
) -> FeatureCoarseRun:
    """Replace the base coarse-alignment step with DIFT window search."""

    run = _FeatureCoarseAligner(
        context,
        config or FeatureCoarseAlignmentConfig(),
    ).run()
    context.state.extras[RESULT_KEY] = run
    return run


class _FeatureCoarseAligner:
    def __init__(
        self,
        context: base.CropContext,
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
        self.period = base.source_period(context)
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
        state.aligned_rows = base.BoxRows(aligned_boxes, aligned_indices)
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
    context: base.CropContext,
    vis: base.VisOptions,
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
            base.output_path(context, "feature_coarse_alignment.jpg"),
            overlay,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    if vis.display:
        base._display_bgr(overlay, "DIFT sliding-window coarse alignment")


def _render_overlay(
    context: base.CropContext,
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
