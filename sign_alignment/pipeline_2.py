"""Experimental DIFT sliding-window coarse alignment.

This module leaves ``pipeline.py`` unchanged and only replaces its coarse
text-row alignment step. Exact same-label detection matches remain anchors.
Unmatched text signs are searched in fixed-size windows along the matched
detection-row baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from . import pipeline as base
from .box import Box, Boxes, SignCandidate
from .dift_align import DiftMatchConfig, source_foreground_mask


Runner = base.Runner
Step = base.Step
VisOptions = base.VisOptions


@dataclass
class FeatureCoarseAlignmentConfig:
    step_px: float = 100.0
    search_margin_px: float = 100.0
    window_width: Optional[float] = None
    window_height: Optional[float] = None
    max_candidates_per_row: Optional[int] = None
    assignment_min_score: float = 0.0
    progress_every: int = 10
    print_match_timings: bool = True
    match: DiftMatchConfig = field(default_factory=DiftMatchConfig)

    def __post_init__(self) -> None:
        if self.step_px <= 0:
            raise ValueError("step_px must be positive")
        if self.window_width is not None and self.window_width <= 0:
            raise ValueError("window_width must be positive")
        if self.window_height is not None and self.window_height <= 0:
            raise ValueError("window_height must be positive")
        if (
            self.max_candidates_per_row is not None
            and self.max_candidates_per_row <= 0
        ):
            raise ValueError("max_candidates_per_row must be positive")


@dataclass
class CropContext(base.CropContext):
    feature_coarse_alignment: FeatureCoarseAlignmentConfig = field(
        default_factory=FeatureCoarseAlignmentConfig
    )


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


@dataclass(frozen=True)
class FeatureMatchTiming:
    candidate_idx: int
    text_idx: int
    sign_name: str
    elapsed_seconds: float
    score: float
    n_matches: int
    n_inliers: int
    message: str = ""


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
class FeatureRowTiming:
    matches: List[FeatureMatchTiming] = field(default_factory=list)
    logical_pairs: int = 0
    crop_features: int = 0
    crop_feature_seconds: float = 0.0


@dataclass
class FeatureRowAlignmentResult:
    text_row_idx: int
    det_row_idx: int
    baseline: Tuple[float, float]
    anchors: Dict[int, int]
    unmatched_text_indices: List[int]
    candidates: List[SlidingWindow]
    scores: FeatureScoreGrid
    timing: FeatureRowTiming
    assignments: List[FeatureSignAssignment]

    @property
    def score_matrix(self) -> np.ndarray:
        return self.scores.score


@dataclass
class FeatureCoarseRun:
    rows: Dict[int, FeatureRowAlignmentResult]
    window_size: Tuple[int, int]
    total_seconds: float

    @property
    def timing(self) -> Dict[str, object]:
        match_seconds = [
            item.elapsed_seconds
            for row in self.rows.values()
            for item in row.timing.matches
        ]
        computed_pairs = len(match_seconds)
        logical_pairs = sum(row.timing.logical_pairs for row in self.rows.values())
        total_match_seconds = float(sum(match_seconds))
        return {
            "logical_score_pair_count": logical_pairs,
            "computed_pair_count": computed_pairs,
            "cached_pair_count": logical_pairs - computed_pairs,
            "match_total_seconds": total_match_seconds,
            "match_mean_seconds": (
                total_match_seconds / computed_pairs if computed_pairs else 0.0
            ),
            "match_min_seconds": min(match_seconds, default=0.0),
            "match_max_seconds": max(match_seconds, default=0.0),
            "crop_feature_count": sum(
                row.timing.crop_features for row in self.rows.values()
            ),
            "crop_feature_total_seconds": float(sum(
                row.timing.crop_feature_seconds for row in self.rows.values()
            )),
            "coarse_alignment_total_seconds": self.total_seconds,
        }


def align_text_rows_with_feature_search(context: CropContext) -> None:
    """Coarsely align unmatched text signs using DIFT sliding-window scores."""
    _FeatureCoarseAligner(context).run()


class _FeatureCoarseAligner:
    def __init__(self, context: CropContext):
        self.context = context
        self.state = context.state
        self.config = context.feature_coarse_alignment
        self.runtime = context.dift
        self.source = self.runtime.source
        if self.source is None:
            raise ValueError("DiftRuntime.source must be set")
        self.period = base._source_period(context)
        self.window_width = int(round(
            self.config.window_width
            if self.config.window_width is not None
            else self.state.detections.avg_width
        ))
        self.window_height = int(round(
            self.config.window_height
            if self.config.window_height is not None
            else self.state.detections.avg_height
        ))

    def run(self) -> None:
        _synchronize_cuda()
        started = time.perf_counter()
        s = self.state
        det_rows = s.det_rows.as_dict()
        text_rows = s.text_rows.as_dict()
        aligned_boxes = Boxes(tablet=s.crop_tablet)
        aligned_row_indices = [[] for _ in range(len(s.text_rows))]
        results: Dict[int, FeatureRowAlignmentResult] = {}

        for text_row_idx in sorted(s.row_sign_matches):
            if text_row_idx not in s.text_to_det:
                continue
            det_row_idx = s.text_to_det[text_row_idx]
            text_boxes = text_rows[text_row_idx]
            det_boxes = det_rows[det_row_idx]
            result, row_boxes = self.align_row(
                text_row_idx,
                det_row_idx,
                text_boxes,
                det_boxes,
            )
            results[text_row_idx] = result
            for box in row_boxes:
                aligned_row_indices[text_row_idx].append(len(aligned_boxes))
                aligned_boxes.append(box)

        _synchronize_cuda()
        s.aligned_boxes = aligned_boxes
        s.aligned_rows = base.BoxRows(aligned_boxes, aligned_row_indices)
        s.feature_coarse = FeatureCoarseRun(
            rows=results,
            window_size=(self.window_width, self.window_height),
            total_seconds=time.perf_counter() - started,
        )

    def align_row(
        self,
        text_row_idx: int,
        det_row_idx: int,
        text_boxes: List[Box],
        det_boxes: List[Box],
    ) -> Tuple[FeatureRowAlignmentResult, List[Box]]:
        sign_matches = self.state.row_sign_matches[text_row_idx]
        anchors = _exact_anchor_map(text_boxes, det_boxes, sign_matches)
        unmatched = [idx for idx in range(len(text_boxes)) if idx not in anchors]
        baseline = _fit_row_baseline(det_boxes)
        expected_centers = _expected_text_centers(
            len(text_boxes),
            anchors,
            det_boxes,
            float(self.window_width),
        )
        candidates = _build_sliding_windows(
            unmatched,
            anchors,
            {text_idx: det_boxes[det_idx] for text_idx, det_idx in anchors.items()},
            expected_centers,
            baseline,
            self.state.crop_tablet.img.shape[:2],
            (self.window_width, self.window_height),
            self.config.step_px,
            self.config.search_margin_px,
            self.config.max_candidates_per_row,
        )
        print(
            f"  [Feature coarse] text row {text_row_idx} -> det row {det_row_idx}: "
            f"{len(anchors)} anchors, {len(unmatched)} unmatched signs, "
            f"{len(candidates)} crop windows"
        )

        scores, timing = self.score_candidates(
            text_boxes,
            unmatched,
            candidates,
            progress_label=f"text row {text_row_idx}",
        )
        selected = _ordered_score_assignment(
            unmatched,
            candidates,
            scores.score,
            self.config.assignment_min_score,
        )
        fallback_boxes = base.align_text_row_to_detection(
            text_boxes=text_boxes,
            det_boxes=det_boxes,
            matches=sign_matches,
            avg_width=float(self.window_width),
            avg_height=float(self.window_height),
        )
        unmatched_rows = {
            text_idx: matrix_row
            for matrix_row, text_idx in enumerate(unmatched)
        }
        assignments: List[FeatureSignAssignment] = []
        row_boxes: List[Box] = []

        for text_idx, text_box in enumerate(text_boxes):
            if text_idx in anchors:
                det_box = det_boxes[anchors[text_idx]]
                box = Box(
                    x1=det_box.x1,
                    y1=det_box.y1,
                    x2=det_box.x2,
                    y2=det_box.y2,
                    candidates=[SignCandidate(
                        sign=text_box.sign,
                        score=det_box.score,
                    )],
                    tablet=text_box.tablet,
                )
                assignment = FeatureSignAssignment(
                    text_idx,
                    text_box.sign_name,
                    "anchor",
                    box,
                    score=1.0,
                )
            elif text_idx in selected:
                candidate_idx = selected[text_idx]
                candidate = candidates[candidate_idx]
                matrix_row = unmatched_rows[text_idx]
                score = float(scores.score[matrix_row, candidate_idx])
                box = Box(
                    x1=candidate.x1,
                    y1=candidate.y1,
                    x2=candidate.x2,
                    y2=candidate.y2,
                    candidates=[SignCandidate(
                        sign=text_box.sign,
                        score=score,
                    )],
                    tablet=text_box.tablet,
                )
                assignment = FeatureSignAssignment(
                    text_idx,
                    text_box.sign_name,
                    "feature",
                    box,
                    score=score,
                    geometry=float(scores.geometry[matrix_row, candidate_idx]),
                    support=float(scores.support[matrix_row, candidate_idx]),
                    inlier_score=float(scores.inlier_score[matrix_row, candidate_idx]),
                    n_matches=int(scores.matches[matrix_row, candidate_idx]),
                    n_inliers=int(scores.inliers[matrix_row, candidate_idx]),
                    candidate_idx=candidate_idx,
                )
            else:
                box = fallback_boxes[text_idx]
                assignment = FeatureSignAssignment(
                    text_idx,
                    text_box.sign_name,
                    "fallback",
                    box,
                )
            assignments.append(assignment)
            row_boxes.append(box)

        return FeatureRowAlignmentResult(
            text_row_idx=text_row_idx,
            det_row_idx=det_row_idx,
            baseline=baseline,
            anchors=anchors,
            unmatched_text_indices=unmatched,
            candidates=candidates,
            scores=scores,
            timing=timing,
            assignments=assignments,
        ), row_boxes

    def score_candidates(
        self,
        text_boxes: List[Box],
        unmatched: List[int],
        candidates: List[SlidingWindow],
        progress_label: str,
    ) -> Tuple[FeatureScoreGrid, FeatureRowTiming]:
        scores = FeatureScoreGrid.empty(len(unmatched), len(candidates))
        timing = FeatureRowTiming()
        if not unmatched or not candidates:
            return scores, timing

        source_items = []
        for text_idx in unmatched:
            sign = text_boxes[text_idx].sign
            source_img = self.source.get(sign.name, self.period)
            source_feature = self.runtime.get_sign_feature(sign, self.period)
            source_items.append((source_img, source_feature))
        available = sum(
            source_img is not None and source_feature is not None
            for source_img, source_feature in source_items
        )
        timing.logical_pairs = available * len(candidates)
        if not available:
            return scores, timing

        for candidate_idx, candidate in enumerate(candidates):
            crop_box = Box(
                x1=candidate.x1,
                y1=candidate.y1,
                x2=candidate.x2,
                y2=candidate.y2,
                candidates=[SignCandidate(
                    sign=text_boxes[unmatched[0]].sign,
                    score=1.0,
                )],
                tablet=self.state.crop_tablet,
            )
            crop = crop_box.crop_image()
            _synchronize_cuda()
            started = time.perf_counter()
            crop_features = self.runtime.featurize_image(crop)
            _synchronize_cuda()
            timing.crop_feature_seconds += time.perf_counter() - started
            timing.crop_features += 1

            matches_by_sign = {}
            for matrix_row, text_idx in enumerate(unmatched):
                source_img, source_feature = source_items[matrix_row]
                if source_img is None or source_feature is None:
                    continue
                sign_name = text_boxes[text_idx].sign_name
                result = matches_by_sign.get(sign_name)
                if result is None:
                    _synchronize_cuda()
                    started = time.perf_counter()
                    result = self.runtime.match(
                        source_feature,
                        crop_features,
                        source_img.shape[:2],
                        crop.shape[:2],
                        self.config.match,
                        src_foreground_mask=source_foreground_mask(
                            source_img,
                            source_feature.shape[-2:],
                        ),
                    )
                    _synchronize_cuda()
                    matches_by_sign[sign_name] = result
                    timing.matches.append(FeatureMatchTiming(
                        candidate_idx=candidate_idx,
                        text_idx=text_idx,
                        sign_name=sign_name,
                        elapsed_seconds=time.perf_counter() - started,
                        score=result.coarse_score,
                        n_matches=result.n_matches,
                        n_inliers=result.n_inliers,
                        message=result.message,
                    ))

                scores.score[matrix_row, candidate_idx] = result.coarse_score
                scores.geometry[matrix_row, candidate_idx] = result.geometry_score
                scores.support[matrix_row, candidate_idx] = result.support_score
                scores.inlier_score[matrix_row, candidate_idx] = result.inlier_score
                scores.matches[matrix_row, candidate_idx] = result.n_matches
                scores.inliers[matrix_row, candidate_idx] = result.n_inliers

            completed = candidate_idx + 1
            if (
                self.config.progress_every > 0
                and (
                    completed % self.config.progress_every == 0
                    or completed == len(candidates)
                )
            ):
                print(
                    f"    [Feature coarse] {progress_label}: "
                    f"scored {completed}/{len(candidates)} windows"
                )

        return scores, timing


def _exact_anchor_map(
    text_boxes: List[Box],
    det_boxes: List[Box],
    sign_matches: List[Tuple[int, int]],
) -> Dict[int, int]:
    return {
        text_idx: det_idx
        for text_idx, det_idx in sign_matches
        if text_boxes[text_idx].sign_name == det_boxes[det_idx].sign_name
    }


def _fit_row_baseline(det_boxes: List[Box]) -> Tuple[float, float]:
    if len(det_boxes) >= 2:
        xs = np.asarray([box.cx for box in det_boxes], dtype=np.float64)
        ys = np.asarray([box.cy for box in det_boxes], dtype=np.float64)
        if float(np.ptp(xs)) > 1e-6:
            slope, intercept = np.polyfit(xs, ys, 1)
            return float(slope), float(intercept)
    return 0.0, float(np.mean([box.cy for box in det_boxes]))


def _expected_text_centers(
    num_text: int,
    anchors: Dict[int, int],
    det_boxes: List[Box],
    avg_width: float,
) -> np.ndarray:
    indices = np.arange(num_text, dtype=np.float64)
    anchor_items = sorted(anchors.items())
    if len(anchor_items) >= 2:
        text_idx = np.asarray([item[0] for item in anchor_items], dtype=np.float64)
        det_x = np.asarray(
            [det_boxes[item[1]].cx for item in anchor_items],
            dtype=np.float64,
        )
        slope, intercept = np.polyfit(text_idx, det_x, 1)
        return slope * indices + intercept

    spacing = _estimated_detection_spacing(det_boxes, avg_width)
    if len(anchor_items) == 1:
        text_idx, det_idx = anchor_items[0]
        return det_boxes[det_idx].cx + (indices - text_idx) * spacing

    det_center = float(np.mean([box.cx for box in det_boxes]))
    text_center = (num_text - 1) / 2.0
    return det_center + (indices - text_center) * spacing


def _estimated_detection_spacing(
    det_boxes: List[Box],
    fallback: float,
) -> float:
    centers = np.sort(np.asarray([box.cx for box in det_boxes], dtype=np.float64))
    gaps = np.diff(centers)
    gaps = gaps[gaps > 1e-6]
    return float(np.median(gaps)) if gaps.size else float(fallback)


def _build_sliding_windows(
    unmatched_text_indices: List[int],
    anchors: Dict[int, int],
    anchor_boxes: Dict[int, Box],
    expected_centers: np.ndarray,
    baseline: Tuple[float, float],
    image_shape: Tuple[int, int],
    window_size: Tuple[int, int],
    step_px: float,
    margin_px: float,
    max_candidates: Optional[int],
) -> List[SlidingWindow]:
    if not unmatched_text_indices:
        return []

    image_h, image_w = image_shape
    window_w, window_h = window_size
    min_cx = window_w / 2.0
    max_cx = image_w - window_w / 2.0
    min_cy = window_h / 2.0
    max_cy = image_h - window_h / 2.0
    if max_cx < min_cx or max_cy < min_cy:
        return []

    slope, intercept = baseline
    windows: List[SlidingWindow] = []
    runs = _contiguous_runs(unmatched_text_indices)
    for text_start, text_end in runs:
        left_anchor_idx = text_start - 1 if text_start - 1 in anchors else None
        right_anchor_idx = text_end + 1 if text_end + 1 in anchors else None

        start_x = float(
            np.min(expected_centers[text_start:text_end + 1]) - margin_px
        )
        if left_anchor_idx is not None:
            left_box = anchor_boxes[left_anchor_idx]
            start_x = max(start_x, float(left_box.cx + step_px / 2.0))

        end_x = float(
            np.max(expected_centers[text_start:text_end + 1]) + margin_px
        )
        if right_anchor_idx is not None:
            right_box = anchor_boxes[right_anchor_idx]
            end_x = min(end_x, float(right_box.cx - step_px / 2.0))

        start_x = max(min_cx, start_x)
        end_x = min(max_cx, end_x)
        positions = _sample_interval(
            start_x,
            end_x,
            step_px,
            origin=float(expected_centers[text_start]),
        )
        for cx in positions:
            cy = float(np.clip(slope * cx + intercept, min_cy, max_cy))
            x1 = int(round(cx - window_w / 2.0))
            y1 = int(round(cy - window_h / 2.0))
            x1 = min(max(0, x1), image_w - window_w)
            y1 = min(max(0, y1), image_h - window_h)
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

    deduplicated = {
        (window.x1, window.y1, window.x2, window.y2,
         window.text_start, window.text_end): window
        for window in windows
    }
    windows = sorted(
        deduplicated.values(),
        key=lambda window: (window.cx, window.cy),
    )
    if max_candidates is not None and len(windows) > max_candidates:
        keep = np.linspace(
            0, len(windows) - 1, max_candidates, dtype=np.int64
        )
        windows = [windows[idx] for idx in keep]
    return windows


def _contiguous_runs(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    runs: List[Tuple[int, int]] = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _sample_interval(
    start: float,
    end: float,
    step: float,
    origin: Optional[float] = None,
) -> List[float]:
    if end < start:
        return []
    if origin is None:
        origin = start
    first_k = int(np.ceil((start - origin) / step))
    last_k = int(np.floor((end - origin) / step))
    if first_k > last_k:
        return [(start + end) / 2.0]
    return [
        float(origin + k * step)
        for k in range(first_k, last_k + 1)
    ]


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ordered_score_assignment(
    text_indices: List[int],
    candidates: List[SlidingWindow],
    score_matrix: np.ndarray,
    min_score: float,
) -> Dict[int, int]:
    """Maximize assignment count, then score, preserving left-to-right order."""
    num_text = len(text_indices)
    num_candidates = len(candidates)
    if num_text == 0 or num_candidates == 0:
        return {}

    dp_score = np.full(
        (num_text + 1, num_candidates + 1), -np.inf, dtype=np.float64
    )
    dp_count = np.full(
        (num_text + 1, num_candidates + 1), -1, dtype=np.int32
    )
    back: Dict[Tuple[int, int], Tuple[int, int, str]] = {}
    dp_score[0, 0] = 0.0
    dp_count[0, 0] = 0

    for i in range(num_text + 1):
        for j in range(num_candidates + 1):
            if not np.isfinite(dp_score[i, j]):
                continue
            if i < num_text:
                _update_dp(
                    dp_score, dp_count, back,
                    source=(i, j), target=(i + 1, j),
                    score_delta=0.0, count_delta=0, action="skip_text",
                )
            if j < num_candidates:
                _update_dp(
                    dp_score, dp_count, back,
                    source=(i, j), target=(i, j + 1),
                    score_delta=0.0, count_delta=0, action="skip_candidate",
                )
            if i < num_text and j < num_candidates:
                text_idx = text_indices[i]
                candidate = candidates[j]
                score = float(score_matrix[i, j])
                allowed = (
                    candidate.text_start <= text_idx <= candidate.text_end
                    and score > min_score
                )
                if allowed:
                    _update_dp(
                        dp_score, dp_count, back,
                        source=(i, j), target=(i + 1, j + 1),
                        score_delta=score, count_delta=1, action="assign",
                    )

    selected: Dict[int, int] = {}
    i, j = num_text, num_candidates
    while (i, j) != (0, 0):
        previous = back.get((i, j))
        if previous is None:
            break
        prev_i, prev_j, action = previous
        if action == "assign":
            selected[text_indices[prev_i]] = prev_j
        i, j = prev_i, prev_j
    return selected


def _update_dp(
    dp_score: np.ndarray,
    dp_count: np.ndarray,
    back: Dict[Tuple[int, int], Tuple[int, int, str]],
    source: Tuple[int, int],
    target: Tuple[int, int],
    score_delta: float,
    count_delta: int,
    action: str,
) -> None:
    source_i, source_j = source
    target_i, target_j = target
    score = dp_score[source_i, source_j] + score_delta
    count = dp_count[source_i, source_j] + count_delta
    old_score = dp_score[target_i, target_j]
    old_count = dp_count[target_i, target_j]
    more_assignments = count > old_count
    tied_but_better = count == old_count and score > old_score + 1e-12
    if more_assignments or tied_but_better:
        dp_score[target_i, target_j] = score
        dp_count[target_i, target_j] = count
        back[(target_i, target_j)] = (source_i, source_j, action)


def vis_feature_coarse_alignment(
    context: CropContext,
    vis: VisOptions,
) -> None:
    s = context.state
    run: Optional[FeatureCoarseRun] = s.feature_coarse
    row_results = run.rows if run else {}
    if not row_results:
        if vis.info:
            print("=== Feature coarse alignment: no row results ===")
            if run:
                print(
                    f"whole coarse step={run.total_seconds:.3f}s"
                )
        return

    overlay = _render_feature_coarse_overlay(context, row_results)
    if vis.info:
        config = context.feature_coarse_alignment
        print("=== Experimental Feature Coarse Alignment ===")
        print(
            f"score = sqrt(relaxed IoU * angle) * support; step="
            f"{config.step_px:.0f}px; "
            f"window={run.window_size[0]}x{run.window_size[1]}"
        )
        for text_row_idx, result in sorted(row_results.items()):
            feature_assignments = [
                item for item in result.assignments if item.source == "feature"
            ]
            fallback_assignments = [
                item for item in result.assignments if item.source == "fallback"
            ]
            print(
                f"  Text row {text_row_idx} -> Det row {result.det_row_idx}: "
                f"anchors={len(result.anchors)}, "
                f"unmatched={len(result.unmatched_text_indices)}, "
                f"windows={len(result.candidates)}, "
                f"feature={len(feature_assignments)}, "
                f"fallback={len(fallback_assignments)}"
            )
            row_match_seconds = sum(
                timing.elapsed_seconds for timing in result.timing.matches
            )
            row_computed_count = len(result.timing.matches)
            row_match_mean = (
                row_match_seconds / row_computed_count
                if row_computed_count
                else 0.0
            )
            print(
                f"    timing: logical pairs="
                f"{result.timing.logical_pairs}, "
                f"computed pairs={row_computed_count}, "
                f"match total={row_match_seconds:.3f}s, "
                f"match mean={row_match_mean:.3f}s, "
                f"crop features={result.timing.crop_features} in "
                f"{result.timing.crop_feature_seconds:.3f}s"
            )
            if config.print_match_timings:
                for timing in result.timing.matches:
                    candidate = result.candidates[timing.candidate_idx]
                    status = timing.message or "ok"
                    print(
                        f"      pair crop[{timing.candidate_idx}] "
                        f"({candidate.x1},{candidate.y1},"
                        f"{candidate.x2},{candidate.y2}) x "
                        f"text[{timing.text_idx}] {timing.sign_name}: "
                        f"{timing.elapsed_seconds * 1000.0:.2f} ms, "
                        f"score={timing.score:.3f}, "
                        f"inliers={timing.n_inliers}/{timing.n_matches}, "
                        f"{status}"
                    )
            for item in feature_assignments:
                print(
                    f"    text[{item.text_idx}] {item.sign_name}: "
                    f"score={item.score:.3f} "
                    f"(geometry={item.geometry:.3f}, "
                    f"support={item.support:.3f}, "
                    f"inlier_score={item.inlier_score:.3f}, "
                    f"inliers={item.n_inliers}/{item.n_matches})"
                )
        timing_summary = run.timing
        print("=== Feature Coarse Timing Summary ===")
        print(
            f"logical score pairs="
            f"{timing_summary['logical_score_pair_count']}, "
            f"computed pairs={timing_summary['computed_pair_count']}, "
            f"cache reuses={timing_summary['cached_pair_count']}"
        )
        print(
            f"match total={timing_summary['match_total_seconds']:.3f}s, "
            f"mean={timing_summary['match_mean_seconds'] * 1000.0:.2f}ms, "
            f"min={timing_summary['match_min_seconds'] * 1000.0:.2f}ms, "
            f"max={timing_summary['match_max_seconds'] * 1000.0:.2f}ms"
        )
        print(
            f"crop feature extraction: "
            f"{timing_summary['crop_feature_count']} crops in "
            f"{timing_summary['crop_feature_total_seconds']:.3f}s; "
            f"whole coarse step="
            f"{timing_summary['coarse_alignment_total_seconds']:.3f}s"
        )

    if vis.save:
        cv2.imwrite(
            base._out(context, "feature_coarse_alignment.jpg"),
            overlay,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    if vis.display:
        base._display_bgr(
            overlay,
            "Experimental DIFT sliding-window coarse alignment",
        )
    if vis.display or vis.save:
        _plot_score_matrices(context, row_results, vis)


def _render_feature_coarse_overlay(
    context: CropContext,
    row_results: Dict[int, FeatureRowAlignmentResult],
) -> np.ndarray:
    overlay = _to_bgr(context.state.crop_tablet.img).copy()
    for text_row_idx, result in sorted(row_results.items()):
        for candidate in result.candidates:
            cv2.rectangle(
                overlay,
                (candidate.x1, candidate.y1),
                (candidate.x2, candidate.y2),
                (100, 100, 100),
                1,
                cv2.LINE_AA,
            )
        for assignment in result.assignments:
            color = {
                "anchor": (80, 220, 80),
                "feature": (0, 165, 255),
                "fallback": (255, 220, 0),
            }[assignment.source]
            box = assignment.box
            p1 = (int(round(box.x1)), int(round(box.y1)))
            p2 = (int(round(box.x2)), int(round(box.y2)))
            cv2.rectangle(overlay, p1, p2, color, 2, cv2.LINE_AA)
            label = (
                f"R{text_row_idx}:{assignment.sign_name} "
                f"{assignment.score:.2f}"
                if assignment.source == "feature"
                else f"R{text_row_idx}:{assignment.sign_name} "
                     f"{assignment.source[0].upper()}"
            )
            _draw_label(overlay, label, p1, color)
    return overlay


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    x = max(0, origin[0])
    y = max(12, origin[1] - 4)
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (0, 0, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, color, 1, cv2.LINE_AA,
    )


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    return image.astype(np.uint8)


def _plot_score_matrices(
    context: CropContext,
    row_results: Dict[int, FeatureRowAlignmentResult],
    vis: VisOptions,
) -> None:
    import matplotlib.pyplot as plt

    results = [
        result for result in row_results.values()
        if result.score_matrix.size
    ]
    if not results:
        return
    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(max(10, max(len(r.candidates) for r in results) * 0.45),
                 max(3, len(results) * 3.2)),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, result in zip(axes[:, 0], results):
        heat = axis.imshow(
            result.score_matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        signs = [
            result.assignments[text_idx].sign_name
            for text_idx in result.unmatched_text_indices
        ]
        axis.set_yticks(range(len(signs)), signs)
        x_labels = [f"{candidate.cx:.0f}" for candidate in result.candidates]
        stride = max(1, len(x_labels) // 20)
        ticks = list(range(0, len(x_labels), stride))
        axis.set_xticks(ticks, [x_labels[idx] for idx in ticks], rotation=45)
        axis.set_xlabel("candidate center x")
        axis.set_ylabel("unmatched text sign")
        axis.set_title(
            f"Text row {result.text_row_idx} -> "
            f"Det row {result.det_row_idx}: relaxed IoU + angle geometry"
        )
        fig.colorbar(heat, ax=axis, fraction=0.025, pad=0.02)

    if vis.save:
        fig.savefig(
            base._out(context, "feature_coarse_scores.png"),
            dpi=150,
        )
    if vis.display:
        plt.show()
    else:
        plt.close(fig)
