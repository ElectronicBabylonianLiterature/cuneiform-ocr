"""Multi-angle Hough row detection for near-horizontal point sets.

The detector deliberately contains no ``Box`` or pipeline dependencies.  It
operates on an ``(N, 2)`` array of point centres so the geometric part can be
tested independently of the detector runtime.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class HoughRowDetection:
    """Geometric result returned by :func:`detect_hough_rows`."""

    rows: List[List[int]]
    noise: List[int]
    row_angles_deg: np.ndarray
    row_rhos: np.ndarray
    row_x_ranges: List[Tuple[float, float]]
    global_angle_deg: float
    angles_deg: np.ndarray
    rho_grid: np.ndarray
    parameter_space: np.ndarray
    angle_scores: np.ndarray
    x_origin: float
    initial_row_angles_deg: np.ndarray
    initial_row_rhos: np.ndarray
    angle_curve_angles_deg: np.ndarray
    angle_curve_inlier_mask: np.ndarray
    curve_search_angle_deg: float


@dataclass
class _LineCandidate:
    angle_deg: float
    rho: float
    support: Tuple[int, ...]
    score: float
    residual: float
    x_min: float
    x_max: float


@dataclass
class _AngleCurve:
    coefficients: np.ndarray
    rho_center: float
    rho_scale: float
    inlier_mask: np.ndarray

    def evaluate(self, rhos: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(rhos) - self.rho_center) / self.rho_scale
        design = np.column_stack((
            np.ones_like(normalized),
            normalized,
            normalized ** 2,
        ))
        return design @ self.coefficients


def _line_distances(
    centers: np.ndarray,
    angle_deg: float,
    rho: float,
) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.abs(
        -centers[:, 0] * np.sin(angle)
        + centers[:, 1] * np.cos(angle)
        - rho
    )


def _line_y(angle_deg: float, rho: float, x: float) -> float:
    angle = np.deg2rad(angle_deg)
    cosine = np.cos(angle)
    if abs(cosine) < 1e-12:
        return float("inf")
    return float((rho + x * np.sin(angle)) / cosine)


def _fit_angle_curve(
    rhos: np.ndarray,
    angles_deg: np.ndarray,
    curvature_penalty: float,
) -> _AngleCurve:
    """Fit ``theta(rho)`` with low curvature and zero average slope.

    Rho is normalized symmetrically to ``[-1, 1]`` and the linear term is
    fixed to zero.  The two endpoints therefore have the same fitted angle,
    while the quadratic term still permits a gentle bend between them.
    """
    rhos = np.asarray(rhos, dtype=np.float64)
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    rho_center = (
        float((np.min(rhos) + np.max(rhos)) / 2)
        if len(rhos)
        else 0.0
    )
    rho_scale = max(float(np.ptp(rhos)) / 2, 1.0) if len(rhos) else 1.0
    if not len(rhos):
        return _AngleCurve(
            coefficients=np.zeros(3, dtype=np.float64),
            rho_center=rho_center,
            rho_scale=rho_scale,
            inlier_mask=np.empty(0, dtype=bool),
        )

    normalized = (rhos - rho_center) / rho_scale
    design = np.column_stack((
        np.ones_like(normalized),
        normalized,
        normalized ** 2,
    ))
    # Hard constraint: zero mean derivative across the fitted rho interval.
    design[:, 1] = 0.0
    if len(rhos) == 1:
        design[:, 2] = 0.0
    elif len(rhos) == 2:
        design[:, 2] = 0.0

    penalty = np.diag((0.0, 0.0, max(curvature_penalty, 0.0)))

    def solve(weights: np.ndarray) -> np.ndarray:
        weighted_design = design * weights[:, None]
        lhs = design.T @ weighted_design + penalty
        rhs = design.T @ (weights * angles_deg)
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    weights = np.ones(len(rhos), dtype=np.float64)
    coefficients = solve(weights)
    for _ in range(8):
        residuals = angles_deg - design @ coefficients
        residual_center = float(np.median(residuals))
        deviations = np.abs(residuals - residual_center)
        robust_sigma = max(
            1.4826 * float(np.median(deviations)),
            0.50,
        )
        huber_limit = 1.5 * robust_sigma
        weights = np.minimum(
            1.0,
            huber_limit / np.maximum(deviations, 1e-9),
        )
        coefficients = solve(weights)

    residuals = angles_deg - design @ coefficients
    residual_center = float(np.median(residuals))
    deviations = np.abs(residuals - residual_center)
    robust_sigma = max(
        1.4826 * float(np.median(deviations)),
        0.50,
    )
    inlier_limit = max(2.0, min(4.5, 2.5 * robust_sigma))
    inlier_mask = deviations <= inlier_limit
    if int(inlier_mask.sum()) >= min(3, len(rhos)):
        coefficients = solve(inlier_mask.astype(np.float64))

    return _AngleCurve(
        coefficients=coefficients,
        rho_center=rho_center,
        rho_scale=rho_scale,
        inlier_mask=inlier_mask,
    )


def _fit_bounded_line(
    centers: np.ndarray,
    indices: np.ndarray,
    initial_angle_deg: float,
    angle_range_deg: float,
    inlier_distance: float,
) -> Tuple[float, float]:
    """Fit ``y = ax + b`` with Huber-style reweighting and a hard angle bound."""
    points = centers[indices]
    angle_deg = float(np.clip(
        initial_angle_deg,
        -angle_range_deg,
        angle_range_deg,
    ))
    angle = np.deg2rad(angle_deg)
    rho = float(np.median(
        -points[:, 0] * np.sin(angle)
        + points[:, 1] * np.cos(angle)
    ))

    if len(points) < 2 or np.ptp(points[:, 0]) < 1e-9:
        return angle_deg, rho

    weights = np.ones(len(points), dtype=np.float64)
    design = np.column_stack((points[:, 0], np.ones(len(points))))
    max_slope = float(np.tan(np.deg2rad(angle_range_deg)))

    for _ in range(5):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = points[:, 1] * np.sqrt(weights)
        slope, _ = np.linalg.lstsq(
            weighted_design,
            weighted_y,
            rcond=None,
        )[0]
        slope = float(np.clip(slope, -max_slope, max_slope))
        angle_deg = float(np.clip(
            np.rad2deg(np.arctan(slope)),
            -angle_range_deg,
            angle_range_deg,
        ))
        angle = np.deg2rad(angle_deg)
        projected = (
            -points[:, 0] * np.sin(angle)
            + points[:, 1] * np.cos(angle)
        )
        rho = float(np.median(projected))
        residuals = np.abs(projected - rho)
        robust_sigma = max(
            1.4826 * float(np.median(residuals)),
            inlier_distance * 0.20,
            1e-6,
        )
        huber_limit = min(inlier_distance, 1.5 * robust_sigma)
        weights = np.minimum(
            1.0,
            huber_limit / np.maximum(residuals, 1e-9),
        )

    return angle_deg, rho


def _two_dimensional_peaks(
    parameter_space: np.ndarray,
    min_votes: float,
) -> np.ndarray:
    """Return 3x3 local maxima sorted by accumulator value."""
    if parameter_space.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    rows, columns = parameter_space.shape
    padded = np.pad(
        parameter_space,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=-np.inf,
    )
    neighbour_max = np.full_like(parameter_space, -np.inf)
    strict_neighbour_max = np.full_like(parameter_space, -np.inf)
    for row_offset in range(3):
        for column_offset in range(3):
            view = padded[
                row_offset:row_offset + rows,
                column_offset:column_offset + columns,
            ]
            neighbour_max = np.maximum(neighbour_max, view)
            if row_offset != 1 or column_offset != 1:
                strict_neighbour_max = np.maximum(strict_neighbour_max, view)

    mask = (
        (parameter_space >= neighbour_max)
        & (parameter_space > strict_neighbour_max)
        & (parameter_space >= min_votes)
    )
    peaks = np.argwhere(mask)
    if len(peaks) == 0:
        return peaks
    values = parameter_space[peaks[:, 0], peaks[:, 1]]
    return peaks[np.argsort(values)[::-1]]


def _candidate_from_peak(
    centers: np.ndarray,
    angle_deg: float,
    rho: float,
    peak_votes: float,
    angle_range_deg: float,
    assignment_distance: float,
    scale: float,
    min_row_size: int,
) -> _LineCandidate | None:
    support = np.flatnonzero(
        _line_distances(centers, angle_deg, rho) <= assignment_distance
    )
    if len(support) < min_row_size:
        return None

    # Refit twice so the grid peak supplies the initial basin while the final
    # model is continuous rather than quantized to angle_step_deg.
    for _ in range(2):
        angle_deg, rho = _fit_bounded_line(
            centers,
            support,
            angle_deg,
            angle_range_deg,
            assignment_distance,
        )
        support = np.flatnonzero(
            _line_distances(centers, angle_deg, rho) <= assignment_distance
        )
        if len(support) < min_row_size:
            return None

    distances = _line_distances(centers, angle_deg, rho)[support]
    x_values = centers[support, 0]
    span = float(np.ptp(x_values)) if len(support) > 1 else 0.0
    # Any two points define a Hough line.  Only recover a two-point row when
    # the points form a reasonably local fragment; otherwise unrelated noise
    # at opposite sides of the crop would be paired almost inevitably.
    if len(support) == 2 and not 0.75 * scale <= span <= 6.0 * scale:
        return None
    residual = float(np.median(distances))
    score = (
        float(len(support))
        + 0.10 * min(span / scale, float(len(support)))
        - 0.25 * residual / max(assignment_distance, 1e-9)
        + 0.01 * float(peak_votes)
    )
    return _LineCandidate(
        angle_deg=angle_deg,
        rho=rho,
        support=tuple(int(index) for index in support),
        score=score,
        residual=residual,
        x_min=float(x_values.min()),
        x_max=float(x_values.max()),
    )


def _crosses_selected_line(
    candidate: _LineCandidate,
    selected: _LineCandidate,
) -> bool:
    overlap_min = max(candidate.x_min, selected.x_min)
    overlap_max = min(candidate.x_max, selected.x_max)
    if overlap_max <= overlap_min:
        return False
    left_delta = (
        _line_y(candidate.angle_deg, candidate.rho, overlap_min)
        - _line_y(selected.angle_deg, selected.rho, overlap_min)
    )
    right_delta = (
        _line_y(candidate.angle_deg, candidate.rho, overlap_max)
        - _line_y(selected.angle_deg, selected.rho, overlap_max)
    )
    return left_delta * right_delta <= 0.0


def _select_disjoint_lines(
    centers: np.ndarray,
    candidates: List[_LineCandidate],
    scale: float,
    angle_range_deg: float,
    assignment_distance: float,
    min_line_distance: float,
    min_row_size: int,
) -> List[_LineCandidate]:
    selected: List[_LineCandidate] = []
    claimed: set[int] = set()

    for original in sorted(candidates, key=lambda item: item.score, reverse=True):
        available = np.asarray(
            [index for index in original.support if index not in claimed],
            dtype=np.int64,
        )
        if len(available) < min_row_size:
            continue

        angle_deg, rho = _fit_bounded_line(
            centers,
            available,
            original.angle_deg,
            angle_range_deg,
            assignment_distance,
        )
        distances = _line_distances(centers, angle_deg, rho)
        support = np.asarray([
            index for index in np.flatnonzero(distances <= assignment_distance)
            if index not in claimed
        ], dtype=np.int64)
        if len(support) < min_row_size:
            continue

        x_values = centers[support, 0]
        span = float(np.ptp(x_values)) if len(support) > 1 else 0.0
        if len(support) == 2 and not 0.75 * scale <= span <= 6.0 * scale:
            continue
        candidate = _LineCandidate(
            angle_deg=angle_deg,
            rho=rho,
            support=tuple(int(index) for index in support),
            score=original.score,
            residual=float(np.median(distances[support])),
            x_min=float(x_values.min()),
            x_max=float(x_values.max()),
        )

        duplicate = False
        for existing in selected:
            spans_overlap = (
                min(candidate.x_max, existing.x_max)
                >= max(candidate.x_min, existing.x_min)
            )
            overlap_midpoint = (
                max(candidate.x_min, existing.x_min)
                + min(candidate.x_max, existing.x_max)
            ) / 2
            same_geometric_line = (
                spans_overlap
                and abs(candidate.angle_deg - existing.angle_deg) <= 2.0
                and abs(
                    _line_y(candidate.angle_deg, candidate.rho, overlap_midpoint)
                    - _line_y(existing.angle_deg, existing.rho, overlap_midpoint)
                ) < min_line_distance
            )
            if same_geometric_line or _crosses_selected_line(candidate, existing):
                duplicate = True
                break
        if duplicate:
            continue

        selected.append(candidate)
        claimed.update(candidate.support)

    return selected


def _assign_and_refine(
    centers: np.ndarray,
    lines: List[_LineCandidate],
    angle_range_deg: float,
    assignment_distance: float,
    segment_extension: float,
    min_row_size: int,
) -> List[_LineCandidate]:
    current = list(lines)
    for _ in range(3):
        if not current:
            return []
        distances = np.column_stack([
            _line_distances(centers, line.angle_deg, line.rho)
            for line in current
        ])
        for line_index, line in enumerate(current):
            # A two-point line has zero fitting residual by construction.  A
            # modest complexity penalty prevents it from stealing endpoints
            # that are already well explained by a stronger row.
            if len(line.support) == 2:
                distances[:, line_index] += 0.50 * assignment_distance
            elif len(line.support) == 3:
                distances[:, line_index] += 0.20 * assignment_distance
            outside_segment = (
                (centers[:, 0] < line.x_min - segment_extension)
                | (centers[:, 0] > line.x_max + segment_extension)
            )
            distances[outside_segment, line_index] = np.inf

        assignments = np.argmin(distances, axis=1)
        minimum_distances = distances[np.arange(len(centers)), assignments]
        assignments[minimum_distances > assignment_distance] = -1

        refined: List[_LineCandidate] = []
        for line_index, line in enumerate(current):
            support = np.flatnonzero(assignments == line_index)
            if len(support) < min_row_size:
                continue
            angle_deg, rho = _fit_bounded_line(
                centers,
                support,
                line.angle_deg,
                angle_range_deg,
                assignment_distance,
            )
            line_distances = _line_distances(centers, angle_deg, rho)[support]
            x_values = centers[support, 0]
            refined.append(_LineCandidate(
                angle_deg=angle_deg,
                rho=rho,
                support=tuple(int(index) for index in support),
                score=line.score,
                residual=float(np.median(line_distances)),
                x_min=float(x_values.min()),
                x_max=float(x_values.max()),
            ))
        current = refined

    return current


def _merge_compatible_fragments(
    centers: np.ndarray,
    lines: List[_LineCandidate],
    angle_range_deg: float,
    assignment_distance: float,
    max_segment_gap: float,
) -> List[_LineCandidate]:
    """Merge fragments only when one bounded line explains their full union."""
    current = list(lines)
    while True:
        best_merge = None
        for first_index in range(len(current)):
            for second_index in range(first_index + 1, len(current)):
                first = current[first_index]
                second = current[second_index]
                segment_gap = max(
                    0.0,
                    max(first.x_min, second.x_min)
                    - min(first.x_max, second.x_max),
                )
                if segment_gap > max_segment_gap:
                    continue

                support = np.asarray(sorted(
                    set(first.support) | set(second.support)
                ), dtype=np.int64)
                initial = (
                    first.angle_deg
                    if len(first.support) >= len(second.support)
                    else second.angle_deg
                )
                angle_deg, rho = _fit_bounded_line(
                    centers,
                    support,
                    initial,
                    angle_range_deg,
                    assignment_distance,
                )
                distances = _line_distances(centers, angle_deg, rho)[support]
                if float(np.max(distances)) > assignment_distance:
                    continue

                residual = float(np.median(distances))
                x_values = centers[support, 0]
                merged = _LineCandidate(
                    angle_deg=angle_deg,
                    rho=rho,
                    support=tuple(int(index) for index in support),
                    score=first.score + second.score,
                    residual=residual,
                    x_min=float(x_values.min()),
                    x_max=float(x_values.max()),
                )
                merge_key = (residual, -len(support))
                if best_merge is None or merge_key < best_merge[0]:
                    best_merge = (
                        merge_key,
                        first_index,
                        second_index,
                        merged,
                    )

        if best_merge is None:
            return current
        _, first_index, second_index, merged = best_merge
        current = [
            line for index, line in enumerate(current)
            if index not in (first_index, second_index)
        ]
        current.append(merged)


def _select_refine_and_merge(
    centers: np.ndarray,
    candidates: List[_LineCandidate],
    scale: float,
    angle_range_deg: float,
    assignment_distance: float,
    min_line_distance: float,
    min_row_size: int,
) -> List[_LineCandidate]:
    selected = _select_disjoint_lines(
        centers,
        candidates,
        scale,
        angle_range_deg,
        assignment_distance,
        min_line_distance,
        min_row_size,
    )
    selected = _assign_and_refine(
        centers,
        selected,
        angle_range_deg,
        assignment_distance,
        segment_extension=5.0 * scale,
        min_row_size=min_row_size,
    )
    selected = _merge_compatible_fragments(
        centers,
        selected,
        angle_range_deg,
        assignment_distance,
        max_segment_gap=5.0 * scale,
    )
    return _assign_and_refine(
        centers,
        selected,
        angle_range_deg,
        assignment_distance,
        segment_extension=5.0 * scale,
        min_row_size=min_row_size,
    )


def detect_hough_rows(
    centers: np.ndarray,
    scale: float,
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
) -> HoughRowDetection:
    """Detect rows from point centres with an independent angle per row.

    Candidate lines are local maxima in the full two-dimensional Hough space.
    Every continuous refit is clipped to ``[-angle_range_deg,
    +angle_range_deg]``; no selected row can leave that interval.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1:] != (2,):
        raise ValueError("centers must have shape (N, 2)")
    if angle_range_deg <= 0 or angle_step_deg <= 0:
        raise ValueError("angle range and step must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if rho_step_factor <= 0 or rho_sigma_factor <= 0:
        raise ValueError("rho step and sigma factors must be positive")
    if min_line_distance_factor < 0 or assignment_distance_factor <= 0:
        raise ValueError("line distance factors must be non-negative")
    if min_row_size < 1:
        raise ValueError("min_row_size must be at least one")
    if curve_search_angle_deg <= 0:
        raise ValueError("curve_search_angle_deg must be positive")
    if curve_curvature_penalty < 0:
        raise ValueError("curve_curvature_penalty must be non-negative")

    angles_deg = np.arange(
        -angle_range_deg,
        angle_range_deg + 1e-12,
        angle_step_deg,
        dtype=np.float64,
    )
    if angles_deg[-1] < angle_range_deg - 1e-12:
        angles_deg = np.append(angles_deg, angle_range_deg)
    x_origin = float(np.median(centers[:, 0])) if len(centers) else 0.0
    hough_centers = centers.copy()
    if len(hough_centers):
        hough_centers[:, 0] -= x_origin

    rho_step = scale * rho_step_factor
    rho_sigma = scale * rho_sigma_factor
    assignment_distance = scale * assignment_distance_factor
    min_line_distance = scale * min_line_distance_factor

    projected_by_angle = []
    angle_scores = []
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        rhos = (
            -hough_centers[:, 0] * np.sin(angle)
            + hough_centers[:, 1] * np.cos(angle)
        )
        projected_by_angle.append(rhos)
        if len(hough_centers):
            pairwise_delta = (rhos[:, None] - rhos[None, :]) / rho_sigma
            pairwise_votes = np.exp(-0.5 * pairwise_delta ** 2)
            angle_scores.append(float(
                (pairwise_votes.sum() - len(hough_centers)) / 2
            ))
        else:
            angle_scores.append(0.0)

    if not len(hough_centers):
        return HoughRowDetection(
            rows=[],
            noise=[],
            row_angles_deg=np.empty(0, dtype=np.float64),
            row_rhos=np.empty(0, dtype=np.float64),
            row_x_ranges=[],
            global_angle_deg=0.0,
            angles_deg=angles_deg,
            rho_grid=np.empty(0, dtype=np.float64),
            parameter_space=np.empty((0, len(angles_deg)), dtype=np.float64),
            angle_scores=np.asarray(angle_scores),
            x_origin=x_origin,
            initial_row_angles_deg=np.empty(0, dtype=np.float64),
            initial_row_rhos=np.empty(0, dtype=np.float64),
            angle_curve_angles_deg=np.empty(0, dtype=np.float64),
            angle_curve_inlier_mask=np.empty(0, dtype=bool),
            curve_search_angle_deg=curve_search_angle_deg,
        )

    all_projected_rhos = np.concatenate(projected_by_angle)
    rho_grid = np.arange(
        all_projected_rhos.min() - 2 * rho_sigma,
        all_projected_rhos.max() + 2 * rho_sigma + rho_step,
        rho_step,
    )
    parameter_space = []
    for rhos in projected_by_angle:
        normalized_delta = (
            rho_grid[:, None] - rhos[None, :]
        ) / rho_sigma
        parameter_space.append(
            np.exp(-0.5 * normalized_delta ** 2).sum(axis=1)
        )
    parameter_space = np.asarray(parameter_space).T

    global_angle_idx = int(np.argmax(angle_scores))
    global_angle_deg = float(angles_deg[global_angle_idx])
    peaks = _two_dimensional_peaks(parameter_space, min_peak_votes)
    candidates: List[_LineCandidate] = []
    for rho_index, angle_index in peaks:
        candidate = _candidate_from_peak(
            hough_centers,
            float(angles_deg[angle_index]),
            float(rho_grid[rho_index]),
            float(parameter_space[rho_index, angle_index]),
            angle_range_deg,
            assignment_distance,
            scale,
            min_row_size,
        )
        if candidate is None:
            continue
        candidates.append(candidate)

    initial_selected = _select_refine_and_merge(
        hough_centers,
        candidates,
        scale,
        angle_range_deg,
        assignment_distance,
        min_line_distance,
        min_row_size,
    )
    initial_selected.sort(
        key=lambda line: _line_y(line.angle_deg, line.rho, 0.0)
    )
    initial_angles = np.asarray(
        [line.angle_deg for line in initial_selected],
        dtype=np.float64,
    )
    initial_rhos = np.asarray(
        [line.rho for line in initial_selected],
        dtype=np.float64,
    )
    angle_curve = _fit_angle_curve(
        initial_rhos,
        initial_angles,
        curve_curvature_penalty,
    )
    angle_curve_values = np.clip(
        angle_curve.evaluate(rho_grid),
        -angle_range_deg,
        angle_range_deg,
    )
    curve_reselection_active = len(initial_selected) >= 5

    candidate_rhos = np.asarray(
        [candidate.rho for candidate in candidates],
        dtype=np.float64,
    )
    candidate_curve_angles = np.clip(
        angle_curve.evaluate(candidate_rhos),
        -angle_range_deg,
        angle_range_deg,
    )
    curve_candidates = (
        [
            candidate
            for candidate, expected_angle
            in zip(candidates, candidate_curve_angles)
            if abs(candidate.angle_deg - expected_angle)
            <= curve_search_angle_deg
        ]
        if curve_reselection_active
        else candidates
    )
    selected = _select_refine_and_merge(
        hough_centers,
        curve_candidates,
        scale,
        angle_range_deg,
        assignment_distance,
        min_line_distance,
        min_row_size,
    )

    # Continuous row refits can move off their discrete Hough seeds slightly.
    # Recheck the curve window and reassign once after removing such a line.
    for _ in range(2 if curve_reselection_active else 0):
        selected = [
            line for line in selected
            if abs(
                line.angle_deg
                - float(np.clip(
                    angle_curve.evaluate(np.asarray([line.rho]))[0],
                    -angle_range_deg,
                    angle_range_deg,
                ))
            ) <= curve_search_angle_deg
        ]
        selected = _assign_and_refine(
            hough_centers,
            selected,
            angle_range_deg,
            assignment_distance,
            segment_extension=5.0 * scale,
            min_row_size=min_row_size,
        )

    selected.sort(key=lambda line: _line_y(line.angle_deg, line.rho, 0.0))
    rows: List[List[int]] = []
    row_angles = []
    row_rhos = []
    row_x_ranges = []
    assigned_indices: set[int] = set()
    for line in selected:
        angle = np.deg2rad(line.angle_deg)
        indices = list(line.support)
        indices.sort(key=lambda index: (
            hough_centers[index, 0] * np.cos(angle)
            + hough_centers[index, 1] * np.sin(angle)
        ))
        rows.append(indices)
        row_angles.append(line.angle_deg)
        row_rhos.append(line.rho)
        row_x_ranges.append((
            line.x_min + x_origin,
            line.x_max + x_origin,
        ))
        assigned_indices.update(indices)

    noise = sorted(
        set(range(len(centers))) - assigned_indices,
        key=lambda index: centers[index, 0],
    )
    return HoughRowDetection(
        rows=rows,
        noise=noise,
        row_angles_deg=np.asarray(row_angles, dtype=np.float64),
        row_rhos=np.asarray(row_rhos, dtype=np.float64),
        row_x_ranges=row_x_ranges,
        global_angle_deg=global_angle_deg,
        angles_deg=angles_deg,
        rho_grid=rho_grid,
        parameter_space=parameter_space,
        angle_scores=np.asarray(angle_scores, dtype=np.float64),
        x_origin=x_origin,
        initial_row_angles_deg=initial_angles,
        initial_row_rhos=initial_rhos,
        angle_curve_angles_deg=angle_curve_values,
        angle_curve_inlier_mask=angle_curve.inlier_mask,
        curve_search_angle_deg=curve_search_angle_deg,
    )
