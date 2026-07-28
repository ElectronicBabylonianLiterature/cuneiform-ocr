"""Position-first text-to-candidate attraction prototype.

The fixed experiment contract is documented in
``SIGN_ALIGNMENT_3_CANDIDATE_OPTIMIZATION.md``.  Keep this module direct and
experiment-oriented; it is not a replacement for ``sign_alignment.pipeline``.

This module is deliberately separate from :mod:`sign_alignment.pipeline`.
It consumes the state produced by ``match_rows`` -> ``align_text_rows`` and
stores its results under ``candidate_test_*`` attributes on ``SampleState``.
None of the original alignment fields are overwritten.

The experiment treats detector boxes as fixed physical candidates.  Reliable
same-label matches are hard anchors.  Every other text sign is represented by
a movable box and is softly attracted to nearby, unused candidates while the
temperature is annealed.  Candidate columns have capacity one, and a final
ordered dynamic program converts the soft attraction into a partial one-to-one
matching.  Detector class scores are a bounded *bonus* only: a geometrically
good box with the wrong class remains a valid match and is relabelled with the
text sign.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import cv2
import numpy as np
import torch

from sign_alignment.box import Box, Boxes
from sign_alignment.pipeline import (
    BoxRows,
    CropContext,
    VisOptions,
    gt_boxes_for_visualization,
)
from sign_alignment.visualizer import (
    BboxVisualizer,
    CompositeVisualizer,
    build_sign_match_info as build_sign_match_info_data,
)


# Experiment-wide semantic palette (RGB).  Green is reserved for GT only.
FIXED_CANDIDATE_COLOR = (255, 0, 0)
ANCHOR_COLOR = (180, 80, 255)
MOVABLE_TEXT_COLOR = (64, 128, 255)
CANDIDATE_MATCH_COLOR = (255, 215, 0)
NULL_COLOR = (0, 210, 255)
GT_COLOR = (0, 255, 0)
MOVEMENT_COLOR = (255, 255, 255)


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[::-1]


@dataclass(frozen=True)
class CandidateAttractionConfig:
    """Small, inspectable configuration for the prototype experiment."""

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
    history: list[dict[str, float]]
    free_text_indices: tuple[int, ...]
    free_candidate_indices: tuple[int, ...]
    final_soft_assignment: np.ndarray
    final_pair_cost: np.ndarray
    allowed_edges: np.ndarray
    stage_boxes: list[tuple[float, list[Box]]]


@dataclass
class CandidateAttractionRun:
    config: CandidateAttractionConfig
    boxes: Boxes
    rows: BoxRows
    row_results: dict[int, CandidateRowResult]


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


def _make_row_basis(det_boxes: list[Box], fallback_width: float, fallback_height: float) -> _RowBasis:
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

    history: list[dict[str, float]] = []
    stage_parameter_snapshots: list[tuple[float, np.ndarray]] = []
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

        global_step = 0
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
                    prior_weight * (free_parameters[:, :2] - initial_free[:, :2]).square().sum(dim=1)
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
                    real_mass = float(real_probability.sum(dim=1).mean().cpu()) if free_count else 0.0
                    entropy = float((
                        -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log())
                        .sum(dim=1).mean()
                    ).cpu())
                    history.append({
                        "step": float(global_step),
                        "temperature": float(temperature),
                        "loss": float(loss.detach().cpu()),
                        "attraction": float(attraction.detach().cpu()),
                        "real_mass": real_mass,
                        "entropy": entropy,
                    })
                global_step += 1
            stage_parameter_snapshots.append((
                float(temperature),
                assemble_all().detach().cpu().numpy().copy(),
            ))

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
        final_pair_cost = torch.empty((0, candidate_count), device=device)
        stage_parameter_snapshots = [
            (float(temperature), fixed_parameters_np.copy())
            for temperature in config.temperatures
        ]

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

    stage_boxes = []
    for temperature, stage_parameters in stage_parameter_snapshots:
        boxes_at_stage = []
        for text_idx, text_box in enumerate(text_boxes):
            if text_idx in anchor_candidates:
                box = _copy_candidate_geometry(
                    candidates[anchor_candidates[text_idx]], text_box, score=1.0
                )
            else:
                box = _decode_box(
                    stage_parameters[text_idx], text_box, basis, score=1.0
                )
            boxes_at_stage.append(box)
        stage_boxes.append((temperature, boxes_at_stage))

    return CandidateRowResult(
        text_row_idx=text_row_idx,
        det_row_idx=det_row_idx,
        candidates=candidates,
        assignments=assignments,
        history=history,
        free_text_indices=tuple(free_text_indices),
        free_candidate_indices=tuple(free_candidate_indices),
        final_soft_assignment=final_probability.detach().cpu().numpy(),
        final_pair_cost=final_pair_cost.detach().cpu().numpy(),
        allowed_edges=allowed_np.copy(),
        stage_boxes=stage_boxes,
    )


def run_candidate_attraction(
    context: CropContext,
    config: Optional[CandidateAttractionConfig] = None,
) -> CandidateAttractionRun:
    """Run the isolated prototype and attach only ``candidate_test_*`` state."""

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
    # Dynamic attributes are intentional: the original SampleState/pipeline is
    # left byte-for-byte and field-for-field untouched.
    state.candidate_test_boxes = output_boxes
    state.candidate_test_rows = output_rows
    state.candidate_test_run = run
    return run


def candidate_attraction_records(context: CropContext) -> list[dict]:
    """Flat diagnostics suitable for ``pandas.DataFrame`` in the notebook."""

    run: CandidateAttractionRun = context.state.candidate_test_run
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
    run: CandidateAttractionRun = state.candidate_test_run
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

    state.candidate_test_text_sign_match_info = text_info
    state.candidate_test_det_sign_match_info = det_info


def vis_candidate_pool(context: CropContext, vis: VisOptions) -> None:
    """Show physical candidate grouping and hard-anchor consumption."""

    state = context.state
    run: CandidateAttractionRun = state.candidate_test_run
    candidate_boxes = []
    candidate_colors = {}
    candidate_labels = []
    for row_result in run.row_results.values():
        anchor_indices = {
            assignment.candidate_idx
            for assignment in row_result.assignments
            if assignment.output_status == "anchor"
        }
        for candidate in row_result.candidates:
            candidate_boxes.append(candidate.box)
            candidate_colors[id(candidate.box)] = (
                ANCHOR_COLOR
                if candidate.candidate_idx in anchor_indices
                else FIXED_CANDIDATE_COLOR
            )
            candidate_labels.append((
                candidate.box,
                f"R{row_result.text_row_idx + 1}C{candidate.candidate_idx + 1}",
            ))

    boxes = Boxes(candidate_boxes, tablet=state.crop_tablet)
    visualizer = BboxVisualizer(color=FIXED_CANDIDATE_COLOR)
    visualizer.color_func = lambda box: candidate_colors[id(box)]
    image = visualizer.draw_boxes(
        state.crop_tablet.img.copy(), boxes, show_labels=True, show_scores=True
    )
    for box, label in candidate_labels:
        cv2.putText(
            image,
            label,
            (int(box.x1), max(12, int(box.y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    visualizer.result = image

    if vis.info:
        physical_count = sum(len(row.candidates) for row in run.row_results.values())
        raw_count = sum(
            len(candidate.member_det_indices)
            for row in run.row_results.values()
            for candidate in row.candidates
        )
        anchor_count = sum(
            assignment.output_status == "anchor"
            for row in run.row_results.values()
            for assignment in row.assignments
        )
        print("=== Fixed physical candidate pool ===")
        print(f"  Raw row detections: {raw_count}")
        print(f"  Physical candidates after grouping: {physical_count}")
        print(f"  Candidates consumed by reliable anchors: {anchor_count}")
        print("  Purple=hard anchor candidate, red=available fixed candidate")
    if vis.save:
        os.makedirs(context.output_dir, exist_ok=True)
        visualizer.save(_output_path(context, "candidate_pool.jpg"))
    if vis.display:
        visualizer.display_result(vis_opt="draw")


def vis_candidate_annealing_snapshots(context: CropContext, vis: VisOptions) -> None:
    """Show how continuous text boxes move after every temperature stage."""

    if not (vis.display or vis.save or vis.info):
        return
    import matplotlib.pyplot as plt

    state = context.state
    run: CandidateAttractionRun = state.candidate_test_run
    row_results = list(run.row_results.values())
    stage_count = min(
        (len(row.stage_boxes) for row in row_results),
        default=0,
    )
    panels: list[tuple[str, np.ndarray]] = []

    def candidate_background() -> np.ndarray:
        image = state.crop_tablet.img.copy()
        for row_result in row_results:
            for candidate in row_result.candidates:
                box = candidate.box
                cv2.rectangle(
                    image,
                    (int(box.x1), int(box.y1)),
                    (int(box.x2), int(box.y2)),
                    _bgr(FIXED_CANDIDATE_COLOR),
                    1,
                )
        return image

    for stage_idx in range(stage_count):
        image = candidate_background()
        stage_boxes = []
        colors = {}
        temperature = row_results[0].stage_boxes[stage_idx][0]
        for row_result in row_results:
            _, row_boxes = row_result.stage_boxes[stage_idx]
            anchor_text_indices = {
                assignment.text_idx
                for assignment in row_result.assignments
                if assignment.output_status == "anchor"
            }
            visible_row_boxes = []
            for text_idx, box in enumerate(row_boxes):
                if not _box_fully_inside_image(box):
                    continue
                visible_row_boxes.append(box)
                stage_boxes.append(box)
                colors[id(box)] = (
                    ANCHOR_COLOR
                    if text_idx in anchor_text_indices
                    else MOVABLE_TEXT_COLOR
                )
            ordered = sorted(visible_row_boxes, key=lambda box: box.cx)
            for first, second in zip(ordered, ordered[1:]):
                cv2.line(
                    image,
                    (int(first.cx), int(first.cy)),
                    (int(second.cx), int(second.cy)),
                    (180, 180, 0),
                    1,
                    cv2.LINE_AA,
                )
        boxes = Boxes(stage_boxes, tablet=state.crop_tablet)
        visualizer = BboxVisualizer(color=MOVABLE_TEXT_COLOR)
        visualizer.color_func = lambda box: colors[id(box)]
        rendered = visualizer.draw_boxes(image, boxes, show_labels=False)
        panels.append((f"after T={temperature:g}", rendered))

    final_image = candidate_background()
    final_colors = {}
    for row_result in row_results:
        for assignment in row_result.assignments:
            final_colors[id(assignment.final_box)] = {
                "anchor": ANCHOR_COLOR,
                "candidate": CANDIDATE_MATCH_COLOR,
                "null": NULL_COLOR,
            }[assignment.output_status]
    final_visualizer = BboxVisualizer(color=CANDIDATE_MATCH_COLOR)
    final_visualizer.color_func = lambda box: final_colors[id(box)]
    final_image = final_visualizer.draw_boxes(
        final_image, run.boxes, show_labels=False
    )
    panels.append(("final ordered assignment", final_image))

    column_count = min(3, max(1, len(panels)))
    row_count = int(np.ceil(len(panels) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7 * column_count, 5.5 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (title, image) in zip(axes.flat, panels):
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(title)
        axis.axis("off")
    for axis in list(axes.flat)[len(panels):]:
        axis.axis("off")
    figure.suptitle(
        "Annealed attraction: red=fixed candidates, purple=anchors, "
        "blue=movable text, yellow=final candidate match"
    )
    if vis.info:
        print(f"=== Annealing snapshots: {stage_count} temperature stages + final ===")
    if vis.save:
        os.makedirs(context.output_dir, exist_ok=True)
        figure.savefig(
            _output_path(context, "candidate_annealing_snapshots.png"),
            dpi=150,
            bbox_inches="tight",
        )
    if vis.display:
        plt.show()
    else:
        plt.close(figure)


def vis_candidate_assignment_matrices(context: CropContext, vis: VisOptions) -> None:
    """Plot final soft mass, pair costs, gates, and the selected hard edges."""

    if not (vis.display or vis.save or vis.info):
        return
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    run: CandidateAttractionRun = context.state.candidate_test_run
    plotted = 0
    for row_result in run.row_results.values():
        if not row_result.free_text_indices:
            continue
        soft = row_result.final_soft_assignment
        cost = row_result.final_pair_cost
        allowed = row_result.allowed_edges
        free_candidate_indices = list(row_result.free_candidate_indices)
        text_labels = [
            f"T{text_idx + 1}:{row_result.assignments[text_idx].sign_name}"
            for text_idx in row_result.free_text_indices
        ]
        candidate_labels = [
            f"C{candidate_idx + 1}:{row_result.candidates[candidate_idx].box.sign_name}"
            for candidate_idx in free_candidate_indices
        ]
        probability_labels = candidate_labels + ["NULL"]

        if cost.shape[1]:
            figure, axes = plt.subplots(
                1,
                2,
                figsize=(
                    max(10, 0.75 * len(probability_labels) + 6),
                    max(4, 0.5 * len(text_labels) + 2),
                ),
                constrained_layout=True,
            )
            probability_axis, cost_axis = axes
        else:
            figure, probability_axis = plt.subplots(
                1,
                1,
                figsize=(5, max(4, 0.5 * len(text_labels) + 2)),
                constrained_layout=True,
            )
            cost_axis = None

        probability_image = probability_axis.imshow(
            soft, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto"
        )
        probability_axis.set_title("final soft assignment")
        probability_axis.set_xticks(range(len(probability_labels)), probability_labels, rotation=45, ha="right")
        probability_axis.set_yticks(range(len(text_labels)), text_labels)
        figure.colorbar(probability_image, ax=probability_axis, fraction=0.046)

        # Mark the ordered-DP result, including NULL, with a red rectangle.
        assignment_by_text = {
            assignment.text_idx: assignment
            for assignment in row_result.assignments
        }
        global_to_free = {
            candidate_idx: free_idx
            for free_idx, candidate_idx in enumerate(free_candidate_indices)
        }
        for free_text_idx, text_idx in enumerate(row_result.free_text_indices):
            assignment = assignment_by_text[text_idx]
            selected_col = (
                global_to_free[assignment.candidate_idx]
                if assignment.candidate_idx is not None
                else len(free_candidate_indices)
            )
            probability_axis.add_patch(Rectangle(
                (selected_col - 0.5, free_text_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=2,
            ))
        if soft.size <= 250:
            for row_idx in range(soft.shape[0]):
                for col_idx in range(soft.shape[1]):
                    probability_axis.text(
                        col_idx,
                        row_idx,
                        f"{soft[row_idx, col_idx]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if soft[row_idx, col_idx] < 0.6 else "white",
                    )

        if cost_axis is not None:
            display_cost = np.ma.masked_where(~allowed, cost)
            cost_image = cost_axis.imshow(display_cost, cmap="magma_r", aspect="auto")
            cost_axis.set_title("position-dominant pair cost (masked=gated)")
            cost_axis.set_xticks(range(len(candidate_labels)), candidate_labels, rotation=45, ha="right")
            cost_axis.set_yticks(range(len(text_labels)), text_labels)
            figure.colorbar(cost_image, ax=cost_axis, fraction=0.046)
        figure.suptitle(
            f"Text row R{row_result.text_row_idx + 1} -> detection row D{row_result.det_row_idx + 1}"
        )
        if vis.save:
            os.makedirs(context.output_dir, exist_ok=True)
            path = _output_path(
                context,
                f"candidate_assignment_matrix_r{row_result.text_row_idx + 1}.png",
            )
            figure.savefig(path, dpi=150, bbox_inches="tight")
        if vis.display:
            plt.show()
        else:
            plt.close(figure)
        plotted += 1
    if vis.info:
        print(f"=== Assignment matrices: {plotted} movable text rows ===")
        print("  Red cell=final ordered-DP choice; NULL is the last probability column.")


def vis_candidate_alignment_diagnostic(context: CropContext, vis: VisOptions) -> None:
    """Render coarse and candidate-result alignment diagnostics side by side."""

    state = context.state
    if not hasattr(state, "candidate_test_text_sign_match_info"):
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
        aligned_rows=state.candidate_test_rows.as_lists(),
        det_sign_match_info=state.candidate_test_det_sign_match_info,
        text_sign_match_info=state.candidate_test_text_sign_match_info,
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
            for info in state.candidate_test_text_sign_match_info.values()
        ]
        print("=== Candidate alignment diagnostic ===")
        print(f"  Same-label/anchor or supported: {statuses.count('same')}")
        print(f"  Accepted position-only diff-label: {statuses.count('diff')}")
        print(f"  NULL/unmatched: {statuses.count('unmatched')}")
        print("  Solid detection boxes + dashed non-same text boxes; solid/dashed row baselines.")
    if vis.save:
        os.makedirs(context.output_dir, exist_ok=True)
        final.save(_output_path(context, "candidate_alignment_diagnostic.jpg"))
        comparison.save(_output_path(context, "candidate_alignment_diagnostic_comparison.jpg"))
    if vis.display:
        final.display_result(vis_opt="draw")
        comparison.display_result(vis_opt="draw")


def vis_candidate_results_comparison(context: CropContext, vis: VisOptions) -> None:
    """Mirror the original PSR 2x2 comparison for the candidate experiment."""

    state = context.state
    image = state.crop_tablet.img
    final_boxes = state.candidate_test_boxes

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
        before.save(_output_path(context, "candidate_coarse_aligned.jpg"))
        after.save(_output_path(context, "candidate_final.jpg"))
        det_overlay.save(_output_path(context, "candidate_overlay_det_final.jpg"))
        gt_overlay.save(_output_path(context, "candidate_overlay_gt_final.jpg"))
        comparison.save(_output_path(context, "candidate_results_comparison.jpg"))
    if vis.display:
        comparison.display_result(vis_opt="draw")


def vis_candidate_parameter_changes(context: CropContext, vis: VisOptions) -> None:
    """Report and plot coarse-to-final geometry changes by assignment status."""

    state = context.state
    run: CandidateAttractionRun = state.candidate_test_run
    rows = []
    filtered_count = 0
    for row_result in run.row_results.values():
        coarse_row = state.aligned_rows.row_boxes(row_result.text_row_idx)
        for assignment in row_result.assignments:
            if not assignment.included_in_result:
                filtered_count += 1
                continue
            before = coarse_row[assignment.text_idx]
            after = assignment.final_box
            rows.append({
                "status": assignment.output_status,
                "dx": after.cx - before.cx,
                "dy": after.cy - before.cy,
                "dw": after.width - before.width,
                "dh": after.height - before.height,
                "movement": float(np.hypot(after.cx - before.cx, after.cy - before.cy)),
            })
    if not rows:
        if vis.info:
            print(
                "No in-image candidate results for parameter-change analysis "
                f"({filtered_count} off-image boxes filtered)."
            )
        return

    if vis.info:
        print("=== Candidate parameter changes (coarse -> final) ===")
        print(f"  Off-image boxes filtered from result: {filtered_count}")
        for status in ("anchor", "candidate", "null"):
            subset = [row for row in rows if row["status"] == status]
            if not subset:
                continue
            print(
                f"  {status:9s}: n={len(subset):3d}, "
                f"movement mean={np.mean([r['movement'] for r in subset]):7.2f}px, "
                f"max={np.max([r['movement'] for r in subset]):7.2f}px"
            )
        for key in ("dx", "dy", "dw", "dh"):
            values = np.asarray([row[key] for row in rows])
            print(
                f"  {key}: mean={values.mean():.2f}, std={values.std():.2f}, "
                f"|max|={np.abs(values).max():.2f}"
            )
    if not (vis.display or vis.save):
        return

    import matplotlib.pyplot as plt

    colors = {"anchor": "tab:purple", "candidate": "tab:orange", "null": "tab:cyan"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for status, color in colors.items():
        subset = [row for row in rows if row["status"] == status]
        if not subset:
            continue
        axes[0, 0].scatter(
            [row["dx"] for row in subset],
            [row["dy"] for row in subset],
            label=status,
            color=color,
            alpha=0.7,
        )
        axes[0, 1].hist(
            [row["movement"] for row in subset],
            bins=20,
            alpha=0.5,
            label=status,
            color=color,
        )
        axes[1, 0].scatter(
            [row["dw"] for row in subset],
            [row["dh"] for row in subset],
            label=status,
            color=color,
            alpha=0.7,
        )
    axes[0, 0].axhline(0, color="black", linewidth=0.7)
    axes[0, 0].axvline(0, color="black", linewidth=0.7)
    axes[0, 0].set(title="center displacement", xlabel="delta cx (px)", ylabel="delta cy (px)")
    axes[0, 1].set(title="movement distribution", xlabel="center movement (px)", ylabel="count")
    axes[1, 0].axhline(0, color="black", linewidth=0.7)
    axes[1, 0].axvline(0, color="black", linewidth=0.7)
    axes[1, 0].set(title="size change", xlabel="delta width (px)", ylabel="delta height (px)")
    counts = [sum(row["status"] == status for row in rows) for status in colors]
    axes[1, 1].bar(list(colors), counts, color=list(colors.values()))
    axes[1, 1].set(title="final assignment counts", ylabel="signs")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.legend()
    figure.suptitle("Candidate optimization: coarse-to-final diagnostics")
    if vis.save:
        os.makedirs(context.output_dir, exist_ok=True)
        figure.savefig(
            _output_path(context, "candidate_parameter_changes.png"),
            dpi=150,
            bbox_inches="tight",
        )
    if vis.display:
        plt.show()
    else:
        plt.close(figure)


def _output_path(context: CropContext, suffix: str) -> str:
    return os.path.join(
        context.output_dir,
        f"{context.task_type}_{context.state.fragment_id}_{suffix}",
    )


def vis_candidate_attraction(context: CropContext, vis: VisOptions) -> None:
    """Overlay fixed candidates, attraction paths, and final assignments."""

    state = context.state
    run: CandidateAttractionRun = state.candidate_test_run
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
        print("=== Candidate attraction prototype ===")
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
        path = _output_path(context, "candidate_attraction.jpg")
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


def vis_candidate_attraction_history(context: CropContext, vis: VisOptions) -> None:
    """Plot mean optimization traces across nontrivial matched rows."""

    run: CandidateAttractionRun = context.state.candidate_test_run
    histories = [result.history for result in run.row_results.values() if result.history]
    if not histories:
        if vis.info:
            print("No movable text boxes; optimization history is empty.")
        return
    common_length = min(map(len, histories))
    steps = np.arange(common_length)
    loss = np.mean([[row[i]["loss"] for i in range(common_length)] for row in histories], axis=0)
    real_mass = np.mean(
        [[row[i]["real_mass"] for i in range(common_length)] for row in histories], axis=0
    )
    entropy = np.mean(
        [[row[i]["entropy"] for i in range(common_length)] for row in histories], axis=0
    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, loss)
    axes[0].set_title("mean total loss")
    axes[1].plot(steps, real_mass)
    axes[1].set_title("mean mass on real candidates")
    axes[2].plot(steps, entropy)
    axes[2].set_title("mean assignment entropy")
    for axis in axes:
        axis.set_xlabel("optimization step")
        axis.grid(alpha=0.25)
    figure.suptitle("Candidate attraction annealing diagnostics")
    figure.tight_layout()
    if vis.save:
        os.makedirs(context.output_dir, exist_ok=True)
        path = _output_path(context, "candidate_attraction_history.png")
        figure.savefig(path, dpi=150, bbox_inches="tight")
        if vis.info:
            print(f"Saved: {os.path.abspath(path)}")
    if vis.display:
        plt.show()
    else:
        plt.close(figure)
