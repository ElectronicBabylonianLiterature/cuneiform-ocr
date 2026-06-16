"""DIFT feature cache and alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .sign import Sign, SignResolver
from .box import Box, Boxes
from .data_source import CanonicalSignSource
from .dift_model import DiftModel


@dataclass
class CanonicalFeatureRecord:
    feature_map: torch.Tensor          # (C, 64, 64), L2-normalized on C
    img: np.ndarray                    # original canonical image (uint8 gray)
    img_size: Tuple[int, int]          # (H, W) of canonical image
    sign_id: str
    sign_name: str


@dataclass(frozen=True)
class DiftMatchConfig:
    max_matches: int = 300
    min_matches: int = 3
    min_support: int = 12
    ransac_threshold: Optional[float] = None


class CanonicalFeatureCache:
    def __init__(
        self,
        source: CanonicalSignSource,
        wrapper,
        disk_dir: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.source = source
        self.wrapper = wrapper
        self.dtype = dtype
        self.disk_dir = Path(disk_dir).expanduser() if disk_dir else None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, CanonicalFeatureRecord] = {}

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, sign: Sign) -> Optional[CanonicalFeatureRecord]:
        sid = self.source.get_id(sign)
        if not sid:
            return None
        rec = self._cache.get(sid)
        if rec is not None:
            return rec

        rec = self._load_from_disk(sign, sid)
        if rec is None:
            rec = self._featurize_sign(sign, sid)
            self._save_to_disk(sign, rec)
        self._cache[sid] = rec
        return rec

    def _disk_path(self, sign: Sign) -> Optional[Path]:
        if self.disk_dir is None:
            return None
        stem = self.source.cache_file_stem(sign)
        if not stem:
            return None
        return self.disk_dir / f"{stem}.pt"

    def _load_from_disk(self, sign: Sign, sid: str) -> Optional[CanonicalFeatureRecord]:
        path = self._disk_path(sign)
        if path is None or not path.exists():
            return None
        data = torch.load(path, map_location="cpu", weights_only=False)
        feature_map = data["feature_map"].to(dtype=self.dtype)
        img = data["img"]
        if not isinstance(img, np.ndarray):
            img = np.asarray(img, dtype=np.uint8)
        return CanonicalFeatureRecord(
            feature_map=feature_map,
            img=img,
            img_size=tuple(data.get("img_size", img.shape[:2])),
            sign_id=data.get("sign_id", sid),
            sign_name=data.get("sign_name", sign.name),
        )

    def _save_to_disk(self, sign: Sign, rec: CanonicalFeatureRecord) -> None:
        path = self._disk_path(sign)
        if path is None:
            return
        payload = {
            "feature_map": rec.feature_map.detach().cpu().to(dtype=self.dtype),
            "img": rec.img,
            "img_size": rec.img_size,
            "sign_id": rec.sign_id,
            "sign_name": rec.sign_name,
        }
        torch.save(payload, path)

    def _featurize_sign(self, sign: Sign, sid: str) -> CanonicalFeatureRecord:
        img = self.source.get_image(sign)
        if img is None:
            raise KeyError(f"{sign.name} has no canonical image")
        return CanonicalFeatureRecord(
            feature_map=self.featurize(img).cpu().to(dtype=self.dtype),
            img=img, img_size=img.shape[:2],
            sign_id=sid, sign_name=sign.name,
        )

    def featurize(self, image: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.wrapper.featurize(
                Image.fromarray(_to_rgb(image))
            ).detach()

    def match(
        self,
        record: CanonicalFeatureRecord,
        image: np.ndarray,
        config: DiftMatchConfig,
    ) -> "DiftCropMatchResult":
        return self.match_features(
            record,
            self.featurize(image),
            image.shape[:2],
            config,
        )

    def match_features(
        self,
        record: CanonicalFeatureRecord,
        feature_map: torch.Tensor,
        image_shape: Tuple[int, int],
        config: DiftMatchConfig,
    ) -> "DiftCropMatchResult":
        return match_dift_features(
            record.feature_map,
            feature_map,
            record.img.shape[:2],
            image_shape,
            config,
        )


@dataclass
class DiftAlignmentConfig:
    feature_viz_max_signs: int = 12
    affine_probe_iteration: Optional[int] = 10
    affine_probe_padding_ratio: float = 0.2
    affine_probe_max_boxes: Optional[int] = None
    affine_probe_thumb: int = 128
    canonical_overlay_max_boxes: Optional[int] = None
    manual_match_max_lines: int = 80
    manual_match_thumb: int = 360
    match: DiftMatchConfig = field(default_factory=DiftMatchConfig)


@dataclass
class CanonicalFeatureSet:
    source: CanonicalSignSource
    cache: CanonicalFeatureCache
    period: str


@dataclass
class DiftRuntime:
    model: DiftModel
    source: CanonicalSignSource
    feature_dir: Optional[str] = None
    config: DiftAlignmentConfig = field(default_factory=DiftAlignmentConfig)
    _sources: Dict[str, CanonicalSignSource] = field(
        default_factory=dict, init=False, repr=False
    )
    _caches: Dict[str, CanonicalFeatureCache] = field(
        default_factory=dict, init=False, repr=False
    )

    def setup(self, period: str) -> CanonicalFeatureSet:
        source = self._sources.get(period)
        if source is None:
            source = self.source.for_period(period)
            self._sources[period] = source

        key = source.cache_namespace()
        cache = self._caches.get(key)
        if cache is None:
            cache = CanonicalFeatureCache(
                source,
                self.model.make_wrapper(),
                disk_dir=self.feature_dir,
            )
            self._caches[key] = cache

        return CanonicalFeatureSet(
            source=source,
            cache=cache,
            period=period,
        )


@dataclass
class DiftAffineProbeResult:
    sign_box: Box
    crop_img: np.ndarray
    crop_offset: Tuple[int, int]
    padded_bbox: Tuple[int, int, int, int]
    canonical_img: Optional[np.ndarray] = None
    match: Optional["DiftCropMatchResult"] = None
    message: str = ""

    @property
    def sign_name(self) -> str:
        return self.sign_box.sign_name

    @property
    def affine(self) -> Optional[np.ndarray]:
        return self.match.affine if self.match else None

    @property
    def affine_full(self) -> Optional[np.ndarray]:
        return _affine_with_offset(self.affine, self.crop_offset)

    @property
    def n_matches(self) -> int:
        return self.match.n_matches if self.match else 0

    @property
    def n_inliers(self) -> int:
        return self.match.n_inliers if self.match else 0

    @property
    def mean_inlier_similarity(self) -> float:
        return self.match.mean_inlier_similarity if self.match else 0.0


@dataclass
class DiftCropMatchResult:
    """Local DIFT matches and an interpretable, uncalibrated match score."""

    affine: Optional[np.ndarray]
    src_points: np.ndarray
    dst_points: np.ndarray
    similarities: np.ndarray
    inlier_mask: np.ndarray
    n_matches: int
    n_inliers: int
    mean_inlier_similarity: float
    semantic_score: float
    global_similarity_score: float
    geometry_score: float
    support_score: float
    score: float
    message: str = ""

    @property
    def coarse_score(self) -> float:
        return self.geometry_score * self.support_score


@dataclass
class CanonicalOverlay:
    iteration: int
    image: np.ndarray
    stats: Dict[str, Any]


@dataclass
class DiftAffineProbe:
    iteration: int
    boxes: Boxes
    results: List[DiftAffineProbeResult]


def canonical_feature_norm_image(feature_map: torch.Tensor) -> np.ndarray:
    fm = feature_map.detach().float().cpu()
    feat = torch.linalg.vector_norm(fm, dim=0).numpy()
    return _normalize01(feat)


def canonical_feature_overlay(record: CanonicalFeatureRecord) -> np.ndarray:
    img = record.img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    heat = canonical_feature_norm_image(record.feature_map)
    heat = cv2.resize(heat, (gray.shape[1], gray.shape[0]),
                      interpolation=cv2.INTER_CUBIC)
    heat_bgr = cv2.applyColorMap((heat * 255).astype(np.uint8),
                                 cv2.COLORMAP_VIRIDIS)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(gray_bgr, 0.45, heat_bgr, 0.55, 0)


def render_canonical_feature_grid(
    rows: List[Tuple[str, CanonicalFeatureRecord]],
    thumb: int = 120,
) -> np.ndarray:
    """Render [canonical | feature norm | overlay] rows for canonical signs."""
    grid_rows = []
    for name, rec in rows:
        grid_rows.append([
            (name[:18], _to_bgr(rec.img)),
            ("DIFT feature norm", _heat_to_bgr(canonical_feature_norm_image(rec.feature_map))),
            ("overlay", canonical_feature_overlay(rec)),
        ])
    return _render_grid(grid_rows, thumb, "No canonical features", header_h=24)


def collect_detected_canonical_feature_rows(
    boxes: Boxes,
    cache: CanonicalFeatureCache,
    max_signs: int = 12,
) -> Tuple[List[Tuple[str, CanonicalFeatureRecord]], List[str], int]:
    detected_names = list(dict.fromkeys(
        sb.sign_name for sb in boxes
    ))
    rows: List[Tuple[str, CanonicalFeatureRecord]] = []
    missing: List[str] = []
    for name in detected_names:
        rec = cache.get(SignResolver.from_name(name))
        if rec is None:
            missing.append(name)
            continue
        rows.append((name, rec))
        if len(rows) >= max_signs:
            break
    return rows, missing, len(detected_names)


def render_canonical_sign_overlay(
    image: np.ndarray,
    boxes: Boxes,
    cache: CanonicalFeatureCache,
    max_boxes: Optional[int] = None,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Paste canonical sign images into the current sign boxes."""
    overlay = _to_bgr(image).copy()
    stats: Dict[str, Any] = {
        "total": 0,
        "pasted": 0,
        "missing": 0,
        "skipped": 0,
        "missing_names": [],
    }

    if max_boxes is not None:
        boxes = boxes[:max_boxes]

    missing_names = []
    for sb in boxes:
        stats["total"] += 1
        rec = cache.get(sb.sign)
        if rec is None:
            stats["missing"] += 1
            missing_names.append(sb.sign_name)
            if draw_boxes:
                _draw_sign_box(overlay, sb, color=(140, 140, 140),
                               draw_label=draw_labels)
            continue

        pasted = _paste_canonical_into_box(overlay, rec.img, sb)
        if pasted:
            stats["pasted"] += 1
        else:
            stats["skipped"] += 1

        if draw_boxes:
            _draw_sign_box(overlay, sb, color=_OPTIMIZED_BBOX_COLOR,
                           draw_label=draw_labels)

    stats["missing_names"] = list(dict.fromkeys(missing_names))
    return overlay, stats


def build_dift_affine_probe(
    boxes: Boxes,
    cache: CanonicalFeatureCache,
    config: DiftAlignmentConfig,
) -> List[DiftAffineProbeResult]:
    """Estimate canonical-to-crop affine transforms for current sign boxes."""
    results: List[DiftAffineProbeResult] = []
    if config.affine_probe_max_boxes is not None:
        boxes = boxes[:config.affine_probe_max_boxes]

    for sb in boxes:
        padded_bbox = sb.crop_bounds(config.affine_probe_padding_ratio)
        crop = sb.crop_image(config.affine_probe_padding_ratio)
        offset = padded_bbox[:2]
        rec = cache.get(sb.sign)
        if rec is None:
            results.append(DiftAffineProbeResult(
                sign_box=sb.copy(),
                crop_img=crop,
                crop_offset=offset,
                padded_bbox=padded_bbox,
                message="missing canonical",
            ))
            continue

        match = cache.match(rec, crop, config.match)
        results.append(DiftAffineProbeResult(
            sign_box=sb.copy(),
            crop_img=crop,
            crop_offset=offset,
            padded_bbox=padded_bbox,
            canonical_img=rec.img,
            match=match,
            message=match.message,
        ))
    return results


def match_dift_features(
    src_ft: torch.Tensor,
    dst_ft: torch.Tensor,
    src_img_shape: Tuple[int, int],
    dst_img_shape: Tuple[int, int],
    config: DiftMatchConfig = DiftMatchConfig(),
) -> DiftCropMatchResult:
    """Match local features and score semantic plus geometric agreement.

    The score is not a calibrated probability. It combines mean cosine
    similarity of affine inliers with their ratio among mutual matches and a
    small-match penalty:

        score = semantic * sqrt(geometry * support)
    """
    src_pts, dst_pts, sims, global_similarity = _best_buddies_points(
        src_ft,
        dst_ft,
        src_img_shape,
        dst_img_shape,
        max_matches=config.max_matches,
    )
    n_matches = len(src_pts)
    if n_matches < config.min_matches:
        return _crop_match_result(
            src_pts,
            dst_pts,
            sims,
            global_similarity_score=global_similarity,
            message=f"need at least {config.min_matches} mutual matches",
            min_support=config.min_support,
        )

    threshold = config.ransac_threshold
    if threshold is None:
        threshold = max(3.0, 0.06 * max(dst_img_shape))

    src32 = src_pts.astype(np.float32)
    dst32 = dst_pts.astype(np.float32)
    affine = inliers = None
    for estimator in (cv2.estimateAffine2D, cv2.estimateAffinePartial2D):
        affine, inliers = estimator(
            src32,
            dst32,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(threshold),
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if affine is not None:
            break
    if affine is None:
        return _crop_match_result(
            src_pts,
            dst_pts,
            sims,
            global_similarity_score=global_similarity,
            message="affine RANSAC failed",
            min_support=config.min_support,
        )

    mask = (
        inliers.ravel().astype(bool)
        if inliers is not None
        else np.ones(n_matches, dtype=bool)
    )
    return _crop_match_result(
        src_pts,
        dst_pts,
        sims,
        global_similarity_score=global_similarity,
        affine=affine,
        inlier_mask=mask,
        min_support=config.min_support,
    )


def _crop_match_result(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    similarities: np.ndarray,
    global_similarity_score: float = 0.0,
    affine: Optional[np.ndarray] = None,
    inlier_mask: Optional[np.ndarray] = None,
    min_support: int = 12,
    message: str = "",
) -> DiftCropMatchResult:
    n_matches = len(src_points)
    if inlier_mask is None:
        inlier_mask = np.zeros(n_matches, dtype=bool)
    else:
        inlier_mask = np.asarray(inlier_mask, dtype=bool)
    n_inliers = int(inlier_mask.sum())
    mean_similarity = (
        float(similarities[inlier_mask].mean()) if n_inliers else 0.0
    )
    semantic = float(np.clip(mean_similarity, 0.0, 1.0))
    global_similarity = float(np.clip(global_similarity_score, 0.0, 1.0))
    geometry = float(n_inliers / n_matches) if n_matches else 0.0
    support = float(min(1.0, n_inliers / max(1, min_support)))
    score = float(semantic * np.sqrt(geometry * support))
    return DiftCropMatchResult(
        affine=affine,
        src_points=src_points,
        dst_points=dst_points,
        similarities=similarities,
        inlier_mask=inlier_mask,
        n_matches=n_matches,
        n_inliers=n_inliers,
        mean_inlier_similarity=mean_similarity,
        semantic_score=semantic,
        global_similarity_score=global_similarity,
        geometry_score=geometry,
        support_score=support,
        score=score,
        message=message,
    )


_OPTIMIZED_BBOX_COLOR = (0, 165, 255)      # orange, BGR
_TRANSFORMED_BBOX_COLOR = (255, 0, 255)    # magenta, BGR
_CENTER_LINK_COLOR = (235, 235, 235)


def render_dift_affine_probe(
    image: np.ndarray,
    boxes: Boxes,
    results: List[DiftAffineProbeResult],
    iteration: Optional[int],
    thumb: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (overlay image, per-box grid image) for affine diagnostics."""
    overlay = _to_bgr(image).copy()
    for sb in boxes:
        _draw_sign_box(overlay, sb, _OPTIMIZED_BBOX_COLOR,
                       draw_label=True, thickness=2)
    for res in results:
        if res.affine_full is None or res.canonical_img is None:
            continue
        warped = _transformed_image_corners(
            res.canonical_img.shape[:2], res.affine_full
        )
        cv2.polylines(overlay, [warped.astype(np.int32)], isClosed=True,
                      color=_TRANSFORMED_BBOX_COLOR,
                      thickness=2, lineType=cv2.LINE_AA)
        before_center = _as_int_point((res.sign_box.cx, res.sign_box.cy))
        after_center = _as_int_point(
            _transformed_image_center(res.canonical_img.shape[:2], res.affine_full)
        )
        _draw_dashed_line(overlay, before_center, after_center, _CENTER_LINK_COLOR,
                          thickness=2)
        _draw_center_marker(overlay, before_center, _OPTIMIZED_BBOX_COLOR)
        _draw_center_marker(overlay, after_center, _TRANSFORMED_BBOX_COLOR)

    grid_rows = []
    for res in results:
        crop_vis = _draw_local_bbox(res.crop_img, res.sign_box, res.crop_offset)
        canon_vis = _to_bgr(res.canonical_img) if res.canonical_img is not None else None
        warp_vis = _warp_overlay(res)
        status = (
            f"{res.sign_name[:10]} {res.n_inliers}/{res.n_matches} "
            f"{res.mean_inlier_similarity:.2f}"
            if res.affine is not None
            else f"{res.sign_name[:10]} {res.message}"
        )
        grid_rows.append([
            (f"crop iter {iteration}", crop_vis),
            ("canonical", canon_vis),
            (status, warp_vis),
        ])

    grid = _render_grid(
        grid_rows,
        thumb,
        f"No DIFT affine probe results at iter {iteration}",
        header_h=30,
        bg=25,
        cell_bg=34,
    )
    return overlay, grid


def render_dift_feature_matches(
    canonical_img: np.ndarray,
    crop_img: np.ndarray,
    result: DiftCropMatchResult,
    thumb: int = 360,
    max_lines: int = 80,
) -> np.ndarray:
    """Render canonical-to-crop mutual matches, highlighting RANSAC inliers."""
    header_h = 72
    gap = 24
    margin = 12
    canvas_w = margin * 2 + thumb * 2 + gap
    canvas_h = header_h + thumb + margin
    canvas = np.full((canvas_h, canvas_w, 3), 28, np.uint8)

    left, left_scale, left_offset = _fit_to_square_with_geometry(
        canonical_img, thumb
    )
    right, right_scale, right_offset = _fit_to_square_with_geometry(
        crop_img, thumb
    )
    left_x = margin
    right_x = margin + thumb + gap
    canvas[header_h:header_h + thumb, left_x:left_x + thumb] = left
    canvas[header_h:header_h + thumb, right_x:right_x + thumb] = right

    _draw_text(
        canvas,
        (
            f"score={result.score:.3f}  semantic={result.semantic_score:.3f}  "
            f"geometry={result.geometry_score:.3f}  support={result.support_score:.3f}"
        ),
        24,
        x=margin,
        scale=0.5,
    )
    _draw_text(
        canvas,
        (
            f"green=inlier, red=outlier  "
            f"inliers={result.n_inliers}/{result.n_matches}"
        ),
        50,
        x=margin,
        scale=0.45,
    )
    _draw_text(canvas, "canonical", header_h - 7, x=left_x, scale=0.43)
    _draw_text(canvas, "selected crop", header_h - 7, x=right_x, scale=0.43)

    indices = _display_match_indices(result, max_lines)
    for idx in indices:
        src_x, src_y = result.src_points[idx]
        dst_x, dst_y = result.dst_points[idx]
        p1 = (
            int(round(left_x + left_offset[0] + src_x * left_scale)),
            int(round(header_h + left_offset[1] + src_y * left_scale)),
        )
        p2 = (
            int(round(right_x + right_offset[0] + dst_x * right_scale)),
            int(round(header_h + right_offset[1] + dst_y * right_scale)),
        )
        is_inlier = bool(result.inlier_mask[idx])
        color = (80, 220, 100) if is_inlier else (90, 90, 210)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, color, -1, cv2.LINE_AA)
    return canvas


def render_dift_crop_warp(
    canonical_img: np.ndarray,
    crop_img: np.ndarray,
    result: DiftCropMatchResult,
) -> np.ndarray:
    """Overlay the affine-warped canonical image on the selected crop."""
    crop = _to_bgr(crop_img)
    if result.affine is None:
        return crop.copy()

    return _blend_affine(canonical_img, crop, result.affine)


def _display_match_indices(
    result: DiftCropMatchResult,
    max_lines: int,
) -> np.ndarray:
    if result.n_matches == 0 or max_lines <= 0:
        return np.zeros(0, dtype=np.int64)

    order = np.argsort(result.similarities)[::-1]
    inliers = order[result.inlier_mask[order]]
    outliers = order[~result.inlier_mask[order]]
    max_inliers = min(len(inliers), max(1, int(max_lines * 0.75)))
    max_outliers = min(len(outliers), max_lines - max_inliers)
    if max_inliers + max_outliers < max_lines:
        max_inliers = min(len(inliers), max_lines - max_outliers)

    # Draw outliers first so accepted correspondences remain legible.
    return np.concatenate([outliers[:max_outliers], inliers[:max_inliers]])


def _best_buddies_points(
    src_ft: torch.Tensor,
    dst_ft: torch.Tensor,
    src_img_shape: Tuple[int, int],
    dst_img_shape: Tuple[int, int],
    max_matches: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Vectorized ProtoSnap best-buddies with per-image pixel coordinates.

    ProtoSnap's helper assumes one square image size for both sides and uses a
    quadratic Python loop. This equivalent keeps similarities for scoring and
    maps each feature grid to its own source image shape.
    """
    src = src_ft.detach().float()
    dst = dst_ft.detach().float()
    device = dst.device
    src = src.to(device=device)

    c = src.shape[0]
    src_flat = src.reshape(c, -1)
    dst_flat = dst.reshape(c, -1)
    sim = src_flat.T @ dst_flat
    global_similarity = 0.5 * (
        sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean()
    )

    src_to_dst = sim.argmax(dim=1)
    dst_to_src = sim.argmax(dim=0)
    src_idx = torch.arange(sim.shape[0], device=device)
    is_mutual = dst_to_src[src_to_dst] == src_idx
    mutual_src = src_idx[is_mutual]
    mutual_dst = src_to_dst[is_mutual]
    mutual_sim = sim[mutual_src, mutual_dst]

    if mutual_src.numel() > max_matches:
        vals, keep = torch.topk(mutual_sim, k=max_matches)
        mutual_src = mutual_src[keep]
        mutual_dst = mutual_dst[keep]
        mutual_sim = vals

    src_pts = _feature_indices_to_image_points(
        mutual_src.detach().cpu().numpy(),
        src.shape[1], src.shape[2],
        src_img_shape[0], src_img_shape[1],
    )
    dst_pts = _feature_indices_to_image_points(
        mutual_dst.detach().cpu().numpy(),
        dst.shape[1], dst.shape[2],
        dst_img_shape[0], dst_img_shape[1],
    )
    return src_pts, dst_pts, mutual_sim.detach().cpu().numpy(), float(global_similarity.detach().cpu())


def _feature_indices_to_image_points(
    indices: np.ndarray,
    grid_h: int,
    grid_w: int,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    ys = indices // grid_w
    xs = indices % grid_w
    pts_x = (xs + 0.5) / grid_w * img_w
    pts_y = (ys + 0.5) / grid_h * img_h
    return np.stack([pts_x, pts_y], axis=1).astype(np.float32)


def _warp_overlay(res: DiftAffineProbeResult) -> np.ndarray:
    crop = _to_bgr(res.crop_img)
    if res.canonical_img is None or res.affine is None:
        return _draw_local_bbox(crop, res.sign_box, res.crop_offset)
    out = _blend_affine(res.canonical_img, crop, res.affine)
    _draw_local_bbox_inplace(out, res.sign_box, res.crop_offset,
                             _OPTIMIZED_BBOX_COLOR)
    warped_corners = _transformed_image_corners(
        res.canonical_img.shape[:2], res.affine
    )
    cv2.polylines(out, [warped_corners.astype(np.int32)], isClosed=True,
                  color=_TRANSFORMED_BBOX_COLOR, thickness=2,
                  lineType=cv2.LINE_AA)
    before_center = _as_int_point(
        (res.sign_box.cx - res.crop_offset[0], res.sign_box.cy - res.crop_offset[1])
    )
    after_center = _as_int_point(
        _transformed_image_center(res.canonical_img.shape[:2], res.affine)
    )
    _draw_dashed_line(out, before_center, after_center, _CENTER_LINK_COLOR,
                      thickness=1)
    _draw_center_marker(out, before_center, _OPTIMIZED_BBOX_COLOR, radius=3)
    _draw_center_marker(out, after_center, _TRANSFORMED_BBOX_COLOR, radius=3)
    return out


def _blend_affine(
    source: np.ndarray,
    target: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    source = _to_bgr(source)
    target = _to_bgr(target)
    h, w = target.shape[:2]
    warped = cv2.warpAffine(source, affine, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(
        np.full(source.shape[:2], 255, np.uint8),
        affine,
        (w, h),
        flags=cv2.INTER_NEAREST,
    ) > 0
    out = target.copy()
    out[mask] = (
        target[mask].astype(np.float32) * 0.55
        + warped[mask].astype(np.float32) * 0.45
    ).astype(np.uint8)
    return out


def _affine_with_offset(
    affine: Optional[np.ndarray],
    offset: Tuple[int, int],
) -> Optional[np.ndarray]:
    if affine is None:
        return None
    full = affine.copy()
    full[:, 2] += np.array(offset, dtype=full.dtype)
    return full


def _draw_local_bbox(
    img: np.ndarray,
    sb: Box,
    offset: Tuple[int, int],
    color: Tuple[int, int, int] = _OPTIMIZED_BBOX_COLOR,
) -> np.ndarray:
    out = _to_bgr(img).copy()
    _draw_local_bbox_inplace(out, sb, offset, color)
    return out


def _draw_local_bbox_inplace(
    out: np.ndarray,
    sb: Box,
    offset: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    ox, oy = offset
    p1 = (int(round(sb.x1 - ox)), int(round(sb.y1 - oy)))
    p2 = (int(round(sb.x2 - ox)), int(round(sb.y2 - oy)))
    cv2.rectangle(out, p1, p2, color, 2, cv2.LINE_AA)


def _draw_sign_box(
    out: np.ndarray,
    sb: Box,
    color: Tuple[int, int, int],
    draw_label: bool = True,
    thickness: int = 1,
) -> None:
    p1 = (int(round(sb.x1)), int(round(sb.y1)))
    p2 = (int(round(sb.x2)), int(round(sb.y2)))
    cv2.rectangle(out, p1, p2, color, thickness, cv2.LINE_AA)
    if draw_label:
        _draw_text(out, sb.sign_name[:10], max(12, p1[1] - 5),
                   x=max(0, p1[0]), color=color, scale=0.42)


def _paste_canonical_into_box(
    out: np.ndarray,
    canonical_img: np.ndarray,
    sb: Box,
) -> bool:
    img_h, img_w = out.shape[:2]
    x1 = int(np.floor(sb.x1))
    y1 = int(np.floor(sb.y1))
    x2 = int(np.ceil(sb.x2))
    y2 = int(np.ceil(sb.y2))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return False

    clip_x1 = max(0, x1)
    clip_y1 = max(0, y1)
    clip_x2 = min(img_w, x2)
    clip_y2 = min(img_h, y2)
    if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
        return False

    canonical_bgr = _to_bgr(canonical_img)
    patch = cv2.resize(canonical_bgr, (box_w, box_h),
                       interpolation=cv2.INTER_AREA)

    px1 = clip_x1 - x1
    py1 = clip_y1 - y1
    px2 = px1 + (clip_x2 - clip_x1)
    py2 = py1 + (clip_y2 - clip_y1)

    roi = out[clip_y1:clip_y2, clip_x1:clip_x2]
    roi[:] = patch[py1:py2, px1:px2]
    return True


def _transformed_image_corners(
    img_shape: Tuple[int, int],
    affine: np.ndarray,
) -> np.ndarray:
    h, w = img_shape[:2]
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32)
    return cv2.transform(corners, affine)[0]


def _transformed_image_center(
    img_shape: Tuple[int, int],
    affine: np.ndarray,
) -> Tuple[float, float]:
    h, w = img_shape[:2]
    center = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
    pt = cv2.transform(center, affine)[0, 0]
    return float(pt[0]), float(pt[1])


def _as_int_point(pt: Tuple[float, float]) -> Tuple[int, int]:
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _draw_center_marker(
    img: np.ndarray,
    center: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int = 4,
) -> None:
    cv2.circle(img, center, radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)


def _draw_dashed_line(
    img: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash: int = 10,
    gap: int = 7,
) -> None:
    start = np.array(p1, dtype=np.float32)
    end = np.array(p2, dtype=np.float32)
    dist = float(np.linalg.norm(end - start))
    if dist < 1e-3:
        return
    direction = (end - start) / dist
    step = dash + gap
    for t in np.arange(0.0, dist, step):
        seg_start = start + direction * t
        seg_end = start + direction * min(t + dash, dist)
        cv2.line(img, _as_int_point(seg_start), _as_int_point(seg_end),
                 color, thickness, cv2.LINE_AA)


def _heat_to_bgr(heat: np.ndarray) -> np.ndarray:
    cmap = getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET)
    return cv2.applyColorMap((_normalize01(heat) * 255).astype(np.uint8),
                             cmap)


def _fit_to_square(img: Optional[np.ndarray], size: int) -> np.ndarray:
    canvas, _, _ = _fit_to_square_with_geometry(img, size)
    return canvas


def _fit_to_square_with_geometry(
    img: Optional[np.ndarray],
    size: int,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    canvas = np.full((size, size, 3), 42, np.uint8)
    if img is None:
        return canvas, 1.0, (0, 0)
    im = _to_bgr(img)
    h, w = im.shape[:2]
    if h <= 0 or w <= 0:
        return canvas, 1.0, (0, 0)
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, scale, (x0, y0)


def _blank_panel(width: int, height: int, text: str) -> np.ndarray:
    panel = np.full((height, width, 3), 30, np.uint8)
    _draw_text(panel, text, height // 2, color=(220, 220, 220), scale=0.5)
    return panel


def _render_grid(
    rows: List[List[Tuple[str, Optional[np.ndarray]]]],
    thumb: int,
    empty_text: str,
    header_h: int = 24,
    gap: int = 6,
    bg: int = 28,
    cell_bg: int = 35,
) -> np.ndarray:
    if not rows:
        return _blank_panel(thumb * 3 + 24, thumb + header_h + 18, empty_text)

    n_cols = max(len(row) for row in rows)
    cell_h = header_h + thumb
    width = gap + n_cols * (thumb + gap)
    height = gap + len(rows) * (cell_h + gap)
    grid = np.full((height, width, 3), bg, np.uint8)
    label_y = max(12, header_h - 7)

    for r, row in enumerate(rows):
        y0 = gap + r * (cell_h + gap)
        for c, (label, img) in enumerate(row):
            x0 = gap + c * (thumb + gap)
            cell = np.full((cell_h, thumb, 3), cell_bg, np.uint8)
            _draw_text(cell, label, label_y, color=(220, 220, 220), scale=0.38)
            cell[header_h:, :] = _fit_to_square(img, thumb)
            grid[y0:y0 + cell_h, x0:x0 + thumb] = cell
    return grid


def _draw_text(
    canvas: np.ndarray,
    text: str,
    y: int,
    color: Tuple[int, int, int] = (220, 220, 220),
    scale: float = 0.4,
    x: int = 4,
) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def _to_bgr(img: Optional[np.ndarray]) -> np.ndarray:
    if img is None:
        return np.zeros((1, 1, 3), np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    return arr.astype(np.uint8)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2RGB)


def _normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mn = float(arr.min()) if arr.size else 0.0
    ptp = float(np.ptp(arr)) if arr.size else 0.0
    return (arr - mn) / (ptp + 1e-6)
