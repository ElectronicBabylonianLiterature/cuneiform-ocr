"""DIFT feature cache and alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .sign import Sign, SignResolver
from .box import Box, Boxes
from .data_source import CanonicalSignSource


@dataclass
class CanonicalFeatureRecord:
    feature_map: torch.Tensor          # (C, 64, 64), L2-normalized on C
    img: np.ndarray                    # original canonical image (uint8 gray)
    img_size: Tuple[int, int]          # (H, W) of canonical image
    sign_id: str
    sign_name: str


class CanonicalFeatureCache:
    def __init__(self, source: CanonicalSignSource, wrapper, img_size: int = 512,
                 store_device: str = "cpu", store_dtype: torch.dtype = torch.float16,
                 disk_dir: Optional[str] = None):
        self.source = source
        self.wrapper = wrapper
        self.img_size = img_size
        self.store_device = store_device
        self.store_dtype = store_dtype
        self.disk_dir = Path(disk_dir).expanduser() if disk_dir else None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, CanonicalFeatureRecord] = {}
        self._ordered_ids: List[str] = []

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
        return self._remember(sid, rec)

    def precompute_all(self, verbose: bool = True,
                       limit: Optional[int] = None,
                       progress_every: int = 50) -> Dict[str, int]:
        names = self.source.list_sign_names()
        return self.precompute(names, verbose=verbose, limit=limit,
                               progress_every=progress_every)

    def precompute(self, sign_names: Iterable[str], verbose: bool = True,
                   limit: Optional[int] = None,
                   progress_every: int = 50) -> Dict[str, int]:
        names = list(dict.fromkeys(sign_names))
        if limit is not None:
            names = names[:limit]

        stats = {"total": len(names), "computed": 0, "cached": 0, "disk_cached": 0}

        for i, name in enumerate(names, start=1):
            sign = SignResolver.from_name(name)
            sid = self.source.get_id(sign)
            if sid is None:
                raise KeyError(f"{name} is listed by the source but has no source id")

            if sid in self._cache:
                stats["cached"] += 1
            else:
                rec = self._load_from_disk(sign, sid)
                if rec is not None:
                    stats["disk_cached"] += 1
                else:
                    rec = self._featurize_sign(sign, sid)
                    self._save_to_disk(sign, rec)
                    stats["computed"] += 1
                self._remember(sid, rec)

            self._report_precompute(i, len(names), stats, verbose, progress_every)
        return stats

    def _remember(self, sid: str, rec: CanonicalFeatureRecord) -> CanonicalFeatureRecord:
        self._cache[sid] = rec
        if sid not in self._ordered_ids:
            self._ordered_ids.append(sid)
        return rec

    @staticmethod
    def _report_precompute(
        i: int,
        total: int,
        stats: Dict[str, int],
        verbose: bool,
        progress_every: int,
    ) -> None:
        if verbose and (i % progress_every == 0 or i == total):
            print(f"    [precompute] {i}/{total} "
                  f"(computed={stats['computed']}, cached={stats['cached']}, "
                  f"disk={stats['disk_cached']})")

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
        try:
            data = torch.load(
                path, map_location=self.store_device, weights_only=False
            )
        except TypeError:
            data = torch.load(path, map_location=self.store_device)
        feature_map = data["feature_map"].to(
            device=self.store_device, dtype=self.store_dtype
        )
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
            "feature_map": rec.feature_map.detach().to(
                device=self.store_device, dtype=self.store_dtype
            ),
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
        with torch.no_grad():
            fm = self.wrapper.featurize(Image.fromarray(_to_rgb(img)))
        fm_cpu = fm.detach().to(device=self.store_device, dtype=self.store_dtype,
                                non_blocking=False)
        del fm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return CanonicalFeatureRecord(
            feature_map=fm_cpu,
            img=img, img_size=img.shape[:2],
            sign_id=sid, sign_name=sign.name,
        )

    @property
    def ordered_records(self) -> List[CanonicalFeatureRecord]:
        return [self._cache[sid] for sid in self._ordered_ids]

    @property
    def ready(self) -> bool:
        return len(self._cache) > 0


@dataclass
class DiftAlignmentConfig:
    precompute_limit: Optional[int] = None
    feature_viz_max_signs: int = 12
    affine_probe_iteration: Optional[int] = 10
    affine_probe_padding_ratio: float = 0.2
    affine_probe_max_boxes: Optional[int] = None
    affine_probe_max_matches: int = 300
    affine_probe_min_matches: int = 3
    affine_probe_ransac_threshold: Optional[float] = None
    affine_probe_thumb: int = 128
    canonical_overlay_max_boxes: Optional[int] = None


@dataclass
class DiftAffineProbeResult:
    sign_name: str
    sign_box: Box
    crop_img: np.ndarray
    crop_offset: Tuple[int, int]
    padded_bbox: Tuple[int, int, int, int]
    canonical_img: Optional[np.ndarray]
    affine: Optional[np.ndarray]
    affine_full: Optional[np.ndarray]
    n_matches: int
    n_inliers: int
    mean_inlier_similarity: float
    message: str = ""


def _probe_result(
    sb: Box,
    crop_img: np.ndarray,
    crop_offset: Tuple[int, int],
    padded_bbox: Tuple[int, int, int, int],
    canonical_img: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    affine_full: Optional[np.ndarray] = None,
    n_matches: int = 0,
    n_inliers: int = 0,
    mean_inlier_similarity: float = 0.0,
    message: str = "",
) -> DiftAffineProbeResult:
    return DiftAffineProbeResult(
        sign_name=sb.sign_name,
        sign_box=sb.copy(),
        crop_img=crop_img,
        crop_offset=crop_offset,
        padded_bbox=padded_bbox,
        canonical_img=canonical_img,
        affine=affine,
        affine_full=affine_full,
        n_matches=n_matches,
        n_inliers=n_inliers,
        mean_inlier_similarity=mean_inlier_similarity,
        message=message,
    )


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
    boxes: Optional[Boxes],
    cache: Optional[CanonicalFeatureCache],
    max_signs: int = 12,
) -> Tuple[List[Tuple[str, CanonicalFeatureRecord]], List[str], int]:
    if boxes is None or cache is None:
        return [], [], 0

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
    if image is None or boxes is None or cache is None:
        return overlay, stats

    if max_boxes is not None:
        boxes = boxes[:max_boxes]

    missing_names = []
    for sb in boxes:
        stats["total"] += 1
        rec = cache.get(sb.sign)
        if rec is None or rec.img is None:
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
    padding_ratio: float = 0.2,
    max_boxes: Optional[int] = None,
    max_matches: int = 300,
    min_matches: int = 3,
    ransac_threshold: Optional[float] = None,
) -> List[DiftAffineProbeResult]:
    """Estimate canonical-to-crop affine transforms for current sign boxes."""
    if boxes is None or cache is None:
        return []

    results: List[DiftAffineProbeResult] = []
    if max_boxes is not None:
        boxes = boxes[:max_boxes]

    for sb in boxes:
        crop, offset, padded_bbox = crop_sign_box(sb, padding_ratio)
        rec = cache.get(sb.sign)
        if rec is None:
            results.append(_probe_result(sb, crop, offset, padded_bbox, message="missing canonical"))
            continue

        crop_ft = _featurize_image(cache.wrapper, crop)
        affine, n_matches, n_inliers, mean_sim = estimate_dift_affine(
            rec.feature_map,
            crop_ft,
            src_img_shape=rec.img.shape[:2],
            dst_img_shape=crop.shape[:2],
            max_matches=max_matches,
            min_matches=min_matches,
            ransac_threshold=ransac_threshold,
        )
        affine_full = _affine_with_offset(affine, offset)
        results.append(_probe_result(
            sb, crop, offset, padded_bbox,
            canonical_img=rec.img,
            affine=affine,
            affine_full=affine_full,
            n_matches=n_matches,
            n_inliers=n_inliers,
            mean_inlier_similarity=mean_sim,
            message="" if affine is not None else "affine failed",
        ))
    return results


def crop_sign_box(
    sb: Box,
    padding_ratio: float,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    crop = sb.crop_image(padding_ratio)
    x1, y1, x2, y2 = sb.crop_bounds(padding_ratio)
    return crop, (x1, y1), (x1, y1, x2, y2)


def estimate_dift_affine(
    src_ft: torch.Tensor,
    dst_ft: torch.Tensor,
    src_img_shape: Tuple[int, int],
    dst_img_shape: Tuple[int, int],
    max_matches: int = 300,
    min_matches: int = 3,
    ransac_threshold: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], int, int, float]:
    """Estimate a 2x3 affine from source canonical pixels to crop pixels."""
    src_pts, dst_pts, sims = _best_buddies_points(
        src_ft, dst_ft, src_img_shape, dst_img_shape, max_matches=max_matches
    )
    n_matches = len(src_pts)
    if n_matches < min_matches:
        return None, n_matches, 0, 0.0

    threshold = ransac_threshold
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
        return None, n_matches, 0, 0.0

    mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(n_matches, dtype=bool)
    n_inliers = int(mask.sum())
    mean_sim = float(sims[mask].mean()) if n_inliers else 0.0
    return affine, n_matches, n_inliers, mean_sim


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


def _featurize_image(wrapper, img: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        return wrapper.featurize(Image.fromarray(_to_rgb(img))).detach()


def _best_buddies_points(
    src_ft: torch.Tensor,
    dst_ft: torch.Tensor,
    src_img_shape: Tuple[int, int],
    dst_img_shape: Tuple[int, int],
    max_matches: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = src_ft.detach().float()
    dst = dst_ft.detach().float()
    device = dst.device
    src = src.to(device=device)

    c = src.shape[0]
    src_flat = src.reshape(c, -1)
    dst_flat = dst.reshape(c, -1)
    sim = src_flat.T @ dst_flat

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
    return src_pts, dst_pts, mutual_sim.detach().cpu().numpy()


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
    src = _to_bgr(res.canonical_img)
    h, w = crop.shape[:2]
    warped = cv2.warpAffine(src, res.affine, (w, h), flags=cv2.INTER_LINEAR)
    mask_src = np.full(src.shape[:2], 255, np.uint8)
    mask = cv2.warpAffine(mask_src, res.affine, (w, h), flags=cv2.INTER_NEAREST) > 0
    out = crop.copy()
    out[mask] = (crop[mask].astype(np.float32) * 0.55
                 + warped[mask].astype(np.float32) * 0.45).astype(np.uint8)
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
    canvas = np.full((size, size, 3), 42, np.uint8)
    if img is None:
        return canvas
    im = _to_bgr(img)
    h, w = im.shape[:2]
    if h <= 0 or w <= 0:
        return canvas
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


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
