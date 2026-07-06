"""DIFT feature cache and alignment diagnostics."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .sign import Sign
from .box import Box, Boxes
from .data_source import DataSource


@dataclass
class ImageView:
    """Small RGB view over PIL, OpenCV/numpy and tensor images.

    Numpy inputs are treated as OpenCV-style BGR/BGRA by default. PIL inputs
    keep their native RGB interpretation. Callers with true RGB numpy arrays can
    create an ImageView with ``assume_bgr=False`` and pass it to
    ``DiftRuntime.featurize_image``.
    """

    src_img: Any
    pil_rgb: Image.Image
    dtype: Optional[Any] = None
    shape: Optional[Tuple[int, ...]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    @classmethod
    def from_any(cls, img: Any, *, assume_bgr: bool = True) -> "ImageView":
        if isinstance(img, ImageView):
            return img

        if isinstance(img, Image.Image):
            arr = np.asarray(img)
            return cls(
                src_img=img,
                pil_rgb=img.convert("RGB"),
                dtype=arr.dtype,
                shape=arr.shape,
                min_value=float(arr.min()) if arr.size else None,
                max_value=float(arr.max()) if arr.size else None,
            )

        if isinstance(img, torch.Tensor):
            tensor = img.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)
            arr = tensor.numpy()
        else:
            arr = np.asarray(img)

        raw_min = float(arr.min()) if arr.size else None
        raw_max = float(arr.max()) if arr.size else None
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32, copy=False)
            if arr.size and raw_min is not None and raw_max is not None:
                if 0.0 <= raw_min and raw_max <= 1.0:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            rgb = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if assume_bgr else arr
        elif arr.ndim == 3 and arr.shape[2] == 4:
            code = cv2.COLOR_BGRA2RGB if assume_bgr else cv2.COLOR_RGBA2RGB
            rgb = cv2.cvtColor(arr, code)
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        return cls(
            src_img=img,
            pil_rgb=Image.fromarray(rgb).convert("RGB"),
            dtype=arr.dtype,
            shape=arr.shape,
            min_value=raw_min,
            max_value=raw_max,
        )

    def as_pil(self) -> Image.Image:
        return self.pil_rgb

    def as_rgb_numpy(self) -> np.ndarray:
        return np.asarray(self.pil_rgb)

    def as_bgr_numpy(self) -> np.ndarray:
        return cv2.cvtColor(self.as_rgb_numpy(), cv2.COLOR_RGB2BGR)

    def as_gray_numpy(self) -> np.ndarray:
        return cv2.cvtColor(self.as_rgb_numpy(), cv2.COLOR_RGB2GRAY)

    def as_tensor_chw(self) -> torch.Tensor:
        return torch.from_numpy(self.as_rgb_numpy()).permute(2, 0, 1)


@dataclass(frozen=True)
class DiftMatchConfig:
    max_matches: int = 300
    min_matches: int = 3
    min_support: int = 12
    ransac_threshold: Optional[float] = None


@dataclass
class DiftAlignmentConfig:
    feature_viz_max_signs: int = 12
    affine_probe_iteration: Optional[int] = 10
    affine_probe_padding_ratio: float = 0.2
    affine_probe_max_boxes: Optional[int] = None
    affine_probe_thumb: int = 128
    source_overlay_max_boxes: Optional[int] = None
    manual_match_max_lines: int = 80
    manual_match_thumb: int = 360
    match: DiftMatchConfig = field(default_factory=DiftMatchConfig)


@dataclass
class DiftMatchResult:
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
    sim_withoutbg: float
    geometry_score: float
    support_score: float
    score: float
    coarse_score: float
    message: str = ""


@dataclass
class DiftRuntime:
    checkpoint: str
    feature_dir: Optional[str] = None
    config: DiftAlignmentConfig = field(default_factory=DiftAlignmentConfig)
    img_size: int = 512
    prompt: str = ""
    source: Optional[DataSource] = None
    dtype: torch.dtype = torch.float16
    feature_cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    sd_featurizer: Any = field(init=False, repr=False)
    dift_wrapper: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from .dift_src.dift import DiftWrapper, SDFeaturizer
        self.sd_featurizer = SDFeaturizer(sd_id=self.checkpoint)
        self.dift_wrapper = DiftWrapper(
            Namespace(prompt=self.prompt, img_size=self.img_size),
            dift=self.sd_featurizer,
        )

    def get_sign_feature(
        self,
        sign: Sign,
        period: str,
    ) -> Optional[torch.Tensor]:
        source = self.source
        if source is None:
            raise ValueError("DiftRuntime.source must be set")
        sid = self._sign_id(sign, period, source)

        feature_map = self.feature_cache.get(sid)
        if feature_map is not None:
            return feature_map

        feature_map = self.load_from_disk(sid)
        if feature_map is not None:
            self.feature_cache[sid] = feature_map
            return feature_map

        image = source.get(sign.name, period)
        if image is None:
            return None

        feature_map = self.featurize_image(image).detach().cpu().to(dtype=self.dtype)
        self.feature_cache[sid] = feature_map
        self.save_to_disk(sid, feature_map)
        return feature_map

    def featurize_image(self, image: Any) -> torch.Tensor:
        image_view = ImageView.from_any(image)
        with torch.no_grad():
            return self.dift_wrapper.featurize(image_view.as_pil()).detach()

    def match(
        self,
        src_feature: torch.Tensor,
        dst_feature: torch.Tensor,
        src_img_shape: Tuple[int, int],
        dst_img_shape: Tuple[int, int],
        config: Optional[DiftMatchConfig] = None,
        src_foreground_mask: Optional[np.ndarray] = None,
    ) -> DiftMatchResult:
        """Match local features and score semantic plus geometric agreement."""
        config = config or self.config.match
        src = src_feature.detach().float()
        dst = dst_feature.detach().float()
        device = dst.device
        src = src.to(device=device)

        c = src.shape[0]
        src_flat = src.reshape(c, -1)
        dst_flat = dst.reshape(c, -1)
        sim = src_flat.T @ dst_flat
        global_similarity = 0.5 * (
            sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean()
        )
        sim_withoutbg = _foreground_only_global_similarity(
            sim,
            src_foreground_mask,
            fallback=global_similarity,
        )

        src_to_dst = sim.argmax(dim=1)
        dst_to_src = sim.argmax(dim=0)
        src_idx = torch.arange(sim.shape[0], device=device)
        is_mutual = dst_to_src[src_to_dst] == src_idx
        mutual_src = src_idx[is_mutual]
        mutual_dst = src_to_dst[is_mutual]
        mutual_sim = sim[mutual_src, mutual_dst]

        if mutual_src.numel() > config.max_matches:
            vals, keep = torch.topk(mutual_sim, k=config.max_matches)
            mutual_src = mutual_src[keep]
            mutual_dst = mutual_dst[keep]
            mutual_sim = vals

        src_points = _feature_indices_to_image_points(
            mutual_src.detach().cpu().numpy(),
            src.shape[1],
            src.shape[2],
            src_img_shape[0],
            src_img_shape[1],
        )
        dst_points = _feature_indices_to_image_points(
            mutual_dst.detach().cpu().numpy(),
            dst.shape[1],
            dst.shape[2],
            dst_img_shape[0],
            dst_img_shape[1],
        )
        similarities = mutual_sim.detach().cpu().numpy()
        global_similarity_score = float(global_similarity.detach().cpu())
        sim_withoutbg_score = float(sim_withoutbg.detach().cpu())

        def finish(
            *,
            affine: Optional[np.ndarray] = None,
            inlier_mask: Optional[np.ndarray] = None,
            message: str = "",
        ) -> DiftMatchResult:
            n_matches = len(src_points)
            if inlier_mask is None:
                mask = np.zeros(n_matches, dtype=bool)
            else:
                mask = np.asarray(inlier_mask, dtype=bool)
            n_inliers = int(mask.sum())
            mean_similarity = (
                float(similarities[mask].mean()) if n_inliers else 0.0
            )
            semantic = float(np.clip(mean_similarity, 0.0, 1.0))
            global_score = float(np.clip(global_similarity_score, 0.0, 1.0))
            no_bg_score = float(np.clip(sim_withoutbg_score, 0.0, 1.0))
            geometry = float(n_inliers / n_matches) if n_matches else 0.0
            support = float(min(1.0, n_inliers / max(1, config.min_support)))
            coarse = float(geometry * support)
            score = float(semantic * np.sqrt(coarse))
            return DiftMatchResult(
                affine=affine,
                src_points=src_points,
                dst_points=dst_points,
                similarities=similarities,
                inlier_mask=mask,
                n_matches=n_matches,
                n_inliers=n_inliers,
                mean_inlier_similarity=mean_similarity,
                semantic_score=semantic,
                global_similarity_score=global_score,
                sim_withoutbg=no_bg_score,
                geometry_score=geometry,
                support_score=support,
                score=score,
                coarse_score=coarse,
                message=message,
            )

        n_matches = len(src_points)
        if n_matches < config.min_matches:
            return finish(
                message=f"need at least {config.min_matches} mutual matches"
            )

        threshold = config.ransac_threshold
        if threshold is None:
            threshold = max(3.0, 0.06 * max(dst_img_shape))

        src32 = src_points.astype(np.float32)
        dst32 = dst_points.astype(np.float32)
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
            return finish(message="affine RANSAC failed")

        inlier_mask = (
            inliers.ravel().astype(bool)
            if inliers is not None
            else np.ones(n_matches, dtype=bool)
        )
        return finish(affine=affine, inlier_mask=inlier_mask)

    def _sign_id(self, sign: Sign, period: str, source: DataSource) -> str:
        return f"{source.key()}/{period}/{sign.name}"

    def _make_disk_path(self, sid: str) -> Optional[Path]:
        if not self.feature_dir:
            return None
        feature_dir = Path(self.feature_dir).expanduser()
        feature_dir.mkdir(parents=True, exist_ok=True)
        readable = re.sub(r"[^\w.@+-]+", "_", sid, flags=re.UNICODE).strip("_")
        readable = readable[:160] or "value"
        digest = hashlib.sha1(sid.encode("utf-8")).hexdigest()[:10]
        return feature_dir / f"{readable}_{digest}.pt"

    def load_from_disk(self, sid: str) -> Optional[torch.Tensor]:
        path = self._make_disk_path(sid)
        if path is None or not path.exists():
            return None
        feature = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(feature, torch.Tensor):
            return None
        return feature.to(dtype=self.dtype)

    def save_to_disk(
        self,
        sid: str,
        feature: torch.Tensor,
    ) -> None:
        path = self._make_disk_path(sid)
        if path is None:
            return
        torch.save(feature.detach().cpu().to(dtype=self.dtype), path)


@dataclass
class DiftAffineProbeResult:
    sign_box: Box
    crop_img: np.ndarray
    crop_offset: Tuple[int, int]
    padded_bbox: Tuple[int, int, int, int]
    source_img: Optional[np.ndarray] = None
    match: Optional[DiftMatchResult] = None
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
class SignOverlay:
    iteration: int
    image: np.ndarray
    stats: Dict[str, Any]


@dataclass
class DiftAffineProbe:
    iteration: int
    boxes: Boxes
    results: List[DiftAffineProbeResult]


def source_feature_norm_image(feature_map: torch.Tensor) -> np.ndarray:
    fm = feature_map.detach().float().cpu()
    feat = torch.linalg.vector_norm(fm, dim=0).numpy()
    return _normalize01(feat)


def source_feature_overlay(
    image: np.ndarray,
    feature_map: torch.Tensor,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    heat = source_feature_norm_image(feature_map)
    heat = cv2.resize(heat, (gray.shape[1], gray.shape[0]),
                      interpolation=cv2.INTER_CUBIC)
    heat_bgr = cv2.applyColorMap((heat * 255).astype(np.uint8),
                                 cv2.COLORMAP_VIRIDIS)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(gray_bgr, 0.45, heat_bgr, 0.55, 0)


def source_foreground_mask(
    image: np.ndarray,
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """Return a source-sign foreground mask at feature-grid resolution."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    grid_h, grid_w = map(int, grid_shape)
    if grid_h <= 0 or grid_w <= 0:
        return np.zeros((0, 0), dtype=bool)
    resized = cv2.resize(
        gray.astype(np.uint8),
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA,
    )
    return resized < 255


def render_source_feature_grid(
    rows: List[Tuple[str, np.ndarray, torch.Tensor]],
    thumb: int = 120,
) -> np.ndarray:
    """Render [source image | feature norm | overlay] rows for signs."""
    grid_rows = []
    for name, image, feature_map in rows:
        grid_rows.append([
            (name[:18], _to_bgr(image)),
            ("DIFT feature norm", _heat_to_bgr(source_feature_norm_image(feature_map))),
            ("overlay", source_feature_overlay(image, feature_map)),
        ])
    return _render_grid(grid_rows, thumb, "No source features", header_h=24)


def collect_detected_source_feature_rows(
    boxes: Boxes,
    runtime: DiftRuntime,
    period: str,
    max_signs: int = 12,
) -> Tuple[List[Tuple[str, np.ndarray, torch.Tensor]], List[str], int]:
    source = runtime.source
    if source is None:
        raise ValueError("DiftRuntime.source must be set")
    detected = {}
    for sb in boxes:
        detected.setdefault(sb.sign_name, sb.sign)

    rows: List[Tuple[str, np.ndarray, torch.Tensor]] = []
    missing: List[str] = []
    for name, sign in detected.items():
        image = source.get(sign.name, period)
        feature_map = runtime.get_sign_feature(sign, period)
        if image is None or feature_map is None:
            missing.append(name)
            continue
        rows.append((name, image, feature_map))
        if len(rows) >= max_signs:
            break
    return rows, missing, len(detected)


def render_source_sign_overlay(
    image: np.ndarray,
    boxes: Boxes,
    runtime: DiftRuntime,
    period: str,
    max_boxes: Optional[int] = None,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Paste source sign images into the current sign boxes."""
    source = runtime.source
    if source is None:
        raise ValueError("DiftRuntime.source must be set")
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
        source_img = source.get(sb.sign_name, period)
        if source_img is None:
            stats["missing"] += 1
            missing_names.append(sb.sign_name)
            if draw_boxes:
                _draw_sign_box(overlay, sb, color=(140, 140, 140),
                               draw_label=draw_labels)
            continue

        pasted = _paste_source_into_box(overlay, source_img, sb)
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
    runtime: DiftRuntime,
    period: str,
    config: DiftAlignmentConfig,
) -> List[DiftAffineProbeResult]:
    """Estimate source-to-crop affine transforms for current sign boxes."""
    source = runtime.source
    if source is None:
        raise ValueError("DiftRuntime.source must be set")
    results: List[DiftAffineProbeResult] = []
    if config.affine_probe_max_boxes is not None:
        boxes = boxes[:config.affine_probe_max_boxes]

    for sb in boxes:
        padded_bbox = sb.crop_bounds(config.affine_probe_padding_ratio)
        crop = sb.crop_image(config.affine_probe_padding_ratio)
        offset = padded_bbox[:2]
        source_img = source.get(sb.sign_name, period)
        source_feature = runtime.get_sign_feature(sb.sign, period)
        if source_img is None or source_feature is None:
            results.append(DiftAffineProbeResult(
                sign_box=sb.copy(),
                crop_img=crop,
                crop_offset=offset,
                padded_bbox=padded_bbox,
                message="missing source",
            ))
            continue

        crop_feature = runtime.featurize_image(crop)
        match = runtime.match(
            source_feature,
            crop_feature,
            source_img.shape[:2],
            crop.shape[:2],
            config.match,
            src_foreground_mask=source_foreground_mask(
                source_img,
                source_feature.shape[-2:],
            ),
        )
        results.append(DiftAffineProbeResult(
            sign_box=sb.copy(),
            crop_img=crop,
            crop_offset=offset,
            padded_bbox=padded_bbox,
            source_img=source_img,
            match=match,
            message=match.message,
        ))
    return results


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
        if res.affine_full is None or res.source_img is None:
            continue
        warped = _transformed_image_corners(
            res.source_img.shape[:2], res.affine_full
        )
        cv2.polylines(overlay, [warped.astype(np.int32)], isClosed=True,
                      color=_TRANSFORMED_BBOX_COLOR,
                      thickness=2, lineType=cv2.LINE_AA)
        before_center = _as_int_point((res.sign_box.cx, res.sign_box.cy))
        after_center = _as_int_point(
            _transformed_image_center(res.source_img.shape[:2], res.affine_full)
        )
        _draw_dashed_line(overlay, before_center, after_center, _CENTER_LINK_COLOR,
                          thickness=2)
        _draw_center_marker(overlay, before_center, _OPTIMIZED_BBOX_COLOR)
        _draw_center_marker(overlay, after_center, _TRANSFORMED_BBOX_COLOR)

    grid_rows = []
    for res in results:
        crop_vis = _draw_local_bbox(res.crop_img, res.sign_box, res.crop_offset)
        source_vis = _to_bgr(res.source_img) if res.source_img is not None else None
        warp_vis = _warp_overlay(res)
        status = (
            f"{res.sign_name[:10]} {res.n_inliers}/{res.n_matches} "
            f"{res.mean_inlier_similarity:.2f}"
            if res.affine is not None
            else f"{res.sign_name[:10]} {res.message}"
        )
        grid_rows.append([
            (f"crop iter {iteration}", crop_vis),
            ("source", source_vis),
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
    source_img: np.ndarray,
    crop_img: np.ndarray,
    result: DiftMatchResult,
    thumb: int = 360,
    max_lines: int = 80,
) -> np.ndarray:
    """Render source-to-crop mutual matches, highlighting RANSAC inliers."""
    header_h = 92
    gap = 24
    margin = 12
    canvas_w = margin * 2 + thumb * 2 + gap
    canvas_h = header_h + thumb + margin
    canvas = np.full((canvas_h, canvas_w, 3), 28, np.uint8)

    left, left_scale, left_offset = _fit_to_square_with_geometry(
        source_img, thumb
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
            f"global={result.global_similarity_score:.3f}  "
            f"sim_withoutbg={result.sim_withoutbg:.3f}"
        ),
        48,
        x=margin,
        scale=0.45,
    )
    _draw_text(
        canvas,
        (
            f"green=inlier, red=outlier  "
            f"inliers={result.n_inliers}/{result.n_matches}"
        ),
        72,
        x=margin,
        scale=0.45,
    )
    _draw_text(canvas, "source", header_h - 7, x=left_x, scale=0.43)
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
    source_img: np.ndarray,
    crop_img: np.ndarray,
    result: DiftMatchResult,
) -> np.ndarray:
    """Overlay the affine-warped source image on the selected crop."""
    crop = _to_bgr(crop_img)
    if result.affine is None:
        return crop.copy()

    return _blend_affine(source_img, crop, result.affine)


def _display_match_indices(
    result: DiftMatchResult,
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


def _foreground_only_global_similarity(
    sim: torch.Tensor,
    src_foreground_mask: Optional[np.ndarray],
    fallback: torch.Tensor,
) -> torch.Tensor:
    if src_foreground_mask is None:
        return fallback

    mask = torch.as_tensor(
        np.asarray(src_foreground_mask, dtype=bool).reshape(-1),
        device=sim.device,
        dtype=torch.bool,
    )
    if mask.numel() != sim.shape[0] or not bool(mask.any()):
        return fallback

    foreground_sim = sim[mask]
    return 0.5 * (
        foreground_sim.max(dim=1).values.mean()
        + foreground_sim.max(dim=0).values.mean()
    )


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
    if res.source_img is None or res.affine is None:
        return _draw_local_bbox(crop, res.sign_box, res.crop_offset)
    out = _blend_affine(res.source_img, crop, res.affine)
    _draw_local_bbox_inplace(out, res.sign_box, res.crop_offset,
                             _OPTIMIZED_BBOX_COLOR)
    warped_corners = _transformed_image_corners(
        res.source_img.shape[:2], res.affine
    )
    cv2.polylines(out, [warped_corners.astype(np.int32)], isClosed=True,
                  color=_TRANSFORMED_BBOX_COLOR, thickness=2,
                  lineType=cv2.LINE_AA)
    before_center = _as_int_point(
        (res.sign_box.cx - res.crop_offset[0], res.sign_box.cy - res.crop_offset[1])
    )
    after_center = _as_int_point(
        _transformed_image_center(res.source_img.shape[:2], res.affine)
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


def _paste_source_into_box(
    out: np.ndarray,
    source_img: np.ndarray,
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

    source_bgr = _to_bgr(source_img)
    patch = cv2.resize(source_bgr, (box_w, box_h),
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
    return ImageView.from_any(img).as_bgr_numpy()


def _normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mn = float(arr.min()) if arr.size else 0.0
    ptp = float(np.ptp(arr)) if arr.size else 0.0
    return (arr - mn) / (ptp + 1e-6)
