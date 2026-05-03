"""
ProtoSnap center-freeze refiner.

This module is the *redesigned* ProtoSnap integration. It is no longer a
"replace bbox geometry" stage; it is a **center-only correction + freeze**
component, callable either before PSR optimization (once) or as a
mid-iteration hook during PSR.

Pipeline owns *when*; this module owns *how*.

Stage decomposition (cf. TAU-VAILab/ProtoSnap, arXiv:2502.00129):
    - Stage 1 ("init"):  global affine prototype -> target via DIFT/SIFT
                         best-buddies + RANSAC. We use ONLY this stage.
    - Stage 2 ("optim"): per-stroke skeleton optimization. Discarded here
                         because PSR already optimizes (cx, cy, w, h).

What this module produces, per sign, is a `CenterRefinement`:
    - implied new (cx, cy) computed by mapping the prototype center under
      the affine and shifting from the crop frame back to sub-image coords;
    - an `accepted` flag from the multi-criteria gate (see `default_gate`);
    - diagnostic info (inliers, reproj error, scale, det, etc.).

Two estimators are provided:
    - `OpenCVAffineEstimator`  (default, no GPU/SD weights required)
    - `OfficialProtoSnapInitEstimator` (stub; wire to TAU-VAILab/ProtoSnap)

The transform estimator is a `Protocol`, so swapping is a one-line config
change once the official model is wired up.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .sign import Sign
from .tablet import SignBox, SubTablet

logger = logging.getLogger(__name__)


# ===========================================================================
# Period -> font selection
# ===========================================================================

# Two prototype styles ship with the official ProtoSnap repo:
#   - Santakku       : Old Babylonian
#   - Assurbanipal   : Neo-Assyrian
# Anything not explicitly Old-Babylonian-ish falls back to Assurbanipal.
_PERIOD_TO_FONT: Dict[str, str] = {
    "old babylonian": "Santakku",
    "old babylonian (early)": "Santakku",
    "old assyrian": "Santakku",
    "neo-assyrian": "Assurbanipal",
    "neo-babylonian": "Assurbanipal",
    "middle babylonian": "Assurbanipal",
    "middle assyrian": "Assurbanipal",
}
_DEFAULT_FONT = "Assurbanipal"


def resolve_font_for_period(period: Optional[str]) -> Tuple[str, str]:
    """Map an EBL fragment 'script.period' string to a ProtoSnap font.

    Returns
    -------
    (font, reason)
        font   : 'Santakku' or 'Assurbanipal'
        reason : human-readable string for the info log
    """
    if not period:
        return _DEFAULT_FONT, f"period missing -> default {_DEFAULT_FONT}"
    key = period.strip().lower()
    if key in _PERIOD_TO_FONT:
        font = _PERIOD_TO_FONT[key]
        return font, f'period "{period}" -> {font}'
    return _DEFAULT_FONT, f'period "{period}" not in table -> default {_DEFAULT_FONT}'


# ===========================================================================
# Result types
# ===========================================================================

class SnapStatus(str, Enum):
    OK = "ok"                          # passed all gates
    NO_PROTOTYPE = "no_prototype"      # ABZ -> Unicode missing or PNG missing
    UNCLEAR_SIGN = "unclear_sign"      # 'X' / 'NoABZ0'
    OUT_OF_BOUNDS = "out_of_bounds"    # crop fell outside sub-image
    NO_FEATURES = "no_features"        # SIFT found nothing on crop or prototype
    LOW_MATCHES = "low_matches"        # too few putative matches for RANSAC
    BAD_TRANSFORM = "bad_transform"    # gate rejected: degenerate affine / huge offset
    LOW_INLIERS = "low_inliers"        # gate rejected: inlier ratio / count too low
    DISABLED = "disabled"
    ERROR = "error"


@dataclass(frozen=True)
class InitTransform:
    """Output of Stage-1 init: a global affine prototype -> crop, plus stats."""
    affine: np.ndarray           # (2, 3) affine matrix
    inliers: int
    total_matches: int
    inlier_ratio: float
    median_reproj_err: float     # in crop pixels
    proto_shape: Tuple[int, int] # (H, W) of prototype
    crop_shape: Tuple[int, int]  # (H, W) of crop


@dataclass
class CenterRefinement:
    sign_index: int               # index into the SubTablet's flat sign_boxes
    sign_name: str
    abz: str
    status: SnapStatus
    accepted: bool = False        # gate decision; True iff we want to freeze
    new_cx: float = 0.0           # in sub-image coords
    new_cy: float = 0.0
    delta_cx: float = 0.0         # new - original
    delta_cy: float = 0.0
    transform: Optional[InitTransform] = None
    info: Dict[str, Any] = field(default_factory=dict)
    # Visual assets for per-sign debugging grid (populated by refine_one)
    crop_img: Optional[np.ndarray] = None        # crop around sign from sub-image
    proto_img: Optional[np.ndarray] = None       # prototype image
    crop_offset: Optional[Tuple[int, int]] = None  # (ox, oy): crop top-left in sub-image


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class GateConfig:
    """Multi-criteria gate (G2 + G3 + offset sanity).

    Defaults are tuned for SIFT/RANSAC on cuneiform-sized crops; tighten
    if you swap in DIFT (which produces denser, cleaner matches).
    """
    min_inliers: int = 5
    min_inlier_ratio: float = 0.30
    max_reproj_err_ratio: float = 0.25  # × max(crop_w, crop_h)
    # Affine shape: scale must be in [1/scale_max, scale_max], skew bounded,
    # det > 0 (no flip).
    scale_min: float = 0.5
    scale_max: float = 2.0
    max_skew: float = 0.5
    # Implied center offset must be at most this fraction of max(bbox_w, bbox_h)
    # — otherwise we probably matched the neighbor sign.
    max_offset_ratio: float = 0.6


@dataclass
class ProtoSnapConfig:
    """Static config for the center-freeze refiner.

    repo_root None disables the step gracefully (logs a notice, no-op).
    """
    repo_root: Optional[str] = None
    # Override font; if None we pick from the fragment's API period.
    force_font: Optional[str] = None

    # Crop padding around bbox: pad_pixels = ratio * max(bbox.w, bbox.h)
    crop_padding_ratio: float = 0.1
    # Resize crop+prototype to this size before feature matching (pixels,
    # longest side). None disables resizing.
    feature_img_size: Optional[int] = 256
    # Pixels of white re-padding added after tight-cropping the prototype PNG.
    # Mirrors the official ProtoSnap pipeline (crop white → pad 10 → resize).
    # Set to 0 to disable tight-crop entirely.
    proto_pad: int = 10

    # Estimator selection: "opencv" or "official_protosnap".
    estimator: str = "opencv"
    # When using OpenCV: SIFT (default) or ORB. SIFT is more reliable; if
    # cv2.SIFT_create raises, we fall back to ORB.
    opencv_feature: str = "sift"

    # Gate settings
    gate: GateConfig = field(default_factory=GateConfig)

    # When ProtoSnap is invoked as a mid-PSR hook, freeze at this iteration.
    # `0` reproduces "run once before optimization starts".
    snap_at_iter: int = 10

    log_failures: bool = True


# ===========================================================================
# Prototype bank
# ===========================================================================

def _tight_crop_pad(img: np.ndarray, pad: int) -> np.ndarray:
    """Replicate the official ProtoSnap prototype preprocessing.

    1. Tight-crop white borders (pixels == 255) → content-only rectangle.
    2. Re-add `pad` pixels of white on all sides.

    If the image is entirely white or `pad == 0` with no content, returns
    the original image unchanged.
    """
    if pad == 0:
        return img
    content = img != 255
    rows = np.any(content, axis=1)
    cols = np.any(content, axis=0)
    if not rows.any():
        return img  # blank image — leave as-is
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    cropped = img[r0:r1 + 1, c0:c1 + 1]
    return cv2.copyMakeBorder(cropped, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)


class PrototypeBank:
    """Resolves a Sign to its prototype PNG.

    Layout (as shipped by TAU-VAILab/ProtoSnap):
        <repo_root>/prototypes/<font>/0x120xx/0x12000.png

    Sign names are looked up via <repo_root>/prototypes/metadata.csv, which
    maps Unicode codepoints to sign names and per-font availability flags.
    """

    def __init__(self, config: ProtoSnapConfig, font: str):
        self.config = config
        self.font = font
        self._prototypes_dir: Optional[Path] = None
        self._name_to_hex: Dict[str, str] = {}  # sign_name -> "0x12000"
        self._cache: Dict[str, np.ndarray] = {}  # hex -> image

    def is_ready(self) -> bool:
        if not self.config.repo_root:
            return False
        proto_root = Path(self.config.repo_root) / "prototypes"
        proto_dir = proto_root / self.font
        if not proto_dir.exists():
            logger.warning("prototype dir missing: %s", proto_dir)
            return False
        self._prototypes_dir = proto_dir
        self._load_metadata(proto_root / "metadata.csv")
        return True

    def _load_metadata(self, csv_path: Path) -> None:
        if not csv_path.exists():
            logger.warning("metadata.csv missing: %s", csv_path)
            return
        import csv
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("name", "").strip()
                    if not name:
                        continue
                    hex_val = row.get("hex", "").strip()
                    if not hex_val:
                        continue
                    # Only include signs available in the selected font
                    if row.get(self.font, "False").strip().lower() != "true":
                        continue
                    self._name_to_hex[name] = hex_val
        except Exception as e:
            logger.warning("failed to load metadata.csv %s: %s", csv_path, e)

    def _hex_to_path(self, hex_val: str) -> Path:
        # e.g. "0x1202d" -> subdir "0x120xx", file "0x1202d.png"
        subdir = hex_val[:5] + "xx"
        return self._prototypes_dir / subdir / f"{hex_val}.png"

    def get_prototype_path(self, sign: Sign) -> Optional[Path]:
        if not self._prototypes_dir:
            return None
        hex_val = self._name_to_hex.get(sign.name)
        if hex_val is None:
            return None
        path = self._hex_to_path(hex_val)
        return path if path.exists() else None

    def load_prototype(self, sign: Sign) -> Optional[np.ndarray]:
        hex_val = self._name_to_hex.get(sign.name)
        if hex_val is None:
            return None
        if hex_val in self._cache:
            return self._cache[hex_val]
        path = self.get_prototype_path(sign)
        if path is None:
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = _tight_crop_pad(img, self.config.proto_pad)
        self._cache[hex_val] = img
        return img

    def coverage(self, sign_boxes: List[SignBox]) -> Dict[str, Any]:
        ok = 0
        missing: List[str] = []
        for sb in sign_boxes:
            if self.get_prototype_path(sb.sign) is not None:
                ok += 1
            else:
                missing.append(sb.sign_name)
        top_missing = [n for n, _ in Counter(missing).most_common(10)]
        return {
            "total": len(sign_boxes),
            "with_prototype": ok,
            "metadata_size": len(self._name_to_hex),
            "missing_sign_top10": top_missing,
        }


# ===========================================================================
# Transform estimators (Stage-1 init)
# ===========================================================================

class TransformEstimator(Protocol):
    """Estimate a global affine: prototype-frame -> crop-frame.

    Implementations must be deterministic given the same RNG seed (we set
    cv2.setRNGSeed in OpenCVAffineEstimator).
    """
    def estimate(
        self, crop: np.ndarray, prototype: np.ndarray
    ) -> Optional[InitTransform]: ...


class OpenCVAffineEstimator:
    """SIFT (or ORB) + RANSAC partial-affine.

    Uses `cv2.estimateAffinePartial2D` (4 DoF: rotation+uniform-scale+
    translation), which is robust on small crops where full 6-DoF affine
    over-fits noise. Outputs the partial affine as a 2x3 matrix.
    """

    def __init__(self, feature: str = "sift", img_size: Optional[int] = 256,
                 ransac_thr: float = 4.0, ransac_iters: int = 2000):
        self.feature = feature
        self.img_size = img_size
        self.ransac_thr = ransac_thr
        self.ransac_iters = ransac_iters
        self._detector = self._make_detector(feature)
        self._matcher = self._make_matcher(feature)
        cv2.setRNGSeed(0)

    @staticmethod
    def _make_detector(feature: str):
        if feature == "sift":
            try:
                return cv2.SIFT_create()
            except AttributeError:
                feature = "orb"  # fall through
        if feature == "orb":
            return cv2.ORB_create(nfeatures=2000)
        raise ValueError(f"unknown feature {feature}")

    @staticmethod
    def _make_matcher(feature: str):
        if feature == "sift":
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _resize(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.img_size:
            return img, 1.0
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side <= self.img_size:
            return img, 1.0
        scale = self.img_size / long_side
        new = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        return new, scale

    def estimate(self, crop: np.ndarray, prototype: np.ndarray
                 ) -> Optional[InitTransform]:
        crop_gray = self._to_gray(crop)
        proto_gray = self._to_gray(prototype)
        crop_r, s_crop = self._resize(crop_gray)
        proto_r, s_proto = self._resize(proto_gray)

        kp_p, des_p = self._detector.detectAndCompute(proto_r, None)
        kp_c, des_c = self._detector.detectAndCompute(crop_r, None)
        if des_p is None or des_c is None or len(kp_p) < 4 or len(kp_c) < 4:
            return None

        # KNN + Lowe's ratio test
        try:
            knn = self._matcher.knnMatch(des_p, des_c, k=2)
        except cv2.error:
            return None
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.8 * n.distance:
                good.append(m)
        if len(good) < 4:
            return None

        src = np.float32([kp_p[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # Undo the resize so the affine is in *original* prototype/crop coords
        src = src / s_proto
        dst = dst / s_crop

        A, mask = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thr,
            maxIters=self.ransac_iters,
            confidence=0.99,
            refineIters=10,
        )
        if A is None or mask is None:
            return None

        inlier_mask = mask.ravel().astype(bool)
        n_in = int(inlier_mask.sum())
        n_tot = int(len(good))

        # Median reprojection error on inliers
        if n_in > 0:
            src_in = src[inlier_mask].reshape(-1, 2)
            dst_in = dst[inlier_mask].reshape(-1, 2)
            proj = (A[:, :2] @ src_in.T).T + A[:, 2]
            err = np.linalg.norm(proj - dst_in, axis=1)
            med_err = float(np.median(err))
        else:
            med_err = float("inf")

        return InitTransform(
            affine=A.astype(np.float32),
            inliers=n_in,
            total_matches=n_tot,
            inlier_ratio=n_in / max(n_tot, 1),
            median_reproj_err=med_err,
            proto_shape=proto_gray.shape[:2],
            crop_shape=crop_gray.shape[:2],
        )


class OfficialProtoSnapInitEstimator:
    """Stub for the SD-DIFT based stage-1 init.

    To enable: clone https://github.com/TAU-VAILab/ProtoSnap, install its
    requirements, gdown the SD weights, and fill in `estimate`. The wire-up
    point is the `_match_dift` placeholder. Until then, ``estimate`` always
    returns None and the caller falls back to OpenCV.
    """

    def __init__(self, repo_root: str, weights_dir: str, img_size: int = 256):
        self.repo_root = repo_root
        self.weights_dir = weights_dir
        self.img_size = img_size
        self._model = None

    def _ensure_model_loaded(self):
        # TODO(protosnap): import SDFeaturizer from cloned repo and load weights.
        raise NotImplementedError(
            "OfficialProtoSnapInitEstimator: wire up TAU-VAILab/ProtoSnap "
            "SDFeaturizer here. See main.py initialize() in their repo."
        )

    def estimate(self, crop, prototype):
        # Until wired up: signal "not available", caller may fall back.
        return None


def make_estimator(config: ProtoSnapConfig) -> TransformEstimator:
    if config.estimator == "official_protosnap":
        if not config.repo_root:
            logger.warning("official_protosnap requested but repo_root is None; using opencv")
            return OpenCVAffineEstimator(
                feature=config.opencv_feature, img_size=config.feature_img_size)
        # Try to construct; if it raises NotImplementedError, fall back.
        try:
            est = OfficialProtoSnapInitEstimator(
                repo_root=config.repo_root,
                weights_dir=str(Path(config.repo_root) / "weights" / "SD_with_prompt"),
                img_size=config.feature_img_size or 256,
            )
            est._ensure_model_loaded()
            return est
        except NotImplementedError:
            logger.warning("OfficialProtoSnapInitEstimator not wired up; falling back to opencv")
            return OpenCVAffineEstimator(
                feature=config.opencv_feature, img_size=config.feature_img_size)
    return OpenCVAffineEstimator(
        feature=config.opencv_feature, img_size=config.feature_img_size)


# ===========================================================================
# Gate (reliability filter)
# ===========================================================================

def default_gate(t: InitTransform, bbox: SignBox, gc: GateConfig,
                 implied_offset_px: float) -> Tuple[bool, str, Dict[str, float]]:
    """Multi-criteria gate. Returns (accepted, reason, diagnostics).

    Combines:
      - inlier count / ratio
      - reprojection error (relative to crop size)
      - affine shape (scale bounds, det positive, skew bounded)
      - implied center-offset sanity (relative to bbox size)

    Replace by passing a custom callable to CenterFreezeRefiner if you want
    a different gate; this function is also the place where you'd add an
    ablation switch.
    """
    H, W = t.crop_shape
    crop_diag = max(H, W)
    bbox_diag = max(bbox.width, bbox.height)

    # Affine shape diagnostics
    A = t.affine[:, :2]
    sx = float(np.linalg.norm(A[:, 0]))
    sy = float(np.linalg.norm(A[:, 1]))
    det = float(np.linalg.det(A))
    if sx > 0 and sy > 0:
        skew = float(abs(np.dot(A[:, 0], A[:, 1])) / (sx * sy))
    else:
        skew = 1.0

    diag = {
        "inliers": float(t.inliers),
        "inlier_ratio": float(t.inlier_ratio),
        "med_reproj_err": float(t.median_reproj_err),
        "scale_x": sx,
        "scale_y": sy,
        "det": det,
        "skew": skew,
        "implied_offset_px": float(implied_offset_px),
        "crop_diag": float(crop_diag),
        "bbox_diag": float(bbox_diag),
    }

    if t.inliers < gc.min_inliers:
        return False, f"inliers {t.inliers} < {gc.min_inliers}", diag
    if t.inlier_ratio < gc.min_inlier_ratio:
        return False, f"inlier_ratio {t.inlier_ratio:.2f} < {gc.min_inlier_ratio}", diag
    if t.median_reproj_err > gc.max_reproj_err_ratio * crop_diag:
        return False, (f"reproj_err {t.median_reproj_err:.1f} > "
                       f"{gc.max_reproj_err_ratio} * {crop_diag:.0f}"), diag
    if not (gc.scale_min <= sx <= gc.scale_max):
        return False, f"scale_x {sx:.2f} out of [{gc.scale_min}, {gc.scale_max}]", diag
    if not (gc.scale_min <= sy <= gc.scale_max):
        return False, f"scale_y {sy:.2f} out of [{gc.scale_min}, {gc.scale_max}]", diag
    if det <= 0:
        return False, f"det {det:.2f} <= 0 (mirror)", diag
    if skew > gc.max_skew:
        return False, f"skew {skew:.2f} > {gc.max_skew}", diag
    if bbox_diag > 0 and implied_offset_px > gc.max_offset_ratio * bbox_diag:
        return False, (f"offset {implied_offset_px:.1f} > "
                       f"{gc.max_offset_ratio} * {bbox_diag:.0f}"), diag

    return True, "ok", diag


# ===========================================================================
# Center-freeze refiner
# ===========================================================================

class CenterFreezeRefiner:
    """Per-sign center correction + gate.

    Stateless w.r.t. the optimizer; the pipeline turns `accepted=True`
    refinements into `optimizer.freeze_centers(...)` calls.
    """

    def __init__(
        self,
        config: ProtoSnapConfig,
        bank: PrototypeBank,
        estimator: TransformEstimator,
        gate: Optional[Callable[..., Tuple[bool, str, Dict]]] = None,
    ):
        self.config = config
        self.bank = bank
        self.estimator = estimator
        self.gate = gate or default_gate

    # ---- single-sign refinement ----

    def refine_one(self, image: np.ndarray, sb: SignBox, sign_idx: int,
                   sub_image_shape: Tuple[int, int]) -> CenterRefinement:
        out = CenterRefinement(
            sign_index=sign_idx, sign_name=sb.sign_name, abz=sb.abz_name,
            status=SnapStatus.ERROR, accepted=False,
            new_cx=sb.cx, new_cy=sb.cy,
        )

        if sb.abz_name in ("X", "NoABZ0"):
            out.status = SnapStatus.UNCLEAR_SIGN
            return out

        # Load prototype and crop early so both are available in the returned
        # CenterRefinement for the debugging visualization, even on early-out paths.
        out.proto_img = self.bank.load_prototype(sb.sign)

        _crop, (ox, oy) = self._crop_around(image, sb, sub_image_shape)
        if _crop is not None:
            out.crop_img = _crop
            out.crop_offset = (ox, oy)

        if out.proto_img is None:
            out.status = SnapStatus.NO_PROTOTYPE
            return out

        if _crop is None:
            out.status = SnapStatus.OUT_OF_BOUNDS
            return out

        crop = _crop

        try:
            t = self.estimator.estimate(crop, out.proto_img)
        except Exception as e:
            if self.config.log_failures:
                logger.exception("estimator failed on %s", sb.sign_name)
            out.status = SnapStatus.ERROR
            out.info["exception"] = str(e)
            return out

        if t is None:
            out.status = SnapStatus.NO_FEATURES
            return out
        if t.total_matches < 4:
            out.status = SnapStatus.LOW_MATCHES
            out.transform = t
            return out

        # Map prototype center -> crop coords -> sub-image coords
        ph, pw = t.proto_shape
        proto_center = np.array([pw / 2.0, ph / 2.0], dtype=np.float32)
        in_crop = t.affine[:, :2] @ proto_center + t.affine[:, 2]
        new_cx = float(in_crop[0]) + ox
        new_cy = float(in_crop[1]) + oy

        implied_offset = float(np.hypot(new_cx - sb.cx, new_cy - sb.cy))
        accepted, reason, diag = self.gate(t, sb, self.config.gate, implied_offset)

        out.transform = t
        out.new_cx = new_cx
        out.new_cy = new_cy
        out.delta_cx = new_cx - sb.cx
        out.delta_cy = new_cy - sb.cy
        out.info.update(diag)
        out.info["gate_reason"] = reason

        if not accepted:
            if reason.startswith("inliers ") or reason.startswith("inlier_ratio "):
                out.status = SnapStatus.LOW_INLIERS
            else:
                out.status = SnapStatus.BAD_TRANSFORM
            out.accepted = False
        else:
            out.status = SnapStatus.OK
            out.accepted = True
        return out

    # ---- batch over a SubTablet ----

    def refine_subtablet(
        self, sub_tablet: SubTablet,
    ) -> List[CenterRefinement]:
        if sub_tablet.img is None:
            raise ValueError("refine_subtablet requires sub_tablet.img to be set")
        H, W = sub_tablet.img.shape[:2]
        results: List[CenterRefinement] = []
        for i, sb in enumerate(sub_tablet.sign_boxes):
            r = self.refine_one(sub_tablet.img, sb, i, (H, W))
            results.append(r)
        return results

    # ---- internals ----

    def _crop_around(self, image: np.ndarray, sb: SignBox,
                     sub_image_shape: Tuple[int, int]
                     ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        H, W = sub_image_shape
        pad = max(sb.width, sb.height) * self.config.crop_padding_ratio
        x1 = int(max(0, sb.x1 - pad))
        y1 = int(max(0, sb.y1 - pad))
        x2 = int(min(W, sb.x2 + pad))
        y2 = int(min(H, sb.y2 + pad))
        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)
        return image[y1:y2, x1:x2], (x1, y1)


# ===========================================================================
# Factory
# ===========================================================================

def make_refiner(
    config: Optional[ProtoSnapConfig], period: Optional[str] = None,
) -> Tuple[Optional[CenterFreezeRefiner], Dict[str, Any]]:
    """Build a CenterFreezeRefiner if assets are reachable, else (None, meta).

    `meta` always contains the resolved period / font / reason for logging,
    even when the refiner is None — pipeline uses it to print info.
    """
    meta: Dict[str, Any] = {
        "enabled": False,
        "period": period,
        "font": None,
        "font_reason": None,
        "repo_root": getattr(config, "repo_root", None) if config else None,
        "estimator": None,
    }
    if config is None:
        meta["font_reason"] = "ProtoSnap disabled (config is None)"
        return None, meta

    if config.force_font:
        font, reason = config.force_font, f"forced via config -> {config.force_font}"
    else:
        font, reason = resolve_font_for_period(period)
    meta["font"] = font
    meta["font_reason"] = reason

    bank = PrototypeBank(config, font)
    if not bank.is_ready():
        meta["font_reason"] += " | bank not ready (repo_root / prototypes missing)"
        return None, meta

    estimator = make_estimator(config)
    meta["estimator"] = type(estimator).__name__
    meta["enabled"] = True
    return CenterFreezeRefiner(config, bank, estimator), meta


# ===========================================================================
# Debugging visualization: per-sign ProtoSnap grid
# ===========================================================================

def render_refinement_grid(
    refinements: List[CenterRefinement],
    thumb: int = 128,
    cols: int = 5,
) -> np.ndarray:
    """Build a BGR grid image showing per-sign ProtoSnap results.

    Each panel contains:
      Left  — prototype thumbnail (gray).
      Right — crop thumbnail with original center (cyan circle) and
              ProtoSnap-estimated center (green cross = accepted,
              red cross = rejected) connected by an arrow.
    Header text: sign_name, status tag, Δcx / Δcy.
    Footer text: gate reason or status string.
    Colored border: green = accepted, red = rejected/error, gray = no-proto.

    Returns
    -------
    np.ndarray  BGR uint8 image ready for cv2.imwrite / matplotlib display.
    """
    HDR_H  = 36   # header bar height
    FTR_H  = 22   # footer bar height
    GAP    = 2    # gap between proto and crop thumbnails
    MARGIN = 6    # space between cells in the grid
    BORDER = 3    # colored border thickness around each cell

    PANEL_W = 2 * thumb + GAP
    PANEL_H = HDR_H + thumb + FTR_H
    CELL_W  = PANEL_W + 2 * BORDER + MARGIN
    CELL_H  = PANEL_H + 2 * BORDER + MARGIN

    _GREEN = (80, 200, 80)    # accepted
    _RED   = (80, 80, 200)    # rejected / error  (BGR)
    _GRAY  = (140, 140, 140)  # no-prototype / unclear / disabled

    _NO_PROTO_STATUSES = {
        SnapStatus.NO_PROTOTYPE, SnapStatus.UNCLEAR_SIGN, SnapStatus.DISABLED
    }

    def _border_color(r: CenterRefinement) -> Tuple[int, int, int]:
        if r.accepted:
            return _GREEN
        if r.status in _NO_PROTO_STATUSES:
            return _GRAY
        return _RED

    def _fit(img: Optional[np.ndarray]
             ) -> Tuple[np.ndarray, float, int, int]:
        """Fit *img* into a thumb×thumb BGR canvas.

        Returns (canvas, scale, x_offset, y_offset) where the image was
        placed at (x_offset, y_offset) inside the canvas after scaling.
        """
        canvas = np.full((thumb, thumb, 3), 40, dtype=np.uint8)
        if img is None:
            return canvas, 1.0, 0, 0
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        if max(h, w) == 0:
            return canvas, 1.0, 0, 0
        sc = thumb / max(h, w)
        nw = max(1, int(round(w * sc)))
        nh = max(1, int(round(h * sc)))
        rsz = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        yo, xo = (thumb - nh) // 2, (thumb - nw) // 2
        canvas[yo:yo + nh, xo:xo + nw] = rsz
        return canvas, sc, xo, yo

    def _text(canvas: np.ndarray, txt: str, y: int,
              color=(210, 210, 210), scale: float = 0.38) -> None:
        """Draw text with thin black outline for readability on any bg."""
        cv2.putText(canvas, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    def _marker(canvas: np.ndarray, pt: Tuple[float, float],
                color: Tuple[int, int, int], kind: int, size: int = 10) -> None:
        x, y = int(round(pt[0])), int(round(pt[1]))
        h, w = canvas.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            cv2.drawMarker(canvas, (x, y), color, kind, size, 2, cv2.LINE_AA)

    def _circle(canvas: np.ndarray, pt: Tuple[float, float],
                color: Tuple[int, int, int], radius: int = 5) -> None:
        x, y = int(round(pt[0])), int(round(pt[1]))
        h, w = canvas.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), radius, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), radius, color, 1, cv2.LINE_AA)

    def _make_panel(r: CenterRefinement) -> np.ndarray:
        panel = np.full((PANEL_H, PANEL_W, 3), 25, dtype=np.uint8)
        bc = _border_color(r)

        # ── Header ──────────────────────────────────────────────────────────
        tag = "[OK]" if r.accepted else f"[{r.status.value[:4].upper()}]"
        delta = f"dx={r.delta_cx:+.1f} dy={r.delta_cy:+.1f}"
        hdr = np.full((HDR_H, PANEL_W, 3), 50, dtype=np.uint8)
        name_str = r.sign_name[:12]
        _text(hdr, f"{name_str:<12} {tag}", HDR_H - 22, bc, scale=0.40)
        _text(hdr, delta, HDR_H - 6, (180, 180, 180), scale=0.36)
        panel[:HDR_H] = hdr

        # ── Prototype thumbnail ──────────────────────────────────────────────
        proto_canvas, _, _, _ = _fit(r.proto_img)
        panel[HDR_H:HDR_H + thumb, :thumb] = proto_canvas

        # ── Crop thumbnail with center markers ───────────────────────────────
        crop_canvas, sc, xo, yo = _fit(r.crop_img)

        if r.crop_offset is not None and r.crop_img is not None:
            ox, oy_off = r.crop_offset

            def _to_th(cx: float, cy: float) -> Tuple[float, float]:
                return cx * sc + xo, cy * sc + yo

            # Original center (sign bbox center before ProtoSnap)
            orig_cx_c = (r.new_cx - r.delta_cx) - ox
            orig_cy_c = (r.new_cy - r.delta_cy) - oy_off
            pt_orig = _to_th(orig_cx_c, orig_cy_c)
            _circle(crop_canvas, pt_orig, (0, 210, 210), radius=5)  # cyan

            # ProtoSnap-estimated center (only if transform actually ran)
            if r.transform is not None:
                new_cx_c = r.new_cx - ox
                new_cy_c = r.new_cy - oy_off
                pt_new = _to_th(new_cx_c, new_cy_c)
                nc = _GREEN if r.accepted else _RED
                _marker(crop_canvas, pt_new, nc, cv2.MARKER_CROSS, 14)
                # Arrow from original to new
                p0 = (int(round(pt_orig[0])), int(round(pt_orig[1])))
                p1 = (int(round(pt_new[0])), int(round(pt_new[1])))
                if abs(p1[0] - p0[0]) + abs(p1[1] - p0[1]) > 2:
                    cv2.arrowedLine(crop_canvas, p0, p1, nc, 1,
                                    tipLength=0.3, line_type=cv2.LINE_AA)

        panel[HDR_H:HDR_H + thumb, thumb + GAP:] = crop_canvas

        # ── Footer ──────────────────────────────────────────────────────────
        reason = str(r.info.get("gate_reason", r.status.value))
        if len(reason) > 44:
            reason = reason[:42] + ".."
        ftr = np.full((FTR_H, PANEL_W, 3), 38, dtype=np.uint8)
        _text(ftr, reason, FTR_H - 5, (170, 170, 170), scale=0.33)
        panel[HDR_H + thumb:] = ftr

        return panel

    # ── Tile into grid ───────────────────────────────────────────────────────
    n = len(refinements)
    if n == 0:
        return np.full((CELL_H, CELL_W, 3), 15, dtype=np.uint8)

    rows = (n + cols - 1) // cols
    grid = np.full((MARGIN + rows * CELL_H,
                    MARGIN + cols * CELL_W, 3), 15, dtype=np.uint8)

    for idx, r in enumerate(refinements):
        ri, ci = divmod(idx, cols)
        y0 = MARGIN + ri * CELL_H
        x0 = MARGIN + ci * CELL_W
        panel = _make_panel(r)
        bc = _border_color(r)
        # Bordered cell
        bh, bw = PANEL_H + 2 * BORDER, PANEL_W + 2 * BORDER
        cell = np.zeros((bh, bw, 3), dtype=np.uint8)
        cv2.rectangle(cell, (0, 0), (bw - 1, bh - 1), bc, BORDER)
        cell[BORDER:BORDER + PANEL_H, BORDER:BORDER + PANEL_W] = panel
        grid[y0:y0 + bh, x0:x0 + bw] = cell

    return grid


# ===========================================================================
# Helpers for the pipeline / state-dict serialization
# ===========================================================================

def summarize_refinements(results: List[CenterRefinement]) -> Dict[str, Any]:
    counts = Counter(r.status.value for r in results)
    accepted = sum(1 for r in results if r.accepted)
    return {"total": len(results), "accepted": accepted, "by_status": dict(counts)}


def collect_used_prototype_paths(
    results: List[CenterRefinement], bank: PrototypeBank,
    sign_boxes: List[SignBox],
) -> List[str]:
    """List of unique prototype PNGs that were actually consulted, for state_dict."""
    paths = set()
    for r in results:
        if r.transform is None:
            continue
        sb = sign_boxes[r.sign_index]
        p = bank.get_prototype_path(sb.sign)
        if p is not None:
            paths.add(str(p))
    return sorted(paths)
