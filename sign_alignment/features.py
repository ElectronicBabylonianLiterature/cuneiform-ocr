"""Feature extraction and matching framework — minimal skeleton.

Paradigm A (sparse): SIFTExtractor → SparseFeatures → NNMatcher
Paradigm B (dense):  DenseExtractor → DenseFeatures  → BestBuddiesMatcher

Both produce MatchResult (src_pts, dst_pts, scores) in image pixel coords.
Visualisation functions return BGR uint8 numpy arrays (no display/save).
"""

from __future__ import annotations

import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SparseFeatures:
    keypoints:   np.ndarray      # (N, 2) float32  (x, y)
    descriptors: np.ndarray      # (N, D) float32
    scales:      np.ndarray      # (N,)   float32  — detection scale = circle radius*2
    angles:      np.ndarray      # (N,)   float32  — dominant orientation in degrees
    responses:   np.ndarray      # (N,)   float32  — DoG response (strength)
    image_size:  Tuple[int, int] # (W, H)

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoints)

    def top_n(self, n: int) -> "SparseFeatures":
        """Return new SparseFeatures keeping only the n strongest by response."""
        if n >= self.n_keypoints:
            return self
        idx = np.argsort(self.responses)[::-1][:n]
        return SparseFeatures(
            keypoints=self.keypoints[idx],
            descriptors=self.descriptors[idx],
            scales=self.scales[idx],
            angles=self.angles[idx],
            responses=self.responses[idx],
            image_size=self.image_size,
        )


@dataclass
class DenseFeatures:
    feature_map: Union[np.ndarray, "torch.Tensor"]  # (C, H_feat, W_feat)
    image_size:  Tuple[int, int]                     # (W, H) original image

    def feat_to_image_coords(self, fx: float, fy: float) -> Tuple[float, float]:
        fw, fh = self.feature_map.shape[-1], self.feature_map.shape[-2]
        iw, ih = self.image_size
        return fx / fw * iw, fy / fh * ih

    def image_to_feat_coords(self, x: float, y: float) -> Tuple[float, float]:
        fw, fh = self.feature_map.shape[-1], self.feature_map.shape[-2]
        iw, ih = self.image_size
        return x / iw * fw, y / ih * fh


@dataclass
class MatchResult:
    src_pts: np.ndarray   # (M, 2) float32  (x, y) in source image
    dst_pts: np.ndarray   # (M, 2) float32  (x, y) in destination image
    scores:  np.ndarray   # (M,)   float32  higher = better

    @property
    def n_matches(self) -> int:
        return len(self.src_pts)

    @classmethod
    def empty(cls) -> "MatchResult":
        z2 = np.zeros((0, 2), dtype=np.float32)
        return cls(src_pts=z2, dst_pts=z2.copy(), scores=np.zeros(0, np.float32))


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray) -> Union[SparseFeatures, DenseFeatures]: ...

    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


class SIFTExtractor(FeatureExtractor):
    def __init__(self):
        self._sift = cv2.SIFT_create()

    def extract(self, image: np.ndarray) -> SparseFeatures:
        gray = self._gray(image)
        h, w = gray.shape[:2]
        kps, descs = self._sift.detectAndCompute(gray, None)
        if not kps:
            return SparseFeatures(
                keypoints=np.zeros((0, 2), np.float32),
                descriptors=np.zeros((0, 128), np.float32),
                scales=np.zeros(0, np.float32),
                angles=np.zeros(0, np.float32),
                responses=np.zeros(0, np.float32),
                image_size=(w, h),
            )
        return SparseFeatures(
            keypoints=np.array([kp.pt       for kp in kps], np.float32),
            descriptors=descs.astype(np.float32),
            scales=np.array([kp.size     for kp in kps], np.float32),
            angles=np.array([kp.angle    for kp in kps], np.float32),
            responses=np.array([kp.response for kp in kps], np.float32),
            image_size=(w, h),
        )


class DenseExtractor(FeatureExtractor, ABC):
    """Subclass and implement extract() to return DenseFeatures."""
    @abstractmethod
    def extract(self, image: np.ndarray) -> DenseFeatures: ...


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------

class FeatureMatcher(ABC):
    @abstractmethod
    def match(self, feat_a, feat_b) -> MatchResult: ...


class NNMatcher(FeatureMatcher):
    """Simple BFMatcher (crossCheck) for SparseFeatures — no ratio filtering."""

    def __init__(self):
        self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def match(self, feat_a: SparseFeatures, feat_b: SparseFeatures) -> MatchResult:
        if feat_a.n_keypoints == 0 or feat_b.n_keypoints == 0:
            return MatchResult.empty()
        try:
            ms = self._matcher.match(feat_a.descriptors, feat_b.descriptors)
        except cv2.error:
            return MatchResult.empty()
        if not ms:
            return MatchResult.empty()
        ia = np.array([m.queryIdx for m in ms], np.int32)
        ib = np.array([m.trainIdx for m in ms], np.int32)
        return MatchResult(
            src_pts=feat_a.keypoints[ia],
            dst_pts=feat_b.keypoints[ib],
            scores=-np.array([m.distance for m in ms], np.float32),
        )


class BestBuddiesMatcher(FeatureMatcher):
    """Mutual cosine-NN matcher for DenseFeatures. Requires torch."""

    def match(self, feat_a: DenseFeatures, feat_b: DenseFeatures) -> MatchResult:
        if not _TORCH:
            raise RuntimeError("BestBuddiesMatcher requires torch.")

        def to_t(fm):
            return (torch.from_numpy(fm.copy()) if isinstance(fm, np.ndarray) else fm).float()

        fa = F.normalize(to_t(feat_a.feature_map), dim=0)  # (C, Ha, Wa)
        fb = F.normalize(to_t(feat_b.feature_map), dim=0)  # (C, Hb, Wb)
        C, Ha, Wa = fa.shape
        Hb, Wb = fb.shape[-2], fb.shape[-1]

        sim = fa.view(C, -1).T @ fb.view(C, -1)             # (Na, Nb)
        nn_a2b = sim.argmax(dim=1).cpu().numpy()
        nn_b2a = sim.argmax(dim=0).cpu().numpy()
        sim_np = sim.cpu().numpy()

        src_pts, dst_pts, scores = [], [], []
        for ia in range(Ha * Wa):
            ib = int(nn_a2b[ia])
            if int(nn_b2a[ib]) != ia:
                continue
            fy_a, fx_a = divmod(ia, Wa)
            fy_b, fx_b = divmod(ib, Wb)
            src_pts.append(feat_a.feat_to_image_coords(fx_a + 0.5, fy_a + 0.5))
            dst_pts.append(feat_b.feat_to_image_coords(fx_b + 0.5, fy_b + 0.5))
            scores.append(float(sim_np[ia, ib]))

        if not src_pts:
            return MatchResult.empty()
        return MatchResult(
            src_pts=np.array(src_pts, np.float32),
            dst_pts=np.array(dst_pts, np.float32),
            scores=np.array(scores, np.float32),
        )


# ---------------------------------------------------------------------------
# Visualisation — return BGR uint8 np.ndarray
# ---------------------------------------------------------------------------

def draw_keypoints(image: np.ndarray, features: SparseFeatures,
                   color: Tuple[int, int, int] = (0, 200, 255),
                   max_draw: int = None,
                   highlight_pts: np.ndarray = None,
                   highlight_color: Tuple[int, int, int] = (50, 255, 50)) -> np.ndarray:
    """Draw keypoints with optional highlighted points (e.g. matched) drawn on top.

    highlight_pts are drawn regardless of max_draw and in a distinct color.
    """
    if max_draw is not None:
        features = features.top_n(max_draw)
    vis = _bgr(image)
    cv_kps = [cv2.KeyPoint(float(features.keypoints[i, 0]),
                           float(features.keypoints[i, 1]),
                           float(features.scales[i]),
                           float(features.angles[i]))
              for i in range(features.n_keypoints)]
    cv2.drawKeypoints(vis, cv_kps, vis, color=color,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Draw highlighted points (matched keypoints) on top, bypass max_draw limit
    if highlight_pts is not None and len(highlight_pts) > 0:
        for pt in highlight_pts:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(vis, (x, y), 7, (0, 0, 0), -1, cv2.LINE_AA)        # black outline
            cv2.circle(vis, (x, y), 5, highlight_color, -1, cv2.LINE_AA)  # filled dot
    return vis


def draw_matches(img_a: np.ndarray, img_b: np.ndarray,
                 matches: MatchResult, max_draw: int = 100) -> np.ndarray:
    a, b = _bgr(img_a), _bgr(img_b)
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    canvas = np.full((max(ha, hb), wa + wb, 3), 30, np.uint8)
    canvas[:ha, :wa] = a
    canvas[:hb, wa:] = b
    rng = np.random.default_rng(42)
    for i in range(min(matches.n_matches, max_draw)):
        c = tuple(int(v) for v in rng.integers(80, 220, 3))
        pt_a = (int(matches.src_pts[i, 0]),      int(matches.src_pts[i, 1]))
        pt_b = (int(matches.dst_pts[i, 0]) + wa, int(matches.dst_pts[i, 1]))
        cv2.circle(canvas, pt_a, 5, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt_b, 5, c, -1, cv2.LINE_AA)
        cv2.line(canvas, pt_a, pt_b, c, 2, cv2.LINE_AA)
    return canvas


def draw_dense_features(image: np.ndarray, features: DenseFeatures,
                        alpha: float = 0.55) -> np.ndarray:
    """PCA of feature map → RGB, alpha-blended over image."""
    fm = features.feature_map
    if _TORCH and isinstance(fm, torch.Tensor):
        fm = fm.detach().cpu().numpy()
    C, Hf, Wf = fm.shape
    flat = fm.reshape(C, -1).T.astype(np.float32)
    centered = flat - flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:3].T                           # (N, 3)
    lo, hi = proj.min(0), proj.max(0)
    proj = (proj - lo) / (hi - lo + 1e-8) * 255.0
    bgr_feat = proj.reshape(Hf, Wf, 3)[:, :, ::-1].astype(np.uint8)
    bgr_feat = cv2.resize(bgr_feat, features.image_size, interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(_bgr(image), 1.0 - alpha, bgr_feat, alpha, 0.0)


def _bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()
