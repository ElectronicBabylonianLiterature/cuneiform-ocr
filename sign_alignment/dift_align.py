"""
Gap-driven DIFT identification of missing cuneiform signs.

This module identifies signs that the object detector missed, using ONLY
the detection-row layout as a prior — no text-label guidance. The flow is:

    1. Pre-compute DIFT features for the ENTIRE canonical sign inventory
       (eager, cached on the CropContext so it amortizes across samples).
    2. Scan each detection row's coordinates: where adjacent boxes are
       further apart than expected by avg sign width, insert one or more
       pending bboxes at evenly-spaced positions inside the gap.
    3. For each pending bbox, crop the sub-image, run DIFT, and score
       it against every cached prototype. The highest-scoring sign is
       assigned to that bbox.
    4. The discovered (assigned) bboxes are appended to
       `sub_image.detections` BEFORE the PSR optimizer is constructed,
       so the GMM data term naturally sees them — no PSR-internal hooks.

The canonical source is intentionally pluggable. For now it wraps the
ProtoSnap prototype bank; future code can plug in eBL canonical signs
by implementing `CanonicalSignSource`.

Important: we do NOT use ProtoSnap's `CenterFreezeRefiner` (refining
already-detected signs). We only reuse ProtoSnap's prototype PNGs and
the SD-DIFT feature extractor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .bounding_box import BoundingBox
from .sign import Sign, SignResolver
from .tablet import SignBox, SubTablet
from .protosnap import (
    PrototypeBank,
    ProtoSnapConfig,
    resolve_font_for_period,
)


# ===========================================================================
# Canonical sign source — where the prototype/canonical image comes from.
# ===========================================================================

class CanonicalSignSource(ABC):
    """Registry of canonical images per Sign + a stable id for caching."""

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def get_image(self, sign: Sign) -> Optional[np.ndarray]:
        """Canonical image (grayscale uint8) for `sign`, already preprocessed."""

    @abstractmethod
    def get_id(self, sign: Sign) -> Optional[str]:
        """Stable cache key for this sign (returns None if not served)."""

    @abstractmethod
    def list_sign_names(self) -> List[str]:
        """Names of every sign this source can serve.

        Used for eager full-inventory precomputation. Empty list = no signs.
        """

    def describe(self) -> str:
        return type(self).__name__


class ProtoSnapCanonicalSource(CanonicalSignSource):
    """Canonical source backed by the ProtoSnap prototype bank."""

    def __init__(self, config: ProtoSnapConfig, font: str):
        self.config = config
        self.font = font
        self._bank = PrototypeBank(config, font)
        self._ready = self._bank.is_ready()

    def is_ready(self) -> bool:
        return self._ready

    def get_image(self, sign: Sign) -> Optional[np.ndarray]:
        return self._bank.load_prototype(sign) if self._ready else None

    def get_id(self, sign: Sign) -> Optional[str]:
        if not self._ready:
            return None
        hex_val = self._bank._name_to_hex.get(sign.name)
        return f"{self.font}/{hex_val}" if hex_val else None

    def list_sign_names(self) -> List[str]:
        return list(self._bank._name_to_hex.keys()) if self._ready else []

    @property
    def bank(self) -> PrototypeBank:
        return self._bank

    def describe(self) -> str:
        if not self._ready:
            return f"ProtoSnap ({self.font}) — not ready"
        return (f"ProtoSnap font={self.font}, "
                f"metadata_size={len(self._bank._name_to_hex)}")


class EBLCanonicalSource(CanonicalSignSource):
    """Placeholder for the future eBL canonical-sign retrieval."""

    def __init__(self, period: Optional[str]):
        self.period = period

    def is_ready(self) -> bool:
        return False

    def get_image(self, sign): return None
    def get_id(self, sign): return None
    def list_sign_names(self): return []

    def describe(self) -> str:
        return f"EBL canonical source (period={self.period!r}) — not implemented"


def make_canonical_source(
    period: Optional[str], config: ProtoSnapConfig,
) -> Tuple[Optional[CanonicalSignSource], str]:
    """Default canonical source for `period` (currently ProtoSnap-backed).

    Returns (source, reason). `reason` is a short log-friendly string.
    """
    if config is None or not getattr(config, "repo_root", None):
        return None, "ProtoSnapConfig.repo_root not set"
    if config.force_font:
        font, reason = config.force_font, f"forced font -> {config.force_font}"
    else:
        font, reason = resolve_font_for_period(period)
    source = ProtoSnapCanonicalSource(config, font)
    if not source.is_ready():
        return None, f"{reason} | bank not ready"
    return source, reason


# ===========================================================================
# DIFT feature cache + eager full-inventory precompute
# ===========================================================================

@dataclass
class CanonicalFeatureRecord:
    """One canonical sign's DIFT features + source image."""
    feature_map: torch.Tensor          # (C, 64, 64), L2-normalized on C
    img: np.ndarray                    # original canonical image (uint8 gray)
    img_size: Tuple[int, int]          # (H, W) of canonical image
    sign_id: str
    sign_name: str


class CanonicalFeatureCache:
    """DIFT feature cache for canonical signs.

    DIFT features are (1920, 64, 64) per prototype, ~30 MB in fp32 — far
    too big to keep all ~900 prototype features on a single 16 GB GPU.
    To keep the inventory searchable in memory, features are stored on
    the **CPU in fp16** (about 15 MB each, ~14 GB total for the whole
    Assurbanipal set) and copied to the active device on each scoring
    call. We never build a stacked tensor.

    Supports both lazy `get(sign)` (back-compat) and eager
    `precompute(signs)` (preferred — gap-driven identification needs
    the whole inventory available upfront).
    """

    def __init__(self, source: CanonicalSignSource, wrapper, img_size: int = 512,
                 store_device: str = "cpu", store_dtype: torch.dtype = torch.float16):
        self.source = source
        self.wrapper = wrapper
        self.img_size = img_size
        self.store_device = store_device
        self.store_dtype = store_dtype
        self._cache: Dict[str, CanonicalFeatureRecord] = {}
        self._ordered_ids: List[str] = []

    def __len__(self) -> int:
        return len(self._cache)

    # ---- single-sign lookup (back-compat / on-demand) -----------------

    def get(self, sign: Sign) -> Optional[CanonicalFeatureRecord]:
        sid = self.source.get_id(sign)
        if not sid:
            return None
        rec = self._cache.get(sid)
        if rec is not None:
            return rec
        rec = self._featurize_sign(sign, sid)
        if rec is not None:
            self._cache[sid] = rec
            self._ordered_ids.append(sid)
        return rec

    # ---- eager full-inventory precompute -------------------------------

    def precompute_all(self, verbose: bool = True,
                       limit: Optional[int] = None,
                       progress_every: int = 50) -> Dict[str, int]:
        """Featurize every sign the source can serve.

        Returns a small stats dict: {'total', 'computed', 'cached',
        'missing_image', 'errors'}.
        """
        names = self.source.list_sign_names()
        return self.precompute(names, verbose=verbose, limit=limit,
                               progress_every=progress_every)

    def precompute(self, sign_names: Iterable[str], verbose: bool = True,
                   limit: Optional[int] = None,
                   progress_every: int = 50) -> Dict[str, int]:
        """Featurize each sign by name; idempotent.

        Args:
            sign_names: names to featurize.
            limit:      cap on count (None = all).
            verbose:    print progress every `progress_every` items.
        """
        names = list(dict.fromkeys(sign_names))  # de-dup, preserve order
        if limit is not None:
            names = names[:limit]

        stats = {"total": len(names), "computed": 0, "cached": 0,
                 "missing_image": 0, "errors": 0}

        for i, name in enumerate(names, start=1):
            sign = SignResolver.from_name(name)
            sid = self.source.get_id(sign)
            if sid is None:
                stats["missing_image"] += 1
                continue
            if sid in self._cache:
                stats["cached"] += 1
            else:
                try:
                    rec = self._featurize_sign(sign, sid)
                except Exception as e:
                    stats["errors"] += 1
                    if verbose:
                        print(f"    [precompute] {name}: error: {e}")
                    continue
                if rec is None:
                    stats["missing_image"] += 1
                    continue
                self._cache[sid] = rec
                self._ordered_ids.append(sid)
                stats["computed"] += 1

            if verbose and (i % progress_every == 0 or i == len(names)):
                print(f"    [precompute] {i}/{len(names)} "
                      f"(computed={stats['computed']}, cached={stats['cached']}, "
                      f"missing={stats['missing_image']}, errors={stats['errors']})")
        return stats

    def _featurize_sign(self, sign: Sign, sid: str) -> Optional[CanonicalFeatureRecord]:
        img = self.source.get_image(sign)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 \
              else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with torch.no_grad():
            fm = self.wrapper.featurize(pil)
        # Move to CPU + fp16 so all 900 features fit in RAM, not VRAM.
        fm_cpu = fm.detach().to(device=self.store_device, dtype=self.store_dtype,
                                non_blocking=False)
        # Free GPU memory eagerly between featurizations.
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


# ===========================================================================
# Gap detection: where do we expect to insert a "pending" bbox?
# ===========================================================================

@dataclass
class GapDetectionConfig:
    # If centre-to-centre distance / avg_width rounds to N (>= 2), there are
    # N-1 missing signs in that row gap. Tolerance band: a ratio close to 2 is
    # interpreted as "1 missing", close to 3 as "2 missing", etc.
    # tolerance_ratio: how close to integer N to count as N (e.g. 0.4 means
    # round to N if |stride/avg_w - N| <= tolerance_ratio).
    tolerance_ratio: float = 0.4
    # Maximum signs to insert in one gap (guards against single huge gap
    # generating many wild candidates).
    max_inserts_per_gap: int = 3
    # Padding around the candidate bbox when cropping for DIFT featurization
    # (fraction of bbox size).
    crop_padding_ratio: float = 0.3
    # If a row has only one detection, we cannot interpolate any gap — skip.
    require_two_anchors: bool = True


@dataclass
class GapCandidate:
    """A pending bbox identified from row-coordinate gaps.

    No sign label yet — `SignIdentifier` assigns one. `crop_img` is the
    sub-image patch used for DIFT featurization; `crop_offset` is its
    top-left corner in sub-image coords (so we can map features back to
    sub-image pixel space).
    """
    cx: float
    cy: float
    width: float
    height: float
    row_idx: int
    insert_idx: int                 # which insert within the gap (0..N-1)
    n_inserts: int                  # total inserts placed in this gap
    left_anchor_col: int            # col_idx of the left detected box
    right_anchor_col: int           # col_idx of the right detected box
    expected_stride: float          # the stride (avg sign width) used
    crop_img: Optional[np.ndarray] = None
    crop_offset: Optional[Tuple[int, int]] = None


class GapDetector:
    """Coordinate-based gap finder.

    For each detected row (sorted by x), scan adjacent box pairs. If the
    centre-to-centre distance is approximately N×avg_width (with N >= 2),
    insert N-1 candidate bboxes spaced evenly between the two anchors.
    The candidate bbox uses avg_width × avg_height; the centre is placed
    on the line connecting the two anchors (so it follows the row's slope).
    """

    def __init__(self, config: Optional[GapDetectionConfig] = None):
        self.config = config or GapDetectionConfig()

    def find_gaps(self, sub_tablet_detection: SubTablet,
                  image: np.ndarray) -> List[GapCandidate]:
        cfg = self.config
        avg_w = float(sub_tablet_detection.avg_width or 0.0)
        avg_h = float(sub_tablet_detection.avg_height or 0.0)
        if avg_w <= 0 or avg_h <= 0:
            return []

        H, W = image.shape[:2]
        gaps: List[GapCandidate] = []

        rows = sub_tablet_detection.get_rows_dict()
        for row_idx, row_boxes in rows.items():
            if row_idx < 0:
                continue
            if cfg.require_two_anchors and len(row_boxes) < 2:
                continue

            # Boxes already sorted by col_idx (== sort by cx). Defensive sort.
            row_sorted: List[SignBox] = sorted(row_boxes, key=lambda sb: sb.cx)
            for i in range(len(row_sorted) - 1):
                left = row_sorted[i]
                right = row_sorted[i + 1]
                stride = right.cx - left.cx
                if stride <= 0:
                    continue
                # How many sign-widths fit in this stride?
                n_units = stride / avg_w
                n_total = int(round(n_units))
                # Acceptance: must round to >= 2 (= at least 1 missing sign)
                # AND the rounding error must be within tolerance.
                if n_total < 2:
                    continue
                if abs(n_units - n_total) > cfg.tolerance_ratio:
                    continue
                n_insert = min(n_total - 1, cfg.max_inserts_per_gap)

                # Place inserts evenly along the line from left to right
                # centre. y interpolates linearly along the row's actual slope.
                for j in range(n_insert):
                    t = (j + 1) / (n_insert + 1)
                    cx = left.cx + t * (right.cx - left.cx)
                    cy = left.cy + t * (right.cy - left.cy)
                    crop, off = self._crop(image, cx, cy, avg_w, avg_h,
                                            (H, W), cfg.crop_padding_ratio)
                    gaps.append(GapCandidate(
                        cx=cx, cy=cy, width=avg_w, height=avg_h,
                        row_idx=row_idx,
                        insert_idx=j, n_inserts=n_insert,
                        left_anchor_col=left.col_idx,
                        right_anchor_col=right.col_idx,
                        expected_stride=avg_w,
                        crop_img=crop, crop_offset=off,
                    ))
        return gaps

    @staticmethod
    def _crop(image: np.ndarray, cx: float, cy: float, w: float, h: float,
              sub_image_shape: Tuple[int, int], pad_ratio: float
              ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        H, W = sub_image_shape
        pad_x = w * pad_ratio
        pad_y = h * pad_ratio
        x1 = int(max(0, round(cx - w / 2 - pad_x)))
        y1 = int(max(0, round(cy - h / 2 - pad_y)))
        x2 = int(min(W, round(cx + w / 2 + pad_x)))
        y2 = int(min(H, round(cy + h / 2 + pad_y)))
        if x2 <= x1 or y2 <= y1:
            return None, (0, 0)
        return image[y1:y2, x1:x2].copy(), (x1, y1)


# ===========================================================================
# Sign identification: rank cached prototypes by DIFT-feature similarity.
# ===========================================================================

@dataclass
class IdentificationConfig:
    # Number of top results to return for diagnostics / visualisation.
    top_k: int = 5
    # K used in "mean of top-K best-buddy similarities". Higher = more
    # robust but tends to converge across all prototypes; lower = noisy.
    score_top_k: int = 20
    # Below this score the assignment is marked as low-confidence (still
    # included as a discovery, but flagged for downstream filtering).
    min_score: float = 0.0


@dataclass
class IdentificationResult:
    """One ranked sign candidate for a crop."""
    sign_name: str
    score: float
    record: CanonicalFeatureRecord


class SignIdentifier:
    """DIFT feature scorer: crop -> ranked list of canonical signs.

    Score: mean cosine similarity of the top-K mutual-NN (best-buddy)
    pairs between prototype and crop feature grids. Cheap (a single
    matmul + two argmaxes per prototype) and discriminative.
    """

    def __init__(self, cache: CanonicalFeatureCache,
                 config: Optional[IdentificationConfig] = None):
        self.cache = cache
        self.config = config or IdentificationConfig()

    def featurize_crop(self, crop: np.ndarray) -> torch.Tensor:
        if crop.ndim == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with torch.no_grad():
            return self.cache.wrapper.featurize(pil).detach()

    def identify(self, crop: np.ndarray) -> List[IdentificationResult]:
        """Return all cached prototypes sorted by score (best first).

        Caller typically takes [0] as the assigned label; the rest are
        kept for diagnostics / visualization.
        """
        if not self.cache.ready:
            return []
        crop_ft = self.featurize_crop(crop)
        # Move crop features to fp16 too, so the matmul uses uniform dtype
        # (saves both VRAM and FLOPs).
        crop_ft = crop_ft.to(dtype=torch.float16)
        device = crop_ft.device
        scored: List[IdentificationResult] = []
        for rec in self.cache.ordered_records:
            # Bring the prototype to the same device just for this scoring.
            proto_ft = rec.feature_map.to(device=device, dtype=torch.float16,
                                          non_blocking=True)
            s = self._score(proto_ft, crop_ft)
            scored.append(IdentificationResult(rec.sign_name, s, rec))
            # Release the temporary copy.
            del proto_ft
        scored.sort(key=lambda r: -r.score)
        return scored

    def _score(self, proto_ft: torch.Tensor, crop_ft: torch.Tensor) -> float:
        # proto_ft, crop_ft: (C, 64, 64), L2-normalized on C.
        # Compute (Hp*Wp, Ht*Wt) cosine similarity matrix by matmul.
        C = proto_ft.shape[0]
        p_flat = proto_ft.reshape(C, -1)        # (C, 4096)
        t_flat = crop_ft.reshape(C, -1)         # (C, 4096)
        sim = p_flat.T @ t_flat                  # (4096, 4096) fp16

        # Mutual NN (best-buddy) mask.
        p2t = sim.argmax(dim=1)
        t2p = sim.argmax(dim=0)
        idx = torch.arange(sim.shape[0], device=sim.device)
        is_mutual = (t2p[p2t] == idx)
        if not is_mutual.any():
            return 0.0

        bb_sims = sim[idx[is_mutual], p2t[is_mutual]]
        k = min(self.config.score_top_k, int(bb_sims.numel()))
        if k <= 0:
            return 0.0
        return float(torch.topk(bb_sims.float(), k).values.mean().item())


# ===========================================================================
# Discovery: gap detection + identification + summary container
# ===========================================================================

@dataclass
class GapDiscovery:
    """A gap candidate after sign identification."""
    gap: GapCandidate
    best: IdentificationResult
    top_k: List[IdentificationResult] = field(default_factory=list)
    low_confidence: bool = False

    @property
    def sign_name(self) -> str: return self.best.sign_name
    @property
    def score(self) -> float: return self.best.score

    def to_bounding_box(self) -> BoundingBox:
        """Convert into a BoundingBox so PSR can treat this as a detection."""
        sign = SignResolver.from_name(self.best.sign_name)
        return BoundingBox(
            x1=self.gap.cx - self.gap.width / 2,
            y1=self.gap.cy - self.gap.height / 2,
            x2=self.gap.cx + self.gap.width / 2,
            y2=self.gap.cy + self.gap.height / 2,
            score=float(self.best.score),
            sign=sign,
        )


@dataclass
class DiscoveryConfig:
    gap: GapDetectionConfig = field(default_factory=GapDetectionConfig)
    identification: IdentificationConfig = field(default_factory=IdentificationConfig)
    # Eager precompute control.
    precompute_full_inventory: bool = True
    precompute_limit: Optional[int] = None   # cap (None = all)
    # If True, also include text-derived sign names in precompute even when
    # they're not part of the full inventory (no-op for ProtoSnap source).
    precompute_include_text: bool = True


def summarize_discoveries(discoveries: List[GapDiscovery],
                          n_gaps_total: int) -> Dict[str, Any]:
    confident = sum(1 for d in discoveries if not d.low_confidence)
    low = sum(1 for d in discoveries if d.low_confidence)
    by_sign = Counter(d.sign_name for d in discoveries)
    return {
        "gaps_total": n_gaps_total,
        "discoveries": len(discoveries),
        "confident": confident,
        "low_confidence": low,
        "by_sign_top10": dict(by_sign.most_common(10)),
    }


# ===========================================================================
# Visualisation: per-gap grid (crop + top-K prototypes)
# ===========================================================================

def render_discovery_grid(discoveries: List[GapDiscovery],
                          thumb: int = 96, top_k_show: int = 3) -> np.ndarray:
    """One row per discovery: [crop | top-1 proto (assigned) | top-2 | ...].

    Each cell is annotated with the sign name and the DIFT score.  The
    assigned (top-1) cell gets a green border, the rest gray.
    """
    if not discoveries:
        return np.full((thumb + 30, thumb * (top_k_show + 1) + 20, 3), 15, np.uint8)

    HDR_H = 22
    BORDER = 2
    GAP = 4
    CELL = thumb + 2 * BORDER + HDR_H + GAP
    n_cols = top_k_show + 1   # crop + K prototypes
    n_rows = len(discoveries)
    W = GAP + n_cols * (thumb + 2 * BORDER + GAP)
    Hh = GAP + n_rows * CELL
    grid = np.full((Hh, W, 3), 18, np.uint8)

    GREEN = (80, 200, 80)
    GRAY = (110, 110, 110)
    AMBER = (40, 170, 220)

    def _fit(img: np.ndarray) -> np.ndarray:
        canvas = np.full((thumb, thumb, 3), 40, np.uint8)
        if img is None:
            return canvas
        im = img
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        if max(h, w) == 0:
            return canvas
        sc = thumb / max(h, w)
        nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
        rsz = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        yo, xo = (thumb - nh) // 2, (thumb - nw) // 2
        canvas[yo:yo + nh, xo:xo + nw] = rsz
        return canvas

    def _text(canvas, txt, y, color=(220, 220, 220), scale=0.36, x=4):
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    for r, disc in enumerate(discoveries):
        y0 = GAP + r * CELL
        x0 = GAP

        # column 0: crop with header (row/col + score)
        crop_cell_w = thumb + 2 * BORDER
        cell = np.full((HDR_H + thumb, crop_cell_w, 3), 24, np.uint8)
        _text(cell, f"R{disc.gap.row_idx} insert{disc.gap.insert_idx+1}/"
                    f"{disc.gap.n_inserts}", HDR_H - 6,
              AMBER if disc.low_confidence else GREEN, scale=0.36)
        cell[HDR_H:, BORDER:BORDER + thumb] = _fit(_crop_to_show(disc))
        grid[y0:y0 + cell.shape[0], x0:x0 + cell.shape[1]] = cell

        # column 1..K: top-K prototypes (column 1 = assigned)
        for k in range(top_k_show):
            xk = x0 + (k + 1) * (thumb + 2 * BORDER + GAP)
            color = GREEN if k == 0 else GRAY
            if disc.low_confidence and k == 0:
                color = AMBER
            sub = disc.top_k[k] if k < len(disc.top_k) else None
            cell = np.full((HDR_H + thumb, crop_cell_w, 3), 30, np.uint8)
            cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, cell.shape[0] - 1),
                          color, BORDER)
            if sub is not None:
                name = sub.sign_name[:12]
                _text(cell, f"{name}  {sub.score:.3f}",
                      HDR_H - 6, color, scale=0.34)
                cell[HDR_H:, BORDER:BORDER + thumb] = _fit(sub.record.img)
            grid[y0:y0 + cell.shape[0], xk:xk + cell.shape[1]] = cell

    return grid


def _crop_to_show(disc: GapDiscovery) -> Optional[np.ndarray]:
    return disc.gap.crop_img
