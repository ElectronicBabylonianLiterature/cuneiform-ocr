"""Shared utilities for the anchor / baseline scripts.

Design points:

* All anchors operate on the headline evaluation CSV (per-fragment detected
  and reference sign sequences). This avoids re-loading the detection JSON
  and re-querying MongoDB for the ground-truth — the CSV already contains
  exactly what the headline matcher consumed.
* The same n-gram pipeline as the headline (`FragmentModel` with the default
  ``n_values=(1, 2, 3)``) is used so that anchor scores are directly
  comparable to the 0.232 headline. Do not bypass ``preprocess`` /
  ``postprocess``; rebuild the FragmentModel for each sequence and pull the
  n-gram set out via ``get_ngrams``.
* The 173-class detection vocabulary is loaded from
  ``data_processing.sign_resolver.CLASSES_ABZ``. ``X`` and ``NoABZ0`` are
  both the UnclearSign bucket and are treated as out-of-vocabulary for the
  purpose of UB1's label-mapping ceiling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent  # repo root (cuneiform-ocr/)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `ebl_ngrams` lives in the sibling ngram-matcher checkout. The
# `cuneiform-ocr` conda env doesn't have it installed as a package — fall
# back to importing from source if the package import fails.
_NGRAM_SRC = ROOT.parent / "ngram-matcher" / "src"
try:
    from ebl_ngrams import FragmentModel  # noqa: E402
except ModuleNotFoundError:
    if _NGRAM_SRC.is_dir() and str(_NGRAM_SRC) not in sys.path:
        sys.path.insert(0, str(_NGRAM_SRC))
    from ebl_ngrams import FragmentModel  # noqa: E402

from data_processing.sign_resolver import CLASSES_ABZ  # noqa: E402


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

INPUT_CSV = (
    ROOT
    / "evaluation_output_new_line_det_0.8_0.35_2_0.006_20260225"
    / "evaluation_results.csv"
)
OUTPUT_DIR = (
    ROOT
    / "evaluation_output_new_line_det_0.8_0.35_2_0.006_20260525-anchor-test-generate-lbs-ubs"
)

# Keep legacy aliases so any external code using the old names still works.
DEFAULT_INPUT_CSV = INPUT_CSV
DEFAULT_OUTPUT_DIR = OUTPUT_DIR


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# The detection head is constrained to this set of ABZ codes. `X` and
# `NoABZ0` are the UnclearSign bucket (all <90-instance classes merged) and
# are deliberately *excluded* from the 173-class vocabulary used for UB1.
# The matcher's `postprocess` also drops any n-gram containing `X`, so
# treating UnclearSign as OOV here matches the headline's behavior end-to-end.
VOCAB_173: Set[str] = {abz for abz in CLASSES_ABZ if abz not in ("X", "NoABZ0")}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_headline_csv(path: Path = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    """Load the headline evaluation CSV with sensible dtypes / NaN handling."""
    df = pd.read_csv(path, dtype={"fragment_id": str, "filename": str})
    df["detected_signs"] = df["detected_signs"].fillna("").astype(str)
    df["ground_truth_signs"] = df["ground_truth_signs"].fillna("").astype(str)
    return df


# ---------------------------------------------------------------------------
# N-gram construction and scoring
# ---------------------------------------------------------------------------

def build_ngrams(signs: str, fragment_id: str = "") -> Set[Tuple[str, ...]]:
    """Return the headline-equivalent n-gram set for a sign sequence.

    Empty / whitespace-only sequences map to an empty set so the caller can
    short-circuit without constructing a FragmentModel.
    """
    if not signs or not signs.strip():
        return set()
    fm = FragmentModel(id_=fragment_id, signs=signs)
    # `get_ngrams()` with no args returns the union over `n_values`
    # (1, 2, 3) by default — matching the headline.
    return set(fm.get_ngrams())


def score_against(
    pred_ngrams: Set[Tuple[str, ...]],
    ref_ngrams: Set[Tuple[str, ...]],
) -> Tuple[float, int]:
    """Mirror ``FragmentModel.match`` (no length weighting)."""
    a, b = len(pred_ngrams), len(ref_ngrams)
    if a == 0 or b == 0:
        return 0.0, 0
    overlap = len(pred_ngrams & ref_ngrams)
    return overlap / min(a, b), overlap


def build_ngram_cache(
    df: pd.DataFrame,
    col: str,
    desc: str = "",
) -> List[Set[Tuple[str, ...]]]:
    """Build an ordered list of n-gram sets, one entry per row of *df*.

    Returns a list (not a dict) so callers can index by row position for
    fast permutation-based shuffling.
    """
    out: List[Set[Tuple[str, ...]]] = []
    desc = desc or f"ngrams[{col}]"
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        out.append(build_ngrams(row[col], row["fragment_id"]))
    return out


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high). Empty input → (nan, nan, nan)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(arr.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Output dir helper
# ---------------------------------------------------------------------------

def ensure_output_dir(path: Path = DEFAULT_OUTPUT_DIR) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Headline sanity reproduction (used by --sanity flags in each anchor script)
# ---------------------------------------------------------------------------

def reproduce_headline(df: pd.DataFrame) -> Tuple[float, float]:
    """Rerun the matcher on (detected_signs, ground_truth_signs) to check
    that our common-utilities path reproduces the CSV's stored mean score.

    Returns (computed_mean, csv_mean). Both should be within ~1e-6 of each
    other; anything larger means something drifted and the anchor results
    will not be comparable to the headline.
    """
    pred = build_ngram_cache(df, "detected_signs", desc="sanity[pred]")
    ref = build_ngram_cache(df, "ground_truth_signs", desc="sanity[ref]")
    scores = np.empty(len(df), dtype=float)
    for i, (a, b) in enumerate(zip(pred, ref)):
        scores[i], _ = score_against(a, b)
    return float(scores.mean()), float(df["score"].mean())
