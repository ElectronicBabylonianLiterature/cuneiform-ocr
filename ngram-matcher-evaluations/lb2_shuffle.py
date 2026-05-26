"""LB2 — cross-fragment shuffle baseline.

For each fragment ``f`` in the headline subset, score the model's predicted
sequence ``S_pred(f)`` against the reference sequence ``S_ref(f')`` of a
different randomly-chosen fragment ``f'``. Repeat with ``M`` independent
derangement seeds. Report mean ± std across seeds, plus a representative
bootstrap CI from one run (matching Table 7's CI methodology).

Per-fragment, per-seed scores are saved to a Parquet long-format file so
A1-style stratification can be reapplied later without re-running the matcher.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    INPUT_CSV,
    OUTPUT_DIR,
    bootstrap_mean_ci,
    build_ngram_cache,
    ensure_output_dir,
    load_headline_csv,
    score_against,
)

# ============================================================================
# Configuration
# ============================================================================

INPUT_CSV = INPUT_CSV       # headline evaluation CSV
OUTPUT_DIR = OUTPUT_DIR     # where results are written
M_SEEDS = 30                # number of independent derangement seeds
BASE_SEED = 0               # first seed value
N_RESAMPLES = 10_000        # bootstrap resample count


def derangement(n: int, rng: np.random.Generator, max_retries: int = 50) -> np.ndarray:
    """Return a permutation of [0..n) with no fixed point.

    The probability of a random permutation being a derangement is ~1/e for
    large n, so a few retries are usually enough. As a robust fallback we
    fix any residual fixed point by swapping with the next index.
    """
    for _ in range(max_retries):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    # Robust fallback: rotate by 1 then resolve fixed points pairwise.
    perm = np.arange(n)
    rng.shuffle(perm)
    fixed = np.where(perm == np.arange(n))[0]
    for idx in fixed:
        swap = (idx + 1) % n
        perm[idx], perm[swap] = perm[swap], perm[idx]
    assert not np.any(perm == np.arange(n)), "Derangement fallback failed"
    return perm


def run_one_shuffle(
    pred_ngrams: List[set],
    ref_ngrams: List[set],
    perm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(pred_ngrams)
    scores = np.empty(n, dtype=float)
    overlaps = np.empty(n, dtype=float)
    for i in range(n):
        s, o = score_against(pred_ngrams[i], ref_ngrams[int(perm[i])])
        scores[i] = s
        overlaps[i] = o
    return scores, overlaps


def main() -> None:
    out_dir = ensure_output_dir(OUTPUT_DIR)
    df = load_headline_csv(INPUT_CSV)

    pred_cache = build_ngram_cache(df, "detected_signs", desc="ngrams[pred]")
    ref_cache = build_ngram_cache(df, "ground_truth_signs", desc="ngrams[ref]")
    n = len(df)
    fragment_ids = df["fragment_id"].tolist()

    per_seed_score_mean: List[float] = []
    per_seed_overlap_mean: List[float] = []
    per_seed_records: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    for k in range(M_SEEDS):
        seed = BASE_SEED + k
        rng = np.random.default_rng(seed)
        perm = derangement(n, rng)
        scores, overlaps = run_one_shuffle(pred_cache, ref_cache, perm)
        per_seed_score_mean.append(float(scores.mean()))
        per_seed_overlap_mean.append(float(overlaps.mean()))
        per_seed_records.append((seed, scores, overlaps, perm))
        print(
            f"[LB2] seed={seed}  mean_score={scores.mean():.4f}  "
            f"mean_overlap={overlaps.mean():.3f}"
        )

    # Long-format dump (seed, fragment_id, paired_with, score, overlap)
    rows = []
    for seed, scores, overlaps, perm in per_seed_records:
        for i in range(n):
            rows.append((
                seed,
                fragment_ids[i],
                fragment_ids[int(perm[i])],
                float(scores[i]),
                float(overlaps[i]),
            ))
    long_df = pd.DataFrame(
        rows, columns=["seed", "fragment_id", "paired_with", "score", "overlap"]
    )
    long_path = out_dir / "lb2_per_fragment_seed.parquet"
    long_df.to_parquet(long_path, index=False)
    print(f"Saved: {long_path}")

    rep_seed, rep_scores, rep_overlaps, _ = per_seed_records[0]
    _, ci_lo, ci_hi = bootstrap_mean_ci(
        rep_scores, n_resamples=N_RESAMPLES, seed=BASE_SEED
    )

    per_seed_score_mean_arr = np.asarray(per_seed_score_mean)
    per_seed_overlap_mean_arr = np.asarray(per_seed_overlap_mean)
    summary = {
        "anchor": "LB2_shuffle",
        "subset": "headline_eval",
        "n_fragments": n,
        "m_seeds": M_SEEDS,
        "score_mean_across_seeds": float(per_seed_score_mean_arr.mean()),
        "score_std_across_seeds": float(per_seed_score_mean_arr.std(ddof=1))
        if M_SEEDS > 1 else 0.0,
        "score_median_first_seed": float(np.median(rep_scores)),
        "score_ci_low_first_seed": ci_lo,
        "score_ci_high_first_seed": ci_hi,
        "overlap_mean_across_seeds": float(per_seed_overlap_mean_arr.mean()),
        "overlap_std_across_seeds": float(per_seed_overlap_mean_arr.std(ddof=1))
        if M_SEEDS > 1 else 0.0,
        "overlap_median_first_seed": float(np.median(rep_overlaps)),
        "representative_seed": int(rep_seed),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = out_dir / "lb2_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
