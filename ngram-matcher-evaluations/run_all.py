"""Aggregate the per-anchor summary CSVs into one Table 8-ready CSV.

Reads each anchor's ``*_summary.csv`` (if present) from the output directory,
plus the headline numbers from the input CSV, and writes
``anchors_summary.csv`` in the format requested by Section 4 of the spec:

    anchor, subset, n_fragments, score_mean, score_median,
    score_ci_low, score_ci_high, overlap_mean

This does not re-run any baselines — it only aggregates whatever each
anchor script has produced.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    INPUT_CSV,
    OUTPUT_DIR,
    bootstrap_mean_ci,
    ensure_output_dir,
    load_headline_csv,
)

# ============================================================================
# Configuration
# ============================================================================

INPUT_CSV = INPUT_CSV       # headline evaluation CSV
OUTPUT_DIR = OUTPUT_DIR     # directory containing per-anchor summary CSVs
N_RESAMPLES = 10_000        # bootstrap resample count
SEED = 0                    # random seed for bootstrap


COLUMNS = [
    "anchor", "subset", "n_fragments",
    "score_mean", "score_median",
    "score_ci_low", "score_ci_high",
    "overlap_mean",
]


def _headline_row(df: pd.DataFrame, n_resamples: int, seed: int) -> dict:
    scores = df["score"].to_numpy()
    mean, lo, hi = bootstrap_mean_ci(scores, n_resamples=n_resamples, seed=seed)
    return {
        "anchor": "headline",
        "subset": "headline_eval",
        "n_fragments": int(len(scores)),
        "score_mean": mean,
        "score_median": float(np.median(scores)),
        "score_ci_low": lo,
        "score_ci_high": hi,
        "overlap_mean": float(df["overlap"].mean()),
    }


def _read_seed_summary(path: Path, anchor: str, subset: str) -> dict | None:
    if not path.exists():
        return None
    row = pd.read_csv(path).iloc[0].to_dict()
    return {
        "anchor": anchor,
        "subset": subset,
        "n_fragments": int(row["n_fragments"]),
        "score_mean": float(row["score_mean_across_seeds"]),
        "score_median": float(row["score_median_first_seed"]),
        "score_ci_low": float(row["score_ci_low_first_seed"]),
        "score_ci_high": float(row["score_ci_high_first_seed"]),
        "overlap_mean": float(row["overlap_mean_across_seeds"]),
    }


def _read_simple_summary(path: Path) -> List[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        {k: r[k] for k in COLUMNS if k in r}
        for r in df.to_dict(orient="records")
    ]


def main() -> None:
    out_dir = ensure_output_dir(OUTPUT_DIR)
    df = load_headline_csv(INPUT_CSV)

    rows: List[dict] = [_headline_row(df, N_RESAMPLES, SEED)]

    lb1 = _read_seed_summary(out_dir / "lb1_summary.csv", "LB1_random_pred", "headline_eval")
    if lb1: rows.append(lb1)

    lb2 = _read_seed_summary(out_dir / "lb2_summary.csv", "LB2_shuffle", "headline_eval")
    if lb2: rows.append(lb2)

    rows.extend(_read_simple_summary(out_dir / "ub1_summary.csv"))
    rows.extend(_read_simple_summary(out_dir / "ub2_summary.csv"))

    final = pd.DataFrame(rows)
    # Reorder columns deterministically
    for c in COLUMNS:
        if c not in final.columns:
            final[c] = float("nan")
    final = final[COLUMNS]

    out_path = out_dir / "anchors_summary.csv"
    final.to_csv(out_path, index=False)
    print(final.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
