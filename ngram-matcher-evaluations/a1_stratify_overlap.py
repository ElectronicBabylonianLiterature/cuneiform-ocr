"""A1 — stratify per-fragment match scores by overlap size.

Pure post-hoc analysis of the headline CSV: no new matcher computation, no
ngram cache. Bins fragments by overlap size and reports mean / median /
95% bootstrap CI per bin.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
OUTPUT_DIR = OUTPUT_DIR     # where results are written
N_RESAMPLES = 10_000        # bootstrap resample count
SEED = 0                    # random seed for bootstrap

# (lo, hi) is half-open [lo, hi); hi=None means open-ended.
BINS = [
    (1, 5, "1-4"),
    (5, 10, "5-9"),
    (10, 20, "10-19"),
    (20, 50, "20-49"),
    (50, None, "50+"),
]


def stratify(
    df: pd.DataFrame,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    overall_scores = df["score"].to_numpy()
    overall_mean, overall_lo, overall_hi = bootstrap_mean_ci(
        overall_scores, n_resamples=n_resamples, seed=seed
    )
    rows.append({
        "overlap_bin": "ALL",
        "n_fragments": int(len(overall_scores)),
        "score_mean": overall_mean,
        "score_median": float(np.median(overall_scores)),
        "score_ci_low": overall_lo,
        "score_ci_high": overall_hi,
    })
    # Also surface the overlap == 0 fragments separately — they dominate the
    # left tail of the headline distribution and are easy to lose sight of
    # otherwise.
    zero_mask = df["overlap"] == 0
    rows.append({
        "overlap_bin": "0",
        "n_fragments": int(zero_mask.sum()),
        "score_mean": float(df.loc[zero_mask, "score"].mean()) if zero_mask.any() else float("nan"),
        "score_median": float(df.loc[zero_mask, "score"].median()) if zero_mask.any() else float("nan"),
        "score_ci_low": float("nan"),
        "score_ci_high": float("nan"),
    })
    for lo, hi, label in BINS:
        if hi is None:
            sub = df[df["overlap"] >= lo]
        else:
            sub = df[(df["overlap"] >= lo) & (df["overlap"] < hi)]
        scores = sub["score"].to_numpy()
        if len(scores) == 0:
            rows.append({
                "overlap_bin": label,
                "n_fragments": 0,
                "score_mean": float("nan"),
                "score_median": float("nan"),
                "score_ci_low": float("nan"),
                "score_ci_high": float("nan"),
            })
            continue
        mean, ci_lo, ci_hi = bootstrap_mean_ci(scores, n_resamples=n_resamples, seed=seed)
        rows.append({
            "overlap_bin": label,
            "n_fragments": int(len(scores)),
            "score_mean": mean,
            "score_median": float(np.median(scores)),
            "score_ci_low": ci_lo,
            "score_ci_high": ci_hi,
        })
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = ensure_output_dir(OUTPUT_DIR)
    df = load_headline_csv(INPUT_CSV)

    table = stratify(df, n_resamples=N_RESAMPLES, seed=SEED)
    print(table.to_string(index=False))

    out_path = out_dir / "a1_stratification.csv"
    table.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
