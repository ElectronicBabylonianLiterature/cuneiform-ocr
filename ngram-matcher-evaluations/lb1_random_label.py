"""LB1 — random-label baseline (P_pred version).

For each detection in ``S_pred(f)``, resample the label iid from the empirical
class-frequency distribution ``P_pred`` (computed across all detected
sequences in the headline run). Bounding-box / DBSCAN line-grouping
structure is preserved implicitly: we keep the existing per-fragment flat
sequence, including ``\\n`` line separators, and only swap token labels.
This is equivalent to "preserve bbox, resample label, rerun line_signs"
because the DBSCAN line-grouping in ``data_processing.line_process`` is
label-independent (clusters by centroid only).

P_train (the training-set class-frequency distribution) is intentionally
not implemented — only P_pred. If LB1 vs LB2 differ noticeably and the
team wants P_train as a second cross-check, the COCO training annotations
will need to be loaded from the remote mount.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    INPUT_CSV,
    OUTPUT_DIR,
    bootstrap_mean_ci,
    build_ngrams,
    ensure_output_dir,
    load_headline_csv,
    score_against,
)

# ============================================================================
# Configuration
# ============================================================================

INPUT_CSV = INPUT_CSV       # headline evaluation CSV
OUTPUT_DIR = OUTPUT_DIR     # where results are written
M_SEEDS = 30                # number of independent random seeds
BASE_SEED = 0               # first seed value
N_RESAMPLES = 10_000        # bootstrap resample count


def compute_p_pred(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    counter: Counter = Counter()
    for s in df["detected_signs"]:
        if not s:
            continue
        for line in s.split("\n"):
            counter.update(line.split())
    if not counter:
        raise RuntimeError("No detected tokens found — cannot estimate P_pred.")
    labels = list(counter.keys())
    counts = np.array([counter[lab] for lab in labels], dtype=float)
    probs = counts / counts.sum()
    return labels, probs


def relabel(
    signs: str,
    labels: List[str],
    probs: np.ndarray,
    rng: np.random.Generator,
) -> str:
    if not signs:
        return ""
    out_lines: List[str] = []
    for line in signs.split("\n"):
        toks = line.split()
        if not toks:
            out_lines.append("")
            continue
        idx = rng.choice(len(labels), size=len(toks), p=probs)
        out_lines.append(" ".join(labels[i] for i in idx))
    return "\n".join(out_lines)


def main() -> None:
    out_dir = ensure_output_dir(OUTPUT_DIR)
    df = load_headline_csv(INPUT_CSV)

    labels, probs = compute_p_pred(df)
    top5 = sorted(zip(probs, labels), reverse=True)[:5]
    print(f"[LB1] |vocab| = {len(labels)}  top-5 = {[(l, round(float(p), 4)) for p, l in top5]}")

    # Build ref ngrams once (they are constant across seeds).
    fragment_ids = df["fragment_id"].tolist()
    ref_ngrams_list: List[set] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ngrams[ref]"):
        ref_ngrams_list.append(build_ngrams(row["ground_truth_signs"], row["fragment_id"]))

    per_seed_score_mean: List[float] = []
    per_seed_overlap_mean: List[float] = []
    per_seed_records: List[Tuple[int, np.ndarray, np.ndarray]] = []

    detected = df["detected_signs"].tolist()
    n = len(df)
    for k in range(M_SEEDS):
        seed = BASE_SEED + k
        rng = np.random.default_rng(seed)
        scores = np.empty(n, dtype=float)
        overlaps = np.empty(n, dtype=float)
        for i, det in enumerate(detected):
            relabeled = relabel(det, labels, probs, rng)
            pred_ngrams = build_ngrams(relabeled, fragment_ids[i])
            s, o = score_against(pred_ngrams, ref_ngrams_list[i])
            scores[i] = s
            overlaps[i] = o
        per_seed_score_mean.append(float(scores.mean()))
        per_seed_overlap_mean.append(float(overlaps.mean()))
        per_seed_records.append((seed, scores, overlaps))
        print(
            f"[LB1] seed={seed}  mean_score={scores.mean():.4f}  "
            f"mean_overlap={overlaps.mean():.3f}"
        )

    # Long-format dump (seed, fragment_id, score, overlap)
    rows = []
    for seed, scores, overlaps in per_seed_records:
        for i in range(n):
            rows.append((seed, fragment_ids[i], float(scores[i]), float(overlaps[i])))
    long_df = pd.DataFrame(rows, columns=["seed", "fragment_id", "score", "overlap"])
    long_path = out_dir / "lb1_per_fragment_seed.parquet"
    long_df.to_parquet(long_path, index=False)
    print(f"Saved: {long_path}")

    rep_seed, rep_scores, rep_overlaps = per_seed_records[0]
    _, ci_lo, ci_hi = bootstrap_mean_ci(
        rep_scores, n_resamples=N_RESAMPLES, seed=BASE_SEED
    )
    per_seed_score_mean_arr = np.asarray(per_seed_score_mean)
    per_seed_overlap_mean_arr = np.asarray(per_seed_overlap_mean)
    summary = {
        "anchor": "LB1_random_pred",
        "subset": "headline_eval",
        "n_fragments": n,
        "m_seeds": M_SEEDS,
        "p_pred_vocab_size": len(labels),
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
    summary_path = out_dir / "lb1_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
