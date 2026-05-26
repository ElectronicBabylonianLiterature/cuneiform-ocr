"""UB1 — 173-class vocabulary ceiling.

The detection head only outputs ABZ codes in ``VOCAB_173``. The ground truth
reference, however, comes from ``SignToABZResolver`` which resolves *any*
ABZ via MongoDB — so reference sequences can contain ABZs outside the
detection vocabulary.

UB1 measures the ceiling imposed by that vocabulary mismatch:

    1. Take the reference ``S_ref(f)`` (the same sequence the headline used).
    2. Map every token not in ``VOCAB_173`` to ``X`` (UnclearSign) — exactly
       what the detector would have done if it had recognized that token
       perfectly but the vocabulary forced it to be merged.
    3. Score the mapped sequence against the unmapped reference under the
       same matcher as the headline.

Two numbers come out of this:

* ``score_mean`` — the headline metric ``|A∩B| / min(|A|, |B|)`` between the
  mapped reference (A) and the raw reference (B). Because the matcher's
  ``postprocess`` drops every n-gram containing ``X`` from *both* sides and
  ``A ⊆ B`` after that drop, this is mathematically forced to be 1.0
  whenever A is non-empty. The headline metric is therefore *insensitive*
  to the 173-class collapse; UB1 ≈ 1.0 documents that fact.

* ``recall_mean`` — ``|A∩B| / |B|``, i.e. the fraction of reference n-grams
  that survive the vocabulary mapping. This is the *actual* ceiling loss
  caused by the 173-class collapse. Reported as a supplementary diagnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    INPUT_CSV,
    OUTPUT_DIR,
    VOCAB_173,
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
N_RESAMPLES = 10_000        # bootstrap resample count
SEED = 0                    # random seed for bootstrap


def map_to_173(signs: str) -> str:
    """Replace tokens not in ``VOCAB_173`` with ``X``, preserving line breaks."""
    if not signs:
        return ""
    out_lines: List[str] = []
    for line in signs.split("\n"):
        toks = line.split()
        if not toks:
            out_lines.append("")
            continue
        out_lines.append(" ".join(t if t in VOCAB_173 else "X" for t in toks))
    return "\n".join(out_lines)


def main() -> None:
    out_dir = ensure_output_dir(OUTPUT_DIR)
    df = load_headline_csv(INPUT_CSV)

    print(f"[UB1] |VOCAB_173| = {len(VOCAB_173)} (unique ABZ codes)")

    records = []
    scores: List[float] = []
    overlaps: List[float] = []
    recalls: List[float] = []
    oov_fracs: List[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="UB1"):
        ref = row["ground_truth_signs"]
        if not ref.strip():
            continue
        toks = ref.split()
        oov = sum(1 for t in toks if t not in VOCAB_173)
        oov_frac = oov / len(toks) if toks else 0.0
        oov_fracs.append(oov_frac)

        mapped = map_to_173(ref)
        ref_ngrams = build_ngrams(ref, fragment_id=f"{row['fragment_id']}_ref")
        mapped_ngrams = build_ngrams(
            mapped, fragment_id=f"{row['fragment_id']}_mapped"
        )
        s, o = score_against(mapped_ngrams, ref_ngrams)

        # Recall: how much of the reference's n-gram budget survives the
        # 173-class collapse. This is the *non-trivial* ceiling — the
        # headline metric (`score`) is forced to 1.0 by the matcher's
        # X-dropping postprocess, see module docstring.
        recall = o / len(ref_ngrams) if ref_ngrams else 0.0

        scores.append(s)
        overlaps.append(o)
        recalls.append(recall)
        records.append({
            "fragment_id": row["fragment_id"],
            "score": s,
            "overlap": o,
            "recall": recall,
            "n_ref_ngrams": len(ref_ngrams),
            "n_mapped_ngrams": len(mapped_ngrams),
            "n_tokens": len(toks),
            "n_oov": oov,
            "oov_frac": oov_frac,
        })

    scores_arr = np.asarray(scores)
    overlaps_arr = np.asarray(overlaps)
    recalls_arr = np.asarray(recalls)
    oov_arr = np.asarray(oov_fracs)

    mean, ci_lo, ci_hi = bootstrap_mean_ci(
        scores_arr, n_resamples=N_RESAMPLES, seed=SEED
    )
    r_mean, r_lo, r_hi = bootstrap_mean_ci(
        recalls_arr, n_resamples=N_RESAMPLES, seed=SEED
    )
    summary = {
        "anchor": "UB1_label_map",
        "subset": "headline_eval",
        "n_fragments": int(len(scores_arr)),
        "score_mean": mean,
        "score_median": float(np.median(scores_arr)),
        "score_ci_low": ci_lo,
        "score_ci_high": ci_hi,
        "overlap_mean": float(overlaps_arr.mean()),
        "overlap_median": float(np.median(overlaps_arr)),
        "recall_mean": r_mean,
        "recall_median": float(np.median(recalls_arr)),
        "recall_ci_low": r_lo,
        "recall_ci_high": r_hi,
        "oov_token_frac_mean": float(oov_arr.mean()),
        "oov_token_frac_median": float(np.median(oov_arr)),
    }

    per_frag = pd.DataFrame(records)
    per_frag_path = out_dir / "ub1_per_fragment.parquet"
    per_frag.to_parquet(per_frag_path, index=False)
    summary_df = pd.DataFrame([summary])
    summary_path = out_dir / "ub1_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved: {per_frag_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
