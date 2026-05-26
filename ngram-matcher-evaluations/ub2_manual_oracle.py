"""UB2 — manual-annotation oracle.

Uses human-annotated bounding boxes as if they were the detector output:
runs the unchanged DBSCAN line-grouping + L→R sequencing pipeline (the same
``line_signs`` function the headline run uses), then matches against the
eBL transliteration reference.

Default config aligns with the headline inference run:
    eps=0.35, min_samples=2, lambda_weight=0.006

Subset construction (matches the spec):
    - Pull every fragment that has manual bounding-box annotations in the
      COCO val split (``instances_val2017.json``). The train split is
      excluded — those fragments were used to train the detector.
    - Intersect with the headline CSV's fragment_id set (i.e., fragments
      that also have a transliteration). The headline CSV already excludes
      training fragments, so the intersection naturally drops them.
    - The resulting subset is what gets scored.

Two rows are produced:
    UB2_oracle           — manual bboxes → line_signs → match against GT
    UB2_model_on_same    — headline score column restricted to the same subset

The COCO JSON path lives on the remote sshfs mount which can flake. If the
file is missing, the script fails early with a clear message.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

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

INPUT_CSV = INPUT_CSV           # headline evaluation CSV
OUTPUT_DIR = OUTPUT_DIR         # where results are written
COCO_JSON = Path(
    "~/erc-work-data/coco-recognitioin-2025-09-25/data-coco/coco/annotations/instances_val2017.json"
).expanduser()                  # COCO val annotations (needs remote mount)
EPS = 0.35                      # DBSCAN epsilon (must match headline)
MIN_SAMPLES = 2                 # DBSCAN min_samples (must match headline)
LAMBDA_WEIGHT = 0.006           # line-sort lambda (must match headline)
N_RESAMPLES = 10_000            # bootstrap resample count
SEED = 0                        # random seed for bootstrap


def _resolve_category_to_abz(cat_name: str, classes: List[str]) -> str:
    """Map a COCO category name to an ABZ code.

    Tries:
        1. If the name is already in ``classes`` (e.g., ``ABZ58``), use it directly.
        2. Otherwise try to interpret the name as a 0-based class index.
        3. Otherwise return ``'X'`` (UnclearSign).
    """
    if cat_name in classes:
        return cat_name
    try:
        idx = int(cat_name)
        if 0 <= idx < len(classes):
            return classes[idx]
    except (TypeError, ValueError):
        pass
    return "X"


def load_coco_annotations(
    coco_json: Path, classes: List[str]
) -> Dict[str, List[Tuple[List[float], str]]]:
    """Return {fragment_id -> [(bbox_xyxy, abz_code), ...]} from a COCO JSON file.

    Bboxes are converted from COCO ``[x, y, w, h]`` to xyxy in line with
    ``line_signs``'s ``_BoxWrapper`` (which expects ``[x1, y1, x2, y2]``).

    ``fragment_id`` is derived from the image filename's stem (``foo.jpg`` →
    ``foo``) so it lines up with the headline CSV.
    """
    try:
        exists = coco_json.exists()
    except OSError as e:
        raise FileNotFoundError(
            f"Cannot stat COCO JSON ({e!s}). The sshfs mount at "
            f"{coco_json.parent} is likely down; remount and retry, or pass "
            "--coco-json with a local override."
        ) from e
    if not exists:
        raise FileNotFoundError(
            f"COCO JSON not found: {coco_json}\n"
            "Pass --coco-json with a local override if the mount path differs."
        )
    with coco_json.open() as f:
        data = json.load(f)

    cat_id_to_abz = {
        c["id"]: _resolve_category_to_abz(str(c["name"]), classes)
        for c in data["categories"]
    }
    image_id_to_frag = {
        img["id"]: Path(img["file_name"]).stem for img in data["images"]
    }

    out: Dict[str, List[Tuple[List[float], str]]] = {}
    for ann in data["annotations"]:
        frag = image_id_to_frag.get(ann["image_id"])
        if frag is None:
            continue
        x, y, w, h = ann["bbox"]
        bbox_xyxy = [float(x), float(y), float(x + w), float(y + h)]
        abz = cat_id_to_abz.get(ann["category_id"], "X")
        out.setdefault(frag, []).append((bbox_xyxy, abz))
    return out


def manual_to_sequence(
    boxes_with_abz: List[Tuple[List[float], str]],
    classes: List[str],
    eps: float,
    min_samples: int,
    lambda_weight: float,
) -> str:
    """Convert manual annotations into the canonical sequenced string.

    Uses ``data_processing.line_process.line_signs`` so the line-grouping and
    L→R sort logic exactly matches the headline run.
    """
    from data_processing.line_process import line_signs

    if not boxes_with_abz:
        return ""

    bboxes = [bb for bb, _ in boxes_with_abz]
    # `line_signs` indexes ``classes`` by label, so build a label list whose
    # entries are direct indices into the same `classes` table.
    label_indices = []
    for _, abz in boxes_with_abz:
        try:
            label_indices.append(classes.index(abz))
        except ValueError:
            # ABZ not in the 173-class list — fall back to UnclearSign 'X'.
            label_indices.append(classes.index("X"))

    return line_signs(
        bboxes=bboxes,
        labels=label_indices,
        classes=classes,
        scores=None,
        eps=eps,
        min_samples=min_samples,
        lambda_weight=lambda_weight,
        return_bboxes=False,
    )


def main() -> None:
    out_dir = ensure_output_dir(OUTPUT_DIR)
    df = load_headline_csv(INPUT_CSV)

    # Pull the canonical 173-class list once.
    from data_processing.sign_resolver import CLASSES_ABZ

    print(f"[UB2] loading manual annotations from {COCO_JSON}")
    anns = load_coco_annotations(COCO_JSON, CLASSES_ABZ)
    print(f"[UB2] manual-annotated fragments in COCO val: {len(anns)}")

    headline_ids = set(df["fragment_id"].tolist())
    subset_ids = sorted(headline_ids & set(anns.keys()))
    print(
        f"[UB2] usable subset (has manual anno AND transliteration AND non-train): "
        f"{len(subset_ids)} fragments"
    )
    if not subset_ids:
        raise SystemExit(
            "Empty UB2 subset. Check that COCO image filenames match headline "
            "fragment_ids (e.g. BM.30024.jpg ↔ fragment_id BM.30024)."
        )

    df_subset = df.set_index("fragment_id").loc[subset_ids].reset_index()

    oracle_records = []
    for frag in tqdm(subset_ids, desc="UB2 oracle"):
        seq = manual_to_sequence(
            anns[frag],
            classes=CLASSES_ABZ,
            eps=EPS,
            min_samples=MIN_SAMPLES,
            lambda_weight=LAMBDA_WEIGHT,
        )
        ref = df_subset.loc[df_subset["fragment_id"] == frag, "ground_truth_signs"].iat[0]
        pred_ngrams = build_ngrams(seq, fragment_id=f"{frag}_oracle")
        ref_ngrams = build_ngrams(ref, fragment_id=f"{frag}_ref")
        s, o = score_against(pred_ngrams, ref_ngrams)
        oracle_records.append({
            "fragment_id": frag,
            "score": s,
            "overlap": o,
            "n_manual_signs": len(anns[frag]),
        })

    oracle_df = pd.DataFrame(oracle_records)
    oracle_path = out_dir / "ub2_oracle_per_fragment.parquet"
    oracle_df.to_parquet(oracle_path, index=False)

    oracle_scores = oracle_df["score"].to_numpy()
    oracle_overlaps = oracle_df["overlap"].to_numpy()
    o_mean, o_lo, o_hi = bootstrap_mean_ci(
        oracle_scores, n_resamples=N_RESAMPLES, seed=SEED
    )

    # Comparison row: the model's headline score restricted to the same subset.
    model_scores = df_subset["score"].to_numpy()
    model_overlaps = df_subset["overlap"].to_numpy()
    m_mean, m_lo, m_hi = bootstrap_mean_ci(
        model_scores, n_resamples=N_RESAMPLES, seed=SEED
    )

    summary_df = pd.DataFrame([
        {
            "anchor": "UB2_oracle",
            "subset": "manual_anno_nontrain",
            "n_fragments": int(len(oracle_scores)),
            "score_mean": o_mean,
            "score_median": float(np.median(oracle_scores)),
            "score_ci_low": o_lo,
            "score_ci_high": o_hi,
            "overlap_mean": float(oracle_overlaps.mean()),
            "overlap_median": float(np.median(oracle_overlaps)),
        },
        {
            "anchor": "UB2_model_on_same",
            "subset": "manual_anno_nontrain",
            "n_fragments": int(len(model_scores)),
            "score_mean": m_mean,
            "score_median": float(np.median(model_scores)),
            "score_ci_low": m_lo,
            "score_ci_high": m_hi,
            "overlap_mean": float(model_overlaps.mean()),
            "overlap_median": float(np.median(model_overlaps)),
        },
    ])
    summary_path = out_dir / "ub2_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved: {oracle_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
