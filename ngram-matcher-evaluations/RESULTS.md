# Interpretive Anchors for the N-gram Match Score — Results

This document summarises the five anchors implemented in this directory.
Each section describes:

1. The question the anchor answers.
2. How the score is computed (with pointers to the relevant code).
3. The number it produced on the 26 833-fragment headline subset.
4. What that number means for the paper.

All anchors share the headline configuration (detection confidence ≥ 0.8;
DBSCAN ε = 0.35, min_samples = 2, λ = 0.006), use the same matcher
([ebl_ngrams `FragmentModel`](../../ngram-matcher/src/ebl_ngrams/fragment_model.py))
with default `n_values = (1, 2, 3)`, and operate on the per-fragment
sequences cached in
[evaluation_output_new_line_det_0.8_0.35_2_0.006_20260225/evaluation_results.csv](../evaluation_output_new_line_det_0.8_0.35_2_0.006_20260225/evaluation_results.csv).

## Headline reference

| Metric | Mean | Median | 95% CI (bootstrap, 10 000 resamples) |
|---|---|---|---|
| Match Score | **0.2317** | 0.204 | [0.2291, 0.2343] |
| Overlap Size | 13.42 | 5.00 | – |

n = 26 833 evaluated fragments (training fragments and fragments without
a transliteration are already excluded by the evaluation CSV).

## Matcher behaviour worth knowing

Two implementation details of
[`ebl_ngrams`](../../ngram-matcher/src/ebl_ngrams/document_model.py) shape
how each anchor is interpreted:

* **n_values = (1, 2, 3) by default.** The headline score and every anchor
  are computed over the *union* of 1-, 2- and 3-grams, not pure trigrams.
* **`postprocess` drops every n-gram containing `'X'`** (UnclearSign).
  Both the predicted and the reference n-gram sets are filtered, so any
  baseline that maps tokens to `X` (UB1) or that resamples to a vocabulary
  containing `X` (LB1) has those n-grams removed on the prediction side.

Together these two behaviours mean the matcher already absorbs the
173-class `UnclearSign` merge, which UB1 below quantifies precisely.

---

## LB1 — Random-label baseline (P_pred)

**Question.** How much of the headline 0.232 can be explained by Zipfian
class-frequency alone, with no per-fragment visual signal?

**Procedure** ([lb1_random_label.py](lb1_random_label.py)).

1. Estimate `P_pred` — the empirical class-frequency distribution over all
   detected tokens in the headline run (162 distinct ABZ codes, `'X'` is
   the most frequent at 9.8 %).
2. For each fragment, take the headline detected sequence and resample
   every token i.i.d. from `P_pred`. Line breaks (`\n`) are preserved
   verbatim — this is equivalent to "keep DBSCAN line clusters, relabel
   tokens" because line clustering depends only on bbox centroids
   ([data_processing/line_process.py:detect_rows_dbscan](../data_processing/line_process.py)).
3. Run the standard matcher against the unchanged reference.
4. Repeat with `M = 30` independent seeds; report mean ± std across
   seeds and the bootstrap CI from one representative run.

**Result.**

| Metric | Value |
|---|---|
| Score, mean across seeds (std) | **0.0695** ± 0.0005 |
| Score, 95 % bootstrap CI (seed 0) | [0.0681, 0.0704] |
| Overlap, mean across seeds (std) | 6.22 ± 0.01 |
| `P_pred` vocab size | 165 (incl. `'X'`) |

**Interpretation.** Pure class-frequency reproduces ≈30 % of the headline
score (0.07 ÷ 0.23). `'X'` carries 9.8 % of `P_pred` mass and any
resampled n-gram containing `X` is dropped by `postprocess`, which keeps
LB1 below LB2: the random baseline is partially self-suppressing under
the matcher.

---

## LB2 — Cross-fragment shuffle baseline

**Question.** How much of the headline is fragment-specific recognition
signal, versus the average co-occurrence statistics of cuneiform text?

**Procedure** ([lb2_shuffle.py](lb2_shuffle.py)).

1. Build n-gram sets for every fragment's predicted and reference
   sequences once and cache them.
2. For each of `M = 30` seeds, draw a derangement `π` (permutation with
   no fixed points) of the 26 833 fragments and compute the matcher score
   between `S_pred(f)` and `S_ref(π(f))` — i.e. the predicted signs of
   fragment `f` against the *wrong* transliteration.
3. Report mean ± std across seeds, plus the bootstrap CI of the first
   seed for direct comparison with the headline Table 7.

**Result.**

| Metric | Value |
|---|---|
| Score, mean across seeds (std) | **0.1102** ± 0.0008 |
| Score, 95 % bootstrap CI (seed 0) | [0.1086, 0.1128] |
| Overlap, mean across seeds (std) | 2.99 ± 0.02 |

**Interpretation.** LB2 is the most informative interpretive anchor.

* Headline − LB2 = 0.2317 − 0.1102 = **0.1215**: this is the part of the
  headline that depends on the prediction actually matching its *own*
  transliteration, not just the base statistics of Mesopotamian text.
* LB2 ÷ headline = 0.48, so roughly half of the headline 0.232 reflects
  generic co-occurrence and the other half reflects fragment-specific
  visual recognition.
* LB2 sits cleanly inside the spec's pre-registered expectation range
  [0.10, 0.18], on the lower end — that is, the model carries real signal.

---

## UB1 — 173-class vocabulary ceiling

**Question.** What is the ceiling imposed by the 173-class detection
vocabulary if visual recognition were otherwise perfect?

**Procedure** ([ub1_label_map.py](ub1_label_map.py)).

1. Build `VOCAB_173` from [`CLASSES_ABZ`](../data_processing/sign_resolver.py)
   minus `'X'` and `'NoABZ0'` (both represent the UnclearSign bucket).
   163 unique codes remain after deduplication.
2. For every fragment, map any reference token outside `VOCAB_173` to
   `'X'`; this simulates a detector that knows every token but is forced
   to merge anything out-of-vocabulary into UnclearSign.
3. Score the mapped sequence (`A`) against the original reference (`B`)
   under the headline matcher.
4. Also compute and report the **recall** `|A ∩ B| / |B|`, which exposes
   the actual n-gram loss caused by the vocabulary collapse.

**Result.**

| Metric | Value |
|---|---|
| Headline-metric score, mean | **0.9991** ± (very narrow) |
| Score, 95 % bootstrap CI | [0.9987, 0.9994] |
| **Recall, mean** | **0.910** |
| Recall, 95 % bootstrap CI | [0.9083, 0.9115] |
| OOV token fraction, mean | 4.6 % |

**Interpretation.** The headline metric is effectively *insensitive* to
the 173-class collapse: the matcher's `postprocess` drops every n-gram
that contains `'X'` on the prediction side, so the prediction n-gram set
is always a subset of the reference n-gram set and the score
`|A ∩ B| / min(|A|, |B|)` saturates at 1.0.

The meaningful number is the recall: **9 % of reference n-grams are lost
to the vocabulary collapse**. The token-level OOV rate is only 4.6 %, but
each OOV token can knock out up to six 1/2/3-grams it participates in,
which roughly doubles the loss at the n-gram level.

What this means for the paper:

* Reporting only the score gives a misleading ceiling of 1.0 — vocab
  is *not* a free pass under our matcher.
* Reporting the recall lets the paper say "an oracle that uses the 173-
  class scheme recovers 91 % of the reference n-grams; the remaining 9 %
  is the structural ceiling imposed by the label space, independent of
  visual detection quality."

---

## A1 — Stratification by overlap size

**Question.** Is the headline 0.232 dragged down by tiny fragments
(where the metric is noisy and the denominator small), or does it hold
up on long, well-preserved tablets?

**Procedure** ([a1_stratify_overlap.py](a1_stratify_overlap.py)). Pure
post-hoc analysis of the headline CSV: bin fragments by overlap size and
bootstrap the mean score per bin (10 000 resamples).

**Result.**

| Overlap bin | n fragments | Score mean | 95 % bootstrap CI |
|---|---|---|---|
| ALL | 26 833 | 0.2317 | [0.2291, 0.2343] |
| 0 | 4 661 | 0.000 | – |
| 1 – 4 | 8 075 | **0.348** | [0.342, 0.355] |
| 5 – 9 | 4 388 | 0.233 | [0.230, 0.236] |
| 10 – 19 | 3 999 | 0.240 | [0.237, 0.243] |
| 20 – 49 | 3 848 | 0.246 | [0.244, 0.249] |
| 50+ | 1 862 | 0.255 | [0.252, 0.258] |

**Interpretation.**

* **17 % of fragments have zero overlap** and contribute 0.0 to the mean.
  Removing them lifts the mean to ≈0.28; the headline 0.232 is partly a
  "miss rate" measurement.
* **The 1–4 bin (30 % of fragments) inflates the mean to 0.348** because
  the `min(|A|, |B|)` denominator collapses to a tiny integer, so even
  one matching unigram produces a large fractional score. This is a
  metric artefact, not a sign of strong performance on small fragments.
* **For overlap ≥ 5 (40 % of fragments) the score stabilises around
  0.24 – 0.25** with a gentle upward trend as fragments get longer. This
  is the most honest "typical" performance number.

The paper can use A1 to argue that small fragments add noise in *both*
directions (zero overlap drags the mean down; tiny-denominator
inflation pushes it up) and that the metric is most discriminative for
tablets with ≥10 ground-truth signs.

---

## UB2 — Manual-annotation oracle (pending)

**Status.** Code is implemented in [ub2_manual_oracle.py](ub2_manual_oracle.py)
and verified end-to-end on a synthetic COCO JSON. The full run is
blocked on the `erc-work-data` sshfs mount being online so that
`instances_val2017.json` can be read. Once the mount is back:

```bash
python ngram-matcher-evaluations/ub2_manual_oracle.py
```

The script will load the COCO val annotations, intersect with the
headline subset to get the fragments that have both manual bboxes and a
transliteration, run the canonical
[`data_processing.line_process.line_signs`](../data_processing/line_process.py)
with the headline DBSCAN parameters, and produce two rows:

* `UB2_oracle` — perfect detector score on the subset.
* `UB2_model_on_same` — the headline model's score on the same subset
  (so the oracle-vs-model gap is measured on identical data).

---

## Summary table (Table 8 candidate)

| Anchor | Subset | n | Score mean | 95 % CI | Overlap mean |
|---|---|---|---|---|---|
| headline | headline_eval | 26 833 | 0.2317 | [0.2291, 0.2343] | 13.42 |
| LB1_random_pred | headline_eval | 26 833 | 0.0695 | [0.0681, 0.0704] | 6.22 |
| LB2_shuffle | headline_eval | 26 833 | 0.1102 | [0.1086, 0.1128] | 2.99 |
| UB1_label_map | headline_eval | 26 833 | 0.9991 | [0.9987, 0.9994] | 94.78 |
| UB1_label_map (recall) | headline_eval | 26 833 | 0.910 | [0.9083, 0.9115] | – |
| UB2_oracle | manual_anno_nontrain | – | pending | – | – |
| UB2_model_on_same | manual_anno_nontrain | – | pending | – | – |

Raw data and per-fragment scores live next to this file (`*_summary.csv`,
`*_per_fragment*.parquet`).

## Suggested paper text

> To place the headline Match Score of 0.232 in context we computed four
> reference anchors (Table 8). A random-label baseline drawing tokens
> i.i.d. from the empirical detection-class distribution (LB1) yields
> 0.070, while a cross-fragment shuffle baseline that pairs each
> predicted sequence with a *different* fragment's transliteration (LB2)
> yields 0.110. The 0.122 gap between LB2 and the headline is the
> portion of the score that genuinely tracks fragment identity rather
> than the base co-occurrence statistics of cuneiform text. On the
> upper-bound side, a label-vocabulary oracle (UB1) recovers 91 % of the
> reference 1/2/3-grams under the 173-class scheme, indicating that ≈9 %
> of the reference's discriminative content is structurally unreachable
> by the detector regardless of visual performance.
