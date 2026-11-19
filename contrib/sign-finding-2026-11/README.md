# Sign finding: two detectors trained further, and tiled inference (November 2026)

This folder is about **sign finding only**: boxes around the signs on a tablet photograph. It adds no sign
classifier and no line reading.

## What is here

| File | What it is |
|---|---|
| `tiled_inference_detr.py` | Runs this repository's single-stage Deformable DETR tile by tile over a whole photograph. |
| `tiled_inference_yolo.py` | The same for a YOLO11m sign finder (one class, "sign"); it can also choose the scale of a photograph by itself. |
| `classes173.json` | The 173 class names of the published DETR, in the model's order. |
| `SHA256SUMS` | Checksums of the two weight files. |
| `DATA.md` | What the two sign finders were trained on, and how to use the lists below for a held-out test. |
| `trained-on-sign-detector-u1.tsv` | Every tablet in the training or the validation of `sign-detector-u1` (19,938 identifiers with their other names). |
| `trained-on-detr173-ft1b-e9.tsv` | Every fragment in our further training of the DETR (548). |
| `test-fragments.tsv` | The 24 test fragments of the table below. |

The weights are not in git. They are attached to the release
[`sign-finding-2026-11-19`](https://github.com/dangerousfood/cuneiform-ocr/releases/tag/sign-finding-2026-11-19) of this fork:

| Weights | Size | What it is |
|---|---|---|
| `detr173-ft1b-e9.safetensors` | 165,801,788 bytes | The published 173-class Deformable DETR ([Zenodo 10.5281/zenodo.17395154](https://doi.org/10.5281/zenodo.17395154)) with all weights trained further (see below). A plain state dict: it loads into `configs/detr.py` with no missing and no unexpected key. |
| `sign-detector-u1.pt` | 40,513,516 bytes | A YOLO11m detector (Ultralytics 8.4), one class, trained on 640 x 640 tiles in which a sign is about 64 pixels tall. |

## Why tiles

In the published detections ([Zenodo 10.5281/zenodo.18863050](https://doi.org/10.5281/zenodo.18863050)) many photographs
carry exactly 100 or 200 boxes, and on tablets with many signs most signs have no box. Run tile by tile at a fixed sign
height (64 pixels), the **same published weights** find far more signs: on the test fragments below, 72.8% of the
hand-boxed signs (at 10 false boxes per 100 hand boxes) against 20.6% / 38.3% in the published detections (thresholds
0.8 / 0.4, which carry 1.5 / 4.8 false boxes per 100; 20 of the 24 fragments have published detections). That part needs
no new weights, only `tiled_inference_detr.py`.

## Results

**Test fragments.** 24 fragments hand-annotated on the eBL platform (8,292 hand-drawn sign boxes of the types HasSign,
Number, Damaged, PartiallyBroken, UnclearSign) that are in none of eBL's public data releases up to December 2025 and
in none of the training sets of the two models below. Twelve are first-millennium (4,765 boxes), five
second-millennium (1,700) and seven third-millennium (1,827).

**Measure.** A hand box is *found* when a predicted box matches it one to one at IoU 0.5 or more. *Extra* boxes are
predicted boxes inside the annotated area that touch no hand box (IoU under 0.1), counted per 100 hand boxes. Every
detector is read at the score threshold at which it leaves **10 extra boxes per 100 hand boxes**, so the figures compare
signs found at an equal number of false boxes. All three are run tiled with signs at 64 pixels.

| Signs found (%) | All 24 fragments | First millennium | Second millennium | Third millennium |
|---|---|---|---|---|
| Published DETR weights, tiled | 72.8 | 84.2 | 73.9 | 42.6 |
| `detr173-ft1b-e9` (DETR trained further) | **85.1** | **94.8** | 79.8 | **53.7** |
| `sign-detector-u1` (YOLO11m) | 82.7 | 85.1 | **91.3** | 52.9 |

| At the same point | Narrow signs found (%) | Two signs under one box (%) |
|---|---|---|
| Published DETR weights, tiled | 43.2 | 12.2 |
| `detr173-ft1b-e9` | 67.7 | 5.3 |
| `sign-detector-u1` | 44.5 | 11.3 |

(Narrow = a hand box under 0.6 of the fragment's median box width; 675 of the 8,292.)

**What these figures do not show.**
- The test set is small: 24 fragments; the second- and third-millennium columns rest on 5 and 7 fragments.
- The scale of each photograph (the factor that makes its signs 64 pixels tall) was taken from the hand boxes, the only
  number taken from them. With the scale chosen automatically the figures will differ somewhat: on the 24 test
  photographs the scale search of `tiled_inference_yolo.py` lands between 0.83 and 1.04 of the hand-box scale (median 0.93).
- The copy of the DETR training (after pass 9 of 30) was chosen with these fragments in view, so its figures are a
  little optimistic.
- The class output of `detr173-ft1b-e9` was trained with sign names but **not evaluated** here. Treat it as a sign
  finder.
- Inside a full reading pipeline a finder that finds more signs does not automatically read better: the stages behind
  it have to be fitted to its boxes.

## How the weights were made

- **`detr173-ft1b-e9`**: starts from the published 173-class weights; all weights trained with AdamW at 2e-5 (backbone,
  sampling offsets and reference points at a tenth of that), cosine schedule, batch 4, 800-pixel tiles with the signs
  scaled to 64 pixels, tile sizes 704 to 896, HSV jitter, no flips. Training tiles: 2,533 tiles from 296 fragments
  hand-annotated on the eBL platform that are not in the 2023 release (41,682 boxes), plus 2,160 tiles of the 2023
  release. This is the copy after pass 9.
- **`sign-detector-u1`**: YOLO11m from the COCO weights, 30 epochs, batch 32, on about 50,000 tiles from CDLI and eBL
  photographs of all periods. Most of its boxes are machine-made: a detector's boxes aligned with the published
  transliteration of the same tablet. Hand-boxed eBL tiles and background tiles are mixed in.

Which tablets these were, tablet by tablet, is in `DATA.md` and the three lists beside it (identifiers and counts only),
so that a test set of tablets the models have not seen can be built.

## Use

```bash
# YOLO11m; the scale of each photograph is searched, and written for the DETR script
python tiled_inference_yolo.py sign-detector-u1.pt photos/*.jpg --out boxes-yolo.json --write-scales scales.json

# Deformable DETR, tiled (this repository's mmdetection setup; 173-class config)
python tiled_inference_detr.py ../../configs/detr.py detr173-ft1b-e9.safetensors classes173.json photos scales.json 64 boxes-detr.json
```

Both scripts were run as committed on photographs of the test set; `tiled_inference_detr.py` gives the same boxes as
the script the figures above were made with. Both write boxes in the photograph's own pixels. Thresholds that left 10 extra boxes per 100 hand boxes on the test
fragments: about 0.25 for `sign-detector-u1`; for the DETR copy choose it on your own material (its scores run lower
than the YOLO's; around 0.2).

## Terms and credits

- `detr173-ft1b-e9` is derived from the published weights by Wentao Che and the eBL team
  ([Zenodo 10.5281/zenodo.17395154](https://doi.org/10.5281/zenodo.17395154), CC BY 4.0) and is offered under the same
  licence, CC BY 4.0, with that attribution.
- `sign-detector-u1` was trained with Ultralytics YOLO, which is licensed AGPL-3.0.
- No photograph is redistributed here. The training photographs belong to their collections and stay under their own
  terms: photographs served by the eBL platform (among them photographs © The Trustees of the British Museum,
  CC BY-NC-SA 4.0) and photographs from the Cuneiform Digital Library Initiative. The hand annotations are the eBL
  project's. Please treat the weights as research material.
