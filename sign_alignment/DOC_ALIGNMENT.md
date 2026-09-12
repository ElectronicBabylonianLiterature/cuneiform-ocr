---
generated_by: OpenAI Codex GPT-5.6 Sol
last_update: 2026-09-12
---

# Sign Alignment Code Documentation

This document describes the current `sign_alignment` implementation in this
workspace. The main entry point is
[`signs_alignment.ipynb`](../signs_alignment.ipynb), and the evaluation entry
point is [`evaluate_alignment.py`](../evaluate_alignment.py).

> The current primary workflow is **Hough row detection + row/sign DP matching
> + fixed-candidate attraction**. PSR, the DIFT affine probe, and DIFT
> sliding-window coarse alignment are optional supplementary or experimental
> workflows; they are not the default final stage of the main notebook.

## 1. Task and Current Workflow

Sign alignment combines two complementary sources:

- image detections provide candidate sign locations, sizes, labels, and scores;
- eBL transliterations provide a more reliable sign order without image
  coordinates.

The main difficulties are tablet damage, missed and false detections,
classification confusion, different text/image row counts, and duplicate
detections from overlapping crops.

The current notebook runs these stages:

1. Load the local image and ground truth; fetch and parse the eBL
   transliteration.
2. Detect with DETR/MMDetection, optionally using SAHI sliced inference; split
   the full image into `SubTablet` objects and retain crop-local detections.
3. Optionally reclassify DETR box crops with ResNet18 and fuse the two labels by
   a confidence rule.
4. Freeze `candidate_boxes`, create the iteration-dependent `optimize_boxes`,
   and create `text_boxes`/`text_rows` from the transliteration.
5. Run multi-angle Hough row detection over the fixed candidate centers, create
   `candidate_rows`, and initialize `optimize_rows` plus candidate provenance.
6. In each iteration, run row DP, within-row sign DP, candidate-backed exact
   anchor selection, coarse text-row alignment, and fixed-candidate attraction.
7. Commit the complete attraction output as the next iteration's
   `optimize_boxes`/`optimize_rows` while preserving provenance to the original
   fixed candidates.

The primary data flow is:

```text
image + GT + eBL text
        |
        v
DETR / optional SAHI ----> optional ResNet18 label fusion
        |
        +--> candidate_boxes (fixed geometry and hypotheses)
        +--> optimize_boxes  (mutable between iterations)
text --> text_boxes / text_rows
        |
Hough(candidate_boxes) --> candidate_rows --> optimize_rows
        |
        v
row DP --> sign DP --> candidate-backed anchors --> coarse alignment
        |
        v
fixed-candidate attraction --> next optimize state / final candidate run
```

## 2. Source Layout

| File | Responsibility |
|---|---|
| [`pipeline.py`](pipeline.py) | State model, step functions, primary attraction algorithm, supplements, and `Runner` |
| [`detector.py`](detector.py) | MMDetection/DETR, SAHI, crop detection, and result caching |
| [`classifier.py`](classifier.py) | ResNet18 crop classification and its class table |
| [`data_source.py`](data_source.py) | Local/COCO data, eBL API, text parsing, and prototype/canonical image sources |
| [`box.py`](box.py) | `SignCandidate`, `Box`, `Boxes`, and coordinate conversion |
| [`tablet.py`](tablet.py) | `Tablet`, `SubTablet`, and root/local coordinate transforms |
| [`sign.py`](sign.py) | `Sign` and ABZ/name/index resolution |
| [`psr_optimizer.py`](psr_optimizer.py) | Optional GMM point-set registration |
| [`dift_align.py`](dift_align.py) | SD-DIFT feature caching, matching, scoring, affine diagnostics, and rendering |
| [`visualizer.py`](visualizer.py) | Box, row, mapping, diagnostic, and composite visualizations |
| [`hyperparam.py`](hyperparam.py) | Coordinate-wise PSR hyperparameter search |
| [`../data_processing/hough_row_detection.py`](../data_processing/hough_row_detection.py) | Pipeline-independent multi-angle Hough row detection |
| [`../data_processing/line_process.py`](../data_processing/line_process.py) | Row/sign DP and baseline coarse alignment |

Package-level exports are defined in [`__init__.py`](__init__.py). Some
experimental APIs only live in `sign_alignment.pipeline`, so importing that
module explicitly is recommended.

## 3. Core Data Model

### 3.1 `Sign` and Class Candidates

`Sign` is immutable:

```python
Sign(abz: str, name: str, idx: int)
```

`SignResolver` converts among ABZ codes, canonical sign names, and detector class
indices. Unknown classes fall back to `UnclearSign`/`X`.

A physical box may carry several classification hypotheses:

```python
SignCandidate(sign: Sign, score: float)
```

`Box.best_candidate` selects the highest-scoring hypothesis; `sign`, `score`,
and `sign_name` are convenience views of it. The detector stores
near-identical geometry as multiple candidates on one `Box`; attraction also
clusters boxes into physical candidates later.

### 3.2 `Tablet` and `SubTablet`

`Tablet.img` is an OpenCV-style BGR `numpy.ndarray`, and `shape` returns
`(height, width)`.

`SubTablet` additionally stores:

- `parent`: the parent image coordinate system;
- `offset_in_parent=(x, y)`: its offset from that parent;
- `mask`: the crop's tablet contour mask, optionally used by PSR boundary loss;
- `offset_in_root`, `to_root()`, and `from_root()`: transforms that also support
  nested crops.

### 3.3 `Box` and `Boxes`

`Box` stores corners `(x1, y1, x2, y2)` and exposes `cx`, `cy`, `width`, and
`height` as properties. Every box belongs to one `Tablet`/`SubTablet` and must
contain at least one `SignCandidate`.

Common operations:

- `Box.from_center(...)`: construct from center and size;
- `copy()` / `translate(dx, dy)`;
- `to_tablet(target)`: convert between root and crop coordinates;
- `crop_bounds()` / `crop_image()`: prepare classifier or DIFT crops.

`Boxes` is a `list[Box]` constrained to one `tablet`. Appending a box from a
different coordinate system raises `ValueError`. It also provides
`avg_width`, `avg_height`, `avg_size`, and batch `to_tablet()`. Empty
collections use `80.0` as the average width and height fallback.

`boxes_in_crop()` retains boxes whose centers lie in the crop and clips their
corners to the image bounds.

### 3.4 `BoxRows`

`BoxRows` stores topology without copying boxes:

```python
BoxRows(
    boxes=<Boxes>,
    rows=[[0, 1, 2], [3, 4]],
    noise=[5],
)
```

Each entry in `rows` is a flat index into `boxes`; the row id is the outer-list
index, and the column id is the position inside the row. Its main helpers are
`row_boxes()`, `as_dict()`, `as_lists()`, `sign_sequences()`, `counts()`, and
`with_boxes()`. `with_boxes()` only accepts an equal-length, index-compatible
replacement collection.

### 3.5 `SampleState`, `CropContext`, and `Runner`

`SampleState` has three conceptual layers:

- fixed full-image data: `full_tablet`, `full_gt_boxes`, `full_detections`, and
  the transliteration;
- selected-crop data: `tablet`, `det_boxes`, and `gt_boxes`;
- current iteration: candidate/optimize/text rows, matches, coarse alignment,
  attraction, PSR/DIFT results, and `extras`.

The most important collections have these semantics:

| Field | Changes between iterations | Meaning |
|---|---:|---|
| `det_boxes` | No | Selected-crop detections; label fusion may update them before freezing |
| `candidate_boxes` | No | Candidate pool frozen by `create_box_sets()` |
| `candidate_rows` | No | Hough topology over the fixed candidate pool |
| `optimize_boxes` | Yes | Input to this iteration's matching and output of the previous attraction run |
| `optimize_rows` | Yes | Current topology over `optimize_boxes` |
| `text_boxes` / `text_rows` | No | Fixed topology created from the transliteration |
| `aligned_boxes` / `aligned_rows` | Rebuilt each iteration | Coarse result organized by text-row topology |

Two provenance maps prevent index drift across repeated iterations:

- `optimize_to_candidate`: current flat optimize-box index → fixed candidate-box
  index;
- `optimize_row_to_candidate`: current optimize-row index → fixed candidate-row
  index.

`CropContext` injects the detector, classifier, data sources, output settings,
and optional PSR/DIFT dependencies while holding the current `SampleState`.

`Runner.run()` executes ordered `Step(name, run, visualize)` objects.
`advance_iteration=True` increments `optimization_iteration` exactly once before
the entire step batch. `choose_sample()` resets fragment state. After detection,
`choose_crop()` switches crops and `init_crop()` invalidates all crop-local and
iteration-local results.

## 4. Data Loading and Text Parsing

### 4.1 Local Data

`LocalDataSource(annotations_dir)` expects:

```text
annotations_dir/
├── imgs/<fragment_id>.jpg|jpeg|png
└── annotations/gt_<fragment_id>.txt
```

Each GT line is `x,y,w,h,sign_name`. `LocalTestDataSource` instead reads COCO
`annotations/instances_val2017.json` and `val2017/`, skipping `iscrowd`
annotations.

### 4.2 eBL Transliteration

`EBLAPISource` requests `/fragments/{fragment_id}`, caches JSON in memory, and
retries failures. Evaluation data with sub-tablet names can use
`strip_subtablet_suffix=True` to remove a trailing `-number`.

`SignTextParser.parse_text_lines()`:

- processes only `TextLine` objects;
- recursively expands `Word`, `AkkadianWord`, `GreekWord`, `Determinative`, and
  related containers;
- handles `Reading`, `Logogram`, `Number`, `CompoundGrapheme`, `Grapheme`,
  `UnclearSign`, `Variant`/`Variant2`, and `Divider`;
- treats `LineBreak` as an output row boundary without producing empty rows;
- removes `BROKEN_AWAY` tokens when `filter_broken=True`;
- resolves readings/subscripts through `SignAPIResolver`, which queries the eBL
  signs API and writes `.sign_api_cache.json`; failures try local
  `SignResolver` and finally fall back to the upper-case name.

`load_data()` retains both forms as `text_lines` and `text_lines_unfiltered`.

## 5. Detection, Cropping, and Label Fusion

### 5.1 DETR/MMDetection

`ModelConfig` selects the config, checkpoint, and device.
`TabletImageDetector.detect()`:

1. calls [`divide_tablet_photo()`](../data_processing/divide_photos.py) when
   `is_crop_itself=False`, receiving image pieces, offsets, and masks; otherwise
   it creates one full-image crop;
2. wraps every crop in `SubTablet`;
3. runs `SingleImageDetector` locally and filters with `score > threshold`;
4. stores parallel `crop_tablets` and `crop_boxes` lists;
5. converts every crop result through `Box.to_tablet()` to build
   `full_detections` in root coordinates.

Direct MMDetection inference is cached by image shape, dtype, and SHA-256
content hash. Outputs whose four coordinates differ by at most 2 px are grouped
as multiple class candidates on one box.

### 5.2 SAHI

With `use_sahi=True`, a full-image prediction first determines:

```text
slice_size = max(1, int(initial_detections.avg_size / box_slice_ratio))
```

Sliced prediction then uses 20% vertical and horizontal overlap. Its cache key
also includes the threshold and `box_slice_ratio`. `slice_height` and
`slice_width` remain available on the detector for diagnostics.

### 5.3 ResNet18 Classification Fusion

`SignClassifier` resizes each detected crop to `232×232`, center-crops to
`224×224`, applies ImageNet normalization, and batches inference through a
custom ResNet18 checkpoint. Probabilities for duplicate output sign names are
summed. Selection is restricted to `COMMON_SIGN_NAMES`, the intersection of
the ResNet and DETR label vocabularies.

For each shared DETR class, `improve_classification()` applies this rule:

1. if DETR reaches the threshold, retain DETR;
2. otherwise, if the classifier reaches the threshold or beats the DETR score,
   use the classifier;
3. otherwise retain DETR.

Non-shared DETR classes are unchanged. When the classifier wins, the box's
candidate list is replaced by one new candidate. Detailed decisions are stored
in `state.extras["classification_improvement"]`.

## 6. Row Detection and Sequence Matching

### 6.1 Text-Box Initialization

`create_box_sets()` runs after optional label fusion:

- `candidate_boxes = det_boxes.copy()` freezes the original pool;
- `optimize_boxes = det_boxes.copy()` creates mutable iteration input;
- `Boxes.from_text_lines()` builds a regular grid from full-image average
  detection width/height;
- the complete text grid is translated to the selected crop's detection
  centroid;
- `BoxRows.from_text_lines()` creates topology from the original line lengths.

Text-box coordinates are only an initialization. Their row ids always match the
parsed transliteration row ids.

### 6.2 Multi-angle Hough Row Detection

`detect_rows()` currently calls `BoxRows.detect_using_hough()`; it does not use
the legacy DBSCAN path. The underlying `detect_hough_rows()` only depends on an
`N×2` center array. Its defaults search `[-15°, +15°]` in `1°` steps, with
distance scales derived from the median detection height.

The implementation:

- accumulates equal-weight Gaussian votes in the full `(rho, theta)` space;
- selects strict 2-D local maxima instead of a single global-angle slice;
- assigns disjoint support and continuously refits bounded-angle robust lines;
- rejects implausible distant two-point rows, crossing rows, and duplicates,
  and merges compatible fragments;
- fits a low-curvature, zero-average-slope angle curve through the first row
  selection;
- reselects inside the curve's local angle window, orders rows vertically, and
  sorts signs by x within a row;
- places unassigned detections in `noise`.

The returned `BoxRows` carries accumulator, strict-neighbor maxima, peaks, row
angles, and angle-curve diagnostics used by `vis_detected_rows_info()`.

### 6.3 Row-level DP

`match_rows()` aligns `optimize_rows.sign_sequences()` and
`text_rows.sign_sequences()` with `match_rows_dp()` using:

```python
skip_text_penalty=0.5
skip_det_penalty=1.0
skip_small_det_penalty=0.2
small_det_threshold=1
similarity_method="jaccard"
```

It stores `matches: list[(text_row_idx, optimize_row_idx)]` and uses
`create_row_mapping()` to build `text_to_optimize` and `optimize_to_text`.
Every row-matching pass invalidates downstream sign matches, alignment, PSR,
and attraction state so stale results cannot leak between iterations.

### 6.4 Within-row Sign DP and Anchors

`match_signs_in_rows()` runs:

```python
match_signs_in_row_dp(
    skip_text_penalty=0.5,
    skip_det_penalty=2.0,
    mismatch_cost=0.9,
)
```

Pairs in `row_sign_matches` may have different labels. A pair is added to
`row_anchor_matches` only if:

- the current optimize sign resolves through provenance to a fixed candidate;
- the text label equals that fixed candidate's best label.

Repeated iterations therefore cannot create an unsupported anchor merely
because an optimize box was previously relabeled from text.

### 6.5 Baseline Coarse Alignment

`align_text_rows()` calls `align_text_row_to_detection()`:

- exact anchors copy the matched detection center and size;
- text signs between anchors are interpolated, and signs outside them are
  extrapolated by average width;
- a row with no anchors gets a fitted detection baseline and is centered around
  the detection-row centroid;
- widths are restricted to `[2/3, 4/3]` of the full-image average width.

The outputs are `aligned_boxes` and `aligned_rows`, indexed by text-row
topology. Entirely unmatched text rows remain empty.

## 7. Fixed-candidate Attraction (Primary Optimization)

`run_candidate_attraction()` treats this iteration's `aligned_rows` as movable
text state and the initially frozen `candidate_rows` as the physical candidate
pool, optimizing each matched row independently.

### 7.1 Physical Candidates

Candidate boxes are clustered by IoU and normalized center distance. A
`PhysicalCandidate` contains:

- the highest-scoring member as its geometry representative;
- every original member index;
- the maximum score for each label in the cluster;
- maximum label score as `objectness`.

Two protected reliable anchors are never merged, and geometry is never
averaged. This preserves alternative class evidence without blurring position.

### 7.2 Parameterization and Pair Cost

Each row receives a tangent/normal basis. Boxes are encoded as:

```text
[along / pitch, normal / row_height,
 log(width / reference_width), log(height / reference_height)]
```

Exact anchors remain fixed to candidate geometry. Pair cost for free text boxes
is dominated by along-row and normal displacement. Later temperature stages
also add size, plus weaker objectness, same-class support, and original DP
diff-pair bonuses. The nearest hard anchors provide a broad allowed-candidate
interval.

### 7.3 Soft and Hard Assignment

`capped_soft_assignment()` returns a
`text_count × (candidate_count + 1)` matrix:

- every text row sums to 1;
- every real candidate column has capacity at most 1;
- the final `NULL` column has unlimited capacity.

Multi-temperature Adam jointly optimizes candidate attraction, baseline, gap,
order, initial-center, and size-prior terms. Defaults use temperatures
`(2.0, 1.0, 0.5, 0.25)` with 35 steps at each temperature.

Finally, `ordered_partial_assignment()` uses DP to produce an order-preserving,
one-to-one hard assignment with NULL. Output statuses are:

- `anchor`: exact match using fixed candidate geometry;
- `candidate`: assigned physical candidate geometry with the text label;
- `null`: no candidate, retaining the optimized/extrapolated geometry.

A box that is not fully inside the image receives `included_in_result=False`
and is omitted from the next iteration.

### 7.4 Re-entrant Iteration

`CandidateAttractionRun` retains input rows plus per-row candidates and
assignment diagnostics. At completion, `optimize_boxes`, `optimize_rows`, and
both provenance maps are committed together; `candidate_boxes` and
`candidate_rows` never change. The next iteration reruns row DP, sign DP, and
coarse alignment.

`candidate_attraction_records()` returns rows suitable for a pandas DataFrame,
including input/output status, candidate labels, class support, soft/NULL
probability, movement, and result inclusion.

## 8. Optional Supplementary Workflows

Examples are collected in
[`signs_alignment_supplement.ipynb`](../signs_alignment_supplement.ipynb).

### 8.1 Relabeling Without Geometry Optimization

`create_result_without_optimization()` starts from `det_boxes.copy()` and only
relabels DP pairs that resolve back to a fixed candidate. Positions, sizes, and
detection scores remain unchanged. The result is cached in
`result_without_optimization_boxes`.

### 8.2 PSR

`create_psr_optimizer()` uses:

- Source: `aligned_rows.as_lists()`;
- Target: fixed `candidate_boxes`;
- parameters: `(cx, cy, width, height)` for every source box.

Its total loss is:

```text
L = lambda_data Ldata + lambda_anchor Lanchor + lambda_seq Lseq
  + lambda_height Lheight + lambda_rows Lrows + lambda_boundary Lboundary
```

- `Ldata`: class-weighted GMM negative log-likelihood with uniform noise; the
  default confusion matrix is identity;
- `Lanchor`: distance from centers to their fitted row baseline;
- `Lseq`: changes in adjacent spacing relative to initialization;
- `Lheight`: within-row height variance;
- `Lrows`: asymmetric quadratic/plateau constraint on adjacent-baseline
  spacing relative to average height;
- `Lboundary`: optional penalty when the first box of a row extends beyond the
  contour mask.

The optimizer uses Adam, optional linear sigma annealing, gradient clipping,
and a 10 px minimum width/height. `optimize_psr()` writes to `optimize_boxes` and,
when topology is compatible, rebuilds `optimize_rows` and candidate provenance.

### 8.3 DIFT Runtime and Affine Probe

`DiftRuntime` lazily obtains prototype/canonical images from `source:
DataSource`, extracts SD-DIFT feature maps, and caches them in memory and an
optional `.pt` directory under `source.key()/period/sign` identities.

Available sources:

- `PrototypeSource`: renders a 512×512 prototype using period-specific fonts
  and eBL Unicode data;
- `EBLMongoCanonicalSource`: retrieves canonical images from MongoDB by period,
  form, and centroid constraints.

`DiftRuntime.match()` obtains mutual nearest-neighbor matches, then tries RANSAC
`estimateAffine2D` and falls back to `estimateAffinePartial2D`. The result
records semantic, global, foreground-only, support, inlier, affine-IoU, angle,
deformation, and scale diagnostics.

`optimize_psr_until_dift_probe()` can pause PSR at a configured iteration,
`run_dift_affine_probe()` diagnoses the current boxes, and
`optimize_psr_after_dift_probe()` only runs the remaining iterations. The probe
does not directly update PSR parameters.

### 8.4 DIFT Sliding-window Coarse Alignment

`align_text_rows_with_feature_search()` can replace ordinary
`align_text_rows()`:

- exact anchors still use detection geometry directly;
- non-anchor signs receive sliding windows near the baseline and within anchor
  intervals;
- each crop is featurized once and source features are reused by sign name;
- matching uses `DiftMatchResult.coarse_score = geometry_score × support_score`;
- DP first maximizes assignment count, then total score, while preserving
  left-to-right order;
- signs below `assignment_min_score` fall back to ordinary baseline alignment.

The result is stored in `state.extras["feature_coarse"]`.

## 9. Evaluation and Hyperparameter Search

[`evaluate_alignment.py`](../evaluate_alignment.py) supports five
`PredictionMode` values:

| Mode | Output |
|---|---|
| `DETECTION` | Raw detections, followed by cross-crop per-class NMS |
| `IMPROVED_DETECTION` | Detections with ResNet18 label fusion, then cross-crop per-class NMS |
| `WITHOUT_PSR` | Detection geometry/scores with alignment-driven relabeling |
| `DET_AS_CANDIDATES` | Current fixed-candidate attraction output |
| `PSR` | PSR after baseline coarse alignment |

Predictions are converted to full-image coordinates and greedily matched to GT
by class and IoU. Reports include mAP over IoU 0.50–0.95, precision/recall/F1 at
IoU 0.5, mean matched IoU, and per-class metrics. Non-sign GT can be excluded
from pipeline evaluation/visualization through
`gt_visualization_excluded_prefixes` (default: `SURFACE_`) without mutating the
stored GT collection.

The current script is configured by module-level constants and
`PREDICTION_MODE` inside `__main__`; it does not expose a command-line argument
parser. Check paths, sample limits, and mode before running:

```bash
python3 evaluate_alignment.py
```

`hyperparameter_search()` is PSR-only. It performs two coordinate-wise rounds
over `SEARCH_AXES`, using an injected `eval_fn` that returns mAP, and writes
`evaluation_results/hyperparam_search.json`.

## 10. Visualization and Output Layout

`VisOptions(info, display, save)` independently controls console diagnostics,
interactive display, and file output. `output_path()` organizes files as:

```text
<output_dir>/<fragment_id>/
├── initial/          # input, GT, detections, and label fusion
├── row_detection/    # Hough accumulator and peaks
└── iter_NNN/         # current attraction/PSR/DIFT iteration
```

Main helpers:

- `BboxVisualizer.draw_boxes()`: boxes, labels, and scores;
- `draw_rows()`: row ids, mappings, and baselines;
- `draw_text_mapping()`: text topology and match status;
- `draw_alignment_diagnostic()`: combined image/text diagnostics;
- `CompositeVisualizer.compose()`: multi-panel comparisons;
- `TextVisualizer.save_text()`: parsed transliteration output.

Green is reserved for GT. Fixed candidates, anchors, candidate matches, NULL,
and movement have separate colors defined in `pipeline.py`.

## 11. Minimal Example

Adjust all data and checkpoint paths for the runtime environment:

```python
import sign_alignment.pipeline as pp
from sign_alignment import LocalDataSource, ModelConfig, TabletImageDetector
from sign_alignment.visualizer import ColorConfig

detector = TabletImageDetector(
    model_config=ModelConfig("configs/detr.py", "/path/to/detr.pth", "auto"),
    default_score_threshold=0.0,
    use_sahi=True,
    box_slice_ratio=0.2,
)
classifier = pp.SignClassifier(
    "/path/to/resnet18.pth", device="auto", is_load_now=False
)
context = pp.CropContext(
    tablet_detector=detector,
    sign_classifier=classifier,
    local_source=LocalDataSource("/path/to/filtered_annotations"),
    color_config=ColorConfig,
    output_dir="alignment_results",
)
runner = pp.Runner(context, pp.VisOptions(info=True, display=False, save=True))
runner.choose_sample(name="K.4426")
runner.run([
    pp.Step("load", pp.load_data),
    pp.Step("detect", pp.detect_signs),
    pp.Step("crop GT", pp.transform_gt_to_crop),
    pp.Step("classify", lambda ctx: pp.improve_classification(ctx, 0.5)),
    pp.Step("box sets", pp.create_box_sets),
    pp.Step("rows", pp.detect_rows),
])

iteration = [
    pp.Step("row match", pp.match_rows),
    pp.Step("sign match", pp.match_signs_in_rows),
    pp.Step("coarse align", pp.align_text_rows),
    pp.Step("candidate attraction", pp.run_candidate_attraction),
]
runner.run(iteration, advance_iteration=True)
result = pp.get_candidate_run(context).boxes
```

Note that `runner.choose_crop()` can only initialize a crop after
`detect_signs()` has populated `crop_tablets`. This example relies on
`context.img_idx` (default 1); set it to 0 for a single-crop detector, or call
`runner.choose_crop(0)` explicitly after detection.

## 12. Tests and Maintenance Invariants

Relevant tests:

```bash
python3 -m unittest \
  sign_alignment.test_data_source \
  sign_alignment.test_classification_improvement \
  sign_alignment.test_pipeline_reentry \
  sign_alignment.test_dift_alignment
```

Preserve these invariants when changing the pipeline:

- a `Box`, its `Boxes`, and their `tablet` must use the same coordinate system;
- `BoxRows.rows` contains indices, not copied boxes;
- label fusion must happen before `create_box_sets()` freezes candidates;
- repeated attraction may only change optimize state, never the fixed pool;
- every anchor must be supported by the original fixed candidate label;
- switching crops or rerunning row matching must not reuse stale downstream
  state;
- DIFT numpy input is interpreted as OpenCV BGR/BGRA by default; true RGB numpy
  input must use `ImageView.from_any(..., assume_bgr=False)` explicitly.
