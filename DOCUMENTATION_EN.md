# Sign Alignment Code Documentation

## 1. Task Overview

The **sign alignment** task for cuneiform tablets: establishing correspondences between **detection boxes** on a tablet image (positions known, class labels not always accurate) and scholars' **text transliterations** (sign names accurate, but no position information), ultimately assigning each transliterated sign an accurate bounding box in the image.

The two information sources are complementary: the detection model provides spatial positions, while the text transliteration provides accurate sign sequences. The core challenges of alignment are:
- Tablets may be **damaged** (top/bottom/middle), causing mismatches in the number of text lines and detection rows
- The detection model's **class labels** may differ from the transliteration (OCR confusion)
- Detections may produce **noise boxes** (false positives), and some signs in the text may be undetectable

---

## 2. Overall Pipeline

Using the notebook `signs_alignment.ipynb` as a reference, the full pipeline consists of the following stages:

1. **Data Loading**: Local GT images + annotation boxes; EBL API text transliteration + broken sign filtering
2. **Image Detection**: DETR model detects sign boxes → `Detection`; text transliteration parsed → `text_lines: List[List[str]]`
3. **SubTablet Construction**: Detection-side SubTablet (Target X); text-side SubTablet (Source S, uniform grid + centroid alignment)
4. **Row Detection & Row-level DP Matching**: DBSCAN clustering → `match_rows_dp` → `row_mapping`
5. **Sign-level DP Matching**: `match_signs_in_row_dp` (per-row alignment within each matched row pair)
6. **Coarse Alignment (Baseline Projection)**: `align_text_to_detection_rows` → `sub_tablet_optim`
7. **PSR Fine Optimization**: `PointSetRegistrationOptimizer`, GMM data loss + structural constraints → `sub_tablet_final`

Each stage is described below.

### 2.1 Data Loading

#### Local Ground Truth

Load images and manual annotations from a local directory. Directory structure: `imgs/{id}.jpg` and `annotations/gt_{id}.txt` (format: one line per box `x,y,w,h,sign_name`).

| Operation | Call | Code Location |
|-----------|------|---------------|
| Discover available fragments | `LocalDataSource.get_available_fragments()` | [sign_alignment/data_source.py#L74](sign_alignment/data_source.py#L74) |
| Load image | `LocalDataSource.load_image(sample)` | [sign_alignment/data_source.py#L74](sign_alignment/data_source.py#L74) |
| Load annotation boxes | `LocalDataSource.load_annotation(sample)` | [sign_alignment/data_source.py#L74](sign_alignment/data_source.py#L74) |

#### Text Transliteration Retrieval & Parsing

Retrieve fragment transliteration data from the EBL API, parse the `text.lines` JSON structure, recursively extract sign names, and optionally filter broken signs (`BROKEN_AWAY`).

| Operation | Call | Code Location |
|-----------|------|---------------|
| Fetch fragment JSON | `EBLAPISource.get_fragment_data(sample)` | [sign_alignment/data_source.py#L154](sign_alignment/data_source.py#L154) |
| Parse text lines | `SignTextParser.parse_text_lines(text_data, filter_broken, sign_resolver)` | [sign_alignment/data_source.py#L402](sign_alignment/data_source.py#L402) |
| Parse single token | `SignTextParser._extract_signs_from_token(token, filter_broken)` | [sign_alignment/data_source.py#L299](sign_alignment/data_source.py#L299) |
| Reading → sign name resolution | `SignAPIResolver.resolve(value, sub_index)` | [sign_alignment/data_source.py#L13](sign_alignment/data_source.py#L13) |

Parsing flow: `parse_text_lines` iterates over each line in `text.lines[]`, recursively calling `_extract_signs_from_token` for each `content[]` token. Container types (`Word`, `Determinative`, etc.) always recurse into child `parts`; leaf node types (`Reading`, `Logogram`) check whether `enclosureType` contains `BROKEN_AWAY`; the extracted `(name, subIndex)` is resolved to a standard sign name via `SignAPIResolver` querying the EBL signs API (e.g., `("qa", 2) → "IB"`).

#### Save Text

| Operation | Call | Code Location |
|-----------|------|---------------|
| Save formatted text lines to .txt | `TextVisualizer.save_text(text_lines, path, fragment_id)` | [sign_alignment/visualizer.py#L1067](sign_alignment/visualizer.py#L1067) |

### 2.2 Image Detection

Uses a pretrained DETR model to detect cuneiform signs on tablet images. Since tablet images are typically large, the image is first cropped into multiple overlapping patches, each detected separately, then coordinates are restored to the full image coordinate system.

| Operation | Call | Code Location |
|-----------|------|---------------|
| Create detector | `TabletImageDetector(model_config, score_threshold, keep_crops)` | [sign_alignment/detector.py#L92](sign_alignment/detector.py#L92) |
| Detect full image | `tablet_detector.detect(img)` | [sign_alignment/detector.py#L102](sign_alignment/detector.py#L102) |
| Crop tablet | `divide_tablet_photo(img)` (internal call) | [data_processing/divide_photos.py#L245](data_processing/divide_photos.py#L245) |
| Per-patch detection | `SingleImageDetector.detect(img_piece)` (internal call) | [sign_alignment/detector.py#L82](sign_alignment/detector.py#L82) |
| Get crop info | `tablet_detector.crop_coordinates`, `tablet_detector.get_cropped_images()` | [sign_alignment/detector.py#L92](sign_alignment/detector.py#L92) |
| Compute average sign dimensions | `compute_avg_dimensions(detections)` | [sign_alignment/heatmap.py#L298](sign_alignment/heatmap.py#L298) |
| Transform GT coords to crop region | `transform_gt_to_cropped_region(gt_boxes, crop_info)` | [sign_alignment/heatmap.py#L355](sign_alignment/heatmap.py#L355) |

`TabletImageDetector.detect()` internal flow:
1. `divide_tablet_photo()` crops the image into patches → returns patch list, `crop_coordinates`, and binary contour masks (`return_masks=True`)
2. For each patch, calls `SingleImageDetector.detect()` to run DETR → gets detection boxes in patch coordinates
3. Adds `crop_coordinates[i]`'s `(x, y)` offset to each detection box to restore full image coordinates
4. When `keep_crops=True`, retains each patch's `SingleImage` (containing patch image, patch-coordinate detections, and contour mask)

### 2.3 SubTablet Construction

Wraps detection results and text transliterations into `SubTablet` objects as the two operands for subsequent alignment.

| Operation | Call | Code Location |
|-----------|------|---------------|
| Detection-side (Target X) | `SubTablet.from_detections(img, detections, avg_width, avg_height)` | [sign_alignment/tablet.py#L157](sign_alignment/tablet.py#L157) |
| Text-side (Source S) | `SubTablet.from_text_lines(text_lines, avg_width, avg_height, align_to_detection_centroid=True)` | [sign_alignment/tablet.py#L172](sign_alignment/tablet.py#L172) |

- **Detection-side**: Converts each `BoundingBox` to a `SignBox` (center representation), preserving original position and size
- **Text-side**: Arranges all text signs on a uniform grid using `avg_width` and `avg_height` as spacing (one `text_line` per row), then translates the entire grid so its centroid aligns with the detection box centroid

Note: The current notebook processes a single sub-tablet (`exp_image`), so the SubTablet's `img` and `detections` come from `cropped[exp_image_idx]`.

### 2.4 Row Detection & Row-level Matching

#### Row Detection (DBSCAN)

Runs DBSCAN clustering on all sign boxes of the detection-side SubTablet, grouping signs into "rows" by y-coordinate proximity.

| Operation | Call | Code Location |
|-----------|------|---------------|
| Detect rows on detection-side | `sub_tablet_detection.detect_rows(eps, min_samples, lambda_weight)` | [sign_alignment/tablet.py#L315](sign_alignment/tablet.py#L315) |
| DBSCAN clustering implementation | `detect_rows_dbscan(boxes, eps, min_samples, lambda_weight, avg_width, avg_height)` | [data_processing/line_process.py#L9](data_processing/line_process.py#L9) |

Text-side row information is already set by `from_text_lines` (`row_idx` = text line number), no additional detection needed.

#### DP Row-level Matching

Aligns the detection row sequence with the text row sequence using dynamic programming to find the optimal row-to-row correspondence.

| Operation | Call | Code Location |
|-----------|------|---------------|
| Get per-row sign sequences | `sub_tablet.get_row_sign_sequences()` | [sign_alignment/tablet.py#L394](sign_alignment/tablet.py#L394) |
| DP row matching | `match_rows_dp(det_row_sequences, text_row_sequences, ...)` | [data_processing/line_process.py#L261](data_processing/line_process.py#L261) |
| Create bidirectional mapping | `create_row_mapping(matches, num_text_rows, num_det_rows)` | [data_processing/line_process.py#L377](data_processing/line_process.py#L377) |
| Row similarity computation | `compute_row_similarity(row1_signs, row2_signs, method)` | [data_processing/line_process.py#L216](data_processing/line_process.py#L216) |

### 2.5 Sign-level Matching

Within each matched row pair, performs sign sequence alignment using DP.

| Operation | Call | Code Location |
|-----------|------|---------------|
| Per-row sign DP matching | `match_signs_in_row_dp(detection_signs, text_signs, ...)` | [data_processing/line_process.py#L403](data_processing/line_process.py#L403) |

Output: `row_sign_matches: {text_row_idx: [(text_sign_idx, det_sign_idx), ...]}`, recording the sign-level correspondences within each matched row.

### 2.6 Coarse Alignment (Baseline Projection)

Uses row matching and sign matching information to "project" text-side sign positions onto the detection-side spatial positions.

| Operation | Call | Code Location |
|-----------|------|---------------|
| Get SignBoxes grouped by row | `sub_tablet.get_rows_dict()` | [sign_alignment/tablet.py#L375](sign_alignment/tablet.py#L375) |
| Baseline projection alignment | `align_text_to_detection_rows(det_rows, text_rows, text_to_det, row_sign_matches, avg_width, avg_height)` | [data_processing/line_process.py#L636](data_processing/line_process.py#L636) |
| Single-row alignment (internal) | `align_text_row_to_detection(det_row_boxes, text_row_boxes, sign_matches, avg_width, avg_height)` | [data_processing/line_process.py#L494](data_processing/line_process.py#L494) |

Alignment logic (for each matched row pair):
1. Using matched `(text_sign, det_sign)` pairs as control points, fit a **linear baseline**: $y = a \cdot x + b$
2. **Matched signs**: x takes the corresponding detection box center `det_box.cx`, y is projected onto the baseline `y(cx)`, width takes the detection box width (clamped to `[2/3, 4/3] × avg_width`), height takes `avg_height`
3. **Unmatched signs**: **linearly interpolate** `(cx, cy)` between adjacent matched signs

Output: `aligned_text_boxes: List[SignBox]`, used to construct the coarse-aligned SubTablet (`sub_tablet_optim`).

### 2.7 PSR Fine Optimization

Using the coarse alignment result as initialization, runs GMM-based Point Set Registration optimization to further fine-tune each sign's position and size.

| Operation | Call | Code Location |
|-----------|------|---------------|
| Create optimizer | `PointSetRegistrationOptimizer(sub_tablet_text, target_detections, sigma, w_noise, lambda_*, contour_mask)` | [sign_alignment/psr_optimizer.py#L141](sign_alignment/psr_optimizer.py#L141) |
| Run optimization | `optimizer.optimize(num_iterations, lr, sigma_anneal, ...)` | [sign_alignment/psr_optimizer.py#L702](sign_alignment/psr_optimizer.py#L702) |
| Get final result | `optimizer.get_optimized_subtablet()` | [sign_alignment/psr_optimizer.py#L810](sign_alignment/psr_optimizer.py#L810) |
| Parameter change analysis | `optimizer.get_param_changes()` | [sign_alignment/psr_optimizer.py#L835](sign_alignment/psr_optimizer.py#L835) |
| Loss curve visualization | `optimizer.plot_loss_history()` | [sign_alignment/psr_optimizer.py#L844](sign_alignment/psr_optimizer.py#L844) |

The optimizer uses Adam, with optional sigma annealing (`sigma_anneal=True`, linearly decaying from initial σ to σ/4), width/height clamped ≥ 10px, and gradient clipping to prevent explosion.

> **Note**: The codebase also contains an older [`ElasticChainOptimizer`](sign_alignment/optimizer.py#L52), based on a heatmap template matching approach, which is not used in the current main pipeline.

---

## 3. Core Data Formats

### 3.1 EBL API JSON

Key structure of fragment data retrieved from the EBL API (see [fragment_output_sample.json](fragment_output_sample.json)):

```json
{
  "museumNumber": {"prefix": "HS", "number": "2086"},
  "text": {
    "lines": [
      {
        "prefix": "1.",
        "type": "TextLine",
        "content": [
          {
            "type": "Word",
            "parts": [
              {
                "type": "Reading",
                "name": "ip",
                "subIndex": 1,
                "enclosureType": []
              },
              { "type": "Joiner" },
              {
                "type": "Reading",
                "name": "qa",
                "subIndex": 2,
                "enclosureType": ["BROKEN_AWAY"]
              }
            ]
          }
        ]
      },
      {
        "prefix": "@",
        "type": "SurfaceAtLine",
        "content": [{"value": "obverse", "type": "ValueToken"}]
      }
    ]
  }
}
```

Key field descriptions:

| Field | Description |
|-------|-------------|
| `text.lines[]` | All lines (including `TextLine`, `SurfaceAtLine`, etc.); only `TextLine` type contains signs |
| `content[]` | Token list for each line |
| `type` | Token type: `Word` (container), `Reading` (phonetic reading), `Logogram` (logographic reading), `Determinative` (determinative), `UnclearSign`, `CompoundGrapheme` (compound grapheme), etc. |
| `name` + `subIndex` | Reading value and subscript index, e.g., `("qa", 2)` represents qa₂ |
| `enclosureType` | If it contains `"BROKEN_AWAY"`, the token is within broken brackets `[...]` |
| `parts` | Child token list for container types (recursive structure) |

### 3.2 text_lines (Parsed Text Lines)

Type: `List[List[str]]`

```python
[
  ['DIŠ', 'IB', 'GA', 'TUM', 'IGI', 'LU', 'TUR', 'IR', 'ZE₂', 'DIM', 'KID'],
  ['DIŠ', 'KA', 'LU', 'UM', 'AN', 'UD', 'TUR', 'LUGAL'],
  ['GABA', 'I', 'TUM', 'RI', 'U', 'U', 'U', 'IGI', 'NE', 'NE', 'IN', 'GAR', 'RI', 'U', 'U', 'U', 'MA'],
  ...
]
```

Text file format (saved by `TextVisualizer.save_text`):

```
Fragment: HS.2086
==================================================
Line 1: DIŠ IB GA TUM IGI LU TUR IR ZE₂ DIM KID
Line 2: DIŠ KA LU UM AN UD TUR LUGAL
Line 3: GABA I TUM RI U U U IGI NE NE IN GAR RI U U U MA
...
```

### 3.3 Tablet Image & Sub-tablet Image

**Full tablet image** `img`: `np.ndarray` (H, W, 3) in BGR format, loaded from `LocalDataSource.load_image()`.

**Sub-tablet cropping**: `divide_tablet_photo()` divides the full tablet into multiple (typically overlapping) patches. Each patch is encapsulated as `SingleImage`:

```python
SingleImage(
    img: np.ndarray,          # patch image (h, w, 3)
    detections: Detection,    # detection boxes on this patch (patch coordinates)
    mask: np.ndarray = None   # binary contour mask (h, w), 0/255
)
```

`crop_coordinates[i]` records the patch's position in the full image: `{"x": int, "y": int, "w": int, "h": int}`

The current notebook selects one patch (`exp_image = cropped[exp_image_idx]`) for the subsequent alignment pipeline.

### 3.4 Ground Truth (`GroundTruths`)

Type alias `List[BoundingBox]`. Source: local file `gt_{fragment_id}.txt`.

```python
BoundingBox(x1=34.0, y1=12.0, x2=78.0, y2=56.0, score=1.0,
            sign=Sign(abz='ABZ58', name='TU', idx=0))
```

- GT annotation score is fixed at 1.0
- File format is `x,y,w,h,sign_name` (one box per line), converted to `(x1, y1, x2=x+w, y2=y+h)` during loading

### 3.5 Detection (`Detection`)

Type alias `List[BoundingBox]`, same structure as GT, but `score` is model confidence (filtered by `score_threshold`), coordinates in full image coordinate system.

### 3.6 SignBox & SubTablet

**`SignBox`** (center representation, [sign_alignment/tablet.py#L22](sign_alignment/tablet.py#L22)):

```python
SignBox(
    sign=Sign(abz='ABZ58', name='TU', idx=0),
    score=0.95,
    cx=56.0, cy=34.0,       # center coordinates (pixels)
    width=44.0, height=44.0, # width and height (pixels)
    row_idx=3,               # row index (DBSCAN cluster result or text line number)
    col_idx=1                # column index within the row
)
```

- Interconvertible: `SignBox.from_detection(bbox)` ↔ `sign_box.to_bounding_box()`
- Provides `x1, y1, x2, y2` properties (computed from center + dimensions)

**`SubTablet`** ([sign_alignment/tablet.py#L127](sign_alignment/tablet.py#L127)):

| Field | Type | Description |
|-------|------|-------------|
| `sign_boxes` | `List[SignBox]` | All sign boxes |
| `img` | `np.ndarray \| None` | Associated image |
| `avg_width`, `avg_height` | `float` | Average sign dimensions |
| `name` | `str` | Identifier name (`"detection"` / `"text"` / `"optim"` / `"optimized"`) |

Key methods:

| Method | Return Type | Description |
|--------|-------------|-------------|
| `get_rows()` | `List[List[SignBox]]` | Grouped by `row_idx`, sorted by `cx` |
| `get_rows_dict()` | `Dict[int, List[SignBox]]` | Same as above, keyed by row number ([L375](sign_alignment/tablet.py#L375)) |
| `get_row_sign_sequences()` | `List[List[str]]` | Per-row sign name sequences (for DP matching, [L394](sign_alignment/tablet.py#L394)) |
| `to_detection_list()` | `List[BoundingBox]` | Convert to BoundingBox list (for visualization) |
| `detect_rows(eps, min_samples, lambda_weight)` | `int` | Run DBSCAN, update all `sign_box.row_idx`, return row count |
| `info` | `str` | Summary info string |

### 3.7 Optimization Results

Both coarse alignment and PSR fine optimization results are returned as `SubTablet` objects:

- `sub_tablet_optim` (after coarse alignment): built from `List[SignBox]` returned by `align_text_to_detection_rows`
- `sub_tablet_final` (after PSR optimization): returned by `optimizer.get_optimized_subtablet()`

**Parameter changes**: `optimizer.get_param_changes()` returns `np.ndarray (M, 4)`, each row being `[Δcx, Δcy, Δw, Δh]` (difference before and after PSR).

**Loss history**: `optimizer.loss_components_history` is a list with one dict per iteration:

```python
{'total': 1.23, 'data': 0.95, 'anchor': 0.01, 'seq': 0.12, 'height': 0.003, 'rows': 0.08, 'sigma': 60.0}
```

### 3.8 Output Files

Alignment results are saved in `alignment_results/` and `alignment_results_heatmap*/` directories:

| Filename Pattern | Content |
|-----------------|---------|
| `{id}_3_text.txt` | Formatted text lines (sign name sequences per line) |
| `{id}_info.txt` | Statistical summary (GT/detection/optimized counts, per-crop match scores) |
| `alignment_summary.json` | Batch alignment summary |
| `debug_{id}_*.jpg` | Visualization images for each step |

---

## 4. Alignment Algorithms

### 4.1 DBSCAN Row Detection

**Purpose**: Group scattered detection-side sign boxes into "rows" by y-coordinate proximity.

**Implementation**: [`detect_rows_dbscan()`](data_processing/line_process.py#L9)

Custom distance metric emphasizing y-coordinate proximity:

$$d(A, B) = \sqrt{\lambda \cdot (\Delta x)^2 + (\Delta y)^2}$$

Then normalized by `avg_size = (avg_width + avg_height) / 2`:

$$d_{\text{norm}}(A, B) = \frac{d(A, B)}{\text{avg\_size}}$$

In practice, coordinates are divided by `avg_size` and the x component is scaled by $\sqrt{\lambda}$, then standard Euclidean-distance DBSCAN is applied.

**Typical parameters**:
- `lambda_weight = 0.007`: Extremely small x-direction weight (≈ 0.08 scale factor), clustering depends almost entirely on y-coordinates
- `eps = 0.4`: Normalized distance threshold, actual distance ≈ `eps × avg_size` pixels
- `min_samples = 1`: Allows single-sign rows

**Post-processing**: After clustering, row labels are renumbered **top-to-bottom by average y-coordinate** (0, 1, 2, ...), noise points are labeled -1.

### 4.2 DP Row-level Matching

**Purpose**: Establish optimal correspondence between the detection row sequence (M rows) and the text row sequence (N rows).

**Implementation**: [`match_rows_dp()`](data_processing/line_process.py#L261)

DP state `dp[i][j]`: minimum cost to align the first i text rows with the first j detection rows.

**Three-way transition**:

$$dp[i][j] = \min\begin{cases} dp[i-1][j-1] + (1 - \text{sim}(t_i, d_j)) & \text{match} \\ dp[i-1][j] + p_{\text{skip\_text}} & \text{skip text row} \\ dp[i][j-1] + p_{\text{skip\_det}} & \text{skip detection row} \end{cases}$$

**Special handling**:

| Mechanism | Description |
|-----------|-------------|
| **Free start** | `dp[i][0] = 0`, allows skipping text beginning (top damage) |
| **Free end** | Takes $\min_i dp[i][M]$ at the last column, allows skipping text end (bottom damage) |
| **Small row penalty** | Detection rows with ≤ `small_det_threshold` signs use lower penalty `skip_small_det_penalty` (these are mostly noise) |

**Row similarity** ([`compute_row_similarity()`](data_processing/line_process.py#L216)):

- `'lcs'`: LCS (Longest Common Subsequence) length divided by the average length of the two rows
- `'jaccard'`: Jaccard coefficient of sign sets (intersection / union)

**Output**: `matches: List[(text_row_idx, det_row_idx)]` (matched row pairs) + bidirectional mappings `text_to_det`, `det_to_text`

### 4.3 DP Sign-level Matching

**Purpose**: Align sign sequences within each matched row pair.

**Implementation**: [`match_signs_in_row_dp()`](data_processing/line_process.py#L403)

Three-way transition similar to row-level DP:

- **match**: same sign name cost = 0, different cost = `mismatch_cost` (default 0.9)
- **skip_text**: skip text sign, penalty = `skip_text_penalty`
- **skip_det**: skip detection sign, penalty = `skip_det_penalty`

Output: `sign_matches: List[(text_sign_idx, det_sign_idx)]`.

### 4.4 Coarse Alignment (Baseline Projection)

**Purpose**: Using matched sign pairs, "project" text-side signs onto the detection-side spatial positions.

**Implementation**: [`align_text_to_detection_rows()`](data_processing/line_process.py#L636), internally calls [`align_text_row_to_detection()`](data_processing/line_process.py#L494).

Processing for each matched row pair:

**Step 1 — Fit row baseline**: Using matched pairs' `(det_box.cx, det_box.cy)` as control points, least-squares fit a linear baseline:

$$y_{\text{baseline}}(x) = a \cdot x + b$$

**Step 2 — Matched signs**:
- `cx` ← corresponding detection box center `det_box.cx`
- `cy` ← baseline value $y_{\text{baseline}}(cx)$ (not directly `det_box.cy`, ensuring intra-row y consistency)
- `width` ← detection box width (clamped to `[2/3, 4/3] × avg_width`)
- `height` ← `avg_height`

**Step 3 — Unmatched signs** (present in text but no detection correspondence):
- **Linearly interpolate** `(cx, cy)` between adjacent matched signs
- Unmatched signs at row head/tail are extrapolated at fixed spacing (`avg_width`)

### 4.5 PSR Fine Optimization (GMM Point Set Registration)

**Purpose**: Using coarse alignment as initialization, gradient-optimize each sign's `(cx, cy, width, height)` so that text-side point set S better matches detection-side point set X.

**Implementation**: [`PointSetRegistrationOptimizer`](sign_alignment/psr_optimizer.py#L141)

#### Symbol Definitions

| Symbol | Meaning | Corresponding Parameter |
|--------|---------|------------------------|
| $S = \{s_m\}_{m=1}^M$ | Text-side sign centers (Source, to be optimized) | `params[:, :2]` |
| $X = \{x_n\}_{n=1}^N$ | Detection-side sign centers (Target, fixed) | `X_pos` |
| $w_m, h_m$ | Width and height of the $m$-th Source sign (to be optimized) | `params[:, 2]`, `params[:, 3]` |
| $w$ | Uniform background noise weight, absorbing unmatched detection boxes | `w_noise` (default 0.1) |
| $W_{c_m, c_n}$ | Class match matrix, defaults to identity (only same-class attract) | `confusion_matrix` |
| $\sigma$ | GMM bandwidth, controls attraction range | `sigma` (default `(avg_width + avg_height) / 2`) |
| $R$ | Number of Source rows | `num_rows` |
| $L_r$ | Number of signs in row $r$ | `row_lengths[r]` |
| $\text{mask}$ | Tablet contour binary mask (255=interior, 0=exterior) | `contour_mask` |

Source signs are arranged by row: row $r$ ($r = 0, \ldots, R-1$) contains signs $s_{r,0}, s_{r,1}, \ldots, s_{r, L_r - 1}$.

#### Data Loss (GMM)

Implementation: [`compute_data_loss()`](sign_alignment/psr_optimizer.py#L354)

Treats Source as GMM centers and Target as observed data. The likelihood for each Target point $x_n$ is:

$$p(x_n) = \frac{w}{N} + \frac{1-w}{M} \sum_{m=1}^{M} W_{c_m, c_n} \cdot \mathcal{N}(x_n;\, s_m,\, \sigma^2 I)$$

Where the Gaussian component is:

$$\mathcal{N}(x_n;\, s_m,\, \sigma^2 I) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{\|x_n - s_m\|^2}{2\sigma^2}\right)$$

The data loss is the mean negative log-likelihood:

$$E_{\text{data}} = -\frac{1}{N} \sum_{n=1}^{N} \log p(x_n)$$

Computed in log-space using `logsumexp` for numerical stability.

When `sigma_anneal=True`, σ linearly anneals from its initial value to σ/4 (coarse → fine matching):

$$\sigma(t) = \sigma_0 \cdot (1 - t) + \sigma_{\text{final}} \cdot t, \quad t = \frac{\text{iter}}{\text{num\_iterations} - 1}$$

#### Structural Constraint Losses

##### Anchor Loss (Row Baseline Anchoring)

Implementation: [`compute_anchor_loss()`](sign_alignment/psr_optimizer.py#L403)

For each row $r$, fits a row baseline to the sign centers $(x_{r,j},\, y_{r,j})$ via least squares:

$$y_{\text{base}}^{(r)}(x) = a_r \cdot x + b_r$$

Mean squared deviation of each sign from its row baseline:

$$L_{\text{anchor}} = \frac{1}{R'} \sum_{r:\, L_r \ge 2} \frac{1}{L_r} \sum_{j=0}^{L_r - 1} \left(y_{r,j} - y_{\text{base}}^{(r)}(x_{r,j})\right)^2$$

Where $R'$ is the number of rows with $L_r \ge 2$. Effect: keeps same-row signs' y-coordinates aligned.

##### Seq Loss (Intra-row Spacing Constraint)

Implementation: [`compute_seq_loss()`](sign_alignment/psr_optimizer.py#L440)

For each pair of adjacent signs $(j, j+1)$ within a row, the expected gap is determined by their widths:

$$\text{gap}_{\text{expected}} = \frac{w_{r,j} + w_{r,j+1}}{2}$$

$$\text{gap}_{\text{actual}} = cx_{r,j+1} - cx_{r,j}$$

$$L_{\text{seq}} = \frac{1}{K} \sum_{r} \sum_{j=0}^{L_r - 2} \left(\text{gap}_{\text{actual}} - \text{gap}_{\text{expected}}\right)^2$$

Where $K$ is the total number of adjacent pairs across all rows. Effect: keeps intra-row signs uniformly spaced.

##### Height Loss (Intra-row Height Consistency)

Implementation: [`compute_height_loss()`](sign_alignment/psr_optimizer.py#L468)

Variance of sign heights within each row:

$$L_{\text{height}} = \frac{1}{R'} \sum_{r:\, L_r \ge 2} \text{Var}\!\left(\{h_{r,j}\}_{j=0}^{L_r-1}\right)$$

Effect: keeps sign heights consistent within the same row.

##### Rows Loss (Inter-row Spacing Constraint, Asymmetric Threshold)

Implementation: [`compute_rows_loss()`](sign_alignment/psr_optimizer.py#L513)

For adjacent row pairs $(r, r+1)$, the ideal row spacing is determined by the average heights of the two rows:

$$d_{\text{ideal}}^{(r)} = \frac{\bar{h}_r + \bar{h}_{r+1}}{2}$$

Where $\bar{h}_r = \frac{1}{L_r}\sum_j h_{r,j}$ is the average height of row $r$.

Actual row spacing is computed via the difference of two row baselines at an alignment point:

$$x_{\text{align}} = \max(\min_j x_{r,j},\; \min_j x_{r+1,j})$$

$$d_{\text{actual}}^{(r)} = y_{\text{base}}^{(r+1)}(x_{\text{align}}) - y_{\text{base}}^{(r)}(x_{\text{align}})$$

Deviation $d = d_{\text{actual}} - d_{\text{ideal}}$. Uses **asymmetric thresholds** and **asymmetric plateau values** to separately control penalties for rows too far apart vs. too close:

$$t = \begin{cases} d_{\text{ideal}} \cdot r_{\text{far}} & \text{if } d \ge 0 \quad (\text{rows too far apart}) \\ d_{\text{ideal}} \cdot r_{\text{close}} & \text{if } d < 0 \quad (\text{rows too close}) \end{cases}$$

$$P = \begin{cases} P_{\text{far}} & \text{if } d \ge 0 \\ P_{\text{close}} & \text{if } d < 0 \end{cases}$$

Where $r_{\text{far}}$ = `rows_threshold_ratio_far` (default 1/3), $r_{\text{close}}$ = `rows_threshold_ratio_close` (default 1/2); $P_{\text{far}}$ = `rows_plateau_far` (default 1.0), $P_{\text{close}}$ = `rows_plateau_close` (default 1.0).

Piecewise loss function:

$$\ell(d, t, P) = \begin{cases} P \cdot \left(\dfrac{d}{t}\right)^2 & \text{if } |d| \le t \\ P & \text{if } |d| > t \end{cases}$$

Total inter-row loss is the average over all adjacent row pairs:

$$L_{\text{rows}} = \frac{1}{R - 1} \sum_{r=0}^{R-2} \ell\!\left(d_{\text{actual}}^{(r)} - d_{\text{ideal}}^{(r)},\; t^{(r)},\; P^{(r)}\right)$$

Within the threshold: quadratic penalty (smooth gradients). Beyond the threshold: constant plateau (no further penalty increase, preventing extreme deviations from dominating gradients). Different $P_{\text{far}}$ and $P_{\text{close}}$ allow different maximum penalty magnitudes for rows too close vs. too far apart.

##### Boundary Loss (Contour Boundary Constraint)

Implementation: [`compute_boundary_loss()`](sign_alignment/psr_optimizer.py#L582)

**Purpose**: Prevent the leftmost (first) sign's bounding box in each row from extending beyond the tablet contour region (in top, left, bottom directions).

**Input**: `contour_mask` — binary mask generated by `divide_tablet_photo(return_masks=True)` (255 = tablet interior, 0 = exterior), stored in `SingleImage.mask`.

**Scope**: Only constrains the **first sign** in each row (`col_idx == 0`, i.e., leftmost), since cuneiform is written left-to-right, and the leftmost sign is closest to the row start edge and most likely to extend beyond the tablet contour boundary due to alignment offsets.

**Computation**:

For each row $r$'s first sign $s_{r,0}$, compute the Intersection over Region (IoR) between its bounding box $(x_1, y_1, x_2, y_2)$ and the contour mask:

$$\text{IoR}_r = \frac{\text{Area}(\text{bbox}_{r,0} \cap \text{contour})}{\text{Area}(\text{bbox}_{r,0})}$$

When bbox is fully inside the contour, IoR = 1 (no penalty); when bbox is partially or fully outside, IoR < 1.

In practice, IoR is computed by uniformly sampling $8 \times 8$ points within the (image-clamped) bbox, querying mask values, and averaging to get a sampled interior fraction $\hat{f}$, then multiplying by an area correction factor (accounting for the portion outside image bounds):

$$\text{IoR}_r = \hat{f}_r \cdot \frac{\text{clamped\_area}}{\text{bbox\_area}}$$

Where clamped\_area is the area after clamping to image bounds, and bbox\_area is the original bbox area. The sampling query part is detached (non-differentiable), but the area ratio participates in the computation graph through bbox's $(w, h)$, maintaining differentiability.

**Loss function** (steep quadratic penalty):

$$L_{\text{boundary}} = \frac{1}{R} \sum_{r=0}^{R-1} k \cdot (1 - \text{IoR}_r)^2$$

Where $k = 1000$ (`boundary_steepness` default) is the steepness factor. When IoR = 1, loss = 0; IoR = 0.5, loss = 250; IoR = 0, loss = 1000. The very large steepness factor ensures that even slight boundary violations produce strong gradients, quickly pushing signs back inside the contour.

#### Total Loss

$$L = \lambda_{\text{data}} \cdot E_{\text{data}} + \lambda_{\text{anchor}} \cdot L_{\text{anchor}} + \lambda_{\text{seq}} \cdot L_{\text{seq}} + \lambda_{\text{height}} \cdot L_{\text{height}} + \lambda_{\text{rows}} \cdot L_{\text{rows}} + \lambda_{\text{boundary}} \cdot L_{\text{boundary}}$$

Total loss computation: [`compute_total_loss()`](sign_alignment/psr_optimizer.py#L681)

#### Optimization Process

The optimization loop is implemented in [`optimize()`](sign_alignment/psr_optimizer.py#L702):
- Optimizer: Adam (`lr` default 1.0)
- Constraints: width, height clamped ≥ 10px after each step
- Gradient safety: `clip_grad_norm` ≤ 1e4, NaN gradients zeroed
- Optional sigma annealing: linearly from `sigma` to `sigma_final` (default σ/4)

#### Loss Function Summary

| Loss | Formula | Input Variables | Shape | Effect | Implementation |
|------|---------|----------------|-------|--------|----------------|
| $E_{\text{data}}$ | $-\frac{1}{N}\sum_n \log p(x_n)$ (GMM negative log-likelihood) | Source-target point pair distances | Far distances approach constant $-\log w$, near distances drop sharply | Attract Source toward Target | [`compute_data_loss()`](sign_alignment/psr_optimizer.py#L354) |
| $L_{\text{anchor}}$ | Mean squared deviation after row baseline $y=ax+b$ fitting | $y - y_{\text{baseline}}$ | Quadratic (parabola) | Keep intra-row y aligned | [`compute_anchor_loss()`](sign_alignment/psr_optimizer.py#L403) |
| $L_{\text{seq}}$ | $(\text{actual\_gap} - \text{expected\_gap})^2$ | Adjacent sign spacing deviation | Quadratic (parabola) | Keep intra-row spacing uniform | [`compute_seq_loss()`](sign_alignment/psr_optimizer.py#L440) |
| $L_{\text{height}}$ | Intra-row height variance $\text{Var}(\{h_j\})$ | Height deviation | Quadratic | Keep intra-row heights consistent | [`compute_height_loss()`](sign_alignment/psr_optimizer.py#L468) |
| $L_{\text{rows}}$ | Asymmetric piecewise function: $P \cdot (d/t)^2$ or $P$ (plateau) | Inter-row spacing deviation | Asymmetric quadratic + asymmetric plateau | Keep inter-row spacing reasonable | [`compute_rows_loss()`](sign_alignment/psr_optimizer.py#L513) |
| $L_{\text{boundary}}$ | $k \cdot (1 - \text{IoR})^2$ ($k=1000$) | bbox-contour mask intersection area ratio | Steep quadratic | Prevent row-start signs from exceeding tablet contour | [`compute_boundary_loss()`](sign_alignment/psr_optimizer.py#L582) |

#### Loss Function Shape Visualization

Generated via `optimizer.plot_loss_curves(save_dir="alignment_loss_functions")`, saved in the `alignment_loss_functions/` directory:

| Loss | Curve Plot | Description |
|------|-----------|-------------|
| $E_{\text{data}}$ | ![loss_data](alignment_loss_functions/loss_data.png) | Negative log-likelihood vs. distance for a single source-target pair. Approaches $-\log w$ plateau at large distances (noise term floor) |
| $L_{\text{anchor}}$ | ![loss_anchor](alignment_loss_functions/loss_anchor.png) | Quadratic penalty for y-deviation from row baseline |
| $L_{\text{seq}}$ | ![loss_seq](alignment_loss_functions/loss_seq.png) | Quadratic penalty for adjacent sign spacing deviation from expected |
| $L_{\text{height}}$ | ![loss_height](alignment_loss_functions/loss_height.png) | Variance growth when a single sign height deviates from row mean (comparison across row lengths) |
| $L_{\text{rows}}$ | ![loss_rows](alignment_loss_functions/loss_rows.png) | Asymmetric piecewise loss: rows too far apart vs. too close use different thresholds, plateau beyond threshold |
| $L_{\text{boundary}}$ | ![loss_boundary](alignment_loss_functions/loss_boundary.png) | Lower IoR (bbox-contour intersection ratio) → higher loss; steep quadratic penalty forces row-start signs inside contour |

---

## 5. Visualization Methods

All visualization tools are defined in [`sign_alignment/visualizer.py`](sign_alignment/visualizer.py).

> **Row numbering convention**: Detection rows use **D#** (D1, D2, ...), text rows use **R#** (R1, R2, ...), both **1-indexed** for display (internal `row_idx` remains 0-indexed). Matches are shown as `D5→R3` (detection row 5 mapped to text row 3) or `R3→D5` (text row 3 mapped to detection row 5).

> **Color consistency**: Three visualizations (`draw_rows` for detection rows, `draw_text_mapping` for text rows, `draw_alignment_diagnostic` for alignment diagnostics) all use **text_row_idx** as input to `_get_row_color()`, ensuring the same row pair uses the same color across all figures.

### 5.1 BboxVisualizer

**Main class** ([`BboxVisualizer`](sign_alignment/visualizer.py#L13)), for drawing detection boxes on images.

#### draw_boxes

[`draw_boxes(img, boxes, show_labels)`](sign_alignment/visualizer.py#L25)

Draws rectangles on the image. Labels are rendered using PIL to support Unicode cuneiform sign names. Label size auto-adapts to box height (1/6 of box height, minimum 12px).

**Use cases**: GT box visualization, detection result visualization, alignment result visualization, overlay comparisons.

#### draw_rows

[`draw_rows(img, boxes, show_labels, show_row_numbers, row_mapping, row_label_prefix, mapped_label_prefix, ...)`](sign_alignment/visualizer.py#L144)

Visualize row structure:
- Each row uses a unique color from the **HSV color wheel** (golden angle 137.5° spacing)
- Same-row sign centers are connected with **colored lines**
- Center points are drawn as colored circle markers
- `show_row_numbers=True`: annotate row numbers on the left margin
- `row_label_prefix`: row label prefix, `"D"` for detection rows, `"R"` for text rows (default `"R"`)
- `mapped_label_prefix`: mapped target prefix, `"D"` or `"R"` (default `"D"`)
- When `row_mapping` is provided: displays mapping relationship `{prefix}{row+1}→{mapped_prefix}{mapped+1}`
- `img=None`: automatically creates a white canvas (for text-side with no background image)

**Use cases**: DBSCAN row detection results (`row_label_prefix="D"`), row matching display.

#### draw_text_mapping

[`draw_text_mapping(img, sign_boxes, row_mapping, sign_match_info, ...)`](sign_alignment/visualizer.py)

Displays text rows' **per-sign matching status**, replacing the previous uniform-color-per-row `draw_rows`:

| Status | Color | Label |
|--------|-------|-------|
| Matched, same sign name | Row primary color (HSV golden angle based on text row) | Text sign name (white text on black background) |
| Matched, different sign name | Row primary color **desaturated** (~40%) | Text sign name (black bg) + detection sign name (dark gray bg RGB(80,80,80), below) |
| Unmatched / entire row unmatched | Gray (128,128,128) | Text sign name (gray bg) |

- Row annotation uses primary color showing `R#→D#` (regardless of per-sign match quality), unmatched rows shown in gray as `R#`
- Center connecting lines and markers maintain existing style

**Use cases**: Text-side match quality overview.

#### draw_alignment_diagnostic

[`draw_alignment_diagnostic(img, detection_sign_boxes, aligned_text_boxes, det_sign_match_info, text_sign_match_info, det_to_text, ...)`](sign_alignment/visualizer.py)

Overlays **coarse alignment results** on the detection image, including detection boxes, placed text sign boxes, and row connecting lines:

| Element | Display Style |
|---------|--------------|
| Detection box — matched, same name | Row primary color, detection sign name (black bg) |
| Detection box — matched, different name | Desaturated row color, detection sign name (dark gray bg) + text sign name (colored bg, below) |
| Detection box — unmatched | Gray, detection sign name (gray bg) |
| Text box — matched different name (interpolated/extrapolated placement) | **Dashed rectangle** (desaturated row color), text sign name (desaturated color bg) |
| Text box — unmatched (interpolated/extrapolated placement) | **Dashed rectangle** (light gray), text sign name (light gray bg) |
| Detection row connecting line | **Solid line**, row primary color |
| Text row connecting line | **Dashed line**, desaturated row color |
| Row annotation | `D#→R#` or `D#` (unmatched row) |

**Use cases**: Coarse alignment quality diagnostics, replacing the previous four-color (green/orange/cyan/magenta) diagnostic figure.

#### display_result / save / show_draw

[`display_result(vis_opt)`](sign_alignment/visualizer.py#L100): Three output modes:

| `vis_opt` | Behavior |
|-----------|----------|
| `"show"` | `cv2.imshow` popup window |
| `"draw"` | `matplotlib.pyplot.show` inline display |
| `"save"` | `cv2.imwrite` save to file |

### 5.2 TextVisualizer

[`TextVisualizer`](sign_alignment/visualizer.py)

- `save_text(text_lines, path, fragment_id)`: writes `List[List[str]]` to a `.txt` file, formatted as `Line N: sign1 sign2 ...`

### 5.3 HeatmapVisualizer

[`HeatmapVisualizer`](sign_alignment/visualizer.py)

Used for the `ElasticChainOptimizer` heatmap approach visualization. Not used in the current main pipeline (PSR).

### 5.4 CompositeVisualizer

[`CompositeVisualizer`](sign_alignment/visualizer.py) — General-purpose composite image tool.

Composes multiple BGR images into a grid-layout composite, supporting titles and custom sizes:

```python
comp = CompositeVisualizer()
comp.compose(
    images=[img1, img2, img3, img4],
    layout=(2, 2),
    titles=["Title A", "Title B", "Title C", "Title D"],
    figsize=(16, 12)
)
comp.display_result(vis_opt="draw")  # or "save"
comp.save("output.jpg")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `images` | `List[np.ndarray]` | BGR image list (typically from each visualizer's `.result`) |
| `layout` | `(rows, cols)` | Grid layout |
| `titles` | `List[str]` | Title for each subplot |
| `figsize` | `(w, h)` | matplotlib figsize |

Returns a composite BGR image, supporting `display_result`, `save`, `show_draw` methods (consistent interface with `BboxVisualizer`).

**Use cases**: Replaces hand-written `plt.subplots(...)` composite code in notebooks, e.g., row matching comparison figures, before/after optimization comparison figures.

### 5.5 Helper Functions

| Function | Description |
|----------|-------------|
| `build_sign_match_info(row_sign_matches, text_to_det, det_rows_dict, optim_sign_boxes)` | Build text-side and detection-side sign-level match info dicts, used by `draw_text_mapping` and `draw_alignment_diagnostic` |
| `_get_row_color(row_idx)` | Module-level helper, generates HSV golden-angle row color (RGB) |
| `_desaturate_color(rgb, factor)` | Reduce RGB color saturation (factor=0→grayscale, 1→original) |
| `_draw_dashed_rect(img, pt1, pt2, color, ...)` | Draw dashed rectangle on OpenCV image |
| `_draw_dashed_line(img, pt1, pt2, color, ...)` | Draw dashed line on OpenCV image |

### 5.6 Loss Curves

[`PointSetRegistrationOptimizer.plot_loss_history()`](sign_alignment/psr_optimizer.py#L844)

Four subplots (2×2 grid):
1. **(0,0) Total loss** over iterations
2. **(0,1) Raw component values** (data, anchor, seq, height, rows)
3. **(1,0) Weighted component values** (multiplied by corresponding lambda, showing actual contribution) + σ annealing curve (if enabled)
4. **(1,1) Log-scale raw component values** (for observing components with large magnitude differences)

### 5.7 Result Comparison

Final results are presented in the notebook (Step 16) as a 2×2 four-panel composite via `CompositeVisualizer`, with each panel also saved individually:

| Position | Content | Individual Save File |
|----------|---------|---------------------|
| (0,0) | Coarse alignment result (cyan boxes) | `debug_{sample}_coarse_aligned.jpg` |
| (0,1) | PSR optimized result (yellow boxes) | `debug_{sample}_final_optimized.jpg` |
| (1,0) | Overlay: detection boxes (red) + final result (yellow) | `debug_{sample}_overlay_det_final.jpg` |
| (1,1) | Overlay: GT boxes (green) + final result (yellow) | `debug_{sample}_overlay_gt_final.jpg` |

Composite image saved as `debug_{sample}_results_comparison.jpg`.

---

## 6. Inference Script

### 6.1 Overview

[signs_alignment_samples.py](signs_alignment_samples.py) is the batch inference script that runs the full PSR alignment pipeline on multiple samples and saves visualization results. Replaces the older heatmap-based `signs_alignment_heatmap.py`.

```bash
python signs_alignment_samples.py
```

### 6.2 Configuration Parameters

All tunable parameters are defined at the top of the script:

| Group | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| Basic | `SAMPLE_LIMIT` | 30 | Number of samples to process |
| Basic | `OUTPUT_DIR` | `"alignment_results"` | Output directory for visualizations |
| DBSCAN | `DBSCAN_EPS` | 0.4 | Distance threshold (normalized) |
| DBSCAN | `DBSCAN_MIN_SAMPLES` | 1 | Minimum neighbor count |
| DBSCAN | `DBSCAN_LAMBDA_WEIGHT` | 0.007 | x-direction weight |
| Row Matching | `ROW_MATCH_SKIP_TEXT_PENALTY` | 0.5 | Penalty for skipping text row |
| Row Matching | `ROW_MATCH_SKIP_DET_PENALTY` | 1.0 | Penalty for skipping detection row |
| Row Matching | `ROW_MATCH_SIMILARITY_METHOD` | `'jaccard'` | Row similarity method |
| Sign Matching | `SIGN_MATCH_MISMATCH_COST` | 0.9 | Cost when sign names differ |
| PSR | `sigma_factor` | 1.5 | σ = avg_width × factor |
| PSR | `w_noise` | 0.1 | Uniform noise weight |
| PSR | `lambda_data` | 2.0 | GMM data loss weight |
| PSR | `lambda_anchor` | 0.01 | Row baseline anchoring loss weight |
| PSR | `lambda_seq` | 0.1 | Intra-row spacing constraint weight |
| PSR | `lambda_height` | 0.01 | Height consistency weight |
| PSR | `lambda_rows` | 5.0 | Inter-row spacing constraint weight |
| PSR | `lambda_boundary` | 1.0 | Contour boundary constraint weight |
| PSR | `num_iterations` | 80 | PSR optimization iterations |
| PSR | `lr` | 1.0 | Adam learning rate |

### 6.3 Processing Flow

For each sample, the script executes the following steps:

1. **Load data**: local image + GT annotations + EBL API text transliteration (parsed via `SignTextParser.parse_text_lines` + `SignAPIResolver`)
2. **Detection**: `TabletImageDetector.detect()` → full-image detection + cropping into sub-tablets
3. **Per-crop processing** (`process_single_crop`):
   - Build detection/text SubTablets
   - DBSCAN row detection → DP row matching → DP sign matching
   - Baseline projection coarse alignment → PSR fine optimization
4. **Coordinate restoration**: convert optimized bboxes from crop coordinates back to full-image coordinates
5. **Save visualizations**

### 6.4 Output Files

Results are saved in the `alignment_results/` directory:

| Filename Pattern | Content |
|-----------------|---------|
| `{id}_1_ground_truth.jpg` | GT boxes (green) |
| `{id}_2_detections.jpg` | Detection boxes (red) |
| `{id}_3_text.txt` | Formatted text lines |
| `{id}_4_optimized.jpg` | PSR optimized boxes (yellow) |
| `{id}_5_det_vs_optimized.jpg` | Detection (red) + optimized (yellow) overlay |
| `{id}_6_optimized_vs_gt.jpg` | GT (green) + optimized (yellow) overlay |
| `{id}_crop{i}_comparison.jpg` | Per-crop 2×2 comparison (coarse/PSR/detection overlay/GT overlay) |
| `{id}_crop{i}_det_rows.jpg` | Detection rows visualization (DBSCAN rows + row matching mapping) |
| `{id}_crop{i}_text_rows_mapped.jpg` | Text rows mapping visualization (per-sign match status) |
| `{id}_crop{i}_rows_side_by_side.jpg` | Detection rows and text rows side-by-side comparison |
| `{id}_crop{i}_alignment_diagnostic.jpg` | Alignment diagnostic (detection boxes + placed text boxes + row lines) |
| `{id}_info.txt` | Statistical summary |
| `alignment_summary.json` | Batch alignment JSON summary |

---

## 7. Evaluation

### 7.1 Overview

[evaluate_alignment.py](evaluate_alignment.py) performs quantitative evaluation of PSR alignment results, computing object-detection-style metrics (mAP, IoU, Precision, Recall), and provides hyperparameter tuning functionality.

```bash
python evaluate_alignment.py
```

### 7.2 Evaluation Metrics

#### IoU Computation

Standard Intersection over Union:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

#### Prediction-GT Matching

Uses a greedy matching strategy: all (pred, gt) pairs are sorted by IoU in descending order, matched sequentially (each GT matches at most one prediction). Two modes are supported:

- **Class-agnostic**: considers only position (IoU), ignores class labels
- **Class-aware**: requires pred and gt sign names to match

#### mAP Computation

Computes precision at standard IoU thresholds [0.5, 0.55, ..., 0.95], averaged to get mAP (COCO-style). Also reports AP@0.5 and AP@0.75.

#### Per-class Metrics

At IoU=0.5 threshold, computes TP/FP/FN per sign class, then per-class Precision, Recall, and F1.

### 7.3 Execution Flow

The script automatically executes three steps:

**STEP 1 — Default Parameter Evaluation**

Runs the full PSR alignment pipeline on `EVAL_SAMPLE_LIMIT` (default 10) samples using `DEFAULT_PSR_PARAMS`, computes and prints metrics, saves to `evaluation_results/evaluation_default.json`.

**STEP 2 — Coordinate-wise Hyperparameter Search**

See §8 Hyperparameter Tuning below.

**STEP 3 — Best Parameter Re-evaluation**

Re-evaluates on the full evaluation set using the best parameters found during search, saves to `evaluation_results/evaluation_best.json`.

### 7.4 Output

| File | Content |
|------|---------|
| `evaluation_results/evaluation_default.json` | Default parameter evaluation results |
| `evaluation_results/evaluation_best.json` | Best parameter evaluation results |
| `evaluation_results/hyperparam_search.json` | Full hyperparameter search results |

---

## 8. Hyperparameter Tuning

### 8.1 Method

Uses **Coordinate-wise Sweep**: adjusts one parameter at a time while keeping all others at their current best values. Compared to full grid search, complexity drops from $O(\prod_i |V_i|)$ to $O(\sum_i |V_i|)$.

Runs two rounds to allow parameters to mutually adapt. Search uses reduced iterations (`num_iterations=40`) for speed.

### 8.2 Search Space

| Parameter | Candidate Values | Description |
|-----------|-----------------|-------------|
| `lambda_data` | [0.5, 1.0, 2.0, 5.0, 10.0] | GMM data loss weight |
| `lambda_anchor` | [0.005, 0.01, 0.05, 0.1] | Row baseline anchoring weight |
| `lambda_seq` | [0.01, 0.05, 0.1, 0.5] | Intra-row spacing constraint weight |
| `lambda_height` | [0.0, 0.005, 0.01, 0.05] | Height consistency weight |
| `lambda_rows` | [1.0, 2.0, 5.0, 10.0] | Inter-row spacing constraint weight |
| `lambda_boundary` | [0.0, 0.5, 1.0, 5.0] | Contour boundary constraint weight |
| `sigma_factor` | [1.0, 1.5, 2.0, 2.5] | GMM bandwidth factor |
| `w_noise` | [0.05, 0.1, 0.2] | Uniform noise weight |

Total evaluations: 2 rounds × (5+4+4+4+4+4+4+3) = 2 × 32 = 64 (minus duplicates for current best values).

### 8.3 Scoring

Uses **class-agnostic mAP** as the scoring metric (higher is better), evaluated on `SEARCH_SAMPLE_LIMIT` (default 5) samples for fast evaluation.

### 8.4 Usage

Running `python evaluate_alignment.py` automatically executes the hyperparameter search (STEP 2). Search results are saved in `evaluation_results/hyperparam_search.json`, containing:

```json
{
  "best_params": { ... },
  "best_mAP": 0.xxxx,
  "all_results": [
    {"round": 1, "param": "lambda_data", "value": 5.0, "mAP": 0.xxxx, ...},
    ...
  ]
}
```

After finding the best parameters, the script automatically re-evaluates on the full evaluation set with full iterations (`num_iterations=80`).

---

## 9. Key Source Files

| File | Key Contents |
|------|-------------|
| [sign_alignment/data_source.py](sign_alignment/data_source.py) | `LocalDataSource`, `EBLAPISource`, `SignTextParser`, `SignAPIResolver` |
| [sign_alignment/detector.py](sign_alignment/detector.py) | `ModelConfig`, `TabletImageDetector`, `SingleImageDetector` |
| [sign_alignment/tablet.py](sign_alignment/tablet.py) | `SignBox`, `SubTablet` |
| [sign_alignment/psr_optimizer.py](sign_alignment/psr_optimizer.py) | `PointSetRegistrationOptimizer` (GMM data loss + structural constraints) |
| [sign_alignment/optimizer.py](sign_alignment/optimizer.py) | `ElasticChainOptimizer` (legacy approach, heatmap-based) |
| [sign_alignment/visualizer.py](sign_alignment/visualizer.py) | `BboxVisualizer`, `TextVisualizer`, `HeatmapVisualizer`, `CompositeVisualizer`, `build_sign_match_info` |
| [sign_alignment/heatmap.py](sign_alignment/heatmap.py) | `compute_avg_dimensions`, `transform_gt_to_cropped_region`, heatmap utilities |
| [sign_alignment/sign.py](sign_alignment/sign.py) | `Sign`, `SignResolver`, `CLASSES_NAME`, `CLASSES_ABZ` |
| [sign_alignment/bounding_box.py](sign_alignment/bounding_box.py) | `BoundingBox`, `Detection`, `GroundTruths` type aliases |
| [data_processing/line_process.py](data_processing/line_process.py) | `detect_rows_dbscan`, `match_rows_dp`, `match_signs_in_row_dp`, `align_text_to_detection_rows` |
| [data_processing/divide_photos.py](data_processing/divide_photos.py) | `divide_tablet_photo` (tablet cropping) |
| [sign_alignment/\_\_init\_\_.py](sign_alignment/__init__.py) | Package public API exports |
| [signs_alignment_samples.py](signs_alignment_samples.py) | Batch inference script (PSR method) |
| [evaluate_alignment.py](evaluate_alignment.py) | Evaluation and hyperparameter tuning script |
| [signs_alignment_heatmap.py](signs_alignment_heatmap.py) | Legacy batch inference script (heatmap method, deprecated) |
