# signs_alignment.ipynb 中 sign match 与相关可视化逻辑报告

## 1. 范围与结论

本报告分析 [signs_alignment.ipynb](signs_alignment.ipynb) 中从行匹配、行内符号匹配到三张关键可视化图的实现链路：

- `text_rows_mapped.jpg`
- `rows_side_by_side.jpg`
- `alignment_diagnostic.jpg`

notebook 本身只调用 pipeline step，真正的匹配和绘制逻辑位于：

- [sign_alignment/pipeline.py](sign_alignment/pipeline.py)
- [sign_alignment/visualizer.py](sign_alignment/visualizer.py)
- [data_processing/line_process.py](data_processing/line_process.py)
- [sign_alignment/tablet.py](sign_alignment/tablet.py)

核心结论：

- sign match 分两层完成：先用 DP 匹配 detection row 与 text row，再在每个已匹配行对内用 DP 匹配符号序列。
- `step_align_text_rows` 会把 text 侧符号投影到 detection 图像坐标，生成 `sub_tablet_aligned`，供 diagnostic overlay 使用。
- `step_build_sign_match_info` 根据行内符号匹配结果生成两个视角的 match-info 字典，随后绘制 `text_rows_mapped`、`rows_side_by_side`、`alignment_diagnostic`。
- 颜色语义统一为：same 使用行主色，diff 使用去饱和行色，unmatched 使用灰色。
- `alignment_diagnostic.jpg` 中实线 bbox/实线中心线来自 detection；虚线 bbox/虚线中心线来自投影后的 text overlay。饱和行色通常表示 same-label 匹配或 detection 行主线，去饱和表示 diff-label 或 text overlay，灰色表示 unmatched。

## 2. notebook 执行顺序

`signs_alignment.ipynb` 中相关 cell 的顺序如下：

1. `step_detect_rows`：对 detection sub-tablet 做 DBSCAN 行检测，并报告 text sub-tablet 行。
2. `step_match_rows`：DP 行匹配，建立 `text_to_det` 与 `det_to_text`。
3. `step_visualize_detection_rows`：绘制 detection 行图，并缓存为 `s.det_row_vis_image`。
4. `step_match_signs_in_rows`：对每个已匹配行对做行内符号 DP 匹配。
5. `step_align_text_rows`：按匹配结果把 text 行粗对齐到 detection 坐标，生成 `s.sub_tablet_aligned`。
6. `step_build_sign_match_info`：生成 match-info 并绘制三张图。

对应 notebook 行在 [signs_alignment.ipynb:194](signs_alignment.ipynb#L194) 到 [signs_alignment.ipynb:251](signs_alignment.ipynb#L251)。

## 3. 使用的数据结构

### 3.1 SignBox

`SignBox` 是符号框的统一表示，定义于 [sign_alignment/tablet.py:17](sign_alignment/tablet.py#L17)。关键字段：

| 字段 | 含义 |
| --- | --- |
| `sign` | `Sign` 对象，包含符号名、ABZ 名、类别索引 |
| `score` | detection 置信度或 text 侧默认分数 |
| `cx`, `cy` | 框中心点坐标 |
| `width`, `height` | 框宽高 |
| `row_idx` | 行号 |
| `col_idx` | 行内列号 |
| `sign_name` | `sign.name` 的属性封装 |

`SignBox` 用中心点加宽高保存，但通过 `x1/y1/x2/y2` 属性提供绘制矩形所需的 corner 坐标，见 [sign_alignment/tablet.py:71](sign_alignment/tablet.py#L71)。

### 3.2 SubTablet

`SubTablet` 持有一组 `SignBox` 和可选图像。与本报告相关的实例包括：

| state 字段 | 来源 | 用途 |
| --- | --- | --- |
| `s.sub_tablet_detection` | 检测结果 | detection 行序列、detection 框绘制、diagnostic 背景 |
| `s.sub_tablet_text` | eBL/API text lines 生成的虚拟网格 | text 行序列、`text_rows_mapped` 绘制 |
| `s.sub_tablet_aligned` | `align_text_to_detection_rows` 输出 | text 框投影到 detection 坐标后的 overlay |

`SubTablet.detect_rows()` 会更新每个 `SignBox.row_idx`，并按行内 x 坐标排序赋值 `col_idx`，见 [sign_alignment/tablet.py:251](sign_alignment/tablet.py#L251)。

`get_row_sign_sequences()` 返回每行的符号名序列，是 DP 匹配的输入，见 [sign_alignment/tablet.py:330](sign_alignment/tablet.py#L330)。

## 4. Sign match 逻辑

### 4.1 行级匹配

`StepMatchRows.run()` 读取 detection/text 两侧的行符号序列：

```python
det_row_sequences = s.sub_tablet_detection.get_row_sign_sequences()
text_row_sequences = s.sub_tablet_text.get_row_sign_sequences()
```

然后调用 `match_rows_dp()`，当前 notebook pipeline 使用参数见 [sign_alignment/pipeline.py:445](sign_alignment/pipeline.py#L445)：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `skip_text_penalty` | `0.5` | 跳过 text 行的代价 |
| `skip_det_penalty` | `1` | 跳过 detection 行的代价 |
| `skip_small_det_penalty` | `0.2` | 跳过小 detection 行的低代价 |
| `small_det_threshold` | `1` | detection 行符号数小于等于 1 时视为小行 |
| `similarity_method` | `"jaccard"` | 行相似度使用符号集合 Jaccard |

`match_rows_dp()` 定义于 [data_processing/line_process.py:261](data_processing/line_process.py#L261)。其 DP 状态为 `dp[i][j]`，表示 text 前 `i` 行与 detection 前 `j` 行对齐的最小代价。每格有三种转移：

| 转移 | 代价 |
| --- | --- |
| match | `dp[i-1, j-1] + (1 - similarity)` |
| skip_text | `dp[i-1, j] + skip_text_penalty` |
| skip_det | `dp[i, j-1] + skip_det_penalty`，小 detection 行使用 `skip_small_det_penalty` |

初始化和结束处理：

- `dp[i, 0] = 0`，允许开头 text 行免费跳过。
- `dp[0, j]` 对 detection 行跳过计罚。
- 在最后一列取最小值，允许结尾 text 行跳过。

输出：

- `s.matches`: `[(text_row_idx, det_row_idx), ...]`
- `s.text_to_det`: `{text_row_idx: det_row_idx}`
- `s.det_to_text`: `{det_row_idx: text_row_idx}`

映射由 `create_row_mapping()` 生成，见 [data_processing/line_process.py:377](data_processing/line_process.py#L377)。

### 4.2 行内符号匹配

`StepMatchSignsInRows.run()` 对 `s.matches` 中每个行对调用 `match_signs_in_row_dp()`，见 [sign_alignment/pipeline.py:518](sign_alignment/pipeline.py#L518)。

当前参数：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `skip_text_penalty` | `0.5` | 跳过 text 符号 |
| `skip_det_penalty` | `2.0` | 跳过 detection 符号 |
| `mismatch_cost` | `0.9` | text/detection 符号名不同时的匹配代价 |

`match_signs_in_row_dp()` 定义于 [data_processing/line_process.py:403](data_processing/line_process.py#L403)。它与行级 DP 类似，但 match cost 更简单：

- 符号名相同：`0.0`
- 符号名不同：`mismatch_cost`

输出存入：

```python
s.row_sign_matches = {
    text_row_idx: [(text_sign_idx, det_sign_idx), ...],
    ...
}
```

注意：`row_sign_matches` 中可以包含 diff-label 匹配，因为 DP 允许用 `mismatch_cost=0.9` 建立不同符号名之间的对应。

### 4.3 text 行粗对齐到 detection 坐标

`StepAlignTextRows.run()` 调用 `align_text_to_detection_rows()`，将 text rows 投影到 detection 图像坐标，见 [sign_alignment/pipeline.py:558](sign_alignment/pipeline.py#L558)。

输入：

- `det_rows=s.sub_tablet_detection.get_rows_dict()`
- `text_rows=s.sub_tablet_text.get_rows_dict()`
- `text_to_det=s.text_to_det`
- `row_sign_matches=s.row_sign_matches`
- `avg_width=s.avg_width`
- `avg_height=s.avg_height`

核心函数是 `align_text_row_to_detection()`，见 [data_processing/line_process.py:494](data_processing/line_process.py#L494)。

当前代码的实际策略：

1. 只把 same-label 的行内匹配作为 anchors，即 `text_sign_boxes[text_idx].sign_name == detection_sign_boxes[det_idx].sign_name`。
2. 若 anchors 数量大于等于 2，用 anchor detection boxes 的 `(cx, cy)` 拟合一条线性 baseline。
3. 若只有 1 个 anchor，baseline 为水平线，`intercept` 等于该 anchor 的 `cy`。
4. 若没有 anchor，则用该 detection 行全部符号拟合 baseline；单个 detection 符号时 baseline 为水平线。
5. anchor text sign 直接使用对应 detection box 的 `cx/cy/width/height`。
6. 非 anchor text sign 在左右 anchors 之间线性插值；只有单侧 anchor 时按 `avg_width` 外推，`cy` 来自 baseline；无 anchors 时围绕 detection 行中心按 `avg_width` 展开。
7. 非 anchor 的 `width/height` 使用 `avg_width/avg_height`。

重要注意：`min_width_ratio` 和 `max_width_ratio` 被传入函数，但当前实现没有实际 clamp 宽度。

输出 `aligned_text_boxes` 被封装成：

```python
s.sub_tablet_aligned = SubTablet(
    sign_boxes=aligned_text_boxes,
    img=s.sub_tablet_detection.img,
    name="aligned",
    avg_width=s.avg_width,
    avg_height=s.avg_height,
)
```

见 [sign_alignment/pipeline.py:570](sign_alignment/pipeline.py#L570)。

## 5. sign match info 构造

`StepBuildSignMatchInfo.run()` 调用 `build_sign_match_info()`，见 [sign_alignment/pipeline.py:598](sign_alignment/pipeline.py#L598)。

输入：

| 参数 | 来源 | 含义 |
| --- | --- | --- |
| `row_sign_matches` | `s.row_sign_matches` | text 行内符号索引到 detection 行内符号索引 |
| `text_to_det` | `s.text_to_det` | text 行号到 detection 行号 |
| `det_rows_dict` | `s.sub_tablet_detection.get_rows_dict()` | detection 行号到 detection `SignBox` 列表 |
| `optim_sign_boxes` | `s.sub_tablet_aligned.sign_boxes` | 已投影到 detection 坐标的 text `SignBox` |

`build_sign_match_info()` 定义于 [sign_alignment/visualizer.py:840](sign_alignment/visualizer.py#L840)。

它会先把行内匹配转换成全局 key：

```python
match_pairs[(text_row_idx, text_sign_idx)] = (det_row_idx, det_sign_idx)
```

然后构造两个视角的字典：

### 5.1 text_sign_match_info

key 是 `(text_row_idx, text_col_idx)`，value 形如：

```python
{"status": "same" | "diff" | "unmatched", "det_sign_name": str | None}
```

构造逻辑：

- text key 在 `match_pairs` 中，且 text/detection 符号名相同：`same`
- text key 在 `match_pairs` 中，但符号名不同：`diff`
- text key 不在 `match_pairs` 中：`unmatched`

见 [sign_alignment/visualizer.py:865](sign_alignment/visualizer.py#L865)。

### 5.2 det_sign_match_info

key 是 `(det_row_idx, det_col_idx)`，value 形如：

```python
{"status": "same" | "diff" | "unmatched", "text_sign_name": str | None}
```

构造逻辑与 text 视角对称，见 [sign_alignment/visualizer.py:885](sign_alignment/visualizer.py#L885)。

## 6. text_rows_mapped.jpg 绘制逻辑

触发位置：[sign_alignment/pipeline.py:623](sign_alignment/pipeline.py#L623)。

```python
text_row_vis.draw_text_mapping(
    img=None,
    sign_boxes=s.sub_tablet_text.sign_boxes,
    row_mapping=s.text_to_det,
    sign_match_info=s.text_sign_match_info,
    mapped_label_prefix="D",
)
```

实际绘制函数：[sign_alignment/visualizer.py:399](sign_alignment/visualizer.py#L399)。

### 6.1 使用的信息

| 信息 | 来源 | 用途 |
| --- | --- | --- |
| 原始 text `SignBox` | `s.sub_tablet_text.sign_boxes` | 绘制 text 网格框、text 符号名 |
| 行映射 | `s.text_to_det` | 标注 `R#→D#`，判断行是否 matched |
| text 视角 match info | `s.text_sign_match_info` | 决定每个 text 符号框颜色与 diff 补充标签 |

### 6.2 画布

调用时 `img=None`，所以函数会根据 text boxes 的范围创建白色画布：

- 计算所有 text boxes 的 `min_x/min_y/max_x/max_y`
- 增加 `margin=100`
- 若坐标为负，通过 `offset_x/offset_y` 平移到可见区域

见 [sign_alignment/visualizer.py:431](sign_alignment/visualizer.py#L431)。

### 6.3 矩形框颜色

每个 text box 使用 `_get_sign_color(row_idx, status, row_mapping)`，见 [sign_alignment/visualizer.py:736](sign_alignment/visualizer.py#L736)。

| 条件 | 颜色 |
| --- | --- |
| text row 不在 `text_to_det` | 灰色 |
| `status == "same"` | text row 主色 |
| `status == "diff"` | text row 主色降低饱和度 |
| `status == "unmatched"` | 灰色 |

行主色来自 `_get_row_color(row_idx)`，使用 HSV golden angle 生成，见 [sign_alignment/visualizer.py:762](sign_alignment/visualizer.py#L762)。

### 6.4 行连线与中心点

按 `row_idx` 分组并按 `cx` 排序：

- matched row：使用该 text row 主色画实线连接中心点。
- unmatched row：使用灰色。
- 每个符号中心画实心圆，并加白色外圈。

见 [sign_alignment/visualizer.py:472](sign_alignment/visualizer.py#L472)。

### 6.5 标签

每个 text box 内部画主标签：

- 黑底白字，内容为 text 符号名，最多前 10 个字符。
- 若 `status == "diff"` 且存在 `det_sign_name`，在主标签下面再画一个深灰底白字的 detection 符号名。

见 [sign_alignment/visualizer.py:491](sign_alignment/visualizer.py#L491)。

左侧行标注：

- matched text row：`R{row+1}→D{det_row+1}`
- unmatched text row：`R{row+1}`

见 [sign_alignment/visualizer.py:533](sign_alignment/visualizer.py#L533)。

### 6.6 输出

若 `vis.save=True`，保存为：

```text
{context.output_dir}/{context.task_type}_{context.fragment_id}_text_rows_mapped.jpg
```

保存位置由 `_out()` 生成，见 [sign_alignment/pipeline.py:221](sign_alignment/pipeline.py#L221) 和 [sign_alignment/pipeline.py:650](sign_alignment/pipeline.py#L650)。

## 7. rows_side_by_side.jpg 绘制逻辑

触发位置：[sign_alignment/pipeline.py:653](sign_alignment/pipeline.py#L653)。

该图不是重新绘制 sign match，而是组合两张已经存在的图：

1. detection rows 图：由 `step_visualize_detection_rows` 生成并缓存到 `s.det_row_vis_image`。
2. text mapping 图：上节的 `text_row_vis.result`。

### 7.1 detection rows 图来源

`StepVisualizeDetectionRows.visualize()` 调用：

```python
det_row_vis.draw_rows(
    s.sub_tablet_detection.img.copy(),
    s.sub_tablet_detection.sign_boxes,
    show_labels=True,
    show_row_numbers=True,
    row_mapping=s.det_to_text,
    row_label_prefix="D",
    mapped_label_prefix="R",
)
```

见 [sign_alignment/pipeline.py:487](sign_alignment/pipeline.py#L487)。

这张 detection rows 图使用 `draw_rows()` 的通用行绘制逻辑：

- 背景为 detection crop 图像。
- detection boxes 按行着色。
- 行内中心点用实线连接。
- 左侧标注 `D#` 或 `D#→R#`。
- matched detection row 的颜色使用映射到的 text row index，保证与 text mapping 侧颜色一致。

`draw_rows()` 定义于 [sign_alignment/visualizer.py:148](sign_alignment/visualizer.py#L148)。

### 7.2 composite 组合

组合使用 `CompositeVisualizer.compose()`，见 [sign_alignment/visualizer.py:913](sign_alignment/visualizer.py#L913)。

当前参数：

```python
images=[s.det_row_vis_image, text_row_vis.result]
layout=(1, 2)
titles=[
    f"Detection Rows ({len(s.det_row_sequences)} rows)",
    f"Text Mapping ({len(s.text_row_sequences)} rows, {len(s.matches)} matched)",
]
```

见 [sign_alignment/pipeline.py:655](sign_alignment/pipeline.py#L655)。

`CompositeVisualizer` 的行为：

- 以 1 行 2 列排布。
- 为每个 cell 加白色 title bar，默认高度 40。
- 按行最大高度、列最大宽度统一 cell 尺寸。
- 如需缩放，保持原图长宽比并居中放在浅灰背景上。
- cell 之间默认 `padding=4`。

### 7.3 输出

只有当 `s.det_row_vis_image is not None` 时才保存该图。因此必须先执行 `step_visualize_detection_rows`，否则 `step_build_sign_match_info` 不会生成 `rows_side_by_side.jpg`。

保存路径：

```text
{context.output_dir}/{context.task_type}_{context.fragment_id}_rows_side_by_side.jpg
```

见 [sign_alignment/pipeline.py:653](sign_alignment/pipeline.py#L653) 到 [sign_alignment/pipeline.py:664](sign_alignment/pipeline.py#L664)。

## 8. alignment_diagnostic.jpg 绘制逻辑

触发位置：[sign_alignment/pipeline.py:634](sign_alignment/pipeline.py#L634)。

```python
diag_vis.draw_alignment_diagnostic(
    img=s.sub_tablet_detection.img.copy(),
    detection_sign_boxes=s.sub_tablet_detection.sign_boxes,
    aligned_text_boxes=s.sub_tablet_aligned.sign_boxes,
    det_sign_match_info=s.det_sign_match_info,
    text_sign_match_info=s.text_sign_match_info,
    det_to_text=s.det_to_text,
)
```

实际绘制函数：[sign_alignment/visualizer.py:553](sign_alignment/visualizer.py#L553)。

### 8.1 使用的信息

| 信息 | 来源 | 用途 |
| --- | --- | --- |
| detection crop 图像 | `s.sub_tablet_detection.img.copy()` | 背景 |
| detection boxes | `s.sub_tablet_detection.sign_boxes` | 实线 detection 框、detection 标签、实线行中心线 |
| aligned text boxes | `s.sub_tablet_aligned.sign_boxes` | 粗对齐 text overlay、虚线 text 行中心线 |
| detection 视角 match info | `s.det_sign_match_info` | detection 框颜色、detection 标签中的 text 补充名 |
| text 视角 match info | `s.text_sign_match_info` | aligned text box 是否画虚线框、text 标签 |
| `det_to_text` | `s.det_to_text` | detection 行标注 `D#→R#`，以及 detection 行颜色与 text 行颜色同步 |

### 8.1.1 用户问题中的视觉语义速查

以 `alignment_diagnostic.jpg` 为主，图中不同形态的语义如下：

| 图形 | 代码来源 | 语义 |
| --- | --- | --- |
| 正常饱和度 bbox | detection 实线 bbox，`status == "same"` | 该 detection 符号与某个 text 符号成功匹配，且二者符号名相同。颜色是对应 text row 的行主色。 |
| 减少饱和度的实线 bbox | detection 实线 bbox，`status == "diff"` | 该 detection 符号与某个 text 符号被 DP 对齐，但二者符号名不同。它是“有对应关系，但 label 不同”，不是 unmatched。 |
| 灰色虚线 bbox | aligned text 虚线 bbox，`status == "unmatched"` | 这是投影到 detection 坐标中的 text 符号，但行内 DP 没有给它匹配到 detection 符号。 |
| 减少饱和度的虚线 bbox | aligned text 虚线 bbox，`status == "diff"` | 这是投影后的 text 符号位置，并且它被匹配到某个 detection 符号，但二者符号名不同；用虚线表示 text overlay，用去饱和表示 diff-label。 |
| 正常饱和度连线 | detection 行中心实线 | 连接同一 detection row 内按 x 排序的 detection 符号中心点，用来显示 detector 认为的行结构。颜色通过 `det_to_text` 映射为对应 text row 的主色。 |
| 减少饱和度虚线连线 | aligned text 行中心虚线 | 连接投影后的 text row 符号中心点，用来检查 text 行粗对齐后的空间走向。用户列表第 6 条也写成“虚线 bboxes”，若指 bbox，则与上一条“减少饱和度的虚线 bbox”语义相同；若指图中横向虚线，则是这里的 text overlay center line。 |
| 其它语义信息 | 标签、中心点、行号标注 | 见下文：标签显示 detection/text 符号名及 diff 对应名；中心点区分 detection 与 text overlay；左侧 `D#→R#` 显示 detection row 到 text row 的映射。 |

因此，`alignment_diagnostic.jpg` 可以按两条规则快速读图：线型区分来源，实线是 detection、虚线是 aligned text；颜色区分匹配质量，饱和行色偏向 same/行主线，去饱和偏向 diff 或 text overlay，灰色表示 unmatched。

### 8.2 detection boxes

先画 detection boxes，颜色由 `det_sign_match_info[(det_row_idx, det_col_idx)]["status"]` 决定：

| 状态 | 颜色 |
| --- | --- |
| `same` | 对应 text row 主色 |
| `diff` | 对应 text row 主色降低饱和度 |
| `unmatched` | 灰色 |

`det_to_text` 用于把 detection row index 映射成 text row index，保证 diagnostic 中 detection 行颜色与 text mapping 颜色一致。

见 [sign_alignment/visualizer.py:589](sign_alignment/visualizer.py#L589) 和 [sign_alignment/visualizer.py:601](sign_alignment/visualizer.py#L601)。

### 8.3 aligned text boxes overlay

aligned text boxes 只对 `diff` 和 `unmatched` 画虚线矩形：

- `diff`：去饱和 text row 主色虚线框。
- `unmatched`：灰色虚线框。
- `same`：不额外画 text 虚线框，因为 same anchor 已经与 detection box 重合或接近，避免视觉重复。

见 [sign_alignment/visualizer.py:610](sign_alignment/visualizer.py#L610)。

### 8.4 行中心线与中心点

detection 行：

- 按 detection row 分组，按 `cx` 排序。
- 用实线连接 detection centers。
- 画实心圆中心点，外加白色边框。
- 颜色通过 `det_to_text` 映射为 text row 主色。

见 [sign_alignment/visualizer.py:623](sign_alignment/visualizer.py#L623)。

aligned text 行：

- 按 text row 分组，按 `cx` 排序。
- 用虚线连接 aligned text centers。
- 颜色为 text row 主色降低饱和度。
- 中心点半径比 detection 小 1。

见 [sign_alignment/visualizer.py:638](sign_alignment/visualizer.py#L638)。

### 8.5 标签

detection box 标签逻辑：

| 状态 | 标签 |
| --- | --- |
| `same` | 黑底白字，显示 detection 符号名 |
| `diff` | 第一行深灰底显示 detection 符号名；第二行用行主色显示对应 text 符号名 |
| `unmatched` | 灰底白字，显示 detection 符号名 |

见 [sign_alignment/visualizer.py:659](sign_alignment/visualizer.py#L659)。

aligned text box 标签逻辑：

- `diff`：在 aligned text box 上用去饱和行色显示 text 符号名。
- `unmatched`：灰底显示 text 符号名。
- `same`：不额外画 text 标签。

见 [sign_alignment/visualizer.py:689](sign_alignment/visualizer.py#L689)。

左侧 detection 行标注：

- matched detection row：`D{det_row+1}→R{text_row+1}`
- unmatched detection row：`D{det_row+1}`

见 [sign_alignment/visualizer.py:713](sign_alignment/visualizer.py#L713)。

### 8.6 输出

若 `vis.save=True`，保存为：

```text
{context.output_dir}/{context.task_type}_{context.fragment_id}_alignment_diagnostic.jpg
```

见 [sign_alignment/pipeline.py:666](sign_alignment/pipeline.py#L666)。

## 9. 三张图的对照关系

| 图 | 坐标系 | 背景 | 主要目的 | 是否使用 `sub_tablet_aligned` |
| --- | --- | --- | --- | --- |
| `text_rows_mapped.jpg` | text 虚拟网格坐标 | 白底 | 看 text 每个符号是否匹配到 detection，以及 diff 的 detection 名 | 否 |
| `rows_side_by_side.jpg` | 左侧 detection 坐标，右侧 text 虚拟网格坐标 | 左侧图像，右侧白底 | 并排检查 detection 行与 text 行映射 | 否，右侧仍是原始 text 网格 |
| `alignment_diagnostic.jpg` | detection 图像坐标 | detection crop 图像 | 看 detection 框、粗对齐 text 框、same/diff/unmatched 的空间关系 | 是 |

## 10. 读图建议

1. 先看 `rows_side_by_side.jpg`：确认 `D#→R#` 与 `R#→D#` 行映射是否大体合理。
2. 再看 `text_rows_mapped.jpg`：检查 text 侧哪些符号为 same、diff、unmatched。这里更适合观察序列级匹配。
3. 最后看 `alignment_diagnostic.jpg`：检查 same anchors 是否落在 detection 框上，diff/unmatched text 虚线框是否沿 detection 行合理插值或外推。

如果 `alignment_diagnostic.jpg` 中大量 text 虚线框漂移，常见原因是：

- 行级 DP 映射错了，先检查 `rows_side_by_side.jpg`。
- 行内 same-label anchors 太少，粗对齐只能依赖外推或 detection 行整体 baseline。
- detection 行内有噪声框，使行内 DP 或 baseline 受干扰。

## 11. 当前实现中的注意点

- `row_sign_matches` 可能包含 diff-label 匹配，`build_sign_match_info()` 会把这些标成 `diff`，而不是 unmatched。
- `align_text_row_to_detection()` 只用 same-label matches 作为几何 anchors；diff-label matches 只影响 match-info 可视化，不作为 anchor。
- `text_rows_mapped.jpg` 用的是原始 text 网格坐标，不是 aligned 坐标。
- `alignment_diagnostic.jpg` 中 same aligned text boxes 不画额外虚线框，主要看 detection 框本身。
- `rows_side_by_side.jpg` 依赖先前执行 `step_visualize_detection_rows`，否则不会保存。
- 当前代码传入了 `min_width_ratio` 和 `max_width_ratio`，但粗对齐实现没有实际使用这两个参数裁剪宽度。
