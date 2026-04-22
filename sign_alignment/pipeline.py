from dataclasses import dataclass
import os
from typing import Optional

import numpy as np

from sign_alignment.detector import ModelConfig, TabletImageDetector
from sign_alignment.data_source import EBLAPISource, LocalDataSource, GroundTruths, SignAPIResolver, SignTextParser
from sign_alignment.visualizer import BboxVisualizer, ColorConfig, TextVisualizer


@dataclass
class PipelineConfig:
    model_config: ModelConfig
    tablet_detector: TabletImageDetector
    local_source: LocalDataSource
    color_config: ColorConfig
    output_dir: str


class PipelineTools:
    api_source = EBLAPISource()
    sign_resolver = SignAPIResolver()

@dataclass
class SampleState:
    fragment_id: str = None
    img: np.ndarray = None
    gt_boxes: Optional[GroundTruths] = None


# Keep CropContext as a thin wrapper so existing callers still work.
@dataclass
class CropContext:
    config: PipelineConfig
    tools: PipelineTools = PipelineTools()
    state: SampleState = None

    def __post_init__(self):
        if self.state is None:
            self.state = SampleState()

    # Convenience pass-throughs used by Step functions
    @property
    def tablet_detector(self): return self.config.tablet_detector
    @property
    def local_source(self): return self.config.local_source
    @property
    def api_source(self): return self.tools.api_source
    @property
    def color_config(self): return self.config.color_config
    @property
    def output_dir(self): return self.config.output_dir
    @property
    def fragment_id(self): return self.state.fragment_id
    @fragment_id.setter
    def fragment_id(self, v): self.state.fragment_id = v
    @property
    def img(self): return self.state.img
    @img.setter
    def img(self, v): self.state.img = v
    @property
    def gt_boxes(self): return self.state.gt_boxes
    @gt_boxes.setter
    def gt_boxes(self, v): self.state.gt_boxes = v


class Step:
    def __init__(self, name: str, description: str = None, run=None, visualize=None):
        self.name = name
        self.description = description
        self.run = run
        self.visualize = visualize


class Runner:
    def __init__(self, context: CropContext, steps: list[Step]):
        self.context = context
        self.steps = steps

        fragments = context.local_source.get_available_fragments()
        self._fragments = fragments
        print(f"Found {len(fragments)} fragments with both image and annotation")

    def run_single_step(self, step: Step):
        info_message = f"Step: {step.name}"
        if step.description:
            info_message += f" - {step.description}"
        print(info_message)
        if step.run:
            step.run(self.context)
        if step.visualize:
            step.visualize(self.context)

    def choose_sample(self, idx: int):
        fragment_id = self._fragments[idx]
        print(f"Processing sample: {fragment_id}")
        self.context.fragment_id = fragment_id


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

def _load_from_local_source(context: CropContext):
    context.img = context.local_source.load_image(context.fragment_id)
    context.gt_boxes = context.local_source.load_annotation(context.fragment_id)
    print(f"Ground truth boxes: {len(context.gt_boxes)}")


def _visualize_ground_truth(context: CropContext):
    gt_bbox_visualizer = BboxVisualizer(context.color_config.GT_COLOR.value)
    gt_bbox_visualizer.draw_boxes(context.img.copy(), context.gt_boxes)
    gt_bbox_visualizer.save(
        os.path.join(context.output_dir, f"debug_{context.fragment_id}_gt.jpg")
    )

def _load_sign_text_from_api(context: CropContext):
    # get sign text (gt) from text.lines with broken sign filtering
    fragment_data = context.api_source.get_fragment_data(context.fragment_id)
    # Get fragment data and extract text from text.lines (instead of signs field)
    if fragment_data is None:
        raise ValueError(f"No fragment data found for sample {context.fragment_id}")

    text_data = fragment_data.get('text', {})
    text_lines = SignTextParser.parse_text_lines(text_data, filter_broken=True, sign_resolver=context.tools.sign_resolver)
    total_text_signs = sum(len(line) for line in text_lines)
    print(f"  Text lines: {len(text_lines)}, total signs: {total_text_signs}")

    # Also show unfiltered for comparison
    text_lines_unfiltered = SignTextParser.parse_text_lines(text_data, filter_broken=False, sign_resolver=context.tools.sign_resolver)
    total_unfiltered = sum(len(line) for line in text_lines_unfiltered)
    print(f"  Unfiltered: {total_unfiltered} signs, broken signs removed: {total_unfiltered - total_text_signs}")

    # print all text lines
    TextVisualizer.save_text(text_lines, path=os.path.join(context.config.output_dir, f"debug_{context.fragment_id}_text_filtered.txt"), fragment_id=context.fragment_id)
    TextVisualizer.save_text(text_lines_unfiltered, path=os.path.join(context.config.output_dir, f"debug_{context.fragment_id}_text.txt"), fragment_id=context.fragment_id)
 
 
    


step_load_data = Step(
    name="Load Data", 
    run=_load_from_local_source)

step_show_ground_truth = Step(
    name="Show Ground Truth", visualize=_visualize_ground_truth)
step_load_sign_text = Step(
    name="Load Sign Text from API", run=_load_sign_text_from_api)



DEBUG_STEPS = [step_load_data, step_show_ground_truth, step_load_sign_text]