import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from sign_alignment.box import Box, Boxes
from sign_alignment.pipeline import (
    BoxRows,
    CandidateAttractionConfig,
    INITIAL_OUTPUT_CATEGORY,
    ROW_DETECTION_OUTPUT_CATEGORY,
    Runner,
    SampleState,
    Step,
    VisOptions,
    _candidate_ref_for_optimize_sign,
    align_text_rows,
    match_rows,
    match_signs_in_rows,
    output_path,
    run_candidate_attraction,
)
from sign_alignment.sign import SignResolver
from sign_alignment.tablet import Tablet


class ReentrantPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tablet = Tablet(
            img=np.zeros((240, 640, 3), dtype=np.uint8),
            name="synthetic",
        )

    def _boxes(self, names, *, y=120.0):
        return Boxes(
            (
                Box.from_center(
                    cx=80.0 + 100.0 * index,
                    cy=y,
                    width=40.0,
                    height=40.0,
                    sign=SignResolver.from_name(name),
                    tablet=self.tablet,
                )
                for index, name in enumerate(names)
            ),
            tablet=self.tablet,
        )

    def _single_row_state(self, candidate_names, text_names=None):
        candidate_boxes = self._boxes(candidate_names)
        optimize_boxes = candidate_boxes.copy()
        text_boxes = self._boxes(text_names or candidate_names)
        state = SampleState(
            fragment_id="fragment",
            tablet=self.tablet,
            full_detections=candidate_boxes,
            det_boxes=candidate_boxes,
            text_boxes=text_boxes,
            text_rows=BoxRows(text_boxes, [list(range(len(text_boxes)))]),
            candidate_boxes=candidate_boxes,
            candidate_rows=BoxRows(
                candidate_boxes,
                [list(range(len(candidate_boxes)))],
            ),
            optimize_boxes=optimize_boxes,
            optimize_rows=BoxRows(
                optimize_boxes,
                [list(range(len(optimize_boxes)))],
            ),
            optimize_to_candidate={
                index: index for index in range(len(optimize_boxes))
            },
            optimize_row_to_candidate={0: 0},
        )
        return state

    def test_box_rows_replacement_preserves_topology(self):
        first = self._boxes(["A", "BI", "AN"])
        second = first.copy()
        rows = BoxRows(first, [[1, 0], [2]], noise=[2])

        replaced = rows.with_boxes(second)

        self.assertIs(replaced.boxes, second)
        self.assertEqual(replaced.rows, [[1, 0], [2]])
        self.assertEqual(replaced.noise, [2])
        self.assertIsNot(replaced.rows, rows.rows)
        with self.assertRaises(ValueError):
            rows.with_boxes(self._boxes(["A"]))

    def test_iteration_number_is_shared_by_run_and_visualize(self):
        state = SampleState(fragment_id="F")
        output_directory = tempfile.TemporaryDirectory()
        self.addCleanup(output_directory.cleanup)
        context = SimpleNamespace(
            state=state,
            output_dir=output_directory.name,
            task_type="alignment",
        )
        runner = Runner.__new__(Runner)
        runner.context = context
        runner.vis = VisOptions(info=False, display=False, save=False)
        observed = []

        def run_step(ctx):
            observed.append(("run", ctx.state.optimization_iteration))

        def visualize_step(ctx, _):
            observed.append(("visualize", ctx.state.optimization_iteration))

        steps = [Step("iteration", run_step, visualize_step)]
        runner.run(steps)
        runner.run(steps, advance_iteration=True)
        runner.run(steps, advance_iteration=True)

        self.assertEqual(
            observed,
            [
                ("run", 0),
                ("visualize", 0),
                ("run", 1),
                ("visualize", 1),
                ("run", 2),
                ("visualize", 2),
            ],
        )
        state.optimization_iteration = 0
        self.assertEqual(
            output_path(context, "plot.jpg"),
            os.path.join(
                output_directory.name,
                "F",
                "iter_000",
                "alignment_F_plot.jpg",
            ),
        )
        self.assertEqual(
            output_path(
                context,
                "plot.jpg",
                category=INITIAL_OUTPUT_CATEGORY,
            ),
            os.path.join(
                output_directory.name,
                "F",
                "initial",
                "alignment_F_plot.jpg",
            ),
        )
        self.assertEqual(
            output_path(
                context,
                "plot.jpg",
                category=ROW_DETECTION_OUTPUT_CATEGORY,
            ),
            os.path.join(
                output_directory.name,
                "F",
                "row_detection",
                "alignment_F_plot.jpg",
            ),
        )
        state.optimization_iteration = 2
        self.assertEqual(
            output_path(context, "plot.jpg"),
            os.path.join(
                output_directory.name,
                "F",
                "iter_002",
                "alignment_F_plot.jpg",
            ),
        )

    def test_candidate_provenance_uses_nonidentity_row_mapping(self):
        candidate_boxes = self._boxes(["A", "BI"])
        optimize_boxes = candidate_boxes.copy()
        state = SampleState(
            candidate_boxes=candidate_boxes,
            candidate_rows=BoxRows(candidate_boxes, [[0, 1]]),
            optimize_boxes=optimize_boxes,
            optimize_rows=BoxRows(optimize_boxes, [[], [], [], [0, 1]]),
            optimize_to_candidate={0: 0, 1: 1},
            optimize_row_to_candidate={3: 0},
        )

        self.assertEqual(
            _candidate_ref_for_optimize_sign(state, 3, 1),
            (0, 1, 1),
        )

    def test_text_labels_do_not_create_unbacked_anchors(self):
        candidate_boxes = self._boxes(["A", "AN", "KA"])
        optimize_boxes = self._boxes(["A", "BI", "ŠE"])
        text_boxes = self._boxes(["A", "BI", "ŠE"])
        state = SampleState(
            candidate_boxes=candidate_boxes,
            candidate_rows=BoxRows(candidate_boxes, [[0, 1, 2]]),
            optimize_boxes=optimize_boxes,
            optimize_rows=BoxRows(optimize_boxes, [[0, 1, 2]]),
            optimize_to_candidate={0: 0, 1: 1},  # third box came from NULL
            optimize_row_to_candidate={0: 0},
            text_boxes=text_boxes,
            text_rows=BoxRows(text_boxes, [[0, 1, 2]]),
            matches=[(0, 0)],
            optimize_row_sequences=[["A", "BI", "ŠE"]],
            text_row_sequences=[["A", "BI", "ŠE"]],
        )
        context = SimpleNamespace(state=state)

        match_signs_in_rows(context)

        self.assertEqual(state.row_sign_matches[0], [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(state.row_anchor_matches[0], [(0, 0)])

    def test_duplicate_cluster_preserves_the_exact_label_source(self):
        candidate_boxes = self._boxes(["A", "BI"])
        candidate_boxes[1].cx = candidate_boxes[0].cx
        candidate_boxes[1].cy = candidate_boxes[0].cy
        optimize_boxes = candidate_boxes.copy()
        text_boxes = self._boxes(["BI"])
        state = SampleState(
            tablet=self.tablet,
            full_detections=candidate_boxes,
            candidate_boxes=candidate_boxes,
            candidate_rows=BoxRows(candidate_boxes, [[0, 1]]),
            optimize_boxes=optimize_boxes,
            optimize_rows=BoxRows(optimize_boxes, [[0, 1]]),
            optimize_to_candidate={0: 0, 1: 1},
            optimize_row_to_candidate={0: 0},
            text_boxes=text_boxes,
            text_rows=BoxRows(text_boxes, [[0]]),
            matches=[(0, 0)],
            text_to_optimize={0: 0},
            optimize_to_text={0: 0},
            row_sign_matches={0: [(0, 1)]},
            row_anchor_matches={0: [(0, 1)]},
            aligned_boxes=text_boxes.copy(),
        )
        state.aligned_rows = BoxRows(state.aligned_boxes, [[0]])
        context = SimpleNamespace(state=state)

        run_candidate_attraction(
            context,
            CandidateAttractionConfig(
                temperatures=(1.0,),
                steps_per_temperature=0,
                device="cpu",
            ),
        )

        # Geometry comes from the highest-score physical representative, while
        # provenance retains the raw BI detection that made this a true anchor.
        self.assertEqual(state.optimize_to_candidate, {0: 1})
        match_rows(context)
        match_signs_in_rows(context)
        self.assertEqual(state.row_anchor_matches[0], [(0, 0)])

    def test_two_iterations_keep_fixed_candidates_unchanged(self):
        state = self._single_row_state(["A", "BI", "AN"])
        context = SimpleNamespace(state=state)
        candidate_snapshot = [
            (
                box.x1,
                box.y1,
                box.x2,
                box.y2,
                tuple((item.sign.name, item.score) for item in box.candidates),
            )
            for box in state.candidate_boxes
        ]
        config = CandidateAttractionConfig(
            temperatures=(1.0,),
            steps_per_temperature=0,
            device="cpu",
        )

        for iteration in (1, 2):
            state.optimization_iteration = iteration
            match_rows(context)
            match_signs_in_rows(context)
            align_text_rows(context)
            run = run_candidate_attraction(context, config)

            self.assertEqual(state.matches, [(0, 0)])
            self.assertEqual(
                state.row_anchor_matches[0],
                [(0, 0), (1, 1), (2, 2)],
            )
            self.assertEqual(run.iteration, iteration)
            self.assertIs(state.optimize_boxes, run.boxes)
            self.assertIs(state.optimize_rows.boxes, state.optimize_boxes)
            self.assertIs(state.candidate_rows.boxes, state.candidate_boxes)
            self.assertEqual(state.optimize_to_candidate, {0: 0, 1: 1, 2: 2})

        self.assertEqual(
            [
                (
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                    tuple((item.sign.name, item.score) for item in box.candidates),
                )
                for box in state.candidate_boxes
            ],
            candidate_snapshot,
        )


if __name__ == "__main__":
    unittest.main()
