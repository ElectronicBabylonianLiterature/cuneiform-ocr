import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sign_alignment.box import Box, Boxes, SignCandidate
from sign_alignment.classifier import COMMON_SIGN_NAMES, RESNET18_INDEX_TO_SIGN
from sign_alignment.pipeline import (
    SampleState,
    choose_classification,
    evaluate_detection,
    improve_classification,
)
from sign_alignment.sign import SignResolver
from sign_alignment.tablet import Tablet
import evaluate_alignment


class ClassificationImprovementTest(unittest.TestCase):
    def setUp(self):
        self.tablet = Tablet(
            img=np.zeros((100, 100, 3), dtype=np.uint8),
            name="test",
        )

    def _box(self, sign_name, score=1.0):
        return Box.from_center(
            cx=50,
            cy=50,
            width=20,
            height=20,
            sign=SignResolver.from_name(sign_name),
            tablet=self.tablet,
            score=score,
        )

    def test_explicit_mapping_merges_period_classes(self):
        self.assertEqual(len(RESNET18_INDEX_TO_SIGN), 482)
        self.assertGreater(RESNET18_INDEX_TO_SIGN.count("AN"), 1)
        self.assertEqual(len(COMMON_SIGN_NAMES), 123)

    def test_confidence_fusion_rule(self):
        def choose(detr_score, classifier_score):
            detr = SignCandidate(SignResolver.from_name("A"), detr_score)
            return choose_classification(detr, ("AN", classifier_score))

        self.assertEqual(choose(0.8, 0.9)[2], "detr")
        self.assertEqual(choose(0.8, 0.4)[2], "detr")
        self.assertEqual(choose(0.4, 0.8)[2], "classifier")
        self.assertEqual(choose(0.4, 0.3)[2], "detr")
        self.assertEqual(choose(0.3, 0.4)[2], "classifier")

    def test_only_shared_detr_classes_are_changed(self):
        det_boxes = Boxes(
            [self._box("A", 0.3), self._box("AL", 0.2)],
            tablet=self.tablet,
        )
        classifier = SimpleNamespace(
            classify_boxes=lambda _: [("AN", 0.8), ("AN", 0.9)]
        )
        context = SimpleNamespace(
            state=SampleState(det_boxes=det_boxes),
            sign_classifier=classifier,
        )

        improve_classification(context)

        self.assertEqual(det_boxes[0].sign_name, "AN")
        self.assertEqual(det_boxes[1].sign_name, "AL")

    def test_evaluation_selects_the_named_boxes_collection(self):
        gt_boxes = Boxes([self._box("A")], tablet=self.tablet)
        det_boxes = Boxes([self._box("A", 0.9)], tablet=self.tablet)
        optimize_boxes = Boxes([self._box("AN", 0.9)], tablet=self.tablet)
        state = SampleState(
            fragment_id="test",
            tablet=self.tablet,
            gt_boxes=gt_boxes,
            det_boxes=det_boxes,
            optimize_boxes=optimize_boxes,
        )
        context = SimpleNamespace(
            state=state,
            gt_visualization_excluded_prefixes=("SURFACE_",),
        )

        evaluate_detection(context, "det_boxes", "detection")
        self.assertAlmostEqual(
            state.extras["current_detection_evaluation"]["metrics"]["mAP"],
            1.0,
        )

        evaluate_detection(context, "optimize_boxes", "optimized")
        self.assertAlmostEqual(
            state.extras["current_detection_evaluation"]["metrics"]["mAP"],
            0.0,
        )

    def test_batch_alignment_improves_before_creating_candidates(self):
        captured_steps = []
        runner = SimpleNamespace(
            run=lambda steps: captured_steps.extend(step.name for step in steps)
        )

        evaluate_alignment._run_without_psr_alignment_steps(runner)

        self.assertLess(
            captured_steps.index("Improve classification"),
            captured_steps.index("Create box sets"),
        )

    def test_improved_detection_mode_runs_the_same_fusion(self):
        det_boxes = Boxes([self._box("A", 0.3)], tablet=self.tablet)
        full_tablet = self.tablet
        state = SampleState(
            fragment_id="test",
            tablet=self.tablet,
            full_tablet=full_tablet,
            det_boxes=det_boxes,
        )
        context = SimpleNamespace(
            state=state,
            tablet_detector=SimpleNamespace(crop_tablets=[self.tablet]),
        )
        runner = SimpleNamespace(choose_crop=lambda _: None)

        with patch.object(evaluate_alignment, "improve_classification") as improve:
            evaluate_alignment._predict_detection_crops(
                runner,
                context,
                "test",
                improve=True,
            )

        improve.assert_called_once_with(
            context,
            evaluate_alignment.CLASSIFICATION_CONFIDENCE_THRESHOLD,
        )


if __name__ == "__main__":
    unittest.main()
