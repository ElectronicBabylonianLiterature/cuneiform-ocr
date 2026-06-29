import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sign_alignment.dift_align import (
    DiftAlignmentConfig,
    DiftMatchConfig,
    DiftRuntime,
    match_dift_features,
)
from sign_alignment.pipeline import _optimize_psr
from sign_alignment.pipeline_2 import (
    SlidingWindow,
    _ordered_score_assignment,
)


def _identity_features(size: int = 4) -> torch.Tensor:
    count = size * size
    return torch.eye(count).reshape(count, size, size)


class DiftFeatureMatchingTest(unittest.TestCase):
    def test_identity_features_recover_image_scale(self):
        features = _identity_features()
        result = match_dift_features(
            features,
            features,
            (40, 60),
            (80, 120),
            DiftMatchConfig(min_support=4),
        )

        self.assertEqual(result.n_matches, 16)
        self.assertEqual(result.n_inliers, 16)
        np.testing.assert_allclose(
            result.affine,
            np.array([[2, 0, 0], [0, 2, 0]], dtype=np.float64),
            atol=1e-4,
        )
        self.assertAlmostEqual(result.geometry_score, 1.0)
        self.assertAlmostEqual(result.support_score, 1.0)
        self.assertAlmostEqual(result.global_similarity_score, 1.0)

    def test_too_few_matches_returns_scored_failure(self):
        features = _identity_features()
        result = match_dift_features(
            features,
            features,
            (40, 40),
            (40, 40),
            DiftMatchConfig(max_matches=2, min_matches=3),
        )

        self.assertIsNone(result.affine)
        self.assertEqual(result.n_matches, 2)
        self.assertEqual(result.n_inliers, 0)
        self.assertIn("at least 3", result.message)


class OrderedFeatureAssignmentTest(unittest.TestCase):
    def test_assignment_is_one_to_one_and_left_to_right(self):
        candidates = [
            SlidingWindow(i, 0, i + 1, 1, float(i), 0.5, 0, 2)
            for i in range(3)
        ]
        scores = np.array([
            [0.9, 0.2, 0.1],
            [0.8, 0.7, 0.1],
            [0.1, 0.2, 0.95],
        ])

        self.assertEqual(
            _ordered_score_assignment([0, 1, 2], candidates, scores, 0.0),
            {0: 0, 1: 1, 2: 2},
        )


class DiftRuntimeTest(unittest.TestCase):
    def test_reuses_canonical_cache_without_rebuilding_wrapper(self):
        class Source:
            form = "canonical1"

        class Model:
            def __init__(self):
                self.calls = 0

            def make_wrapper(self):
                self.calls += 1
                return object()

        model = Model()
        source = Source()
        runtime = DiftRuntime(model=model)

        first = runtime.setup(source, "Old Babylonian")
        second = runtime.setup(source, "Old Babylonian")

        self.assertIs(first.cache, second.cache)
        self.assertEqual(model.calls, 1)


class PsrIterationTest(unittest.TestCase):
    def test_second_phase_runs_only_remaining_iterations(self):
        class Optimizer:
            def __init__(self):
                self.loss_history = [0.0] * 10
                self.calls = []

            def optimize(self, num_iterations, **kwargs):
                self.calls.append(num_iterations)
                self.loss_history.extend([0.0] * num_iterations)
                return "optimized"

            def get_optimized_boxes(self):
                return "cached"

        optimizer = Optimizer()
        context = SimpleNamespace(
            psr_params={"num_iterations": 80},
            dift=SimpleNamespace(
                config=DiftAlignmentConfig(affine_probe_iteration=10)
            ),
            state=SimpleNamespace(optimizer=optimizer, final_boxes=None),
        )

        _optimize_psr(context)
        _optimize_psr(context)

        self.assertEqual(optimizer.calls, [70])
        self.assertEqual(context.state.final_boxes, "cached")


if __name__ == "__main__":
    unittest.main()
