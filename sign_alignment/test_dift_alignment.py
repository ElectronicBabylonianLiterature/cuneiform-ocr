import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sign_alignment.dift_align import (
    DiftAlignmentConfig,
    DiftMatchConfig,
    DiftRuntime,
    _affine_rectangle_angle_score,
    _dense_deformation_scores,
)
from sign_alignment.data_source import DataSource
from sign_alignment.pipeline import _optimize_psr
from sign_alignment.pipeline_2 import (
    SlidingWindow,
    _ordered_score_assignment,
)
from sign_alignment.sign import SignResolver


def _identity_features(size: int = 4) -> torch.Tensor:
    count = size * size
    return torch.eye(count).reshape(count, size, size)


def _test_runtime(wrapper=None) -> DiftRuntime:
    class Runtime(DiftRuntime):
        def __post_init__(self):
            self.sd_featurizer = object()
            self.dift_wrapper = wrapper if wrapper is not None else object()

    return Runtime(checkpoint="dummy")


class DiftFeatureMatchingTest(unittest.TestCase):
    def test_identity_features_recover_image_scale(self):
        features = _identity_features()
        result = _test_runtime().match(
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
        self.assertAlmostEqual(result.affine_iou, 1.0)
        self.assertAlmostEqual(result.affine_angle_score, 1.0)
        self.assertAlmostEqual(result.support_score, 1.0)
        self.assertAlmostEqual(result.inlier_score, 1.0)
        self.assertAlmostEqual(result.coarse_score, 1.0)
        self.assertAlmostEqual(result.global_similarity_score, 1.0)
        self.assertAlmostEqual(result.sim_withoutbg, 1.0)
        self.assertAlmostEqual(result.certainty_score, 1.0)
        self.assertAlmostEqual(result.bending_energy_score, 1.0)
        self.assertAlmostEqual(result.jacobian_fold_score, 1.0)
        self.assertAlmostEqual(result.local_distortion_score, 1.0)
        self.assertAlmostEqual(result.scale_score, 1.0)

    def test_dense_scores_penalize_scrambled_correspondence_geometry(self):
        size = 4
        count = size * size
        permutation = torch.arange(count)
        permutation[[5, 10]] = permutation[[10, 5]]
        sim = torch.zeros((count, count), dtype=torch.float32)
        sim[torch.arange(count), permutation] = 1.0

        certainty, bending, non_fold, distortion, scale = _dense_deformation_scores(
            sim,
            (size, size),
            (size, size),
            None,
        )

        self.assertAlmostEqual(certainty, 1.0)
        self.assertLess(bending, 0.5)
        self.assertLess(non_fold, 0.8)
        self.assertLess(distortion, 0.8)
        self.assertLess(scale, 0.8)

    def test_affine_angle_score_penalizes_shear(self):
        rectangle = _affine_rectangle_angle_score(
            (40, 60),
            np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        )
        sheared = _affine_rectangle_angle_score(
            (40, 60),
            np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        )

        self.assertAlmostEqual(rectangle, 1.0)
        self.assertAlmostEqual(sheared, 0.5)

    def test_dense_scale_score_penalizes_large_scale_change(self):
        src_size = 3
        dst_size = 9
        sim = torch.zeros(
            (src_size * src_size, dst_size * dst_size),
            dtype=torch.float32,
        )
        for src_y in range(src_size):
            for src_x in range(src_size):
                src_idx = src_y * src_size + src_x
                dst_idx = (src_y + 3) * dst_size + src_x + 3
                sim[src_idx, dst_idx] = 1.0

        *_, scale = _dense_deformation_scores(
            sim,
            (src_size, src_size),
            (dst_size, dst_size),
            None,
        )

        self.assertAlmostEqual(scale, 1.0 / 3.0, places=4)

    def test_dense_scores_ignore_source_background_similarity(self):
        size = 3
        count = size * size
        foreground = np.array([
            [True, True, False],
            [True, True, False],
            [False, False, False],
        ])
        foreground_indices = torch.tensor([0, 1, 3, 4])

        base_sim = torch.zeros((count, count), dtype=torch.float32)
        base_sim[foreground_indices, foreground_indices] = 1.0
        noisy_background_sim = base_sim.clone()
        noisy_background_sim[~torch.from_numpy(foreground.reshape(-1))] = (
            torch.linspace(-20.0, 20.0, 5 * count).reshape(5, count)
        )

        base_scores = _dense_deformation_scores(
            base_sim,
            (size, size),
            (size, size),
            foreground,
        )
        noisy_scores = _dense_deformation_scores(
            noisy_background_sim,
            (size, size),
            (size, size),
            foreground,
        )

        np.testing.assert_allclose(base_scores, noisy_scores, atol=1e-6)

    def test_coarse_score_uses_affine_crop_iou_not_only_inliers(self):
        src = _identity_features(size=2)
        dst = torch.zeros((4, 4, 4), dtype=torch.float32)
        dst[:, 0, 0] = src[:, 0, 0]
        dst[:, 0, 1] = src[:, 0, 1]
        dst[:, 1, 0] = src[:, 1, 0]
        dst[:, 1, 1] = src[:, 1, 1]

        result = _test_runtime().match(
            src,
            dst,
            (40, 60),
            (80, 120),
            DiftMatchConfig(min_support=4),
        )

        self.assertEqual(result.n_matches, 4)
        self.assertEqual(result.n_inliers, 4)
        self.assertAlmostEqual(result.inlier_score, 1.0)
        self.assertAlmostEqual(result.affine_iou, 0.25)
        self.assertAlmostEqual(result.affine_angle_score, 1.0)
        expected_geometry = np.sqrt(0.25 / 0.7)
        self.assertAlmostEqual(result.geometry_score, expected_geometry)
        self.assertAlmostEqual(result.coarse_score, expected_geometry)

    def test_foreground_mask_excludes_prototype_background_similarity(self):
        src = torch.zeros((2, 2, 2), dtype=torch.float32)
        src[:, 0, 0] = torch.tensor([1.0, 0.0])
        src[:, 0, 1] = torch.tensor([0.0, 1.0])
        src[:, 1, 0] = torch.tensor([0.0, 1.0])
        src[:, 1, 1] = torch.tensor([0.0, 1.0])

        dst = torch.zeros((2, 2, 2), dtype=torch.float32)
        dst[0, :, :] = 1.0
        foreground = np.array([
            [True, False],
            [False, False],
        ])

        result = _test_runtime().match(
            src,
            dst,
            (20, 20),
            (20, 20),
            DiftMatchConfig(),
            src_foreground_mask=foreground,
        )

        self.assertAlmostEqual(result.global_similarity_score, 0.625)
        self.assertAlmostEqual(result.sim_withoutbg, 1.0)

    def test_too_few_matches_returns_scored_failure(self):
        features = _identity_features()
        result = _test_runtime().match(
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
    def test_reuses_feature_cache_without_refeaturizing(self):
        class Source(DataSource):
            form = "canonical1"

            def __init__(self):
                self.calls = 0

            def get(self, sign_name: str, period: str):
                self.calls += 1
                return np.zeros((16, 16), dtype=np.uint8)

        class Wrapper:
            def __init__(self):
                self.calls = 0

            def featurize(self, image):
                self.calls += 1
                return torch.ones((4, 2, 2), dtype=torch.float32)

        wrapper = Wrapper()
        runtime = _test_runtime(wrapper)
        source = Source()
        runtime.source = source
        sign = SignResolver.from_name("AN")

        first = runtime.get_sign_feature(sign, "Old Babylonian")
        second = runtime.get_sign_feature(sign, "Old Babylonian")

        self.assertIs(first, second)
        self.assertEqual(source.calls, 1)
        self.assertEqual(wrapper.calls, 1)


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
