import unittest

import numpy as np

from data_processing.hough_row_detection import (
    _fit_angle_curve,
    detect_hough_rows,
)


def _row(angle_deg, y_at_center, x_values, x_center=600.0):
    slope = np.tan(np.deg2rad(angle_deg))
    y_values = slope * (x_values - x_center) + y_at_center
    return np.column_stack((x_values, y_values))


class MultiAngleHoughRowDetectionTest(unittest.TestCase):
    def test_quadratic_angle_curve_robustly_rejects_two_outliers(self):
        rhos = np.linspace(100.0, 2700.0, 28)
        normalized = (rhos - rhos.mean()) / (np.ptp(rhos) / 2)
        expected = -7.0 + 2.0 * normalized ** 2
        angles = expected + 0.3 * np.sin(np.arange(len(rhos)))
        angles[[5, 20]] = [6.0, -14.0]

        curve = _fit_angle_curve(rhos, angles, curvature_penalty=1.0)

        self.assertEqual(np.flatnonzero(~curve.inlier_mask).tolist(), [5, 20])
        self.assertAlmostEqual(curve.coefficients[1], 0.0, places=12)
        self.assertAlmostEqual(
            curve.evaluate(np.array([rhos[0]]))[0],
            curve.evaluate(np.array([rhos[-1]]))[0],
            places=12,
        )
        np.testing.assert_allclose(
            curve.evaluate(rhos)[curve.inlier_mask],
            expected[curve.inlier_mask],
            atol=0.35,
        )

    def test_detects_independent_row_angles(self):
        x_values = np.linspace(50.0, 1150.0, 10)
        centers = np.vstack((
            _row(-10.0, 200.0, x_values),
            _row(-3.0, 400.0, x_values),
            _row(8.0, 600.0, x_values),
        ))

        result = detect_hough_rows(centers, scale=80.0)

        self.assertEqual(len(result.rows), 3)
        self.assertEqual([len(row) for row in result.rows], [10, 10, 10])
        np.testing.assert_allclose(
            result.row_angles_deg,
            np.array([-10.0, -3.0, 8.0]),
            atol=0.25,
        )
        self.assertEqual(result.noise, [])

    def test_keeps_non_parallel_rows_separate_with_missing_centers(self):
        left_and_right = np.array([
            50.0, 170.0, 290.0, 410.0, 790.0, 910.0, 1030.0, 1150.0,
        ])
        centers = np.vstack((
            _row(-9.0, 250.0, left_and_right),
            _row(7.0, 430.0, left_and_right),
        ))

        result = detect_hough_rows(centers, scale=80.0)

        self.assertEqual(len(result.rows), 2)
        self.assertEqual([set(row) for row in result.rows], [
            set(range(8)),
            set(range(8, 16)),
        ])
        np.testing.assert_allclose(
            result.row_angles_deg,
            np.array([-9.0, 7.0]),
            atol=0.25,
        )

    def test_gapped_row_is_not_split_by_nearby_outlier(self):
        centers = np.array([
            [116.72, 2340.16],
            [179.20, 2321.15],
            [568.75, 2262.77],
            [665.83, 2254.18],
            [772.68, 2231.11],
            [816.10, 2222.31],
            [934.80, 2205.94],
            [453.74, 2329.64],  # Nearby point that does not belong to the row.
        ])

        result = detect_hough_rows(centers, scale=90.0)

        self.assertEqual(result.rows, [list(range(7))])
        self.assertEqual(result.noise, [7])
        self.assertAlmostEqual(result.row_angles_deg[0], -9.05, delta=0.5)

    def test_second_pass_removes_peak_outside_curve_window(self):
        x_values = np.linspace(100.0, 1100.0, 8)
        rows = []
        for y_at_center in np.linspace(100.0, 1900.0, 13):
            normalized = (y_at_center - 1000.0) / 900.0
            angle = -7.0 + 1.2 * normalized + 1.5 * normalized ** 2
            rows.append(_row(angle, y_at_center, x_values))
        rows.append(_row(12.0, 625.0, np.array([70.0, 200.0])))

        result = detect_hough_rows(np.vstack(rows), scale=80.0)

        self.assertEqual(len(result.initial_row_angles_deg), 14)
        self.assertEqual(len(result.row_angles_deg), 13)
        self.assertEqual(
            np.flatnonzero(~result.angle_curve_inlier_mask).tolist(),
            [4],
        )
        self.assertEqual(result.noise, [104, 105])
        self.assertTrue(np.all(np.abs(result.row_angles_deg) < 10.0))
        expected_angles = np.interp(
            result.row_rhos,
            result.rho_grid,
            result.angle_curve_angles_deg,
        )
        self.assertTrue(np.all(
            np.abs(result.row_angles_deg - expected_angles) <= 5.0
        ))

    def test_continuous_refit_never_exceeds_angle_range(self):
        x_values = np.linspace(450.0, 750.0, 8)
        centers = _row(20.0, 300.0, x_values)

        result = detect_hough_rows(
            centers,
            scale=100.0,
            angle_range_deg=15.0,
            angle_step_deg=4.0,
        )

        self.assertGreaterEqual(len(result.rows), 1)
        self.assertTrue(np.all(result.row_angles_deg >= -15.0))
        self.assertTrue(np.all(result.row_angles_deg <= 15.0))
        self.assertTrue(np.all(result.angles_deg >= -15.0))
        self.assertTrue(np.all(result.angles_deg <= 15.0))
        self.assertEqual(result.angles_deg[0], -15.0)
        self.assertEqual(result.angles_deg[-1], 15.0)

    def test_empty_input_is_supported(self):
        result = detect_hough_rows(
            np.empty((0, 2), dtype=np.float64),
            scale=80.0,
        )

        self.assertEqual(result.rows, [])
        self.assertEqual(result.noise, [])
        self.assertEqual(result.parameter_space.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
