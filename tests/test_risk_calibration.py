import unittest

import numpy as np
from scipy import stats

from spalignde.inference._calibration import (
    MISMATCH_CALIBRATION_MODE,
    calibrate_mismatch_variance,
)
from spalignde.inference._legacy_core import calibrate_lambdas_empnull_scale


def _binned_statistics(scales, centers=None, n_per_bin=240, df=30.0):
    risks = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
    scales = np.asarray(scales, dtype=float)
    if centers is None:
        centers = np.zeros_like(risks)
    centers = np.asarray(centers, dtype=float)
    q75 = float(stats.t.ppf(0.75, df=df))
    signs = np.tile(np.array([-1.0, 1.0]), n_per_bin // 2)
    statistics = np.concatenate([
        center + signs * q75 * scale
        for center, scale in zip(centers, scales)
    ])
    r01 = np.repeat(risks, n_per_bin)
    return statistics, np.ones(statistics.size, dtype=bool), r01


class RiskCalibrationTests(unittest.TestCase):
    def test_exact_quadratic_local_excess_is_recovered(self):
        risks = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
        scales = np.sqrt(1.0 + 4.0 * risks**2)
        tvec, use, r01 = _binned_statistics(scales, centers=np.full(5, 12.0))
        result = calibrate_mismatch_variance(
            tvec, use, r01, global_score=0.9, df=30,
            bins=5, min_bin_n=100, p_grid=2.0,
        )

        self.assertEqual(result["diag"]["mode"], MISMATCH_CALIBRATION_MODE)
        self.assertEqual(result["diag"]["A"], 0.0)
        self.assertEqual(result["lambda_global_hat"], 0.0)
        self.assertAlmostEqual(result["lambda_local_hat"], 4.0, places=10)
        self.assertAlmostEqual(result["p_best"], 2.0)
        bins = result["diag"]["df_bin"]
        self.assertIn("scale_excess_raw", bins)
        self.assertIn("scale_excess_isotonic", bins)
        self.assertTrue(np.all(np.diff(bins["tail_excess_iso"]) >= -1e-12))
        zero = bins.loc[bins["bin"].eq("zero")].iloc[0]
        self.assertTrue(bool(zero["zero_risk_target_forced_zero"]))
        self.assertEqual(float(zero["tail_excess_raw"]), 0.0)

    def test_within_bin_location_shifts_do_not_change_calibration(self):
        risks = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
        scales = np.sqrt(1.0 + 2.5 * risks**2)
        base = _binned_statistics(scales, centers=np.zeros(5))
        shifted = _binned_statistics(
            scales,
            centers=np.asarray([50.0, -20.0, 125.0, -80.0, 7.0]),
        )
        fit_base = calibrate_mismatch_variance(
            *base, global_score=0.2, df=30,
            bins=5, min_bin_n=100, p_grid=2.0,
        )
        fit_shifted = calibrate_mismatch_variance(
            *shifted, global_score=0.2, df=30,
            bins=5, min_bin_n=100, p_grid=2.0,
        )
        self.assertAlmostEqual(
            fit_base["lambda_local_hat"],
            fit_shifted["lambda_local_hat"],
            places=10,
        )

    def test_global_score_is_provenance_only(self):
        risks = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
        scales = np.sqrt(1.0 + 3.0 * risks**2)
        data = _binned_statistics(scales)
        low = calibrate_mismatch_variance(
            *data, global_score=0.05, df=30,
            bins=5, min_bin_n=100, p_grid=2.0,
        )
        high = calibrate_mismatch_variance(
            *data, global_score=0.95, df=30,
            bins=5, min_bin_n=100, p_grid=2.0,
        )
        self.assertEqual(low["lambda_global_hat"], 0.0)
        self.assertEqual(high["lambda_global_hat"], 0.0)
        self.assertAlmostEqual(
            low["lambda_local_hat"], high["lambda_local_hat"], places=12
        )
        self.assertNotEqual(low["global_score"], high["global_score"])

    def test_student_t_null_mad_has_no_inflation(self):
        data = _binned_statistics(np.ones(5), centers=np.arange(5) * 100.0)
        result = calibrate_mismatch_variance(
            *data, global_score=0.5, df=30,
            bins=5, min_bin_n=100, p_grid=2.0,
        )
        self.assertAlmostEqual(result["lambda_local_hat"], 0.0, places=12)
        self.assertEqual(result["lambda_global_hat"], 0.0)

    def test_zero_risk_target_is_zero_even_with_large_spread(self):
        scales = np.asarray([10.0, 1.2, 1.4, 1.6, 1.8])
        result = calibrate_mismatch_variance(
            *_binned_statistics(scales), global_score=0.4, df=30,
            bins=5, min_bin_n=100, p_grid=2.0,
        )
        zero = result["diag"]["df_bin"].loc[
            lambda frame: frame["bin"].eq("zero")
        ].iloc[0]
        self.assertGreater(float(zero["robust_scale_t"]), 5.0)
        self.assertEqual(float(zero["tail_excess_raw"]), 0.0)

    def test_local_cap_and_isotonic_constraint_are_enforced(self):
        result = calibrate_mismatch_variance(
            *_binned_statistics([1.0, 3.0, 1.2, 4.0, 2.0]),
            global_score=0.4,
            df=30,
            bins=5,
            min_bin_n=100,
            p_grid=2.0,
            lam_local_cap=0.75,
        )
        self.assertLessEqual(result["lambda_local_hat"], 0.75)
        iso = result["diag"]["df_bin"]["tail_excess_iso"].to_numpy(float)
        self.assertTrue(np.all(np.diff(iso) >= -1e-12))

    def test_too_few_points_returns_zero_calibration(self):
        result = calibrate_mismatch_variance(
            np.zeros(100), np.ones(100, bool), np.linspace(0, 1, 100),
            global_score=0.5, df=30, min_bin_n=50,
        )
        self.assertEqual(result["lambda_local_hat"], 0.0)
        self.assertEqual(result["lambda_global_hat"], 0.0)
        self.assertEqual(
            result["diag"]["mode"], "mad_null1_A0_calibration_too_few_ok"
        )

    def test_too_few_retained_bins_returns_zero_calibration(self):
        n = 1200
        result = calibrate_mismatch_variance(
            np.linspace(-2.0, 2.0, n),
            np.ones(n, dtype=bool),
            np.repeat([0.0, 1.0], n // 2),
            global_score=0.5,
            df=30,
            bins=10,
            min_bin_n=100,
        )
        self.assertEqual(result["lambda_local_hat"], 0.0)
        self.assertEqual(result["lambda_global_hat"], 0.0)
        self.assertEqual(
            result["diag"]["mode"], "mad_null1_A0_calibration_too_few_bins"
        )

    def test_masked_and_nonfinite_points_are_ignored(self):
        data = list(_binned_statistics(np.sqrt(1.0 + np.arange(5) ** 2)))
        data[0] = data[0].copy()
        data[1] = data[1].copy()
        data[0][0] = np.nan
        data[1][1] = False
        result = calibrate_mismatch_variance(
            *data, global_score=0.3, df=30,
            bins=5, min_bin_n=100, p_grid=2.0,
        )
        self.assertTrue(np.isfinite(result["lambda_local_hat"]))
        self.assertEqual(result["lambda_global_hat"], 0.0)

    def test_input_lengths_must_match(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            calibrate_mismatch_variance(
                np.zeros(10), np.ones(9, bool), np.zeros(10),
                global_score=0.0, df=30,
            )

    def test_legacy_wrapper_uses_the_single_new_implementation(self):
        risks = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
        data = _binned_statistics(np.sqrt(1.0 + 2.0 * risks**2))
        result = calibrate_lambdas_empnull_scale(
            *data, global_score=0.7, df=30,
            bins=5, min_bin_n=100, p_grid=2.0, verbose=False,
        )
        self.assertEqual(result["diag"]["mode"], MISMATCH_CALIBRATION_MODE)
        self.assertEqual(result["lambda_global_hat"], 0.0)


if __name__ == "__main__":
    unittest.main()
