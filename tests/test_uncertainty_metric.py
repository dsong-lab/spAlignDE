from __future__ import annotations

import json
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spAlignDE import uncertainty as uq


class MeanDistanceTests(unittest.TestCase):
    def setUp(self):
        # Symmetric displacements have positive mean distances but zero
        # distance variance: substituting the old metric must be detectable.
        self.centers = np.column_stack([np.arange(4) * 20.0, np.zeros(4)])
        offsets = np.column_stack([[1.0, 2.0, 4.0, 8.0], np.zeros(4)])
        self.mapped = {1: self.centers - offsets, 2: self.centers + offsets}
        self.table, self.distances = uq.compute_repeat_point_variance(
            self.mapped, self.centers[:, 0], self.centers[:, 1]
        )

    def tearDown(self):
        plt.close("all")

    def test_mean_distance_is_not_distance_variance(self):
        np.testing.assert_allclose(self.table["dist_mean"], [1, 2, 4, 8])
        np.testing.assert_allclose(self.table["dist_var"], 0)
        np.testing.assert_allclose(self.distances, [[1, 2, 4, 8]] * 2)

    def test_mean_map_uses_mean_values_and_separate_display_cap(self):
        before = self.table.copy(deep=True)
        fig = uq.plot_distance_mean_map(self.table, self.mapped, self.centers)
        scatter = fig.axes[0].collections[1]
        np.testing.assert_allclose(scatter.get_array(), [1, 2, 4, 8])
        self.assertAlmostEqual(scatter.norm.vmax, 7.88)
        self.assertIn("7.4 (95th percentile; n=1)", fig.axes[0].texts[0].get_text())
        self.assertEqual(fig.axes[1].get_ylabel(), "Mean distance across repeats")
        pd.testing.assert_frame_equal(self.table, before)

    def test_legacy_variance_map_still_plots_variance(self):
        fig = uq.plot_distance_variance_map(self.table, self.mapped, self.centers)
        np.testing.assert_allclose(fig.axes[0].collections[1].get_array(), 0)
        self.assertEqual(fig.axes[1].get_ylabel(), "Distance variance across replicates")

    def test_distribution_defaults_to_mean_distance(self):
        fig, threshold = uq.plot_uncertainty_distribution(self.table)
        self.assertAlmostEqual(threshold, 7.4)
        self.assertEqual(fig.axes[0].get_xlabel(), "Mean distance across repeats")
        _, variance_threshold = uq.plot_uncertainty_distribution(
            self.table, value_col="dist_var"
        )
        self.assertEqual(variance_threshold, 0)

    def test_report_selects_high_points_using_its_primary_metric(self):
        with tempfile.TemporaryDirectory() as output:
            kwargs = dict(
                output_dir=output,
                summary=pd.DataFrame({"repeat": [1, 2]}),
                uncertainty_df=self.table,
            )
            report = json.loads(uq.write_brief_report(**kwargs).read_text())
            self.assertEqual(report["primary_metric"], "dist_mean")
            self.assertEqual(report["median_dist_mean"], 3.0)
            self.assertAlmostEqual(report["p95_dist_mean"], 7.4)
            self.assertEqual(report["p95_dist_var"], 0)
            self.assertEqual(report["n_high_variability_points"], 1)

            median_report = json.loads(
                uq.write_brief_report(**kwargs, high_percentile=50).read_text()
            )
            self.assertEqual(median_report["high_variability_threshold"], 3.0)
            self.assertAlmostEqual(median_report["p95_dist_mean"], 7.4)
            self.assertEqual(median_report["n_high_variability_points"], 2)

            legacy = json.loads(
                uq.write_brief_report(**kwargs, value_col="dist_var").read_text()
            )
            self.assertEqual(legacy["primary_metric"], "dist_var")
            self.assertEqual(legacy["n_high_variability_points"], 4)
