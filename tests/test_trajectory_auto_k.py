import unittest

import numpy as np
import pandas as pd

from spalignde.inference._legacy_core import (
    run_global_spatial_roi_trajectory_clustering,
)


def _trajectory_fixture(n_time: int):
    x, y = np.meshgrid(np.arange(10, dtype=float), np.arange(8, dtype=float))
    xy = np.column_stack([x.ravel(), y.ravel()])
    time_ids = [f"sample_{index}" for index in range(n_time)]
    mu_by_time = {}
    for index, time_id in enumerate(time_ids):
        base = np.sin(xy[:, 0] / 2.5) + 0.15 * np.cos(xy[:, 1] / 2.0)
        regional_trend = (xy[:, 1] >= 4).astype(float) * 0.08 * index
        mu_by_time[time_id] = base + 0.03 * index + regional_trend
    fit = {
        "gene": "synthetic_gene",
        "terrain_data": {
            "muA_adj_by_time": mu_by_time,
            "ref_sample_id": time_ids[0],
        },
    }
    shared = {
        "grid_eval": pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}),
        "grid_spacing": 1.0,
        "R_map": 1.5,
        "ref_sample_id": time_ids[0],
    }
    return fit, shared


FAST_AUTO_OPTIONS = {
    "K_GRID": [2, 3],
    "KMEANS_NSTART_AUTO": 1,
    "KMEANS_NSTART_FULL": 1,
    "KMEANS_ITER_AUTO": 12,
    "KMEANS_ITER_FULL": 12,
    "AUTOK_REFINE_MAXITER": 1,
    "FINAL_LABEL_MAXITER": 1,
}


class TrajectoryAutoKTests(unittest.TestCase):
    def _run_auto(self, n_time, *, time_values=None):
        fit, shared = _trajectory_fixture(n_time)
        return run_global_spatial_roi_trajectory_clustering(
            fit,
            shared,
            K_TRAJ=None,
            auto_k=FAST_AUTO_OPTIONS,
            time_values=time_values,
            do_plot=False,
            seed=7,
            use_stat_reliability=False,
        )

    def test_three_time_points_return_minimum_candidate(self):
        out = self._run_auto(3, time_values=[0.0, 2.0, 5.0])
        self.assertEqual(out["K_TRAJ"], 2)
        self.assertEqual(out["auto_k"]["selection"]["mode"], "minimum_K")
        self.assertEqual(out["auto_k"]["selection"]["dynamic_candidates"], [2])

    def test_four_time_points_use_linear_leave_one_out_dynamic_table(self):
        out = self._run_auto(4, time_values=[0.0, 1.5, 4.0, 8.0])
        table = out["auto_k"]["table"]
        self.assertIn(out["K_TRAJ"], FAST_AUTO_OPTIONS["K_GRID"])
        self.assertIn("mean_dynamic_gain", table)
        self.assertIn("se_dynamic_gain", table)
        self.assertIn("grid_fraction_in_small_components", table)
        self.assertEqual(int(table["selected_final_k"].sum()), 1)

    def test_six_time_points_use_formal_two_stage_table(self):
        out = self._run_auto(6, time_values=[0.0, 1.0, 2.5, 4.0, 7.0, 11.0])
        selection = out["auto_k"]["selection"]
        self.assertIn("best_dynamic_gain", selection)
        self.assertIn("best_dynamic_gain_lower_1SE", selection)
        self.assertIn("dynamic_candidates", selection)
        self.assertEqual(
            selection["rule"],
            "held-out complete-trajectory evidence, then fine-to-coarse first local minimum of R_map-footprint fragmentation",
        )
        self.assertIn("fine_to_coarse_order", selection)
        self.assertIn("fine_to_coarse_scan", selection)
        self.assertIn("fine_to_coarse_rank", out["auto_k"]["table"])
        self.assertIn("fine_to_coarse_visited", out["auto_k"]["table"])

    def test_explicit_time_values_control_order_and_are_retained(self):
        out = self._run_auto(4, time_values=[10.0, 2.0, 7.0, 4.0])
        np.testing.assert_allclose(out["time_values"], [2.0, 4.0, 7.0, 10.0])
        self.assertEqual(
            out["time_ids_all"],
            ["sample_1", "sample_3", "sample_2", "sample_0"],
        )

    def test_explicit_cluster_count_bypasses_auto_selection(self):
        fit, shared = _trajectory_fixture(4)
        out = run_global_spatial_roi_trajectory_clustering(
            fit,
            shared,
            K_TRAJ=3,
            time_values=[0.0, 1.0, 2.0, 3.0],
            do_plot=False,
            seed=7,
            use_stat_reliability=False,
            auto_k=FAST_AUTO_OPTIONS,
        )
        self.assertEqual(out["K_TRAJ"], 3)
        self.assertIsNone(out["auto_k"])

    def test_invalid_time_values_are_rejected(self):
        fit, shared = _trajectory_fixture(4)
        with self.assertRaisesRegex(ValueError, "one value for each"):
            run_global_spatial_roi_trajectory_clustering(
                fit,
                shared,
                K_TRAJ=2,
                time_values=[0.0, 1.0],
                do_plot=False,
                use_stat_reliability=False,
            )

    def test_fixed_seed_is_reproducible(self):
        first = self._run_auto(4, time_values=[0.0, 1.0, 2.0, 3.0])
        second = self._run_auto(4, time_values=[0.0, 1.0, 2.0, 3.0])
        self.assertEqual(first["K_TRAJ"], second["K_TRAJ"])
        key = f"K{first['K_TRAJ']}"
        np.testing.assert_array_equal(
            first["results_by_K"][key]["cluster_full"],
            second["results_by_K"][key]["cluster_full"],
        )


if __name__ == "__main__":
    unittest.main()
