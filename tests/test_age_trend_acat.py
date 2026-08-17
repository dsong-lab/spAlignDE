import unittest

import numpy as np
import pandas as pd

from spalignde import (
    LocalDEResult,
    PreparedInference,
    gene_level_age_trend_acat,
)


def _age_trend_result():
    time_ids = ["age_3.0", "age_6.0", "age_9.0", "age_12.0", "age_15.0"]
    ages = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
    n_grid = 24
    residual_pattern = np.array([0.03, -0.02, 0.01, -0.03, 0.02])
    mu_by_time = {}
    precision_by_time = {}
    use_by_time = {}
    p_by_time = {}
    for index, time_id in enumerate(time_ids):
        grid_offset = np.linspace(-0.2, 0.2, n_grid)
        mu_by_time[time_id] = (
            grid_offset + 0.08 * ages[index] + residual_pattern[index]
        )
        precision_by_time[time_id] = np.full(n_grid, 4.0)
        use_by_time[time_id] = np.ones(n_grid, dtype=bool)
        p_by_time[time_id] = np.linspace(0.1, 0.9, n_grid)
    prepared = PreparedInference(
        shared={},
        data=pd.DataFrame(),
        genes=("gene_a",),
        reference="age_1.0",
        library_size=None,
        density_energy_share=0.25,
        alignment_uncertainty_key=None,
    )
    return LocalDEResult(
        fits={"gene_a": {"terrain_data": {
            "time_ids": time_ids,
            "ref_sample_id": "age_1.0",
            "muA_adj_by_time": mu_by_time,
            "Wv_by_time": precision_by_time,
            "use_by_time": use_by_time,
            "p_by_time": p_by_time,
        }}},
        prepared=prepared,
        alpha=0.05,
        contrast="vs_reference",
        mismatch_aware=True,
        technical_adjustment=True,
    )


class AgeTrendAcatTests(unittest.TestCase):
    def test_age_trend_uses_unsmoothed_adjusted_expression_and_precision(self):
        output = gene_level_age_trend_acat(_age_trend_result(), "gene_a")
        summary = output["summary"]
        self.assertLess(summary["gene_level_trend_acat_p"], 0.05)
        self.assertTrue(summary["reject_global_null"])
        self.assertEqual(summary["n_local_trend_tests"], 24)
        self.assertEqual(summary["trend_input"], "unsmoothed_muA_adj_by_time")
        self.assertEqual(summary["trend_weight"], "mismatch_aware_Wv_by_time")
        self.assertFalse(summary["reference_included_as_age_observation"])
        self.assertEqual(len(output["per_contrast"]), 5)
        self.assertEqual(len(output["per_grid"]), 24)

    def test_explicit_time_values_are_supported(self):
        output = gene_level_age_trend_acat(
            _age_trend_result(),
            "gene_a",
            time_values=[2.0, 4.0, 8.0, 16.0, 32.0],
        )
        self.assertEqual(output["per_grid"]["age_min"].iloc[0], 2.0)
        self.assertEqual(output["per_grid"]["age_max"].iloc[0], 32.0)

    def test_current_fit_arrays_are_required(self):
        result = _age_trend_result()
        result.fits["gene_a"]["terrain_data"].pop("Wv_by_time")
        with self.assertRaisesRegex(ValueError, "rerun fit_local_de"):
            gene_level_age_trend_acat(result, "gene_a")


if __name__ == "__main__":
    unittest.main()
