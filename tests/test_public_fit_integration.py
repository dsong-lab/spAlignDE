import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import spalignde
from spalignde import (
    cluster_trajectories,
    fit_local_de,
    gene_level_age_trend_acat,
    gene_level_acat_pvalue,
    prepare_inference,
)
from spalignde.inference._calibration import (
    MISMATCH_CALIBRATION_MODE,
    MULTI_CONTRAST_CALIBRATION_MODE,
)


class PublicFitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = (
            Path(spalignde.__file__).resolve().parent
            / "datasets"
            / "toy_post_alignment.csv.gz"
        )
        data = pd.read_csv(data_path)
        cls.prepared = prepare_inference(
            data,
            reference="age_3.0",
            genes=["toy_up"],
            library_size=250,
            density_energy_share=0.25,
            n_jobs=2,
            random_state=1,
        )
        common = {
            "genes": ["toy_up"],
            "technical_adjustment": True,
            "cell_type_adjustment": True,
            "random_state": 1,
            "verbose": False,
        }
        cls.naive = fit_local_de(
            cls.prepared,
            mismatch_aware=False,
            **common,
        )
        cls.aware = fit_local_de(
            cls.prepared,
            mismatch_aware=True,
            **common,
        )

    def test_compact_public_fit_retains_local_pvalues_for_acat(self):
        terrain = self.aware.fits["toy_up"]["terrain_data"]
        self.assertIsInstance(terrain.get("p_by_time"), dict)
        self.assertEqual(set(terrain["p_by_time"]), set(terrain["time_ids"]))
        for values in terrain["p_by_time"].values():
            self.assertEqual(len(values), len(self.prepared.shared["grid_eval"]))
            self.assertTrue(np.isfinite(np.asarray(values, dtype=float)).any())

        pvalue = gene_level_acat_pvalue(self.aware, "toy_up")
        self.assertTrue(np.isfinite(pvalue))
        self.assertGreaterEqual(pvalue, 0.0)
        self.assertLessEqual(pvalue, 1.0)

    def test_auto_geometry_worker_count_is_recorded_as_serial(self):
        self.assertEqual(self.prepared.metadata["n_jobs"], 2)
        self.assertEqual(self.prepared.metadata["auto_geometry_n_jobs"], 1)

    def test_public_path_uses_local_only_promoted_calibration(self):
        terrain = self.aware.fits["toy_up"]["terrain_data"]
        calibration = terrain["risk_calibration"]
        self.assertEqual(calibration["method"], MULTI_CONTRAST_CALIBRATION_MODE)
        self.assertEqual(calibration["n_contrasts_total"], 3)
        self.assertEqual(calibration["n_contrasts_valid"], 3)
        for record in calibration["provisional_by_time"].values():
            self.assertTrue(record["valid"])
            self.assertEqual(
                record["calibration"]["diag"]["mode"],
                MISMATCH_CALIBRATION_MODE,
            )
        self.assertEqual(float(calibration["lambda_global_hat"]), 0.0)
        self.assertGreaterEqual(float(calibration["lambda_local_hat"]), 0.0)

        self.assertEqual(
            self.aware.metadata["mismatch_calibration"],
            MULTI_CONTRAST_CALIBRATION_MODE,
        )
        self.assertEqual(
            self.aware.metadata["mismatch_calibration_by_gene"]["toy_up"],
            MULTI_CONTRAST_CALIBRATION_MODE,
        )
        self.assertEqual(
            self.aware.metadata["within_contrast_mismatch_calibration"],
            MISMATCH_CALIBRATION_MODE,
        )
        self.assertNotIn("mismatch_lambda_global_by_gene", self.aware.metadata)
        self.assertIsInstance(terrain["Wv_by_time"], dict)
        self.assertIsInstance(terrain["use_by_time"], dict)

    def test_naive_and_aware_comparison_keeps_composition_setting_fixed(self):
        self.assertTrue(self.naive.metadata["cell_type_adjustment"])
        self.assertTrue(self.aware.metadata["cell_type_adjustment"])
        self.assertEqual(
            self.aware.metadata["cell_type_adjustment_method"],
            "normalized_jensen_shannon_distance",
        )
        terrain = self.aware.fits["toy_up"]["terrain_data"]
        self.assertEqual(
            terrain["celltype_adjustment_info"]["method"],
            "normalized_jensen_shannon_distance",
        )
        for time_id in terrain["time_ids"]:
            phi = np.asarray(terrain["celltype_variance_factor_by_time"][time_id])
            self.assertTrue(np.all(np.isfinite(phi)))
            self.assertTrue(np.all((1.0 <= phi) & (phi <= np.e)))
        self.assertEqual(self.naive.metadata["mismatch_calibration"], "none")
        self.assertEqual(
            self.naive.metadata["mismatch_calibration_by_gene"]["toy_up"],
            "none",
        )

    def test_public_trajectory_api_accepts_explicit_time_values(self):
        time_ids = list(
            self.aware.fits["toy_up"]["terrain_data"]["muA_adj_by_time"]
        )
        trajectory = cluster_trajectories(
            self.aware,
            "toy_up",
            n_clusters=2,
            time_values=np.arange(len(time_ids), dtype=float) * 2.0,
            random_state=1,
        )
        self.assertEqual(trajectory.result["K_TRAJ"], 2)
        self.assertEqual(trajectory.metadata["selected_n_clusters"], 2)
        self.assertIsNone(trajectory.metadata["selection"])

    def test_public_age_trend_api_uses_compact_fit_arrays(self):
        output = gene_level_age_trend_acat(self.aware, "toy_up")
        pvalue = output["summary"]["gene_level_trend_acat_p"]
        self.assertTrue(np.isfinite(pvalue))
        self.assertGreaterEqual(pvalue, 0.0)
        self.assertLessEqual(pvalue, 1.0)
        self.assertFalse(
            output["summary"]["reference_included_as_age_observation"]
        )


if __name__ == "__main__":
    unittest.main()
