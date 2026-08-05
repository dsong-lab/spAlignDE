import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import spalignde
from spalignde import fit_local_de, gene_level_acat_pvalue, prepare_inference
from spalignde.inference._calibration import MISMATCH_CALIBRATION_MODE


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

    def test_public_path_uses_local_only_promoted_calibration(self):
        terrain = self.aware.fits["toy_up"]["terrain_data"]
        calibration = terrain["risk_calibration"]
        detail = calibration["calibration"]
        self.assertIs(detail, calibration["empnull"])
        self.assertEqual(detail["diag"]["mode"], MISMATCH_CALIBRATION_MODE)
        self.assertEqual(float(calibration["lambda_global_hat"]), 0.0)
        self.assertGreaterEqual(float(calibration["lambda_local_hat"]), 0.0)

        self.assertEqual(
            self.aware.metadata["mismatch_calibration"],
            MISMATCH_CALIBRATION_MODE,
        )
        self.assertEqual(
            self.aware.metadata["mismatch_calibration_by_gene"]["toy_up"],
            MISMATCH_CALIBRATION_MODE,
        )
        self.assertEqual(
            self.aware.metadata["mismatch_lambda_global_by_gene"]["toy_up"],
            0.0,
        )

    def test_naive_and_aware_comparison_keeps_support_setting_fixed(self):
        self.assertTrue(self.naive.metadata["cell_type_adjustment"])
        self.assertTrue(self.aware.metadata["cell_type_adjustment"])
        self.assertEqual(self.naive.metadata["mismatch_calibration"], "none")
        self.assertEqual(
            self.naive.metadata["mismatch_calibration_by_gene"]["toy_up"],
            "none",
        )


if __name__ == "__main__":
    unittest.main()
