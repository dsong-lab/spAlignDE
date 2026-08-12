from __future__ import annotations

import unittest

import anndata as ad
import numpy as np
import pandas as pd

import spAlignDE


def make_sample(points: np.ndarray, labels: list[str]) -> ad.AnnData:
    return ad.AnnData(
        X=np.ones((len(points), 3), dtype=np.float32),
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(labels)},
            index=[f"obs_{index}" for index in range(len(points))],
        ),
        obsm={"spatial": np.asarray(points, dtype=float)},
    )


class ATACSTContractTests(unittest.TestCase):
    def test_public_api_is_exported(self):
        from spAlignDE.alignment import align_atac_to_st

        self.assertIs(align_atac_to_st, spAlignDE.align_atac_to_st)
        self.assertTrue(callable(spAlignDE.prealign_atac_to_st))
        self.assertTrue(callable(spAlignDE.plot_atac_st_alignment))

    def test_pairing_weights_are_user_configurable_and_normalized(self):
        from spAlignDE.alignment.atac import _atac_pairing_weights

        config = spAlignDE.ATACSTAlignmentConfig()
        self.assertFalse(config.restore_best_checkpoint)
        default_weights = _atac_pairing_weights(config)
        self.assertEqual(
            default_weights,
            {
                "sdf_corr": 0.35,
                "chamfer_sim": 0.25,
                "area_sim": 0.30,
                "dice": 0.10,
            },
        )
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            _atac_pairing_weights(
                spAlignDE.ATACSTAlignmentConfig(sdf_weight=0.40)
            )

    def test_prealignment_preserves_original_coordinates_and_standardizes_output(self):
        atac_points = np.array(
            [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float
        )
        st_points = np.array(
            [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]],
            dtype=float,
        )
        atac = make_sample(atac_points, ["a", "a", "b", "b"])
        st = make_sample(st_points, ["l", "l", "r", "r"] * 2)
        result = spAlignDE.prealign_atac_to_st(
            atac,
            st,
            config=spAlignDE.ATACSTPrealignmentConfig(
                st_transform=spAlignDE.ManualPrealignmentConfig(),
                atac_transform=spAlignDE.ManualPrealignmentConfig(),
                reference_crop_axis="x",
                reference_crop_side="left",
                reference_crop_quantile=0.5,
                raster_scale=1.0,
                canvas_padding=2,
            ),
        )

        np.testing.assert_array_equal(atac.obsm["spatial"], atac_points)
        np.testing.assert_array_equal(st.obsm["spatial"], st_points)
        self.assertEqual(result.st_reference.n_obs, 4)
        self.assertEqual(result.canvas_shape_hw, (6, 6))
        for column in spAlignDE.REQUIRED_OUTPUT_COLUMNS:
            self.assertIn(column, result.atac.obs)
            self.assertIn(column, result.st_reference.obs)
        self.assertEqual(
            result.atac.uns["spAlignDE"]["atac_to_st"]["prealignment"]
            ["reference_observations_after_crop"],
            4,
        )


if __name__ == "__main__":
    unittest.main()
