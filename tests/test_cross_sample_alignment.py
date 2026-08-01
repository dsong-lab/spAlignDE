from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

import spAlignDE

from ._data import make_cross_sample_adata


class CrossSampleAlignmentTests(unittest.TestCase):
    def test_prealignment_plot_uses_the_recorded_method_label(self):
        adata = make_cross_sample_adata()
        manual = spAlignDE.prealign_cross_sample_manual(
            adata,
            query_sample="query",
            reference_sample="reference",
            config=spAlignDE.ManualPrealignmentConfig(),
            verbose=False,
        )
        automatic = spAlignDE.prealign_cross_sample(
            adata,
            query_sample="query",
            reference_sample="reference",
            verbose=False,
        )

        target = (
            "spAlignDE.alignment.cross_sample.pre_core."
            "plot_cluster_correspondence_prealign_before_after"
        )
        with mock.patch(target) as plot:
            spAlignDE.plot_prealignment_result(manual)
            self.assertEqual(
                plot.call_args.kwargs["method_label"],
                "manual similarity",
            )
        with mock.patch(target) as plot:
            spAlignDE.plot_prealignment_result(automatic)
            self.assertEqual(
                plot.call_args.kwargs["method_label"],
                "cluster-correspondence",
            )

    def test_manual_prealignment_matches_equivalent_automatic_transform(self):
        adata = make_cross_sample_adata()
        automatic = spAlignDE.prealign_cross_sample(
            adata,
            query_sample="query",
            reference_sample="reference",
            verbose=False,
        )
        params = automatic.params
        manual_config = spAlignDE.ManualPrealignmentConfig(
            scale=params["scale"],
            theta_deg=params["theta_deg"],
            translation_x=params["translation_x"],
            translation_y=params["translation_y"],
        )
        manual = spAlignDE.prealign_cross_sample_manual(
            adata,
            query_sample="query",
            reference_sample="reference",
            config=manual_config,
            verbose=False,
        )

        np.testing.assert_allclose(
            manual.adata.obs[["x_prealigned", "y_prealigned"]].to_numpy(),
            automatic.adata.obs[["x_prealigned", "y_prealigned"]].to_numpy(),
        )
        self.assertEqual(
            manual.params["prealign_method"],
            "manual_similarity",
        )
        self.assertEqual(manual.params["n_shared_clusters"], 3)
        np.testing.assert_allclose(
            manual.params["matrix"][:2],
            np.column_stack(
                [
                    params["scale"] * np.asarray(params["R"]),
                    np.asarray(params["t"]),
                ]
            ),
        )

    def test_manual_prealignment_validation_and_reference_stability(self):
        adata = make_cross_sample_adata()
        original = np.asarray(adata.obsm["spatial"]).copy()
        config = spAlignDE.ManualPrealignmentConfig(
            scale=1.2,
            theta_deg=-12.0,
            translation_x=0.4,
            translation_y=-0.7,
        )
        result = spAlignDE.prealign_cross_sample_manual(
            adata,
            query_sample="query",
            reference_sample="reference",
            config=config,
            verbose=False,
        )
        query_mask = (
            result.adata.obs["sample_id"].astype(str) == "query"
        ).to_numpy()
        reference_mask = ~query_mask
        expected = spAlignDE.apply_similarity_transform(
            original[query_mask],
            config,
        )
        np.testing.assert_allclose(
            result.adata.obs.loc[
                query_mask, ["x_prealigned", "y_prealigned"]
            ].to_numpy(),
            expected,
        )
        np.testing.assert_allclose(
            result.adata.obs.loc[
                reference_mask, ["x_prealigned", "y_prealigned"]
            ].to_numpy(),
            original[reference_mask],
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            spAlignDE.apply_similarity_transform(
                original,
                spAlignDE.ManualPrealignmentConfig(scale=0),
            )
        with self.assertRaisesRegex(ValueError, "either"):
            spAlignDE.align_cross_sample(
                adata,
                query_sample="query",
                reference_sample="reference",
                prealignment_config=spAlignDE.PrealignmentConfig(),
                manual_prealignment_config=config,
                verbose=False,
            )

    def test_stepwise_alignment_and_h5ad_roundtrip(self):
        adata = make_cross_sample_adata()
        original_spatial = np.asarray(adata.obsm["spatial"]).copy()

        prealignment = spAlignDE.prealign_cross_sample(
            adata,
            query_sample="query",
            reference_sample="reference",
            verbose=False,
        )
        fields = spAlignDE.rasterize_cross_sample(
            prealignment.adata,
            query_sample="query",
            reference_sample="reference",
            config=spAlignDE.RasterizationConfig(grid_spacing=0.25),
        )
        result = spAlignDE.run_slddmm_alignment(
            prealignment.adata,
            fields,
            config=spAlignDE.SLDDMMConfig(
                iterations=2,
                kernel_scale=1.0,
                velocity_grid_spacing=0.5,
                momentum_lr=1.0,
                minimum_momentum_lr=1.0,
                sigma_regularization=100.0,
            ),
            device="cpu",
            verbose=False,
            prealignment=prealignment,
        )

        self.assertTrue(np.array_equal(adata.obsm["spatial"], original_spatial))
        self.assertFalse(any(column in adata.obs for column in spAlignDE.REQUIRED_OUTPUT_COLUMNS))
        self.assertEqual(fields.shared_clusters, ["0", "1", "2"])
        self.assertEqual(fields.query_image.shape[0], 4)
        self.assertTrue(
            all(column in result.adata.obs for column in spAlignDE.REQUIRED_OUTPUT_COLUMNS)
        )

        reference_mask = result.adata.obs["sample_id"].astype(str) == "reference"
        reference_output = result.adata.obs.loc[
            reference_mask, ["x_aligned", "y_aligned"]
        ].to_numpy()
        np.testing.assert_allclose(reference_output, original_spatial[reference_mask])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aligned.h5ad"
            result.adata.write_h5ad(path)
            restored = ad.read_h5ad(path)
            np.testing.assert_allclose(
                restored.obs[list(spAlignDE.REQUIRED_OUTPUT_COLUMNS)].to_numpy(),
                result.adata.obs[list(spAlignDE.REQUIRED_OUTPUT_COLUMNS)].to_numpy(),
            )

    def test_high_level_wrapper(self):
        adata = make_cross_sample_adata(n_per_cluster=10)
        result = spAlignDE.align_cross_sample(
            adata,
            query_sample="query",
            reference_sample="reference",
            rasterization_config=spAlignDE.RasterizationConfig(
                grid_spacing=0.3
            ),
            slddmm_config=spAlignDE.SLDDMMConfig(
                iterations=1,
                kernel_scale=1.0,
                velocity_grid_spacing=0.5,
                momentum_lr=1.0,
                minimum_momentum_lr=1.0,
                sigma_regularization=100.0,
            ),
            device="cpu",
            verbose=False,
            return_result=True,
        )
        self.assertIsInstance(result, spAlignDE.CrossSampleAlignmentResult)
        self.assertEqual(result.query_sample, "query")
        self.assertEqual(result.reference_sample, "reference")

        figure, axes = spAlignDE.plot_alignment_result(result)
        self.assertEqual(len(axes), 2)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
