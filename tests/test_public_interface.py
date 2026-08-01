from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import numpy as np

import spAlignDE

from ._data import make_cross_sample_adata


class PublicInterfaceTests(unittest.TestCase):
    def test_canonical_import_exposes_inference_subpackage(self):
        from spAlignDE.inference import prepare_inference

        self.assertIs(prepare_inference, spAlignDE.prepare_inference)
        self.assertTrue(callable(spAlignDE.build_visium_inference_table))

    def test_canonical_import_exposes_alignment_subpackage(self):
        from spAlignDE.alignment import interactive_manual_prealignment

        self.assertIs(
            interactive_manual_prealignment,
            spAlignDE.interactive_manual_prealignment,
        )

    def test_canonical_import_exposes_uncertainty_subpackage(self):
        from spAlignDE import uncertainty

        mapped = {
            1: np.asarray([[0.0, 0.0], [1.0, 1.0]]),
            2: np.asarray([[0.0, 2.0], [3.0, 1.0]]),
        }
        summary, distances = uncertainty.compute_repeat_point_variance(
            mapped,
            x_src_new=np.asarray([0.0, 1.0]),
            y_src_new=np.asarray([0.0, 1.0]),
        )

        self.assertEqual(summary.shape[0], 2)
        self.assertEqual(distances.shape, (2, 2))
        np.testing.assert_allclose(summary["dist_var"], [0.0, 0.0])

    def test_static_manual_preview_does_not_mutate_anndata(self):
        adata = make_cross_sample_adata(n_per_cluster=3)
        original = np.asarray(adata.obsm["spatial"]).copy()
        fig, axes = spAlignDE.plot_manual_prealignment_preview(
            adata,
            query_sample="query",
            reference_sample="reference",
            config=spAlignDE.ManualPrealignmentConfig(
                scale=1.1,
                theta_deg=5,
                translation_x=0.2,
                translation_y=-0.1,
            ),
            max_points=4,
        )
        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_title(), "Before")
        np.testing.assert_array_equal(adata.obsm["spatial"], original)
        plt.close(fig)

    def test_interactive_controller_applies_selected_values(self):
        adata = make_cross_sample_adata(n_per_cluster=3)
        ui = spAlignDE.interactive_manual_prealignment(
            adata,
            query_sample="query",
            reference_sample="reference",
            max_points=4,
            display_ui=False,
        )
        ui.controls["scale"].value = 1.05
        ui.controls["translation_x"].value = 0.5
        result = ui.apply(verbose=False)

        self.assertAlmostEqual(ui.selected_config.scale, 1.05)
        self.assertAlmostEqual(result.params["translation_x"], 0.5)
        self.assertIn("spAlignDE", result.adata.uns)
        self.assertNotIn("spAlignDE", adata.uns)


if __name__ == "__main__":
    unittest.main()
