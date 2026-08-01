from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spAlignDE

from ._data import make_cross_sample_adata


HAS_BANKSY = importlib.util.find_spec("banksy") is not None


class SingleClusteringAndAtlasContractTests(unittest.TestCase):
    def make_single_adata(self):
        adata = make_cross_sample_adata(n_per_cluster=8, n_genes=12)
        adata = adata[adata.obs["sample_id"].astype(str) == "reference"].copy()
        del adata.obs["sample_id"]
        return adata

    def test_single_clustering_config_is_public(self):
        config = spAlignDE.SingleClusteringConfig()
        self.assertEqual(config.num_neighbors, 30)
        self.assertTrue(config.refine_boundaries)

    def test_atlas_default_uses_four_levels_and_validated_palette(self):
        config = spAlignDE.STAtlasAlignmentConfig()
        self.assertEqual(config.n_levels, 4)
        self.assertEqual(
            config.minimum_coarse_structures,
            7,
        )
        pairing_weights = (
            config.pairing_weight_sdf,
            config.pairing_weight_chamfer,
            config.pairing_weight_dice,
            config.pairing_weight_area,
            config.pairing_weight_thickness,
            config.pairing_weight_asd,
        )
        self.assertAlmostEqual(sum(pairing_weights), 1.0)
        self.assertGreater(config.pairing_weight_area, config.pairing_weight_dice)
        self.assertGreater(
            config.pairing_weight_thickness,
            config.pairing_weight_chamfer,
        )
        palette = spAlignDE.load_atlas_structure_color_map()
        self.assertEqual(palette["CA3sp"], "#ff9896")
        self.assertEqual(palette["DG-sg"], "#c5b0d5")

    @unittest.skipUnless(HAS_BANKSY, "optional BANKSY dependency is not installed")
    def test_cluster_single_preserves_anndata_contract(self):
        adata = make_cross_sample_adata(n_per_cluster=20, n_genes=30)
        adata = adata[adata.obs["sample_id"].astype(str) == "reference"].copy()
        del adata.obs["sample_id"]
        del adata.obs["cluster"]
        original_spatial = np.asarray(adata.obsm["spatial"]).copy()

        output = spAlignDE.cluster_single(
            adata,
            config=spAlignDE.SingleClusteringConfig(
                num_neighbors=5,
                resolution=0.5,
                refine_boundaries=False,
            ),
        )

        self.assertNotIn("cluster", adata.obs)
        self.assertIn("cluster_raw", output.obs)
        self.assertIn("cluster", output.obs)
        self.assertNotIn("cluster_refined", output.obs)
        np.testing.assert_array_equal(output.obsm["spatial"], original_spatial)

    def test_build_st_cluster_hierarchy_preserves_input(self):
        adata = self.make_single_adata()
        original_obs_columns = list(adata.obs.columns)
        original_spatial = np.asarray(adata.obsm["spatial"]).copy()

        output, hierarchy_columns = spAlignDE.build_st_cluster_hierarchy(
            adata,
            config=spAlignDE.STAtlasAlignmentConfig(
                n_levels=3,
                min_genes=4,
            ),
        )

        self.assertEqual(list(adata.obs.columns), original_obs_columns)
        self.assertEqual(hierarchy_columns, ("cluster_level_k2",))
        self.assertIn("cluster_level_k2", output.obs)
        np.testing.assert_array_equal(output.obsm["spatial"], original_spatial)
        self.assertEqual(
            output.uns["spAlignDE"]["st_atlas_hierarchy"]["cluster_key"],
            "cluster",
        )

    def test_atlas_alignment_requires_cluster_labels(self):
        adata = self.make_single_adata()
        del adata.obs["cluster"]
        with self.assertRaisesRegex(ValueError, "Run single clustering"):
            spAlignDE.build_st_cluster_hierarchy(adata)

    def test_atlas_plots_reuse_pair_and_label_colors(self):
        adata = self.make_single_adata()
        n_obs = adata.n_obs
        adata.obs["x_prealigned"] = np.linspace(0.0, 3.0, n_obs)
        adata.obs["y_prealigned"] = np.linspace(0.0, 3.0, n_obs)
        adata.obs["x_aligned"] = np.linspace(0.2, 2.8, n_obs)
        adata.obs["y_aligned"] = np.linspace(0.2, 2.8, n_obs)
        adata.obs["atlas_label"] = np.where(
            adata.obs["cluster"].astype(str) == "0", 1, 2
        )
        atlas = spAlignDE.AllenCCFReference(
            annotation=np.array(
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 2, 2, 0], [0, 0, 0, 0]]
            ),
            x_coordinates=np.arange(4, dtype=float),
            y_coordinates=np.arange(4, dtype=float),
            structures=pd.DataFrame(),
            slice_index=675,
            voxel_size_x=1.0,
            voxel_size_y=1.0,
            annotation_path=Path("annotation.nrrd"),
            structure_table_path=Path("structures.csv"),
        )
        result = spAlignDE.STAtlasAlignmentResult(
            adata=adata,
            atlas=atlas,
            matched_pairs=pd.DataFrame(
                {
                    "cluster": ["0", "1"],
                    "atlas_labels_union": ["1", "2"],
                    "candidate_name": ["CA3sp", "DG-sg"],
                }
            ),
            stage_summary=pd.DataFrame(),
            prealignment_parameters={},
            hierarchy_columns=(),
        )

        fig, axes = spAlignDE.plot_st_atlas_alignment(result)
        self.assertEqual(
            axes[0].get_title(), "Before iterative LDDMM (all points)"
        )
        self.assertEqual(len(axes[0].images), 2)
        plt.close(fig)

        color_map = spAlignDE.load_atlas_label_color_map(atlas=atlas)
        self.assertEqual(color_map["0"], "#ffffff")
        fig, axes = spAlignDE.plot_atlas_label_transfer(
            result,
            color_map=color_map,
        )
        self.assertEqual(axes[0].get_title(), "Allen atlas labels")
        self.assertEqual(
            axes[1].get_title(), "Aligned ST transferred atlas labels"
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
