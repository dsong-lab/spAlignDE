from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import spAlignDE

from ._data import make_cross_sample_adata


class CrossSampleInputTests(unittest.TestCase):
    def test_loader_replaces_historical_metadata_key(self):
        adata = make_cross_sample_adata()
        adata.uns["spalignde"] = {"prealignment": {"scale": 1.0}}

        loaded = spAlignDE.load_cross_sample_data(adata)

        self.assertNotIn("spalignde", loaded.uns)
        self.assertEqual(
            loaded.uns["spAlignDE"]["prealignment"]["scale"],
            1.0,
        )

    def test_loader_preserves_canonical_metadata_key(self):
        adata = make_cross_sample_adata()
        adata.uns["spAlignDE"] = {"prealignment": {"scale": 1.0}}

        loaded = spAlignDE.load_cross_sample_data(adata)

        self.assertEqual(
            loaded.uns["spAlignDE"]["prealignment"]["scale"],
            1.0,
        )

    def test_valid_contract(self):
        adata = make_cross_sample_adata()
        spAlignDE.validate_cross_sample_anndata(adata)

    def test_missing_sample_key(self):
        adata = make_cross_sample_adata()
        del adata.obs["sample_id"]
        with self.assertRaisesRegex(ValueError, "sample_id"):
            spAlignDE.validate_cross_sample_anndata(adata)

    def test_missing_cluster_key(self):
        adata = make_cross_sample_adata()
        del adata.obs["cluster"]
        with self.assertRaisesRegex(ValueError, "Run joint clustering"):
            spAlignDE.validate_cross_sample_anndata(adata)

    def test_invalid_spatial_shape(self):
        adata = make_cross_sample_adata()
        adata.obsm["spatial_3d"] = np.zeros((adata.n_obs, 3))
        with self.assertRaisesRegex(ValueError, "must have shape"):
            spAlignDE.validate_cross_sample_anndata(
                adata,
                spatial_key="spatial_3d",
            )

    def test_non_unique_observation_names(self):
        adata = make_cross_sample_adata()
        names = adata.obs_names.to_numpy().copy()
        names[1] = names[0]
        adata.obs_names = names
        with self.assertRaisesRegex(ValueError, "globally unique"):
            spAlignDE.validate_cross_sample_anndata(adata)

    def test_load_cross_sample_data_accepts_anndata(self):
        adata = make_cross_sample_adata()
        loaded = spAlignDE.load_cross_sample_data(adata)

        self.assertIsNot(loaded, adata)
        self.assertEqual(loaded.shape, adata.shape)
        self.assertTrue(np.array_equal(loaded.obsm["spatial"], adata.obsm["spatial"]))

    def test_load_cross_sample_data_accepts_paired_csv_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pd.DataFrame(
                {
                    "cell_id": ["a", "b"],
                    "x": [0.0, 1.0],
                    "y": [2.0, 3.0],
                    "section": ["left", "right"],
                }
            ).to_csv(directory / "cell_metadata_reference.csv", index=False)
            pd.DataFrame(
                {
                    "cell_id": ["a", "b"],
                    "gene_a": [1.0, 2.0],
                    "gene_b": [3.0, 4.0],
                }
            ).to_csv(directory / "cell_by_gene_reference.csv", index=False)
            pd.DataFrame(
                {
                    "cell_id": ["c", "d"],
                    "x": [4.0, 5.0],
                    "y": [6.0, 7.0],
                    "section": ["left", "right"],
                }
            ).to_csv(directory / "cell_metadata_query.csv", index=False)
            pd.DataFrame(
                {
                    "cell_id": ["c", "d"],
                    "gene_b": [5.0, 6.0],
                    "gene_c": [7.0, 8.0],
                }
            ).to_csv(directory / "cell_by_gene_query.csv", index=False)

            loaded = spAlignDE.load_cross_sample_data(directory)

        self.assertEqual(loaded.shape, (4, 3))
        self.assertEqual(list(loaded.var_names), ["gene_a", "gene_b", "gene_c"])
        self.assertEqual(
            list(loaded.obs_names),
            ["query_c", "query_d", "reference_a", "reference_b"],
        )
        self.assertEqual(set(loaded.obs["sample_id"]), {"query", "reference"})
        self.assertIn("section", loaded.obs)
        self.assertTrue(
            np.array_equal(
                loaded.obsm["spatial"],
                loaded.obs[["x", "y"]].to_numpy(),
            )
        )
        dense = loaded.X.toarray()
        self.assertTrue(np.array_equal(dense[:2, 0], [0.0, 0.0]))
        self.assertTrue(np.array_equal(dense[2:, 2], [0.0, 0.0]))

    def test_load_h5ad_uses_xy_columns_when_spatial_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "combined.h5ad"
            adata = make_cross_sample_adata()
            spatial = np.asarray(adata.obsm["spatial"]).copy()
            adata.obs["x"] = spatial[:, 0]
            adata.obs["y"] = spatial[:, 1]
            del adata.obsm["spatial"]
            adata.write_h5ad(path)

            loaded = spAlignDE.load_cross_sample_data(path)

        self.assertTrue(np.array_equal(loaded.obsm["spatial"], spatial))


class SingleSampleInputTests(unittest.TestCase):
    def test_load_single_sample_data_accepts_anndata_without_sample_id(self):
        adata = make_cross_sample_adata()[:20].copy()
        del adata.obs["sample_id"]
        del adata.obs["cluster"]

        loaded = spAlignDE.load_single_sample_data(adata)

        self.assertIsNot(loaded, adata)
        self.assertEqual(loaded.shape, adata.shape)
        spAlignDE.validate_single_sample_anndata(loaded)

    def test_read_single_sample_csv_preserves_metadata_and_row_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            metadata_path = directory / "cell_metadata_S2R1.csv"
            expression_path = directory / "cell_by_gene_S2R1.csv"
            pd.DataFrame(
                {
                    "cell_id": ["b", "a"],
                    "x": [1.0, 2.0],
                    "y": [3.0, 4.0],
                    "region": ["left", "right"],
                }
            ).to_csv(metadata_path, index=False)
            pd.DataFrame(
                {
                    "cell_id": ["a", "b"],
                    "gene_a": [10.0, 20.0],
                    "gene_b": [30.0, 40.0],
                }
            ).to_csv(expression_path, index=False)

            loaded = spAlignDE.load_single_sample_data(
                metadata_path,
                expression_csv=expression_path,
            )

        self.assertEqual(list(loaded.obs_names), ["b", "a"])
        self.assertEqual(list(loaded.obs["region"]), ["left", "right"])
        self.assertTrue(np.array_equal(loaded.X.toarray()[:, 0], [20.0, 10.0]))
        self.assertTrue(
            np.array_equal(
                loaded.obsm["spatial"],
                loaded.obs[["x", "y"]].to_numpy(),
            )
        )

    def test_single_sample_validator_requires_clusters_only_when_requested(self):
        adata = make_cross_sample_adata()[:20].copy()
        del adata.obs["cluster"]
        spAlignDE.validate_single_sample_anndata(adata)
        with self.assertRaisesRegex(ValueError, "Run single clustering"):
            spAlignDE.validate_single_sample_anndata(
                adata,
                require_cluster=True,
            )


if __name__ == "__main__":
    unittest.main()
