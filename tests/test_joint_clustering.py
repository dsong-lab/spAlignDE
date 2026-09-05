from __future__ import annotations

import importlib.util
import unittest

import numpy as np

import spAlignDE

from ._data import make_cross_sample_adata


HAS_CLUSTERING = all(
    importlib.util.find_spec(name) is not None
    for name in ("banksy", "harmonypy", "scanpy")
)


@unittest.skipUnless(HAS_CLUSTERING, "optional clustering dependencies are not installed")
class JointClusteringTests(unittest.TestCase):
    def test_cluster_joint_preserves_input_data_model(self):
        adata = make_cross_sample_adata(n_per_cluster=30, n_genes=20)
        del adata.obs["cluster"]
        original_x = adata.X.copy()
        original_spatial = np.asarray(adata.obsm["spatial"]).copy()

        config = spAlignDE.JointClusteringConfig(
            num_neighbors=5,
            pca_dim=5,
            resolution=0.5,
            snn_neighbors=10,
            refine_boundaries=False,
            compute_umap=False,
        )
        self.assertEqual(config.leiden_flavor, "leidenalg")
        self.assertEqual(config.leiden_n_iterations, -1)
        self.assertEqual(config.harmony_device, "cpu")
        self.assertEqual(config.harmony_threads, 1)
        output = spAlignDE.cluster_joint(adata, config=config)
        repeated = spAlignDE.cluster_joint(adata, config=config)

        self.assertNotIn("cluster", adata.obs)
        self.assertIn("cluster", output.obs)
        np.testing.assert_array_equal(output.obs["cluster"], repeated.obs["cluster"])
        self.assertIn("cluster_raw", output.obs)
        self.assertNotIn("cluster_refined", output.obs)
        self.assertTrue(
            np.array_equal(
                output.obs["cluster"].astype(str),
                output.obs["cluster_raw"].astype(str),
            )
        )
        self.assertEqual(output.shape, adata.shape)
        self.assertTrue(np.array_equal(output.obsm["spatial"], original_spatial))
        self.assertEqual((output.X != original_x).nnz, 0)


if __name__ == "__main__":
    unittest.main()
