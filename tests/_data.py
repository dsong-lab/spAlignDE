"""Small deterministic datasets used by the package tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def make_cross_sample_adata(
    *,
    n_per_cluster: int = 20,
    n_genes: int = 12,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 2.2]])
    reference = np.vstack(
        [
            rng.normal(center, 0.18, size=(n_per_cluster, 2))
            for center in centers
        ]
    )
    angle = np.deg2rad(18.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    query = (reference - np.array([0.8, -0.4])) @ rotation / 1.08
    spatial = np.vstack([reference, query])
    labels = np.tile(
        np.repeat(np.arange(len(centers)).astype(str), n_per_cluster),
        2,
    )
    samples = np.repeat(["reference", "query"], len(reference))

    expression = rng.poisson(1, size=(len(spatial), n_genes)).astype(float)
    for cluster in range(len(centers)):
        mask = labels == str(cluster)
        expression[mask, cluster * 2 : cluster * 2 + 2] += 3

    obs = pd.DataFrame(
        {"sample_id": samples, "cluster": labels},
        index=[f"cell_{i}" for i in range(len(spatial))],
    )
    adata = ad.AnnData(X=sparse.csr_matrix(expression), obs=obs)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["spatial"] = spatial
    return adata
