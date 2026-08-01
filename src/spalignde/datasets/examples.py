"""Small deterministic datasets for examples and smoke tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def make_cross_sample_example(
    *,
    n_per_cluster: int = 20,
    n_genes: int = 12,
    random_state: int = 0,
) -> ad.AnnData:
    """Return a small two-sample AnnData object with known shared structures.

    The fixed ``reference`` contains three spatial clusters. The ``query`` is
    generated from the same points by a deterministic similarity transform.
    This dataset is intentionally synthetic and is suitable for installation
    checks, API examples, and the CPU quickstart—not biological inference.

    Parameters
    ----------
    n_per_cluster
        Number of observations per cluster and sample. Must be positive.
    n_genes
        Number of simulated genes. Must be at least six.
    random_state
        Seed controlling coordinates and Poisson counts.

    Returns
    -------
    anndata.AnnData
        Cell-by-gene counts with ``sample_id`` and ``cluster`` in ``obs`` and
        two-dimensional coordinates in ``obsm['spatial']``.
    """
    if int(n_per_cluster) < 1:
        raise ValueError("n_per_cluster must be positive")
    if int(n_genes) < 6:
        raise ValueError("n_genes must be at least 6")

    rng = np.random.default_rng(random_state)
    centers = np.asarray([[0.0, 0.0], [3.0, 0.0], [1.5, 2.2]])
    reference = np.vstack(
        [
            rng.normal(center, 0.18, size=(int(n_per_cluster), 2))
            for center in centers
        ]
    )
    angle = np.deg2rad(18.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    query = (reference - np.asarray([0.8, -0.4])) @ rotation / 1.08
    spatial = np.vstack([reference, query])
    labels = np.tile(
        np.repeat(np.arange(len(centers)).astype(str), int(n_per_cluster)),
        2,
    )
    samples = np.repeat(["reference", "query"], len(reference))

    expression = rng.poisson(1, size=(len(spatial), int(n_genes))).astype(float)
    for cluster in range(len(centers)):
        mask = labels == str(cluster)
        expression[mask, cluster * 2 : cluster * 2 + 2] += 3

    obs = pd.DataFrame(
        {"sample_id": samples, "cluster": labels},
        index=[f"cell_{index}" for index in range(len(spatial))],
    )
    result = ad.AnnData(X=sparse.csr_matrix(expression), obs=obs)
    result.var_names = [f"gene_{index}" for index in range(int(n_genes))]
    result.obsm["spatial"] = spatial
    return result
