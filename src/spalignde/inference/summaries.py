"""Gene-level and multi-sample summaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _legacy_core
from ._types import LocalDEResult, TrajectoryResult


def acat_pvalue(pvalues: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Combine dependent P values with the Cauchy combination test."""

    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return float("nan")
    p = np.clip(p[finite], 1e-15, 1 - 1e-15)
    if weights is None:
        w = np.full(p.size, 1.0 / p.size)
    else:
        w0 = np.asarray(weights, dtype=float)[finite]
        if np.any(w0 < 0) or not np.isfinite(w0).all() or w0.sum() <= 0:
            raise ValueError("weights must be finite, non-negative, and sum to a positive value.")
        w = w0 / w0.sum()
    statistic = np.sum(w * np.tan((0.5 - p) * np.pi))
    return float(0.5 - np.arctan(statistic) / np.pi)


def cluster_trajectories(
    result: LocalDEResult,
    gene: str,
    *,
    n_clusters: int | str = "auto",
    time_values: Sequence[float] | None = None,
    random_state: int | None = None,
) -> TrajectoryResult:
    """Cluster adjusted local-expression trajectories for one gene.

    `n_clusters="auto"` uses the notebook's spatial coherence and complexity
    criterion. An integer fixes the requested number of clusters. `time_values`
    is reserved for the standalone refactor; current ordering follows the
    prepared sample identifiers.
    """

    if gene not in result.fits:
        raise KeyError(f"No fitted result is available for gene {gene!r}.")
    if time_values is not None:
        raise NotImplementedError("Explicit time_values are not yet wired into the legacy clustering kernel.")
    if n_clusters != "auto" and (not isinstance(n_clusters, int) or n_clusters < 2):
        raise ValueError("n_clusters must be 'auto' or an integer of at least 2.")
    out = _legacy_core.run_global_spatial_roi_trajectory_clustering(
        result.fits[gene],
        result.prepared.shared,
        K_TRAJ=None if n_clusters == "auto" else n_clusters,
        do_plot=False,
        seed=random_state,
        core=1,
    )
    return TrajectoryResult(result=out, gene=gene, n_clusters=n_clusters)
