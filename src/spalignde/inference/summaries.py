"""Gene-level and multi-sample summaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import _legacy_core
from ._types import LocalDEResult, TrajectoryResult


def acat_pvalue(pvalues: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Combine dependent P values with the Cauchy combination test.

    The small-P branch uses the reciprocal Cauchy-tail approximation used by
    the manuscript analysis. This avoids loss of precision when evaluating
    ``tan((0.5 - p) * pi)`` very close to ``pi / 2``.
    """

    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p) & (p > 0) & (p < 1)
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
    very_small = p < 1e-12
    statistic = float(np.sum(w[very_small] / (np.pi * p[very_small])))
    statistic += float(
        np.sum(w[~very_small] * np.tan((0.5 - p[~very_small]) * np.pi))
    )
    if statistic > 1e15:
        return float(1.0 / (np.pi * statistic))
    return float(0.5 - np.arctan(statistic) / np.pi)


def gene_level_acat_pvalue(result: LocalDEResult, gene: str) -> float:
    """Return the gene-level ACAT omnibus P value from local grid tests.

    Local P values are combined within each contrast. For results with more
    than one contrast, the contrast-level ACAT P values are combined again
    using weights proportional to their numbers of valid local tests. Adjusted
    q-values are never substituted when local P values are unavailable.
    """

    if not isinstance(result, LocalDEResult):
        raise TypeError("result must be returned by fit_local_de().")
    if gene not in result.fits:
        raise KeyError(f"No fitted result is available for gene {gene!r}.")
    terrain = result.fits[gene].get("terrain_data")
    if not isinstance(terrain, dict):
        raise ValueError(f"The fitted result for {gene!r} has no terrain data.")
    p_by_time = terrain.get("p_by_time")
    if not isinstance(p_by_time, dict):
        raise ValueError(f"The fitted result for {gene!r} has no local P values.")

    time_ids = list(terrain.get("time_ids", [])) or list(p_by_time)
    contrast_pvalues: list[float] = []
    contrast_weights: list[float] = []
    for time_id in time_ids:
        local = p_by_time.get(time_id)
        if local is None:
            local = next(
                (values for key, values in p_by_time.items() if str(key) == str(time_id)),
                None,
            )
        if local is None:
            continue
        local = np.asarray(local, dtype=float).reshape(-1)
        valid = np.isfinite(local) & (local > 0) & (local <= 1)
        if not valid.any():
            continue
        contrast_pvalues.append(acat_pvalue(local[valid]))
        contrast_weights.append(float(valid.sum()))

    if not contrast_pvalues:
        return float("nan")
    if len(contrast_pvalues) == 1:
        return float(contrast_pvalues[0])
    return acat_pvalue(contrast_pvalues, weights=contrast_weights)


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
