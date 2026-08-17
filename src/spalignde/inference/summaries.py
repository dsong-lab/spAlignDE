"""Gene-level and multi-sample summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np
import pandas as pd
from scipy import stats

from . import _legacy_core
from ._types import LocalDEResult, TrajectoryResult


def acat_pvalue(pvalues: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Combine dependent P values with the Cauchy combination test.

    The small-P branch uses the reciprocal Cauchy-tail approximation used by
    the manuscript analysis. This avoids loss of precision when evaluating
    ``tan((0.5 - p) * pi)`` very close to ``pi / 2``.
    """

    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p) & (p >= 0) & (p <= 1)
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


def gene_level_age_trend_acat(
    result: LocalDEResult,
    gene: str,
    *,
    time_values: Sequence[float] | None = None,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Test for a spatially distributed linear age trend in one gene.

    At every retained grid location, this function fits a weighted linear
    regression of the unsmoothed adjusted local expression on age. The
    mismatch-aware precision stored in ``Wv_by_time`` supplies the regression
    weights. Two-sided local slope P values are then combined across the grid
    with ACAT. This pre-trajectory test does not use trajectory smoothing,
    trajectory clusters, or the selected cluster count.

    The returned mapping contains ``summary``, ``per_contrast`` diagnostics,
    and the ``per_grid`` slope tests. ``gene_level_trend_acat_p`` is the global
    P value used for the multi-age application. The older per-contrast spatial
    ACAT combination is retained as ``legacy_any_signal_acat_p`` for diagnostic
    comparison only.

    ``time_values`` must provide one numeric age for each fitted non-reference
    contrast in its stored order. If omitted, the first numeric value in every
    contrast identifier is used. The reference section is not added as an
    extra age observation.
    """

    if not isinstance(result, LocalDEResult):
        raise TypeError("result must be returned by fit_local_de().")
    if gene not in result.fits:
        raise KeyError(f"No fitted result is available for gene {gene!r}.")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")

    terrain = result.fits[gene].get("terrain_data")
    if not isinstance(terrain, dict):
        raise ValueError(f"The fitted result for {gene!r} has no terrain data.")
    time_ids = [str(value) for value in terrain.get("time_ids", [])]
    if not time_ids:
        time_ids = [str(value) for value in terrain.get("muA_adj_by_time", {})]
    if len(time_ids) < 3:
        raise ValueError("At least three fitted non-reference contrasts are required.")

    if time_values is None:
        ages = []
        for time_id in time_ids:
            match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", time_id)
            ages.append(float(match.group(0)) if match is not None else np.nan)
        ages = np.asarray(ages, dtype=float)
        if not np.all(np.isfinite(ages)):
            missing = [time_id for time_id, age in zip(time_ids, ages) if not np.isfinite(age)]
            raise ValueError(
                "Could not infer numeric ages from contrast identifiers: "
                f"{missing}. Supply time_values explicitly."
            )
    else:
        ages = np.asarray(time_values, dtype=float).reshape(-1)
        if ages.size != len(time_ids):
            raise ValueError("time_values must have one value for each fitted contrast.")
        if not np.all(np.isfinite(ages)):
            raise ValueError("time_values must contain only finite numeric values.")
    if np.unique(ages).size < 3:
        raise ValueError("At least three distinct numeric ages are required.")

    mu_by_time = terrain.get("muA_adj_by_time")
    precision_by_time = terrain.get("Wv_by_time")
    use_by_time = terrain.get("use_by_time")
    if not isinstance(mu_by_time, dict):
        raise ValueError("The fit does not retain muA_adj_by_time.")
    if not isinstance(precision_by_time, dict) or not isinstance(use_by_time, dict):
        raise ValueError(
            "The fit does not retain Wv_by_time and use_by_time; rerun "
            "fit_local_de() with the current package version."
        )

    y_columns: list[np.ndarray] = []
    precision_columns: list[np.ndarray] = []
    use_columns: list[np.ndarray] = []
    n_grid: int | None = None
    for time_id in time_ids:
        if time_id not in mu_by_time or time_id not in precision_by_time or time_id not in use_by_time:
            raise ValueError(f"Incomplete age-trend inputs for contrast {time_id!r}.")
        y = np.asarray(mu_by_time[time_id], dtype=float).reshape(-1)
        precision = np.asarray(precision_by_time[time_id], dtype=float).reshape(-1)
        use = np.asarray(use_by_time[time_id], dtype=bool).reshape(-1)
        if n_grid is None:
            n_grid = int(y.size)
        if y.size != n_grid or precision.size != n_grid or use.size != n_grid:
            raise ValueError(f"Inconsistent grid length for contrast {time_id!r}.")
        y_columns.append(y)
        precision_columns.append(precision)
        use_columns.append(use)

    y = np.column_stack(y_columns)
    precision = np.column_stack(precision_columns)
    use = np.column_stack(use_columns)
    age_row = ages.reshape(1, -1)
    valid = use & np.isfinite(y) & np.isfinite(precision) & (precision > 0)
    n_observed = valid.sum(axis=1).astype(int)
    weights = np.where(valid, precision, 0.0)
    y_finite = np.where(valid, y, 0.0)
    sum_weights = weights.sum(axis=1)
    age_mean = np.divide(
        np.sum(weights * age_row, axis=1),
        sum_weights,
        out=np.full(y.shape[0], np.nan),
        where=sum_weights > 0,
    )
    expression_mean = np.divide(
        np.sum(weights * y_finite, axis=1),
        sum_weights,
        out=np.full(y.shape[0], np.nan),
        where=sum_weights > 0,
    )
    centered_age = age_row - age_mean[:, None]
    centered_expression = y - expression_mean[:, None]
    sxx = np.sum(weights * np.where(valid, centered_age**2, 0.0), axis=1)
    sxy = np.sum(
        weights * np.where(valid, centered_age * centered_expression, 0.0),
        axis=1,
    )
    slope = np.divide(sxy, sxx, out=np.full(y.shape[0], np.nan), where=sxx > 0)
    fitted = expression_mean[:, None] + slope[:, None] * centered_age
    residual = np.where(valid, y - fitted, 0.0)
    degrees_freedom = n_observed - 2
    weighted_sse = np.sum(weights * residual**2, axis=1)
    residual_scale = np.divide(
        weighted_sse,
        degrees_freedom,
        out=np.full(y.shape[0], np.nan),
        where=degrees_freedom > 0,
    )
    slope_se = np.sqrt(
        np.divide(
            residual_scale,
            sxx,
            out=np.full(y.shape[0], np.nan),
            where=sxx > 0,
        )
    )
    t_statistic = np.divide(
        slope,
        slope_se,
        out=np.full(y.shape[0], np.nan),
        where=np.isfinite(slope_se) & (slope_se > 0),
    )
    local_p = np.full(y.shape[0], np.nan)
    testable = (degrees_freedom > 0) & np.isfinite(t_statistic)
    local_p[testable] = 2.0 * stats.t.sf(
        np.abs(t_statistic[testable]),
        df=degrees_freedom[testable],
    )
    local_p[testable] = np.clip(local_p[testable], 1e-300, 1.0)

    per_grid = pd.DataFrame({
        "gene": gene,
        "grid_index": np.arange(y.shape[0], dtype=int),
        "age_min": float(np.min(ages)),
        "age_max": float(np.max(ages)),
        "n_ages_used": n_observed,
        "age_slope": slope,
        "age_slope_se": slope_se,
        "age_trend_t": t_statistic,
        "age_trend_df": degrees_freedom,
        "age_trend_p": local_p,
    })
    valid_local_p = np.isfinite(local_p) & (local_p >= 0) & (local_p <= 1)
    trend_p = acat_pvalue(local_p[valid_local_p])

    p_by_time = terrain.get("p_by_time", {})
    diagnostic_rows: list[dict[str, object]] = []
    contrast_pvalues: list[float] = []
    contrast_weights: list[float] = []
    for time_id in time_ids:
        values = p_by_time.get(time_id) if isinstance(p_by_time, dict) else None
        if values is None:
            diagnostic_p = float("nan")
            n_values = 0
            source = "missing"
        else:
            values = np.asarray(values, dtype=float).reshape(-1)
            valid_values = np.isfinite(values) & (values >= 0) & (values <= 1)
            n_values = int(valid_values.sum())
            diagnostic_p = acat_pvalue(values[valid_values])
            source = "p_by_time"
            if np.isfinite(diagnostic_p):
                contrast_pvalues.append(diagnostic_p)
                contrast_weights.append(float(max(n_values, 1)))
        diagnostic_rows.append({
            "gene": gene,
            "time_id": time_id,
            "age": float(ages[time_ids.index(time_id)]),
            "n_pvalues": n_values,
            "p_source": source,
            "acat_p_time": diagnostic_p,
            "neglog10_acat_p_time": (
                -np.log10(max(diagnostic_p, 1e-300))
                if np.isfinite(diagnostic_p)
                else np.nan
            ),
        })
    legacy_p = acat_pvalue(contrast_pvalues, weights=contrast_weights)
    summary = {
        "gene": gene,
        "gene_level_trend_acat_p": trend_p,
        "gene_level_acat_p": trend_p,
        "legacy_any_signal_acat_p": legacy_p,
        "neglog10_gene_p": (
            -np.log10(max(trend_p, 1e-300)) if np.isfinite(trend_p) else np.nan
        ),
        "reject_global_null": bool(np.isfinite(trend_p) and trend_p < float(alpha)),
        "alpha": float(alpha),
        "n_local_trend_tests": int(valid_local_p.sum()),
        "n_times_combined": int(len(contrast_pvalues)),
        "trend_input": "unsmoothed_muA_adj_by_time",
        "trend_weight": "mismatch_aware_Wv_by_time",
        "reference_included_as_age_observation": False,
    }
    return {
        "summary": summary,
        "per_contrast": pd.DataFrame(diagnostic_rows),
        "per_grid": per_grid,
    }


def cluster_trajectories(
    result: LocalDEResult,
    gene: str,
    *,
    n_clusters: int | str = "auto",
    time_values: Sequence[float] | None = None,
    auto_k_options: Mapping[str, object] | None = None,
    random_state: int | None = None,
) -> TrajectoryResult:
    """Cluster adjusted local-expression trajectories for one gene.

    ``n_clusters="auto"`` first asks whether cluster-specific complete
    trajectories improve held-out-time prediction relative to a shared time
    trend. If reliable dynamic evidence is present, the final value is selected
    by scanning the one-SE candidate plateau from fine to coarse resolution.
    The scan continues while sub-``R_map`` fragmentation decreases and retains
    the first fine-side local minimum when the next coarser candidate becomes
    more fragmented. With four or five time points the dynamic check uses
    linear leave-one-time-out fits; with three or fewer it returns the smallest
    candidate value. An integer fixes the requested number of clusters.

    ``time_values`` supplies one numeric value for each trajectory time ID in
    the insertion order retained by the fit. If omitted, numeric suffixes in
    sample identifiers are used when available, otherwise equal spacing is
    assumed. ``auto_k_options`` optionally overrides private screening settings,
    including ``K_GRID`` and ``AUTO_SUBSAMPLE_N``.
    """

    if not isinstance(result, LocalDEResult):
        raise TypeError("result must be returned by fit_local_de().")
    if gene not in result.fits:
        raise KeyError(f"No fitted result is available for gene {gene!r}.")
    if n_clusters != "auto" and (not isinstance(n_clusters, int) or n_clusters < 2):
        raise ValueError("n_clusters must be 'auto' or an integer of at least 2.")
    if auto_k_options is not None and not isinstance(auto_k_options, Mapping):
        raise TypeError("auto_k_options must be a mapping or None.")
    out = _legacy_core.run_global_spatial_roi_trajectory_clustering(
        result.fits[gene],
        result.prepared.shared,
        K_TRAJ=None if n_clusters == "auto" else n_clusters,
        auto_k=None if auto_k_options is None else dict(auto_k_options),
        time_values=time_values,
        do_plot=False,
        seed=random_state,
        core=1,
    )
    return TrajectoryResult(
        result=out,
        gene=gene,
        n_clusters=n_clusters,
        metadata={
            "selected_n_clusters": int(out["K_TRAJ"]),
            "selection": None if out.get("auto_k") is None else out["auto_k"].get("selection"),
        },
    )
