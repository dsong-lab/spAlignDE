"""Plot fitted local spatial differential-expression results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from ._types import LocalDEResult


def _scatter_map(ax, xy, values, *, cmap, vmin=None, vmax=None):
    values = np.asarray(values, dtype=float)
    if values.shape != (xy.shape[0],):
        raise ValueError(
            f"Map length {values.size} does not match the shared grid "
            f"({xy.shape[0]} locations)."
        )
    valid = np.isfinite(values)
    artist = ax.scatter(
        xy[valid, 0],
        xy[valid, 1],
        c=values[valid],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=2.0,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    return artist


def _sample_expression(
    result: LocalDEResult,
    sample_id: str,
    gene: str,
) -> tuple[np.ndarray, np.ndarray]:
    data = result.prepared.data
    sample_mask = data["sample_id"].astype(str).eq(str(sample_id))
    sample = data.loc[sample_mask]
    if sample.empty:
        raise ValueError(f"Sample {sample_id!r} is absent from the prepared data.")

    coordinate_columns = ("x_aligned", "y_aligned")
    if not set(coordinate_columns).issubset(sample.columns):
        coordinate_columns = ("x", "y")
    xy = sample.loc[:, coordinate_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    expression = pd.to_numeric(sample[gene], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(expression)
    return xy[valid], expression[valid]


def _reference_for_contrast(result: LocalDEResult, query_id: str) -> str:
    if result.contrast == "vs_reference":
        return str(result.prepared.reference)

    sample_order = [
        str(result.prepared.reference),
        *[str(value) for value in result.prepared.shared.get("time_ids", [])],
    ]
    try:
        query_position = sample_order.index(str(query_id))
    except ValueError:
        return str(result.prepared.reference)
    return sample_order[max(query_position - 1, 0)]


def _plot_boundary(ax, shared, *, linewidth=0.65):
    geometry = shared.get("poly_with_holes")
    if geometry is not None:
        geometries = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        for item in geometries:
            if hasattr(item, "exterior") and item.exterior is not None:
                boundary = np.asarray(item.exterior.coords, dtype=float)
                ax.plot(boundary[:, 0], boundary[:, 1], color="black", lw=linewidth)
            for interior in getattr(item, "interiors", []):
                boundary = np.asarray(interior.coords, dtype=float)
                ax.plot(
                    boundary[:, 0],
                    boundary[:, 1],
                    color="black",
                    lw=max(0.45, linewidth * 0.7),
                )
        return

    boundary = shared.get("outer_bnd")
    if boundary is None:
        auto_geometry = shared.get("auto_geometry", {}) or {}
        boundary = auto_geometry.get("outer_bnd", auto_geometry.get("outer"))
    if boundary is not None:
        boundary = np.asarray(boundary, dtype=float)
        if boundary.ndim == 2 and boundary.shape[1] >= 2:
            ax.plot(boundary[:, 0], boundary[:, 1], color="black", lw=linewidth)


def _plot_spot_expression(ax, xy, expression, *, shared, vmin, vmax):
    artist = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=np.maximum(expression, 0),
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        s=2.0,
        linewidths=0,
        rasterized=True,
    )
    _plot_boundary(ax, shared, linewidth=0.55)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return artist


def _grid_matrix(xy, values):
    x = np.asarray(xy[:, 0], dtype=float)
    y = np.asarray(xy[:, 1], dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != xy.shape[0]:
        raise ValueError(
            f"Map length {values.size} does not match the shared grid "
            f"({xy.shape[0]} locations)."
        )

    unique_x = np.sort(np.unique(x[np.isfinite(x)]))
    unique_y = np.sort(np.unique(y[np.isfinite(y)]))
    x_index = np.searchsorted(unique_x, x)
    y_index = np.searchsorted(unique_y, y)
    matrix = np.full((len(unique_y), len(unique_x)), np.nan, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    matrix[y_index[valid], x_index[valid]] = values[valid]
    grid_x, grid_y = np.meshgrid(unique_x, unique_y)
    return grid_x, grid_y, matrix


def _plot_local_statistic(
    ax,
    xy,
    statistic,
    significant,
    *,
    shared,
    statistic_limit,
):
    statistic = np.asarray(statistic, dtype=float).copy()
    statistic[~np.isfinite(statistic)] = 0.0
    grid_x, grid_y, statistic_matrix = _grid_matrix(xy, statistic)
    artist = ax.pcolormesh(
        grid_x,
        grid_y,
        np.ma.masked_invalid(statistic_matrix),
        shading="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(
            vmin=-statistic_limit,
            vcenter=0.0,
            vmax=statistic_limit,
        ),
        rasterized=True,
    )

    significant = np.asarray(significant, dtype=bool)
    if np.any(significant):
        _, _, significant_matrix = _grid_matrix(
            xy,
            significant.astype(float),
        )
        try:
            ax.contour(
                grid_x,
                grid_y,
                significant_matrix,
                levels=[0.5],
                colors=["#c53030"],
                linewidths=0.9,
            )
        except ValueError:
            pass

    _plot_boundary(ax, shared, linewidth=0.75)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return artist


def plot_local_result(
    result: LocalDEResult,
    gene: str,
    *,
    show_expression: bool = True,
    invert_y: bool = False,
):
    """Plot sample expression and local statistics with significant contours.

    One row is drawn for each query-reference contrast. When
    ``show_expression=True``, the first two columns contain spot-level
    expression in the reference and query samples on their aligned coordinates.
    The final column shows the shared-grid local statistic, with the boundary
    of FDR-significant locations drawn as a red contour. Set ``invert_y=True``
    for image-style coordinates whose row values increase from top to bottom.
    """

    if not isinstance(result, LocalDEResult):
        raise TypeError("result must be returned by fit_local_de().")
    if gene not in result.fits:
        raise KeyError(f"No fitted result is available for gene {gene!r}.")

    terrain = result.fits[gene].get("terrain_data")
    if not isinstance(terrain, dict):
        raise ValueError(f"The fitted result for {gene!r} has no terrain data.")

    time_ids = list(terrain.get("time_ids", []))
    if not time_ids:
        raise ValueError(f"The fitted result for {gene!r} has no contrasts.")

    grid = result.prepared.shared["grid_eval"]
    xy = grid[["x", "y"]].to_numpy(dtype=float)
    n_columns = 3 if show_expression else 1
    figure, axes = plt.subplots(
        len(time_ids),
        n_columns,
        figsize=(4.0 * n_columns, 3.3 * len(time_ids)),
        squeeze=False,
        constrained_layout=True,
    )

    statistic_maps = [
        np.asarray(terrain["stat_by_time"][time_id], dtype=float)
        for time_id in time_ids
    ]
    finite_statistics = np.concatenate(
        [values[np.isfinite(values)] for values in statistic_maps]
    )
    statistic_limit = (
        float(np.nanpercentile(np.abs(finite_statistics), 99))
        if finite_statistics.size
        else 1.0
    )
    statistic_limit = max(statistic_limit, np.finfo(float).eps)

    expression_values: list[np.ndarray] = []
    expression_maps: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    if show_expression:
        for query_id in time_ids:
            reference_id = _reference_for_contrast(result, str(query_id))
            for sample_id in (reference_id, str(query_id)):
                key = (sample_id, gene)
                if key not in expression_maps:
                    expression_maps[key] = _sample_expression(
                        result,
                        sample_id,
                        gene,
                    )
                    expression_values.append(expression_maps[key][1])
    finite_expression = (
        np.concatenate([values[np.isfinite(values)] for values in expression_values])
        if expression_values
        else np.array([], dtype=float)
    )
    expression_max = (
        float(np.nanpercentile(finite_expression, 99.5))
        if finite_expression.size
        else 1.0
    )
    expression_max = max(expression_max, np.finfo(float).eps)

    for row, time_id in enumerate(time_ids):
        column = 0
        if show_expression:
            reference_id = _reference_for_contrast(result, str(time_id))
            reference_xy, reference_expression = expression_maps[(reference_id, gene)]
            query_xy, query_expression = expression_maps[(str(time_id), gene)]
            reference_artist = _plot_spot_expression(
                axes[row, column],
                reference_xy,
                reference_expression,
                shared=result.prepared.shared,
                vmin=0.0,
                vmax=expression_max,
            )
            axes[row, column].set_title(f"{reference_id}: {gene} expression")
            if invert_y:
                axes[row, column].invert_yaxis()
            column += 1
            _plot_spot_expression(
                axes[row, column],
                query_xy,
                query_expression,
                shared=result.prepared.shared,
                vmin=0.0,
                vmax=expression_max,
            )
            axes[row, column].set_title(f"{time_id}: {gene} expression")
            if invert_y:
                axes[row, column].invert_yaxis()
            figure.colorbar(
                reference_artist,
                ax=[axes[row, column - 1], axes[row, column]],
                fraction=0.035,
                pad=0.02,
            )
            column += 1

        statistic = np.asarray(terrain["stat_by_time"][time_id], dtype=float)
        significant = np.asarray(
            terrain["sig_mask_by_time"][time_id],
            dtype=bool,
        )
        artist = _plot_local_statistic(
            axes[row, column],
            xy,
            statistic,
            significant,
            shared=result.prepared.shared,
            statistic_limit=statistic_limit,
        )
        axes[row, column].set_title(
            f"{time_id}: local statistic\nred = FDR-significant region"
        )
        if invert_y:
            axes[row, column].invert_yaxis()
        figure.colorbar(artist, ax=axes[row, column], fraction=0.046, pad=0.02)

    figure.suptitle(f"{gene}: local spatial differential expression")
    return figure
