"""Reusable functions for the cross-sample prealign + LDDMM notebook pipeline.

This module is tutorial-oriented:
- Load source/target cells from CSV or joint AnnData.
- Build a global similarity pre-alignment.
- Convert point annotations into multi-channel rasters.
- Run diffeomorphic LDDMM registration.
- Map points and summarize alignment quality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import anndata as ad
from matplotlib.lines import Line2D
from scipy.spatial import KDTree


def load_source_target_csv(source_csv: str, target_csv: str, cluster_col: str = "leiden_harmony_refined"):
    """Load source/target CSVs and validate required columns."""
    df_src = pd.read_csv(source_csv, index_col=0)
    df_tgt = pd.read_csv(target_csv, index_col=0)

    required_cols = ["x", "y", cluster_col]
    for name, df in [("source", df_src), ("target", df_tgt)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    x_src = df_src["x"].to_numpy(dtype=np.float64)
    y_src = df_src["y"].to_numpy(dtype=np.float64)
    x_tgt = df_tgt["x"].to_numpy(dtype=np.float64)
    y_tgt = df_tgt["y"].to_numpy(dtype=np.float64)

    return df_src, df_tgt, x_src, y_src, x_tgt, y_tgt


def _validate_input_df(df: pd.DataFrame, name: str, cluster_col: str) -> None:
    required_cols = ["x", "y", cluster_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _attach_cluster_column(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    cluster_csv: str | None,
    name: str,
) -> pd.DataFrame:
    """Ensure df has cluster_col, optionally merging from a cluster CSV by cell_id."""
    if cluster_col in df.columns:
        return df

    fallback_cols = ["leiden_harmony_refined", "leiden_harmony"]
    for col in fallback_cols:
        if col in df.columns:
            out = df.copy()
            out[cluster_col] = out[col].astype(str)
            return out

    if cluster_csv and os.path.exists(cluster_csv):
        cluster_df = pd.read_csv(cluster_csv)
        if "cell_id" not in cluster_df.columns:
            raise ValueError(f"{name} cluster CSV missing 'cell_id': {cluster_csv}")

        source_cluster_col = None
        if cluster_col in cluster_df.columns:
            source_cluster_col = cluster_col
        else:
            for col in fallback_cols:
                if col in cluster_df.columns:
                    source_cluster_col = col
                    break

        if source_cluster_col is None:
            raise ValueError(
                f"{name} cluster CSV missing '{cluster_col}' and fallback cols {fallback_cols}: {cluster_csv}"
            )

        map_df = (
            cluster_df[["cell_id", source_cluster_col]]
            .copy()
            .dropna(subset=["cell_id"])
            .assign(cell_id=lambda x: x["cell_id"].astype(str))
            .drop_duplicates(subset=["cell_id"], keep="first")
            .rename(columns={source_cluster_col: cluster_col})
        )

        out = df.copy()
        out["cell_id"] = out["cell_id"].astype(str)
        out = out.merge(map_df, on="cell_id", how="left")
        if out[cluster_col].isna().all():
            raise ValueError(
                f"{name}: no cluster labels matched by cell_id using {cluster_csv}. "
                "Check that AnnData obs['cell_id'] matches cluster CSV cell_id."
            )
        out[cluster_col] = out[cluster_col].astype(str)
        return out

    available = sorted(df.columns.tolist())
    raise ValueError(
        f"{name} is missing required cluster column '{cluster_col}'. "
        f"Available columns: {available}. "
        "Provide cluster_csv to merge labels by cell_id."
    )


def load_source_target_anndata(
    *,
    source_sample_id: str,
    target_sample_id: str,
    anndata_path: str | None = None,
    adata_input: ad.AnnData | None = None,
    source_cluster_csv: str | None = None,
    target_cluster_csv: str | None = None,
    cluster_col: str = "leiden_harmony_refined",
):
    """Load source/target from a joint AnnData by sample_id."""
    if adata_input is None and not anndata_path:
        raise ValueError("For anndata mode, provide either adata_input or anndata_path.")

    adata = adata_input.copy() if adata_input is not None else ad.read_h5ad(anndata_path)
    obs = adata.obs.copy()

    if "sample_id" not in obs.columns:
        raise ValueError("AnnData obs must include 'sample_id'.")
    obs["sample_id"] = obs["sample_id"].astype(str)

    if "cell_id" not in obs.columns:
        obs["cell_id"] = adata.obs_names.astype(str)

    df_src = obs.loc[obs["sample_id"] == str(source_sample_id)].copy()
    df_tgt = obs.loc[obs["sample_id"] == str(target_sample_id)].copy()

    if df_src.empty:
        raise ValueError(f"No cells found for source_sample_id='{source_sample_id}'.")
    if df_tgt.empty:
        raise ValueError(f"No cells found for target_sample_id='{target_sample_id}'.")

    df_src = _attach_cluster_column(
        df_src,
        cluster_col=cluster_col,
        cluster_csv=source_cluster_csv,
        name="source",
    )
    df_tgt = _attach_cluster_column(
        df_tgt,
        cluster_col=cluster_col,
        cluster_csv=target_cluster_csv,
        name="target",
    )

    _validate_input_df(df_src, "source", cluster_col)
    _validate_input_df(df_tgt, "target", cluster_col)

    x_src = df_src["x"].to_numpy(dtype=np.float64)
    y_src = df_src["y"].to_numpy(dtype=np.float64)
    x_tgt = df_tgt["x"].to_numpy(dtype=np.float64)
    y_tgt = df_tgt["y"].to_numpy(dtype=np.float64)
    return df_src, df_tgt, x_src, y_src, x_tgt, y_tgt


def load_source_target_data(
    *,
    input_mode: str = "csv",
    source_csv: str | None = None,
    target_csv: str | None = None,
    source_sample_id: str | None = None,
    target_sample_id: str | None = None,
    anndata_path: str | None = None,
    adata_input: ad.AnnData | None = None,
    source_cluster_csv: str | None = None,
    target_cluster_csv: str | None = None,
    cluster_col: str = "leiden_harmony_refined",
):
    """Load source/target tables and coordinate vectors for registration.

    Parameters
    ----------
    input_mode : {"csv", "anndata"}
        `csv`: read two separate CSV files.
        `anndata`: split one joint AnnData by `sample_id`.
    source_csv, target_csv : str or None
        CSV paths used in `csv` mode. Must contain `x`, `y`, and `cluster_col`.
    source_sample_id, target_sample_id : str or None
        Sample identifiers used in `anndata` mode.
    anndata_path, adata_input : optional
        One of these must be provided in `anndata` mode.
    source_cluster_csv, target_cluster_csv : str or None
        Optional fallback cluster tables (joined by `cell_id`) when AnnData
        does not include `cluster_col`.
    cluster_col : str
        Cluster annotation column used as semantic channels downstream.

    Returns
    -------
    df_src, df_tgt, x_src, y_src, x_tgt, y_tgt
        Source/target tables and numeric coordinate arrays.
    """
    mode = str(input_mode).lower()
    if mode == "csv":
        if not source_csv or not target_csv:
            raise ValueError("For csv mode, provide source_csv and target_csv.")
        return load_source_target_csv(source_csv, target_csv, cluster_col=cluster_col)
    if mode == "anndata":
        if not source_sample_id or not target_sample_id:
            raise ValueError("For anndata mode, provide source_sample_id and target_sample_id.")
        return load_source_target_anndata(
            source_sample_id=source_sample_id,
            target_sample_id=target_sample_id,
            anndata_path=anndata_path,
            adata_input=adata_input,
            source_cluster_csv=source_cluster_csv,
            target_cluster_csv=target_cluster_csv,
            cluster_col=cluster_col,
        )
    raise ValueError("input_mode must be 'csv' or 'anndata'.")


def print_input_preview(df_src: pd.DataFrame, df_tgt: pd.DataFrame, source_name: str, target_name: str, cluster_col: str = "leiden_harmony_refined") -> None:
    """Print a compact sanity check of loaded source/target inputs."""
    print("Loaded input tables:")
    print(f"  source: {source_name}  shape={df_src.shape}")
    print(f"  target: {target_name}  shape={df_tgt.shape}")
    print("\nSource head:")
    print(df_src[["x", "y", cluster_col]].head())
    print("\nTarget head:")
    print(df_tgt[["x", "y", cluster_col]].head())


def plot_raw_spatial(x_src, y_src, x_tgt, y_tgt) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    ax[0].scatter(x_src, y_src, s=1, alpha=0.25, label="source", color="#1f77b4")
    ax[0].set_title("Source Raw Coordinates")
    ax[0].legend(markerscale=10, loc="lower left")

    ax[1].scatter(x_tgt, y_tgt, s=1, alpha=0.25, label="target", color="#ff7f0e")
    ax[1].set_title("Target Raw Coordinates")
    ax[1].legend(markerscale=10, loc="lower left")

    ax[2].scatter(x_src, y_src, s=1, alpha=0.12, label="source", color="#1f77b4")
    ax[2].scatter(x_tgt, y_tgt, s=1, alpha=0.12, label="target", color="#ff7f0e")
    ax[2].set_title("Overlay (Before Alignment)")
    ax[2].legend(markerscale=10, loc="lower left")

    for a in ax:
        a.set_xlabel("x")
        a.set_ylabel("y")
        a.set_aspect("equal", adjustable="box")

    plt.show()


# Boundary-mask similarity alignment (scale + rotation + translation)
# Ready to run after x_src, y_src, x_tgt, y_tgt are defined.
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import minimize


def build_union_grid(x1, y1, x2, y2, resolution=None, padding=0.02, max_hw=1800):
    x_all = np.concatenate([np.asarray(x1, float), np.asarray(x2, float)])
    y_all = np.concatenate([np.asarray(y1, float), np.asarray(y2, float)])

    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()
    xr, yr = (xmax - xmin), (ymax - ymin)
    if xr == 0:
        xr = 1.0
    if yr == 0:
        yr = 1.0

    xmin -= padding * xr
    xmax += padding * xr
    ymin -= padding * yr
    ymax += padding * yr

    if resolution is None:
        resolution = max(xmax - xmin, ymax - ymin) / max_hw
        resolution = max(resolution, 1e-6)

    nx = int(np.ceil((xmax - xmin) / resolution)) + 1
    ny = int(np.ceil((ymax - ymin) / resolution)) + 1

    xg = xmin + np.arange(nx) * resolution
    yg = ymin + np.arange(ny) * resolution
    return xg, yg, (ny, nx), resolution


def points_to_clean_mask(x, y, xg, yg, shape_hw, close_r=6, open_r=2, dilate_r=0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ny, nx = shape_hw

    ix = np.clip(np.rint((x - xg[0]) / (xg[1] - xg[0])).astype(int), 0, nx - 1)
    iy = np.clip(np.rint((y - yg[0]) / (yg[1] - yg[0])).astype(int), 0, ny - 1)

    m = np.zeros((ny, nx), dtype=bool)
    m[iy, ix] = True

    if close_r > 0:
        st = ndimage.generate_binary_structure(2, 1)
        st = ndimage.iterate_structure(st, int(close_r))
        m = ndimage.binary_closing(m, structure=st)

    m = ndimage.binary_fill_holes(m)

    if open_r > 0:
        st = ndimage.generate_binary_structure(2, 1)
        st = ndimage.iterate_structure(st, int(open_r))
        m = ndimage.binary_opening(m, structure=st)

    if dilate_r > 0:
        st = ndimage.generate_binary_structure(2, 1)
        st = ndimage.iterate_structure(st, int(dilate_r))
        m = ndimage.binary_dilation(m, structure=st)

    return m


def principal_axis_angle(mask, xg, yg):
    yy, xx = np.where(mask)
    if len(xx) < 3:
        return 0.0, np.array([xg.mean(), yg.mean()])

    X = xg[xx]
    Y = yg[yy]
    c = np.array([X.mean(), Y.mean()])
    P = np.stack([X - c[0], Y - c[1]], axis=1)
    cov = (P.T @ P) / max(len(P) - 1, 1)
    w, v = np.linalg.eigh(cov)
    axis = v[:, np.argmax(w)]
    theta = np.arctan2(axis[1], axis[0])
    return theta, c


def apply_similarity(x, y, c_src, c_tgt, s, theta, tx, ty):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    P = np.stack([x, y], axis=1)
    Q = (s * (P - c_src) @ R.T) + c_tgt + np.array([tx, ty])
    return Q[:, 0], Q[:, 1]


def iou_score(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def align_by_boundary_similarity(x_src, y_src, x_tgt, y_tgt, resolution=None, close_r=6, open_r=2, dilate_r=0):
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    x_tgt = np.asarray(x_tgt, float)
    y_tgt = np.asarray(y_tgt, float)

    xg, yg, shape_hw, res = build_union_grid(x_src, y_src, x_tgt, y_tgt, resolution=resolution)
    maskI = points_to_clean_mask(x_src, y_src, xg, yg, shape_hw, close_r=close_r, open_r=open_r, dilate_r=dilate_r)
    maskJ = points_to_clean_mask(x_tgt, y_tgt, xg, yg, shape_hw, close_r=close_r, open_r=open_r, dilate_r=dilate_r)

    thetaI, cI = principal_axis_angle(maskI, xg, yg)
    thetaJ, cJ = principal_axis_angle(maskJ, xg, yg)
    s0 = np.sqrt(max(maskJ.sum(), 1) / max(maskI.sum(), 1))

    span_x = xg[-1] - xg[0]
    span_y = yg[-1] - yg[0]

    def objective(p):
        s, theta, tx, ty = p
        if s <= 0:
            return 1.0
        xA, yA = apply_similarity(x_src, y_src, cI, cJ, s, theta, tx, ty)
        mA = points_to_clean_mask(xA, yA, xg, yg, shape_hw, close_r=close_r, open_r=open_r, dilate_r=dilate_r)
        return 1.0 - iou_score(mA, maskJ)

    bounds = [
        (0.3, 3.0),
        (-np.pi, np.pi),
        (-0.35 * span_x, 0.35 * span_x),
        (-0.35 * span_y, 0.35 * span_y),
    ]

    theta_base = thetaJ - thetaI
    starts = [
        np.array([s0, theta_base, 0.0, 0.0]),
        np.array([s0, theta_base + np.pi, 0.0, 0.0]),
    ]

    best = None
    for x0 in starts:
        res_opt = minimize(objective, x0, method='Powell', bounds=bounds, options={'maxiter': 120, 'xtol': 1e-3, 'ftol': 1e-3})
        if (best is None) or (res_opt.fun < best.fun):
            best = res_opt

    s, theta, tx, ty = best.x
    x_src_new, y_src_new = apply_similarity(x_src, y_src, cI, cJ, s, theta, tx, ty)

    maskI_new = points_to_clean_mask(x_src_new, y_src_new, xg, yg, shape_hw, close_r=close_r, open_r=open_r, dilate_r=dilate_r)
    iou_before = iou_score(maskI, maskJ)
    iou_after = iou_score(maskI_new, maskJ)

    params = {
        'scale': float(s),
        'theta_rad': float(theta),
        'theta_deg': float(np.degrees(theta)),
        'tx': float(tx),
        'ty': float(ty),
        'iou_before': float(iou_before),
        'iou_after': float(iou_after),
        'resolution': float(res),
    }
    return x_src_new, y_src_new, params, maskI, maskI_new, maskJ, xg, yg



def run_boundary_prealign_and_plot(x_src, y_src, x_tgt, y_tgt, resolution=None, close_r=6, open_r=2, dilate_r=0):
    """Estimate global similarity pre-alignment from boundary masks and plot QC.

    This is a robust initialization step before LDDMM. It finds scale/rotation/
    translation by maximizing overlap of cleaned binary masks.

    Parameters
    ----------
    resolution : float or None
        Mask-grid resolution. If None, auto-derived from point extent.
    close_r, open_r, dilate_r : int
        Morphology radii for mask cleaning. Increase `close_r` to fill holes;
        increase `open_r` to remove speckles; use small `dilate_r` if masks
        are too thin/fragmented.
    """
    x_src_new, y_src_new, params, maskI_raw, maskI_aligned, maskJ, xg, yg = align_by_boundary_similarity(
        x_src,
        y_src,
        x_tgt,
        y_tgt,
        resolution=resolution,
        close_r=close_r,
        open_r=open_r,
        dilate_r=dilate_r,
    )

    print("Boundary similarity params:", params)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    extent = [xg[0], xg[-1], yg[0], yg[-1]]
    ax[0].imshow(maskI_raw.astype(float), origin="lower", extent=extent, alpha=0.75)
    ax[0].imshow(maskJ.astype(float), origin="lower", extent=extent, alpha=0.35)
    ax[0].set_title("Mask overlap before")

    ax[1].imshow(maskI_aligned.astype(float), origin="lower", extent=extent, alpha=0.75)
    ax[1].imshow(maskJ.astype(float), origin="lower", extent=extent, alpha=0.35)
    ax[1].set_title("Mask overlap after")

    ax[2].scatter(x_src_new, y_src_new, s=1, alpha=0.15, label="source aligned")
    ax[2].scatter(x_tgt, y_tgt, s=1, alpha=0.15, label="target")
    ax[2].set_title("Point cloud overlay")
    ax[2].legend(markerscale=8)
    for a in ax:
        a.set_aspect("equal", "box")
    plt.tight_layout()

    return x_src_new, y_src_new, params, maskI_raw, maskI_aligned, maskJ, xg, yg


def apply_manual_similarity(x, y, x_ref, y_ref, scale=1.0, theta_deg=0.0, tx=0.0, ty=0.0):
    """Apply a manual similarity transform around source/target centroids.

    Useful for interactive tutorial tuning when automatic boundary pre-alignment
    needs adjustment.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c_src = np.array([np.mean(x), np.mean(y)], float)
    c_tgt = np.array([np.mean(np.asarray(x_ref, float)), np.mean(np.asarray(y_ref, float))], float)

    theta = np.deg2rad(theta_deg)
    return apply_similarity(x, y, c_src, c_tgt, scale, theta, tx, ty)


def plot_prealign_check(x_src_new, y_src_new, x_tgt, y_tgt, title: str = "Pre-align check (manual section)") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_src_new, y_src_new, s=1, alpha=0.15, label="source (pre-aligned)")
    ax.scatter(x_tgt, y_tgt, s=1, alpha=0.15, label="target")
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.legend(markerscale=8)
    plt.tight_layout()


def compute_cluster_centroids(
    df: pd.DataFrame,
    label_col: str,
    x_col: str = "x",
    y_col: str = "y",
    min_cluster_size: int = 1,
) -> pd.DataFrame:
    """Compute per-cluster centroid and size table indexed by cluster label."""
    work = df[[label_col, x_col, y_col]].dropna().copy()
    grouped = (
        work.groupby(label_col, observed=True)
        .agg(
            centroid_x=(x_col, "mean"),
            centroid_y=(y_col, "mean"),
            n_cells=(label_col, "size"),
        )
        .sort_index()
    )
    return grouped[grouped["n_cells"] >= int(min_cluster_size)]


def matched_cluster_centroids(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    label_col: str,
    x_col: str = "x",
    y_col: str = "y",
    min_cluster_size: int = 1,
    weight_mode: str | None = "min_size",
):
    """Build one-to-one source/target centroid correspondences by shared labels."""
    src_cent = compute_cluster_centroids(source_df, label_col, x_col, y_col, min_cluster_size)
    tgt_cent = compute_cluster_centroids(target_df, label_col, x_col, y_col, min_cluster_size)
    shared = src_cent.index.intersection(tgt_cent.index)
    if len(shared) < 2:
        raise ValueError(
            f"Need at least 2 shared clusters after filtering; found {len(shared)}. "
            "Lower min_cluster_size or check label_col."
        )

    src = src_cent.loc[shared, ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    tgt = tgt_cent.loc[shared, ["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    if weight_mode == "min_size":
        weights = np.minimum(src_cent.loc[shared, "n_cells"], tgt_cent.loc[shared, "n_cells"]).to_numpy(dtype=float)
    elif weight_mode == "mean_size":
        weights = 0.5 * (
            src_cent.loc[shared, "n_cells"].to_numpy(dtype=float)
            + tgt_cent.loc[shared, "n_cells"].to_numpy(dtype=float)
        )
    elif weight_mode is None:
        weights = np.ones(len(shared), dtype=float)
    else:
        raise ValueError("weight_mode must be 'min_size', 'mean_size', or None")

    matched = pd.DataFrame(
        {
            "cluster": shared,
            "source_centroid_x": src[:, 0],
            "source_centroid_y": src[:, 1],
            "target_centroid_x": tgt[:, 0],
            "target_centroid_y": tgt[:, 1],
            "source_n_cells": src_cent.loc[shared, "n_cells"].to_numpy(),
            "target_n_cells": tgt_cent.loc[shared, "n_cells"].to_numpy(),
            "weight": weights,
        }
    )
    return src, tgt, weights, matched


def estimate_weighted_procrustes(
    source_points,
    target_points,
    weights=None,
    allow_scaling: bool = True,
    allow_reflection: bool = False,
) -> dict:
    """Estimate weighted 2D Kabsch/Procrustes transform from source to target.

    Transform convention for row-vector coordinates:
    ``aligned = scale * source @ R.T + t``.
    """
    X = np.asarray(source_points, dtype=float)
    Y = np.asarray(target_points, dtype=float)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("source_points and target_points must both have shape (N, 2)")
    if X.shape[0] < 2:
        raise ValueError("at least 2 point correspondences are required")

    if weights is None:
        w = np.ones(X.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != X.shape[0]:
            raise ValueError("weights length must match number of point correspondences")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    if not np.isfinite(X).all() or not np.isfinite(Y).all() or not np.isfinite(w).all():
        raise ValueError("points and weights must be finite")
    if w.sum() <= 0:
        raise ValueError("sum of weights must be positive")

    w = w / w.sum()
    mu_x = (w[:, None] * X).sum(axis=0)
    mu_y = (w[:, None] * Y).sum(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y

    covariance = (w[:, None] * Xc).T @ Yc
    U, singular_values, Vt = np.linalg.svd(covariance)
    R = Vt.T @ U.T

    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        singular_values[-1] *= -1
        R = Vt.T @ U.T

    if allow_scaling:
        source_variance = (w * np.sum(Xc * Xc, axis=1)).sum()
        if source_variance <= np.finfo(float).eps:
            raise ValueError("source landmarks are degenerate; cannot estimate scale")
        scale = float(singular_values.sum() / source_variance)
    else:
        scale = 1.0

    t = mu_y - scale * (R @ mu_x)
    return {"R": R, "t": t, "scale": scale}


def apply_cluster_transform(points, transform: dict) -> np.ndarray:
    """Apply a cluster-correspondence similarity transform to row-vector points."""
    points = np.asarray(points, dtype=float)
    return float(transform["scale"]) * points @ np.asarray(transform["R"], dtype=float).T + np.asarray(
        transform["t"], dtype=float
    )


def summarize_cluster_transform(transform: dict) -> dict:
    """Return JSON-serializable transform parameters."""
    R = np.asarray(transform["R"], dtype=float)
    theta_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return {
        "scale": float(transform["scale"]),
        "theta_deg": theta_deg,
        "translation_x": float(transform["t"][0]),
        "translation_y": float(transform["t"][1]),
        "det_R": float(np.linalg.det(R)),
        "R": R.tolist(),
        "t": np.asarray(transform["t"], dtype=float).tolist(),
    }


def run_cluster_correspondence_prealign(
    df_src: pd.DataFrame,
    df_tgt: pd.DataFrame,
    x_src,
    y_src,
    *,
    source_sample_id: str,
    target_sample_id: str,
    cluster_col: str = "leiden_harmony_refined",
    allow_scaling: bool = True,
    allow_reflection: bool = False,
    use_cluster_size_weights: bool = True,
    min_cluster_size: int = 10,
    out_dir: str | os.PathLike | None = None,
    verbose: bool = True,
):
    """Prealign source to target by weighted shared-cluster centroid matching.

    Returns
    -------
    x_src_new, y_src_new, params, centroid_matches, source_centroids,
    target_centroids, centroids_aligned
    """
    weight_mode = "min_size" if use_cluster_size_weights else None
    source_centroids, target_centroids, centroid_weights, centroid_matches = matched_cluster_centroids(
        df_src,
        df_tgt,
        cluster_col,
        min_cluster_size=min_cluster_size,
        weight_mode=weight_mode,
    )

    transform = estimate_weighted_procrustes(
        source_centroids,
        target_centroids,
        weights=centroid_weights,
        allow_scaling=allow_scaling,
        allow_reflection=allow_reflection,
    )
    source_aligned = apply_cluster_transform(np.column_stack([x_src, y_src]), transform)
    x_src_new = source_aligned[:, 0]
    y_src_new = source_aligned[:, 1]

    centroids_aligned = apply_cluster_transform(source_centroids, transform)
    centroid_residuals = np.linalg.norm(centroids_aligned - target_centroids, axis=1)
    params = summarize_cluster_transform(transform)
    params.update(
        {
            "prealign_method": "shared_cluster_weighted_procrustes",
            "transform_convention": "aligned = scale * source @ R.T + t",
            "source_sample": source_sample_id,
            "target_sample": target_sample_id,
            "cluster_col": cluster_col,
            "n_shared_clusters": int(len(centroid_matches)),
            "weighted_centroid_rmse": float(np.sqrt(np.average(centroid_residuals**2, weights=centroid_weights))),
            "unweighted_centroid_rmse": float(np.sqrt(np.mean(centroid_residuals**2))),
            "allow_scaling": bool(allow_scaling),
            "allow_reflection": bool(allow_reflection),
            "weight_mode": weight_mode,
            "min_cluster_size": int(min_cluster_size),
        }
    )

    centroid_matches = centroid_matches.copy()
    centroid_matches["source_aligned_centroid_x"] = centroids_aligned[:, 0]
    centroid_matches["source_aligned_centroid_y"] = centroids_aligned[:, 1]
    centroid_matches["residual_distance"] = centroid_residuals

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        params_path = out_dir / f"transform_params_{source_sample_id}_to_{target_sample_id}.json"
        centroids_path = out_dir / f"centroid_matches_{source_sample_id}_to_{target_sample_id}.csv"
        with params_path.open("w") as fh:
            json.dump(params, fh, indent=2)
        centroid_matches.to_csv(centroids_path, index=False)
        if verbose:
            print("wrote", params_path)
            print("wrote", centroids_path)

    if verbose:
        print(f"Using cluster-correspondence prealign with {len(centroid_matches)} shared clusters")
        print(json.dumps(params, indent=2))

    return x_src_new, y_src_new, params, centroid_matches, source_centroids, target_centroids, centroids_aligned


def _cluster_prealign_legend_handles(source_label: str, target_label: str, source_centroid_label: str):
    return [
        Line2D([0], [0], marker="o", linestyle="None", markersize=4, color="tab:blue", label=source_label),
        Line2D([0], [0], marker="o", linestyle="None", markersize=4, color="tab:orange", label=target_label),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="black",
            color="black",
            label=source_centroid_label,
        ),
        Line2D([0], [0], marker="x", linestyle="None", markersize=6, color="red", label="target centroids"),
    ]


def plot_cluster_correspondence_prealign_before_after(
    x_src,
    y_src,
    x_src_new,
    y_src_new,
    x_tgt,
    y_tgt,
    source_centroids,
    target_centroids,
    centroids_aligned,
    *,
    source_sample_id: str,
    target_sample_id: str,
    out_dir: str | os.PathLike | None = None,
    method_label: str = "cluster-correspondence",
    point_size: float = 1.0,
    point_alpha: float = 0.08,
    figsize: tuple[float, float] = (12, 6),
):
    """Plot raw-before and prealigned-after overlays for a global pre-alignment.

    Source points are blue and drawn below target points. Target points are
    orange and drawn on top. The y-axis is kept in the original coordinate
    orientation.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].scatter(x_src, y_src, s=point_size, alpha=point_alpha, label=f"source {source_sample_id} raw", color="tab:blue", zorder=1)
    axes[0].scatter(x_tgt, y_tgt, s=point_size, alpha=point_alpha, label=f"target {target_sample_id}", color="tab:orange", zorder=2)
    axes[0].scatter(
        source_centroids[:, 0],
        source_centroids[:, 1],
        s=25,
        marker="o",
        facecolors="none",
        edgecolors="black",
        label="source centroids",
        zorder=3,
    )
    axes[0].scatter(
        target_centroids[:, 0],
        target_centroids[:, 1],
        s=35,
        marker="x",
        c="red",
        label="target centroids",
        zorder=4,
    )
    for a, b in zip(source_centroids, target_centroids):
        axes[0].plot([a[0], b[0]], [a[1], b[1]], color="0.4", lw=0.5, alpha=0.25, zorder=0)
    axes[0].set_title(f"Before {method_label} pre-alignment")

    axes[1].scatter(
        x_src_new,
        y_src_new,
        s=point_size,
        alpha=point_alpha,
        label=f"source {source_sample_id} prealigned",
        color="tab:blue",
        zorder=1,
    )
    axes[1].scatter(x_tgt, y_tgt, s=point_size, alpha=point_alpha, label=f"target {target_sample_id}", color="tab:orange", zorder=2)
    axes[1].scatter(
        centroids_aligned[:, 0],
        centroids_aligned[:, 1],
        s=25,
        marker="o",
        facecolors="none",
        edgecolors="black",
        label="aligned source centroids",
        zorder=3,
    )
    axes[1].scatter(
        target_centroids[:, 0],
        target_centroids[:, 1],
        s=35,
        marker="x",
        c="red",
        label="target centroids",
        zorder=4,
    )
    for a, b in zip(centroids_aligned, target_centroids):
        axes[1].plot([a[0], b[0]], [a[1], b[1]], color="0.4", lw=0.5, alpha=0.35, zorder=0)
    axes[1].set_title(f"After {method_label} pre-alignment")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    axes[0].legend(
        handles=_cluster_prealign_legend_handles(
            f"source {source_sample_id} raw", f"target {target_sample_id}", "source centroids"
        ),
        frameon=False,
    )
    axes[1].legend(
        handles=_cluster_prealign_legend_handles(
            f"source {source_sample_id} prealigned", f"target {target_sample_id}", "aligned source centroids"
        ),
        frameon=False,
    )

    fig_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / f"cluster_centroid_prealign_before_after_{source_sample_id}_to_{target_sample_id}.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=160)
        print("wrote", fig_path)

    plt.show()
    return fig, axes, fig_path


# Re-export general rasterization/LDDMM utilities from the package core.
try:
    from ._slddmm_core import (
        centers_to_edges,
        build_shared_grid,
        rasterize_cluster_channels_on_grid,
        prep_lddmm_multichannel,
        composition_to_rgb,
        build_cluster_labels,
        prepare_rasterization_and_multichannel,
        plot_rasterization_preview,
        affine_from_components,
        clip,
        sample_image_on_coords,
        central_diff,
        jacobian_and_div,
        build_kernel_K,
        apply_K,
        advect_field,
        geodesic_shooting,
        LDDMM_shooting,
        run_lddmm_pipeline_source_target,
        run_lddmm_pipeline,
        map_points_source_to_target,
        map_points_target_to_source,
        as_numpy_1d,
        compute_alignment_metrics,
        plot_alignment_overlays,
        plot_cluster_overlay_before_after,
        build_source_output_table,
        save_alignment_output,
        print_output_preview,
        print_cluster_performance,
        compute_alignment_uncertainty_from_eigenmodes,
        plot_point_uncertainty,
        compute_uncertainty_from_current_alignment,

    )
except ImportError:
    from ._slddmm_core import (
        centers_to_edges,
        build_shared_grid,
        rasterize_cluster_channels_on_grid,
        prep_lddmm_multichannel,
        composition_to_rgb,
        build_cluster_labels,
        prepare_rasterization_and_multichannel,
        plot_rasterization_preview,
        affine_from_components,
        clip,
        sample_image_on_coords,
        central_diff,
        jacobian_and_div,
        build_kernel_K,
        apply_K,
        advect_field,
        geodesic_shooting,
        LDDMM_shooting,
        run_lddmm_pipeline_source_target,
        run_lddmm_pipeline,
        map_points_source_to_target,
        map_points_target_to_source,
        as_numpy_1d,
        compute_alignment_metrics,
        plot_alignment_overlays,
        plot_cluster_overlay_before_after,
        build_source_output_table,
        save_alignment_output,
        print_output_preview,
        print_cluster_performance,
        compute_alignment_uncertainty_from_eigenmodes,
        plot_point_uncertainty,
        compute_uncertainty_from_current_alignment,

    )
