#!/usr/bin/env python3
"""Reusable ST-to-H&E alignment functions.

It includes ST cluster refinement, prealignment helpers, ST/H&E mask
construction, pair scoring, SDT construction, and point-transform wrappers.
LDDMM optimization is provided by the package's shared ``_slddmm_core``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.ndimage import distance_transform_edt as edt, gaussian_filter
from scipy.ndimage import zoom
from scipy.spatial import cKDTree
from skimage import morphology
from skimage.measure import label as cc_label
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

try:
    import torch
except Exception:  # pragma: no cover - required only for LDDMM execution.
    torch = None

try:
    import cv2
except Exception:  # pragma: no cover - only needed by functions that explicitly use OpenCV.
    cv2 = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional until a plot function is called.
    plt = None

from ._slddmm_core import map_points_source_to_target

DEFAULT_MASK_PARAMS = dict(
    BIN_THR=0.08,
    SIGMA_FIX=1.8,
    AUTO_SIG=True,
    SIGMA_MIN=1.0,
    SIGMA_MAX=5.0,
    SIGMA_SCL=1.0,
    CLOSE_R=2,
    REF_CLOSE_R=2,
    REF_OPEN_R=1,
    REF_SMOOTH_MAX=1.2,
    KEEP_LARGEST=False,
    MIN_SIZE=120,
    FILL_HOLES=True,
)

MASK_PARAMS_THIN = dict(
    BIN_THR=0.11,
    SIGMA_FIX=1.0,
    AUTO_SIG=True,
    SIGMA_MIN=0.6,
    SIGMA_MAX=2.2,
    SIGMA_SCL=0.8,
    CLOSE_R=1,
    REF_CLOSE_R=1,
    REF_OPEN_R=0,
    REF_SMOOTH_MAX=0.55,
    KEEP_LARGEST=False,
    MIN_SIZE=50,
    FILL_HOLES=False,
)

W_ALIGN = dict(
    sdf_corr=0.20,
    chamfer_sim=0.25,
    dice=0.15,
    area_sim=0.25,
    asd_sim=0.15,
)

LDDMM_INPUT_CFG = dict(
    mode="equalized_sdt",
    st_cfg=dict(sigma_pre=1.4, thr=0.5, close_r=2, open_r=1, min_area=50),
    he_cfg=dict(sigma_pre=0.4, thr=0.5, close_r=1, open_r=1, min_area=80),
    clip_dist=60.0,
    sigma_sdt=0.9,
    band=4.0,
    tau=2.0,
    area_power=0.8,
    w_min=0.5,
    w_max=2.5,
    global_shape_weight=1.6,
    global_channel_scale=None,
    zoom_scale=0.6,
    weight_mode="embedded",
)
LDDMM_MODEL_CFG = dict(nt=5, a=50.0, p=2.0, expand=2.0, grid_step=6)
LDDMM_OPTIM_CFG = dict(
    niter=300,
    diffeo_start=0,
    lrL=5e-9,
    lrT=5e-2,
    lrM=2e3,
    affine_slowdown=10.0,
    lrM_decay=0.9995,
    lrM_min=200.0,
    grad_clip_m0=None,
    rollback_on_rise=True,
    affine_det_floor=1e-4,
    affine_value_clip=5.0,
)
LDDMM_EM_CFG = dict(update_every=5, start_iter=50)
LDDMM_INTENSITY_CFG = dict(sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=5e5, sigmaP=2e1)
LDDMM_RUNTIME_CFG = dict(dtype="float32")

def select_genes_by_variance_fraction(X_log, frac=0.85, min_genes=50):
    gene_var = X_log.var(axis=0)
    order = np.argsort(gene_var)[::-1]
    var_sorted = gene_var[order]
    tot = var_sorted.sum()
    if tot <= 0:
        return order, X_log.shape[1]
    cum_frac = np.cumsum(var_sorted) / tot
    topN = int(np.searchsorted(cum_frac, frac) + 1)
    topN = max(topN, min_genes)
    topN = min(topN, X_log.shape[1])
    return order[:topN], topN


def auto_k_levels(n_clusters, n_levels=4, min_k=2):
    if n_clusters < 2:
        return [n_clusters]
    if n_levels <= 1:
        return [n_clusters]
    n_levels = min(n_levels, n_clusters)
    ks = []
    for i in range(1, n_levels + 1):
        k = int(round(i * n_clusters / n_levels))
        lower = min_k if i < n_levels else n_clusters
        upper = n_clusters - (n_levels - i)
        k = max(lower, min(k, upper))
        if ks and k <= ks[-1]:
            k = min(n_clusters, ks[-1] + 1)
        ks.append(k)
    ks[-1] = n_clusters
    return ks


def _set_cell_id_index(df, df_name):
    # prefer explicit cell_id column
    if "cell_id" in df.columns:
        cid = df["cell_id"].astype(str).str.strip()
        out = df.drop(columns=["cell_id"]).copy()
        out.index = cid
    # fallback: common exported index column name
    elif "Unnamed: 0" in df.columns:
        cid = df["Unnamed: 0"].astype(str).str.strip()
        out = df.drop(columns=["Unnamed: 0"]).copy()
        out.index = cid
    else:
        # fallback: existing index
        out = df.copy()
        out.index = out.index.astype(str).str.strip()

    out.index.name = "cell_id"
    if out.index.duplicated().any():
        ndup = int(out.index.duplicated().sum())
        raise ValueError(f"{df_name} has duplicated cell_id ({ndup})")
    return out


def build_st_refined_clusters(
    meta_path,
    counts_path,
    out_path=None,
    n_levels=4,
    var_frac=0.75,
    min_genes=50,
    drop_blank=True,
    cluster_col="banksy_cluster",
):
    # 1) Load and align by cell_id
    meta_raw = pd.read_csv(meta_path)
    counts_raw = pd.read_csv(counts_path)

    df1 = _set_cell_id_index(meta_raw, "meta")
    counts = _set_cell_id_index(counts_raw, "counts")

    common = df1.index.intersection(counts.index)
    if len(common) < 2:
        raise ValueError("aligned cells < 2 after matching by cell_id")

    # keep only shared cells, in same order
    df1 = df1.loc[common].copy()
    X = counts.loc[common].copy()

    if drop_blank:
        X = X[[c for c in X.columns if not str(c).lower().startswith("blank")]]

    if cluster_col not in df1.columns:
        raise KeyError(f"{cluster_col} not in metadata columns")

    meta = df1.copy()
    meta[cluster_col] = meta[cluster_col].astype(str)

    # 2) Select informative genes
    X_log = np.log1p(X.values)
    top_idx, topN_use = select_genes_by_variance_fraction(X_log, frac=var_frac, min_genes=min_genes)
    genes_use = X.columns[top_idx]
    X_use = pd.DataFrame(X_log[:, top_idx], index=common, columns=genes_use)

    gene_var_all = X_log.var(axis=0)
    var_kept = gene_var_all[top_idx].sum() / (gene_var_all.sum() + 1e-12)
    print(f"[genes] selected {topN_use}/{X.shape[1]} genes, var_fraction~={var_kept:.3f}")

    # 3) Build hierarchy on pseudo-bulk
    avg = X_use.groupby(meta[cluster_col]).mean()
    avg_z = (avg - avg.mean(0)) / (avg.std(0) + 1e-8)
    avg_z = avg_z.fillna(0.0)

    Z = linkage(avg_z.values, method="ward", metric="euclidean")
    base_clusters = avg_z.index.astype(str).tolist()
    n_base = len(base_clusters)

    # 4) Auto k levels and write refined columns
    ks = auto_k_levels(n_base, n_levels=n_levels, min_k=2)
    ks_gen = [k for k in ks if k < n_base]
    refined_cols = []

    for k in ks_gen:
        col = f"banksy_cluster_refined_k{k}"
        df1[col] = pd.NA

        coarse = fcluster(Z, t=k, criterion="maxclust")
        cl2k = dict(zip(base_clusters, coarse))
        df1.loc[common, col] = meta[cluster_col].map(cl2k).astype(int).values
        refined_cols.append(col)

    if out_path is not None:
        df1.to_csv(out_path, index=True)
        print(f"saved: {out_path}")

    print(f"n base clusters: {n_base}")
    print(f"auto ks (coarse->fine): {ks}")
    print(f"generated ks (< original): {ks_gen}")
    print(f"refined columns: {refined_cols}")
    return df1, ks, refined_cols, Z, avg_z


def plot_cluster_grid(df, plot_cols, x_col="x", y_col="y", ncols=2, s=1, alpha=0.9):
    if plt is None:
        raise ImportError("matplotlib is required for plot_cluster_grid.")
    n = len(plot_cols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for i, col in enumerate(plot_cols):
        ax = axes[i]
        d = df[[x_col, y_col, col]].dropna().copy()
        d[col] = d[col].astype(str)

        labels = sorted(d[col].unique(), key=lambda z: (len(z), z))
        cmap = plt.get_cmap("tab20", max(20, len(labels)))
        color_map = {lab: cmap(j % cmap.N) for j, lab in enumerate(labels)}
        colors = d[col].map(color_map)

        ax.scatter(d[x_col], d[y_col], c=colors, s=s, alpha=alpha, linewidths=0)
        ax.set_title(f"{col} (n={len(labels)})", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)

    # hide empty panels
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def _sample_for_plot(df, max_points=None, random_state=7):
    if max_points is None or len(df) <= int(max_points):
        return df.copy()
    return df.sample(int(max_points), random_state=random_state).copy()


def plot_st_refinement_levels(
    df,
    plot_cols=None,
    x_col="x",
    y_col="y",
    ncols=3,
    max_points=60000,
    point_size=0.25,
    alpha=0.75,
    invert_y=True,
    selected_col=None,
):
    """Plot original and refined ST cluster levels for manual level selection."""
    if plt is None:
        raise ImportError("matplotlib is required for plot_st_refinement_levels.")
    if plot_cols is None:
        refined_cols = sorted(
            [c for c in df.columns if c.startswith("banksy_cluster_refined_k")],
            key=lambda c: int(c.split("_k")[-1]),
        )
        plot_cols = [c for c in ["banksy_cluster"] if c in df.columns] + refined_cols
    if len(plot_cols) == 0:
        raise ValueError("No cluster columns were provided or found.")

    d = _sample_for_plot(df, max_points=max_points)
    n = len(plot_cols)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.7 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for i, col in enumerate(plot_cols):
        ax = axes[i]
        dd = d[[x_col, y_col, col]].dropna().copy()
        dd[col] = dd[col].astype(str)
        labels = sorted(dd[col].unique(), key=lambda z: (len(z), z))
        cmap = plt.get_cmap("tab20", max(20, len(labels)))
        color_map = {lab: cmap(j % cmap.N) for j, lab in enumerate(labels)}
        title = f"{col}\n{len(labels)} clusters"
        if selected_col is not None and col == selected_col:
            title += " (selected)"
        ax.scatter(dd[x_col], dd[y_col], c=dd[col].map(color_map), s=point_size, alpha=alpha, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        if invert_y:
            ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig, axes[:n]


def plot_st_he_overlay_before_after(
    df,
    he_image,
    label_canvas=None,
    label_col="banksy_cluster_refined_k7",
    prealign_cols=("x_prealigned", "y_prealigned"),
    aligned_cols=("x_aligned", "y_aligned"),
    max_points=90000,
    random_state=9,
    point_size=0.42,
    alpha=0.72,
    grayscale_background=True,
    color_by_cluster=True,
    point_color="#0072B2",
):
    """Plot ST points over the H&E image before and after LDDMM alignment."""
    if plt is None:
        raise ImportError("matplotlib is required for plot_st_he_overlay_before_after.")
    from PIL import Image, ImageEnhance

    Image.MAX_IMAGE_PIXELS = None

    if label_col not in df.columns:
        raise KeyError(f"{label_col!r} is not present in df.")
    for col in [*prealign_cols, *aligned_cols]:
        if col not in df.columns:
            raise KeyError(f"{col!r} is not present in df.")

    if label_canvas is not None:
        label_canvas = np.asarray(label_canvas)
        height, width = label_canvas.shape[:2]
    else:
        width = int(np.ceil(np.nanmax(df[[prealign_cols[0], aligned_cols[0]]].to_numpy(float)))) + 1
        height = int(np.ceil(np.nanmax(df[[prealign_cols[1], aligned_cols[1]]].to_numpy(float)))) + 1

    if isinstance(he_image, (str, Path)):
        img = Image.open(he_image).convert("RGB")
    elif isinstance(he_image, Image.Image):
        img = he_image.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(he_image)).convert("RGB")
    img = img.resize((width, height), Image.Resampling.BILINEAR)
    if grayscale_background:
        img = img.convert("L").convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(1.18)
        img = ImageEnhance.Contrast(img).enhance(0.82)
    img_arr = np.asarray(img)

    d = _sample_for_plot(df, max_points=max_points, random_state=random_state)
    if color_by_cluster:
        labs = d[label_col].astype(str)
        levels = sorted(labs.unique(), key=lambda z: (len(z), z))
        palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#00a6d6", "#222222"]
        color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(levels)}
        colors = labs.map(color_map)
    else:
        colors = point_color

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.0), sharex=True, sharey=True)
    panels = [
        (axes[0], prealign_cols, "Before LDDMM: similarity-prealigned ST"),
        (axes[1], aligned_cols, "After LDDMM: deformed ST"),
    ]
    for ax, cols, title in panels:
        ax.imshow(img_arr, origin="upper")
        if label_canvas is not None:
            ax.contour((label_canvas >= 0).astype(float), levels=[0.5], colors="black", linewidths=0.5, alpha=0.45)
        ax.scatter(d[cols[0]], d[cols[1]], c=colors, s=point_size, alpha=alpha, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, width - 1)
        ax.set_ylim(height - 1, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig, axes


def similarity_matrix(scale=1.0, angle_deg=0.0, tx=0.0, ty=0.0, cx=0.0, cy=0.0):
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)

    T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], float)     # move center to origin
    S  = np.array([[scale,0,0],[0,scale,0],[0,0,1]], float)
    R  = np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
    T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]], float)       # move back
    Tt = np.array([[1,0,tx],[0,1,ty],[0,0,1]], float)       # translation
    return Tt @ T2 @ R @ S @ T1


def apply_matrix_xy(df, M, x_col="x", y_col="y", outx="x_prealigned", outy="y_prealigned"):
    P = np.c_[df[x_col].to_numpy(float), df[y_col].to_numpy(float), np.ones(len(df))]
    Q = (M @ P.T).T
    out = df.copy()
    out[outx] = Q[:,0]
    out[outy] = Q[:,1]
    return out


def _plot_axis_array(v, n, axis_idx=None):
    if v is None:
        return np.arange(n, dtype=float)
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and axis_idx is not None:
            v = v[axis_idx]
        else:
            return np.arange(n, dtype=float)
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size >= 2:
        return np.linspace(float(arr[0]), float(arr[-1]), n, dtype=float)
    return np.arange(n, dtype=float)


def _axis_to_numpy(v, n, axis_idx=None):
    if v is None:
        return np.arange(n, dtype=float)
    if isinstance(v, (list, tuple)) and len(v) == 2 and axis_idx is not None:
        v = v[axis_idx]
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size >= 2:
        return np.linspace(float(arr[0]), float(arr[-1]), n, dtype=float)
    return np.arange(n, dtype=float)


def make_phys_to_pix(xJ, yJ, H, W):
    """
    Create a physical-to-pixel coordinate converter based on xJ/yJ canvas.
    """
    x_axis = _axis_to_numpy(xJ, W, axis_idx=1)
    y_axis = _axis_to_numpy(yJ, H, axis_idx=0)
    CANVAS_XMIN, CANVAS_XMAX = float(x_axis[0]), float(x_axis[-1])
    CANVAS_YMIN, CANVAS_YMAX = float(y_axis[0]), float(y_axis[-1])

    sx = (W - 1) / (CANVAS_XMAX - CANVAS_XMIN + 1e-12)
    sy = (H - 1) / (CANVAS_YMAX - CANVAS_YMIN + 1e-12)

    def phys_to_pix_array(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        xi = (x - CANVAS_XMIN) * sx
        yi = (y - CANVAS_YMIN) * sy
        xi = np.clip(np.rint(xi), 0, W - 1).astype(int)
        yi = np.clip(np.rint(yi), 0, H - 1).astype(int)
        return xi, yi

    return phys_to_pix_array


def rasterize_points_soft(
    y_idx, x_idx, h, w,
    sigma=2.0,
    bin_thr=0.15,
    min_obj_frac=0.0,
    close_radius=1,
    fill_holes=True,
    auto_sigma=True,
    sigma_min=1.0, sigma_max=6.0, sigma_scale=0.6
):
    acc = np.zeros((h, w), np.float32)

    if auto_sigma and len(y_idx) >= 5:
        pts = np.stack([y_idx, x_idx], axis=1).astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, h - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, w - 1)
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=5)
        d_med = np.median(d[:, -1])
        sigma = float(np.clip(sigma_scale * d_med, sigma_min, sigma_max))

    if len(y_idx) > 0:
        yy = np.clip(y_idx, 0, h - 1)
        xx = np.clip(x_idx, 0, w - 1)
        np.add.at(acc, (yy, xx), 1.0)

    if sigma and sigma > 0:
        acc = ndimage.gaussian_filter(acc, sigma)

    if acc.max() > 0:
        acc /= acc.max()

    hard = (acc >= bin_thr).astype(np.uint8)

    if close_radius and hard.any():
        se = morphology.disk(int(close_radius))
        hard = morphology.binary_closing(hard.astype(bool), se).astype(np.uint8)

    if fill_holes and hard.any():
        hard = ndimage.binary_fill_holes(hard.astype(bool)).astype(np.uint8)

    if min_obj_frac and hard.any():
        min_pix = int(min_obj_frac * h * w)
        if min_pix > 0:
            hard = morphology.remove_small_objects(
                hard.astype(bool), min_size=min_pix
            ).astype(np.uint8)

    return acc, hard


def refine_edges_morph(
    mask, close_r=2, open_r=1, smooth=1.0,
    keep_largest=False, fill_holes=True
):
    m = mask.astype(bool)

    if close_r > 0:
        m = morphology.binary_closing(m, morphology.disk(int(close_r)))

    if open_r > 0:
        m = morphology.binary_opening(m, morphology.disk(int(open_r)))

    if smooth and smooth > 0:
        f = ndimage.gaussian_filter(m.astype(float), float(smooth))
        m = (f >= 0.5)

    if fill_holes:
        m = ndimage.binary_fill_holes(m)

    if keep_largest and m.any():
        lbl = morphology.label(m)
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        m = (lbl == counts.argmax())

    return m.astype(np.uint8)


def cc_stats(mask_u8, connectivity=2):
    m = mask_u8.astype(bool)
    lab = cc_label(m, connectivity=connectivity)
    n_cc = int(lab.max())
    if n_cc == 0:
        return dict(n_cc=0, largest_frac=0.0)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = counts.max()
    total = counts.sum()
    return dict(n_cc=n_cc, largest_frac=float(largest / (total + 1e-12)))


def keep_cc_by_cumfrac(mask_u8, cumfrac=0.7, max_k=6, min_area=0, connectivity=2):
    m = mask_u8.astype(bool)
    lab = cc_label(m, connectivity=connectivity)
    n = int(lab.max())
    if n == 0:
        return mask_u8

    counts = np.bincount(lab.ravel())
    counts[0] = 0

    ids = np.where(counts >= int(min_area))[0]
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)

    ids_sorted = ids[np.argsort(counts[ids])[::-1]]
    areas_sorted = counts[ids_sorted]
    total = float(areas_sorted.sum()) + 1e-12

    cum = np.cumsum(areas_sorted) / total
    k = int(np.searchsorted(cum, cumfrac) + 1)
    k = int(np.clip(k, 1, max_k))

    keep_ids = ids_sorted[:k]
    out = np.isin(lab, keep_ids)
    return out.astype(np.uint8)


def robust_bbox_metrics(coords, k=8, core_q=0.85):
    """Robust bbox using dense-core points."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return {"bbox_w": np.nan, "bbox_h": np.nan, "bbox_ar": np.nan, "bbox_area": np.nan}

    core = coords
    if len(coords) >= max(k + 1, 20):
        tree = cKDTree(coords)
        d, _ = tree.query(coords, k=k + 1)
        local_scale = np.median(d[:, 1:], axis=1)
        thr = np.quantile(local_scale, core_q)
        core = coords[local_scale <= thr]
        if len(core) < 10:
            core = coords

    x = core[:, 0]
    y = core[:, 1]
    x25, x75 = np.percentile(x, [25, 75])
    y25, y75 = np.percentile(y, [25, 75])

    w = max(float(x75 - x25), 1e-9)
    h = max(float(y75 - y25), 1e-9)
    ar = max(w, h) / min(w, h)
    area = w * h
    return {"bbox_w": w, "bbox_h": h, "bbox_ar": ar, "bbox_area": area}


def grid_thin_points(sub, x_col, y_col, grid_size):
    if len(sub) == 0 or grid_size is None or grid_size <= 0:
        return sub.copy()

    tmp = sub.copy()
    tmp["_gx"] = np.floor(tmp[x_col] / grid_size).astype(int)
    tmp["_gy"] = np.floor(tmp[y_col] / grid_size).astype(int)

    if "avg_knn_dist" in tmp.columns:
        keep_idx = (
            tmp.sort_values("avg_knn_dist", ascending=True)
               .drop_duplicates(subset=["_gx", "_gy"])
               .index
        )
    else:
        keep_idx = tmp.drop_duplicates(subset=["_gx", "_gy"]).index

    return tmp.loc[keep_idx].drop(columns=["_gx", "_gy"])


def filter_cluster_for_mask(
    df,
    label_col,
    x_col='x_prealigned',
    y_col='y_prealigned',
    base_k=20,
    min_points=None,
    detail_area_quantile=0.40,
    area_mode="bbox",          # "bbox" or "n_points"
    detail_mad_k=1.0,
    normal_mad_k=1.2,
    apply_grid_thin=True,
    grid_size_detail=10,
    grid_size_normal=None,
    cluster_specific_params=None,
    force_detail_clusters=None,
    force_normal_clusters=None,
):
    if min_points is None:
        min_points = base_k + 1
    if cluster_specific_params is None:
        cluster_specific_params = {}
    if force_detail_clusters is None:
        force_detail_clusters = set()
    if force_normal_clusters is None:
        force_normal_clusters = set()

    area_rows = []
    for cl in df[label_col].dropna().unique():
        sub = df[df[label_col] == cl]
        coords = sub[[x_col, y_col]].to_numpy()
        bb = robust_bbox_metrics(coords) if len(coords) > 0 else {"bbox_area": np.nan}
        area_rows.append({
            "cluster": str(cl),
            "n_points": int(len(sub)),
            "bbox_area": float(bb["bbox_area"]) if np.isfinite(bb["bbox_area"]) else np.nan,
        })
    area_tbl = pd.DataFrame(area_rows)

    gate_col = "n_points" if area_mode == "n_points" else "bbox_area"
    area_thr = float(area_tbl[gate_col].quantile(detail_area_quantile)) if len(area_tbl) > 0 else -np.inf
    detail_clusters = set(area_tbl.loc[area_tbl[gate_col] <= area_thr, "cluster"].astype(str).tolist())

    print(f"[area-only gate] mode={area_mode}, q={detail_area_quantile}, threshold={area_thr:.3f}, n_detail={len(detail_clusters)}")

    keep_list, rm_list, stats = [], [], []
    clusters = df[label_col].dropna().unique()

    for cl in clusters:
        cl_str = str(cl)
        sub = df[df[label_col] == cl].copy()
        n = len(sub)

        cfg = cluster_specific_params.get(cl_str, cluster_specific_params.get(cl, {}))
        k = cfg.get("k", base_k)

        if n < max(min_points, k + 1):
            sub["avg_knn_dist"] = np.nan
            sub["thr"] = np.nan
            sub["is_removed"] = False
            sub["shape_type"] = "too_small"
            keep_list.append(sub)
            stats.append({
                "cluster": cl, "n": n, "shape_type": "too_small", "shape_reason": "too_small",
                "bbox_area": np.nan, "k": k, "mad_k": np.nan, "threshold": np.nan,
                "kept_before_thin": n, "kept": n, "removed": 0, "removed_ratio": 0.0,
            })
            continue

        coords = sub[[x_col, y_col]].to_numpy()
        bb = robust_bbox_metrics(coords)
        bbox_area = bb["bbox_area"]

        if cl_str in force_detail_clusters:
            is_detail = True
            reason = "forced_detail"
        elif cl_str in force_normal_clusters:
            is_detail = False
            reason = "forced_normal"
        else:
            is_detail = (cl_str in detail_clusters)
            reason = "area_only"

        shape_type = "detail" if is_detail else "normal"
        mad_k = cfg.get("mad_k", detail_mad_k if is_detail else normal_mad_k)
        grid_size = cfg.get("grid_size", grid_size_detail if is_detail else grid_size_normal)

        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=k + 1)
        avg_dists = np.mean(dists[:, 1:], axis=1)

        med = np.median(avg_dists)
        mad = np.median(np.abs(avg_dists - med))
        thr = float(np.quantile(avg_dists, 0.90)) if mad == 0 else float(med + mad_k * mad)
        keep_mask = avg_dists <= thr

        sub["avg_knn_dist"] = avg_dists
        sub["thr"] = thr
        sub["is_removed"] = ~keep_mask
        sub["shape_type"] = shape_type
        sub["shape_reason"] = reason
        sub["bbox_area"] = bbox_area

        sub_keep = sub[keep_mask].copy()
        sub_rm = sub[~keep_mask].copy()
        kept_before_thin = len(sub_keep)

        if apply_grid_thin and len(sub_keep) > 0 and grid_size is not None:
            sub_keep2 = grid_thin_points(sub_keep, x_col, y_col, grid_size=grid_size)
            thinned_out_idx = sub_keep.index.difference(sub_keep2.index)
            if len(thinned_out_idx) > 0:
                extra_rm = sub_keep.loc[thinned_out_idx].copy()
                extra_rm["is_removed"] = True
                sub_rm = pd.concat([sub_rm, extra_rm], axis=0)
            sub_keep = sub_keep2

        keep_list.append(sub_keep)
        rm_list.append(sub_rm)

        stats.append({
            "cluster": cl,
            "n": n,
            "shape_type": shape_type,
            "shape_reason": reason,
            "bbox_area": bbox_area,
            "k": k,
            "mad_k": mad_k,
            "threshold": thr,
            "kept_before_thin": kept_before_thin,
            "kept": len(sub_keep),
            "removed": len(sub_rm),
            "removed_ratio": len(sub_rm) / n,
        })

    df_keep = pd.concat(keep_list, axis=0) if keep_list else df.iloc[0:0].copy()
    df_removed = pd.concat(rm_list, axis=0) if rm_list else df.iloc[0:0].copy()
    stats_df = pd.DataFrame(stats).sort_values(["removed_ratio", "removed"], ascending=False)
    return df_keep, df_removed, stats_df


def build_mask_for_cluster(sub, x_col, y_col, h, w, phys_to_pix_array, params):
    xi, yi = phys_to_pix_array(sub[x_col].to_numpy(), sub[y_col].to_numpy())

    soft, hard = rasterize_points_soft(
        yi, xi, h, w,
        sigma=params["SIGMA_FIX"],
        bin_thr=params["BIN_THR"],
        close_radius=params["CLOSE_R"],
        fill_holes=params["FILL_HOLES"],
        auto_sigma=params["AUTO_SIG"],
        sigma_min=params["SIGMA_MIN"],
        sigma_max=params["SIGMA_MAX"],
        sigma_scale=params["SIGMA_SCL"]
    )

    if hard.sum() == 0:
        return soft, hard

    raw_area = int(hard.sum())
    dyn_smooth = float(np.clip(np.sqrt(raw_area) / 15.0, 1.0, params["REF_SMOOTH_MAX"]))

    mask = refine_edges_morph(
        hard,
        close_r=params["REF_CLOSE_R"],
        open_r=params["REF_OPEN_R"],
        smooth=dyn_smooth,
        keep_largest=params["KEEP_LARGEST"],
        fill_holes=params["FILL_HOLES"]
    )

    mask = remove_small_objects(mask.astype(bool), min_size=int(params["MIN_SIZE"])).astype(np.uint8)
    return soft, mask


def build_cluster_masks(
    df_smooth,
    sl,
    xJ,
    yJ,
    x_col="x_aligned",
    y_col="y_aligned",
    label_col="banksy_cluster_refined",
    params=None,
    params_thin=None,
    shape_type_col="shape_type",
    thin_values=("detail",),
    thin_rule="mode",                   # "mode" or "any"
    cluster_params_normal=None,          # e.g. {"7": {"BIN_THR":0.06}}
    cluster_params_thin=None,            # e.g. {"3": {"MIN_SIZE":120}}
    verbose=True
):
    if params is None:
        params = globals().get("DEFAULT_MASK_PARAMS", {}).copy() or dict(
            BIN_THR=0.08, SIGMA_FIX=1.8, AUTO_SIG=True, SIGMA_MIN=1.0, SIGMA_MAX=5.0,
            SIGMA_SCL=1.0, CLOSE_R=2, REF_CLOSE_R=2, REF_OPEN_R=1, REF_SMOOTH_MAX=1.2,
            KEEP_LARGEST=False, MIN_SIZE=120, FILL_HOLES=True
        )
    if params_thin is None:
        params_thin = params
    if cluster_params_normal is None:
        cluster_params_normal = {}
    if cluster_params_thin is None:
        cluster_params_thin = {}

    H, W = sl.shape
    phys_to_pix_array = make_phys_to_pix(xJ, yJ, H, W)

    st_soft, st_masks, rows = {}, {}, []
    clusters = pd.Categorical(df_smooth[label_col]).categories

    for cl in clusters:
        cl_str = str(cl)
        sub = df_smooth[df_smooth[label_col] == cl]
        if len(sub) == 0:
            continue

        is_detail = False
        if shape_type_col in sub.columns:
            stype = sub[shape_type_col].astype(str)
            if thin_rule == "any":
                is_detail = stype.isin(thin_values).any()
            else:
                mode_vals = stype.mode(dropna=True)
                is_detail = (len(mode_vals) > 0) and (str(mode_vals.iloc[0]) in thin_values)

        base_params = params_thin if is_detail else params
        params_use = dict(base_params)
        override_map = cluster_params_thin if is_detail else cluster_params_normal
        override = override_map.get(cl_str, override_map.get(cl, {}))
        if override:
            params_use.update(override)

        shape_type_used = "detail" if is_detail else "normal"

        soft, mask = build_mask_for_cluster(
            sub,
            x_col=x_col,
            y_col=y_col,
            h=H,
            w=W,
            phys_to_pix_array=phys_to_pix_array,
            params=params_use,
        )

        if mask.sum() == 0:
            continue

        stt = cc_stats(mask)
        if (stt["n_cc"] >= 3) or (stt["largest_frac"] < 0.75):
            min_size = int(max(200, 0.002 * mask.sum()))
            mask = remove_small_objects(mask.astype(bool), min_size=min_size).astype(np.uint8)

            mask = keep_cc_by_cumfrac(
                mask,
                cumfrac=0.70,
                max_k=6,
                min_area=int(max(80, 0.001 * mask.sum()))
            )

            stt2 = cc_stats(mask)
            if verbose:
                print(
                    f"[mask cleanup] cluster={cl_str} "
                    f"n_cc {stt['n_cc']}→{stt2['n_cc']} "
                    f"largest% {stt['largest_frac']:.2f}→{stt2['largest_frac']:.2f}"
                )

            if mask.sum() == 0:
                continue

        st_soft[cl_str] = soft
        st_masks[cl_str] = mask

        rows.append({
            "cluster": cl_str,
            "mask_area": int(mask.sum()),
            "n_points": int(len(sub)),
            "shape_type_used": shape_type_used,
        })

    mask_df = pd.DataFrame(rows)
    if len(mask_df) > 0:
        mask_df = mask_df.sort_values("mask_area", ascending=False).reset_index(drop=True)

    if verbose and len(mask_df) > 0:
        print(mask_df.groupby("shape_type_used")["cluster"].count().rename("n_clusters"))

    return {
        "st_soft": st_soft,
        "st_masks": st_masks,
        "mask_df": mask_df,
        "phys_to_pix_array": phys_to_pix_array
    }


def cc_stats_u8(mask_u8, connectivity=2):
    m = mask_u8.astype(bool)
    if not np.any(m):
        return dict(n_cc=0, largest_frac=0.0, max_cc=0, total=0)
    lbl = cc_label(m, connectivity=connectivity)
    cnt = np.bincount(lbl.ravel())
    cnt[0] = 0
    total = int(cnt.sum())
    max_cc = int(cnt.max()) if total > 0 else 0
    largest_frac = float(max_cc / (total + 1e-12))
    n_cc = int((cnt > 0).sum())
    return dict(n_cc=n_cc, largest_frac=largest_frac, max_cc=max_cc, total=total)


def plot_all_st_masks(
    st_masks,
    cols=6,
    max_rows_per_page=6,     # 每页最多 rows（rows*cols 张）
    connectivity=2,
    sort_key="cluster",      # "cluster" or "area" or "n_cc" or "largest"
    only_bad=False,
    bad_cc_thr=5,
    bad_largest_thr=0.85,
    figsize_per_cell=2.8,
):
    # ---- collect stats ----
    items = []
    for k, m in st_masks.items():
        mu8 = m.astype(np.uint8)
        s = cc_stats_u8(mu8, connectivity=connectivity)
        items.append(dict(
            cluster=str(k),
            mask=mu8,
            area=int(mu8.sum()),
            n_cc=int(s["n_cc"]),
            largest=float(s["largest_frac"])
        ))

    # ---- filter bad if needed ----
    if only_bad:
        items = [it for it in items if (it["n_cc"] >= bad_cc_thr) or (it["largest"] < bad_largest_thr)]

    if len(items) == 0:
        print("[INFO] nothing to plot (after filtering).")
        return

    # ---- sorting ----
    if sort_key == "area":
        items.sort(key=lambda x: -x["area"])
    elif sort_key == "n_cc":
        items.sort(key=lambda x: (-x["n_cc"], x["largest"]))
    elif sort_key == "largest":
        items.sort(key=lambda x: x["largest"])
    else:  # "cluster"
        def _try_int(s):
            try: return int(s)
            except: return s
        items.sort(key=lambda x: _try_int(x["cluster"]))

    # ---- paging ----
    per_page = cols * max_rows_per_page
    n_pages = int(np.ceil(len(items) / per_page))

    for p in range(n_pages):
        sub = items[p*per_page:(p+1)*per_page]
        n = len(sub)
        rows = int(np.ceil(n / cols))

        fig_w = cols * figsize_per_cell
        fig_h = rows * figsize_per_cell
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(-1)

        for ax in axes[n:]:
            ax.axis("off")

        for i, it in enumerate(sub):
            ax = axes[i]
            ax.imshow(it["mask"], cmap="gray")
            ax.set_title(
                f"cl={it['cluster']}\n"
                f"A={it['area']} cc={it['n_cc']}\n"
                f"largest%={it['largest']:.2f}",
                fontsize=9
            )
            ax.axis("off")

        fig.suptitle(f"st_masks | page {p+1}/{n_pages} | total={len(items)}", fontsize=14)
        plt.tight_layout()
        plt.show()

    print("[DONE] plotted", len(items), "clusters",
          "(only_bad=True)" if only_bad else "")


def drop_islands_by_fraction(m_bool, frac_keep=0.05):
    lab, n = ndimage.label(m_bool)
    if n <= 1:
        return m_bool
    sizes = ndimage.sum(m_bool, lab, index=np.arange(1, n + 1))
    max_area = float(sizes.max())
    thr = max_area * float(frac_keep)
    keep_ids = 1 + np.where(sizes >= thr)[0]
    return np.isin(lab, keep_ids)


def drop_islands_topk(m_bool, top_k=2):
    lab, n = ndimage.label(m_bool)
    if n <= top_k:
        return m_bool
    sizes = ndimage.sum(m_bool, lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep_ids = 1 + order[:int(top_k)]
    return np.isin(lab, keep_ids)


def masks_from_label_image(label_img, bg_val=-1, min_area=80):
    lab = np.asarray(label_img)
    if np.issubdtype(lab.dtype, np.floating):
        valid = np.isfinite(lab)
    else:
        valid = (lab != bg_val)

    ids = sorted([int(c) for c in np.unique(lab[valid])])
    out = {}
    for c in ids:
        m = (lab == c).astype(np.uint8)
        if m.sum() >= int(min_area):
            out[c] = m
    return out


def smooth_cluster_mask(
    mask_u8,
    close_r=4,
    open_r=1,
    fill_holes=True,
    smooth_sigma=0.8,
    min_area=120,
    keep_largest=False,
    island_mode="fraction",   # "fraction" | "topk" | "none"
    frac_keep=0.25,
    top_k=2,
    smooth_thr=0.5
):
    m = (mask_u8 > 0)

    # 1) drop tiny fragments first
    if min_area and min_area > 0:
        lab, n = ndimage.label(m)
        if n > 0:
            sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
            keep_ids = 1 + np.where(sizes >= int(min_area))[0]
            m = np.isin(lab, keep_ids)

    # 2) morphology (cv2, same style as reference)
    m_u8 = (m.astype(np.uint8) * 255)
    if close_r and close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, k)
    if open_r and open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1))
        m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, k)

    m = (m_u8 > 0)

    # 3) fill holes
    if fill_holes:
        m = ndimage.binary_fill_holes(m)

    # 4) gaussian boundary smoothing
    if smooth_sigma and smooth_sigma > 0:
        f = ndimage.gaussian_filter(m.astype(np.float32), float(smooth_sigma))
        m = (f >= float(smooth_thr))

    # 5) island cleanup
    if island_mode == "fraction":
        m = drop_islands_by_fraction(m, frac_keep=float(frac_keep))
    elif island_mode == "topk":
        m = drop_islands_topk(m, top_k=int(top_k))
    elif island_mode in (None, "none", False):
        pass
    else:
        raise ValueError(f"Unknown island_mode: {island_mode}")

    # 6) optional largest CC
    if keep_largest:
        lab, n = ndimage.label(m)
        if n > 0:
            sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
            keep = 1 + int(np.argmax(sizes))
            m = (lab == keep)

    return m.astype(np.uint8)


def build_he_masks_from_labels(
    labels_img,                 # labels_merged (preferred) or labels_merged_nobg
    bg_val=-1,
    min_area_raw=80,
    close_r=4,
    open_r=1,
    smooth_sigma=0.8,
    smooth_thr=0.5,
    min_area_clean=120,
    keep_largest=False,
    island_mode="fraction",
    frac_keep=0.25,
    top_k=2
):
    he_raw_masks = masks_from_label_image(labels_img, bg_val=bg_val, min_area=min_area_raw)

    he_masks = {}
    for cid, m in he_raw_masks.items():
        mc = smooth_cluster_mask(
            m,
            close_r=close_r,
            open_r=open_r,
            fill_holes=True,
            smooth_sigma=smooth_sigma,
            min_area=min_area_clean,
            keep_largest=keep_largest,
            island_mode=island_mode,
            frac_keep=frac_keep,
            top_k=top_k,
            smooth_thr=smooth_thr
        )
        if mc.sum() > 0:
            he_masks[cid] = mc

    he_soft = {}
    for cid, m in he_masks.items():
        s = ndimage.gaussian_filter(m.astype(np.float32), sigma=max(0.5, smooth_sigma))
        if s.max() > 0:
            s /= s.max()
        he_soft[cid] = s

    he_df = pd.DataFrame(
        [{"cluster": cid, "area": int(m.sum())} for cid, m in he_masks.items()]
    ).sort_values("area", ascending=False).reset_index(drop=True)

    return he_masks, he_soft, he_df, he_raw_masks


def plot_masks_grid(masks, cols=6, max_show=36, title_prefix="HE "):
    ks = sorted(masks.keys(), key=lambda k: masks[k].sum(), reverse=True)[:max_show]
    rows = int(np.ceil(len(ks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.array(axes).ravel()
    for i, k in enumerate(ks):
        axes[i].imshow(masks[k], cmap="gray", interpolation="nearest")
        axes[i].set_title(f"{title_prefix}{k}\nA={int(masks[k].sum())}")
        axes[i].axis("off")
    for j in range(len(ks), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


def boundary_1px(mask):
    m = mask.astype(bool)
    if not m.any():
        return m
    er = ndimage.binary_erosion(m, structure=np.ones((3,3), dtype=bool))
    return m ^ er


def surface_distance_metrics(mask1, mask2):
    b1 = boundary_1px(mask1)
    b2 = boundary_1px(mask2)
    if not (b1.any() and b2.any()):
        return np.inf, np.inf

    dt1 = ndimage.distance_transform_edt(~b1)
    dt2 = ndimage.distance_transform_edt(~b2)

    d12 = dt2[b1]
    d21 = dt1[b2]
    if d12.size == 0 or d21.size == 0:
        return np.inf, np.inf

    all_d = np.concatenate([d12, d21])
    asd = float(all_d.mean())
    hd = float(all_d.max())
    return asd, hd


def asd_to_sim(asd, d0=15.0):
    if not np.isfinite(asd):
        return 0.0
    return float(np.exp(-asd / (d0 + 1e-8)))


def dice_mask(a, b, eps=1e-12):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    den = a.sum() + b.sum()
    return float(2.0 * inter / (den + eps)) if den > 0 else 0.0


def iou_mask(a, b, eps=1e-12):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / (uni + eps)) if uni > 0 else 0.0


def area_sim_from_masks(a, b, eps=1e-9):
    aa = float(np.sum(a)) + eps
    bb = float(np.sum(b)) + eps
    return float(np.exp(-abs(np.log(aa / bb))))


def sdf_corr_band(mask1, mask2, band=20, clip=None):
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    if not (m1.any() and m2.any()):
        return 0.0

    def sdf(m):
        din = ndimage.distance_transform_edt(m)
        dout = ndimage.distance_transform_edt(~m)
        s = din - dout
        if clip is not None:
            s = np.clip(s, -clip, clip)
        return s

    s1 = sdf(m1)
    s2 = sdf(m2)

    band_mask = (np.abs(s1) <= band) | (np.abs(s2) <= band)
    v1 = s1[band_mask].ravel()
    v2 = s2[band_mask].ravel()

    if v1.size < 10 or v1.std() == 0 or v2.std() == 0:
        return 0.0

    corr = np.corrcoef(v1, v2)[0, 1]
    return float((corr + 1.0) / 2.0)


def boundary_1px(mask):
    m = mask.astype(bool)
    if not m.any():
        return m
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool))
    return m ^ er


def chamfer_similarity(mask1, mask2, d0=30.0):
    b1 = boundary_1px(mask1)
    b2 = boundary_1px(mask2)
    if b1.sum() == 0 or b2.sum() == 0:
        return 0.0, np.inf

    edt1 = ndimage.distance_transform_edt(~b1)
    edt2 = ndimage.distance_transform_edt(~b2)

    d12 = float(edt2[b1].mean())
    d21 = float(edt1[b2].mean())
    avgd = 0.5 * (d12 + d21)

    sim = float(np.exp(-avgd / (d0 + 1e-8)))
    return sim, avgd


def compute_all_metrics(mask_st, mask_he):
    chamfer_sim, chamfer_dist = chamfer_similarity(mask_st, mask_he, d0=30.0)

    asd, hd = surface_distance_metrics(mask_st, mask_he)
    asd_sim = asd_to_sim(asd, d0=15.0)

    return {
        "dice": dice_mask(mask_st, mask_he),
        "iou": iou_mask(mask_st, mask_he),
        "area_sim": area_sim_from_masks(mask_st, mask_he),
        "sdf_corr": sdf_corr_band(mask_st, mask_he, band=20, clip=None),
        "chamfer_sim": chamfer_sim,
        "chamfer_dist": chamfer_dist,
        "asd": asd,
        "hd": hd,
        "asd_sim": asd_sim,
    }


def align_score(metrics, weights=None):
    if weights is None:
        weights = W_ALIGN
    s = 0.0
    for k, w in weights.items():
        s += float(w) * float(metrics.get(k, 0.0))
    return float(np.clip(s, 0.0, 1.0))


def all_pair_scores(st_masks, he_masks, min_intersection=20, weights=None):
    rows = []
    for st_id, ms in st_masks.items():
        msb = ms.astype(bool)
        area_st = int(msb.sum())
        if area_st == 0:
            continue
        for he_id, mh in he_masks.items():
            mhb = mh.astype(bool)
            inter = int(np.logical_and(msb, mhb).sum())
            if inter < min_intersection:
                continue
            met = compute_all_metrics(msb, mhb)
            score = align_score(met, weights=weights)
            rows.append({
                "st": st_id, "he": he_id, "align_score": score,
                "intersection": inter, "area_st": area_st, "area_he": int(mhb.sum()),
                **met
            })
    return pd.DataFrame(rows).sort_values("align_score", ascending=False).reset_index(drop=True)


def greedy_nonoverlap_selection(df_pairs, max_pairs=None):
    used_st, used_he, keep = set(), set(), []
    for _, r in df_pairs.sort_values("align_score", ascending=False).iterrows():
        s, h = r["st"], r["he"]
        if s in used_st or h in used_he:
            continue
        keep.append(r)
        used_st.add(s)
        used_he.add(h)
        if max_pairs is not None and len(keep) >= max_pairs:
            break
    return pd.DataFrame(keep).reset_index(drop=True)


def two_to_one_scores(st_masks, he_masks, df_pairs, topk_per_he=5, weights=None):
    rows = []

    # 按 HE 分组（只考虑top candidate，避免爆炸）
    for he_id, df_he in df_pairs.groupby("he"):
        df_he = df_he.sort_values("align_score", ascending=False).head(topk_per_he)

        st_list = df_he["st"].tolist()

        for s1, s2 in combinations(st_list, 2):
            m1 = st_masks[s1]
            m2 = st_masks[s2]
            mh = he_masks[he_id]

            union_mask = np.logical_or(m1, m2)

            inter = np.logical_and(union_mask, mh).sum()
            if inter < 20:
                continue

            met = compute_all_metrics(union_mask, mh)
            score = align_score(met, weights=weights)

            rows.append({
                "st": f"{s1}+{s2}",
                "he": he_id,
                "align_score": score,
                "type": "2to1",
                **met
            })

    return pd.DataFrame(rows).sort_values("align_score", ascending=False)


def union_masks(mask_list):
    out = np.zeros_like(mask_list[0], dtype=bool)
    for m in mask_list:
        out |= m.astype(bool)
    return out.astype(np.uint8)


def plot_match_pairs(df_match, st_masks, he_masks, k=12, ncols=4):
    d = df_match.head(k).copy()
    n = len(d)
    if n == 0:
        print("No matched pairs to plot.")
        return

    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for i, (_, r) in enumerate(d.iterrows()):
        ax = axes[i // ncols][i % ncols]

        s = r["st"]
        h = r["he"]
        ms = st_masks[s].astype(bool)
        mh = he_masks[h].astype(bool)

        # base HE
        ax.imshow(mh, cmap="gray", alpha=0.35)
        # ST contour (red) + HE contour (cyan)
        ax.contour(mh.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)
        ax.contour(ms.astype(float), levels=[0.5], colors="red", linewidths=1.0)

        title = f"ST {s} ↔ HE {h}\nS={r['align_score']:.3f} D={r['dice']:.2f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def onehot_to_sdt(onehot, clip_dist=60.0, sigma_sdt=1.2):
    C, H, W = onehot.shape
    sdt = np.zeros((C, H, W), np.float32)
    for c in range(C):
        m = onehot[c] > 0
        if not m.any():
            sdt[c] = float(clip_dist)
            continue
        din = edt(m.astype(np.uint8))
        dout = edt((~m).astype(np.uint8))
        d = dout - din
        if sigma_sdt > 0:
            d = gaussian_filter(d, sigma=float(sigma_sdt))
        d = np.clip(d, -clip_dist, clip_dist).astype(np.float32)
        sdt[c] = d / float(clip_dist)
    return sdt


def preprocess_st_mask_channel(
    m,
    sigma_pre=1.4,
    thr=0.5,
    close_r=2,
    open_r=1,
    min_area=50,
):
    # ST masks are point-derived; stronger smoothing improves SDT stability.
    m = m.astype(bool)
    if min_area > 0:
        m = remove_small_objects(m, min_size=int(min_area))
    if close_r > 0:
        m = binary_closing(m, disk(int(close_r)))
    if open_r > 0:
        m = binary_opening(m, disk(int(open_r)))
    f = gaussian_filter(m.astype(np.float32), sigma=float(sigma_pre))
    return (f > float(thr)).astype(np.uint8)


def preprocess_he_mask_channel(
    m,
    sigma_pre=0.4,
    thr=0.5,
    close_r=1,
    open_r=1,
    min_area=80,
):
    # HE masks are image-derived; use light denoise to preserve boundaries.
    m = m.astype(bool)
    if min_area > 0:
        m = remove_small_objects(m, min_size=int(min_area))
    if close_r > 0:
        m = binary_closing(m, disk(int(close_r)))
    if open_r > 0:
        m = binary_opening(m, disk(int(open_r)))
    f = gaussian_filter(m.astype(np.float32), sigma=float(sigma_pre))
    return (f > float(thr)).astype(np.uint8)


def preprocess_onehot_asymmetric(
    I_onehot,
    J_onehot,
    st_cfg=None,
    he_cfg=None,
):
    st_cfg = st_cfg or {}
    he_cfg = he_cfg or {}

    I_clean = np.zeros_like(I_onehot, dtype=np.uint8)
    J_clean = np.zeros_like(J_onehot, dtype=np.uint8)

    for c in range(I_onehot.shape[0]):
        I_clean[c] = preprocess_st_mask_channel(I_onehot[c], **st_cfg)
        J_clean[c] = preprocess_he_mask_channel(J_onehot[c], **he_cfg)

    return I_clean, J_clean


def channel_weights_from_area(onehot, power=0.9, w_min=0.25, w_max=4.5, eps=1e-6):
    area = onehot.reshape(onehot.shape[0], -1).sum(axis=1).astype(np.float32)
    w = 1.0 / np.power(area + eps, power)
    w = w / (w.mean() + eps)
    w = np.clip(w, w_min, w_max)
    w = w / (w.mean() + eps)
    return w.astype(np.float32)


def apply_channel_weights(X, w):
    return X * w[:, None, None]


def _norm_id(x):
    s = str(x).strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except:
        return s


def _get_mask(d, key):
    if key in d: return d[key]
    sk = str(key)
    if sk in d: return d[sk]
    try:
        ik = int(float(key))
        if ik in d: return d[ik]
        sik = str(ik)
        if sik in d: return d[sik]
    except:
        pass
    return None


def _as_list(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    return [v]


def build_pair_onehot_from_masks(df_pairs, he_masks_dict, st_masks_dict, add_other=False):
    he_masks = {_norm_id(k): (v > 0).astype(np.uint8) for k, v in he_masks_dict.items()}
    st_masks = {_norm_id(k): (v > 0).astype(np.uint8) for k, v in st_masks_dict.items()}

    H, W = next(iter(he_masks.values())).shape
    rows = []

    for _, r in df_pairs.reset_index(drop=True).iterrows():
        if "st_group" in r and "he_group" in r:
            st_ids = [_norm_id(x) for x in _as_list(r["st_group"])]
            he_ids = [_norm_id(x) for x in _as_list(r["he_group"])]
        else:
            st_ids = [_norm_id(r["st"])]
            he_ids = [_norm_id(r["he"])]

        # keep only valid ids
        st_ids = [x for x in st_ids if _get_mask(st_masks, x) is not None]
        he_ids = [x for x in he_ids if _get_mask(he_masks, x) is not None]
        if len(st_ids) == 0 or len(he_ids) == 0:
            continue
        rows.append((st_ids, he_ids))

    C = len(rows) + (1 if add_other else 0)
    OTHER_CH = len(rows) if add_other else None

    I_onehot = np.zeros((C, H, W), np.uint8)
    J_onehot = np.zeros((C, H, W), np.uint8)

    used_st, used_he = set(), set()
    for ch, (st_ids, he_ids) in enumerate(rows):
        for sid in st_ids:
            I_onehot[ch] |= _get_mask(st_masks, sid)
            used_st.add(sid)
        for hid in he_ids:
            J_onehot[ch] |= _get_mask(he_masks, hid)
            used_he.add(hid)

    if add_other:
        st_other = np.zeros((H, W), np.uint8)
        he_other = np.zeros((H, W), np.uint8)
        for k, m in st_masks.items():
            if k not in used_st:
                st_other |= m
        for k, m in he_masks.items():
            if k not in used_he:
                he_other |= m
        I_onehot[OTHER_CH] = st_other
        J_onehot[OTHER_CH] = he_other

    return I_onehot, J_onehot, {"C": C, "H": H, "W": W, "OTHER_CH": OTHER_CH}


def _area_balance_weights(source_onehot, target_onehot, power=0.8, w_min=0.5, w_max=2.5, eps=1e-6):
    aS = source_onehot.reshape(source_onehot.shape[0], -1).sum(axis=1).astype(np.float64)
    aT = target_onehot.reshape(target_onehot.shape[0], -1).sum(axis=1).astype(np.float64)
    area = 0.5 * (aS + aT) + eps
    w = (np.median(area) / area) ** power
    w = np.clip(w, w_min, w_max).astype(np.float32)
    return w


def _build_equalized_sdt(source_onehot, target_onehot, clip_dist=4.0, sigma_sdt=0.9, band=4.0, tau=2.0):
    source_sdt = onehot_to_sdt(source_onehot, clip_dist=clip_dist, sigma_sdt=sigma_sdt).astype(np.float32)
    target_sdt = onehot_to_sdt(target_onehot, clip_dist=clip_dist, sigma_sdt=sigma_sdt).astype(np.float32)

    for c in range(source_sdt.shape[0]):
        m = (np.abs(source_sdt[c]) <= band) | (np.abs(target_sdt[c]) <= band)
        if m.any():
            s = np.sqrt(0.5 * (np.mean(source_sdt[c][m] ** 2) + np.mean(target_sdt[c][m] ** 2))) + 1e-6
            source_sdt[c] /= s
            target_sdt[c] /= s

    source_sdt = np.tanh(source_sdt / tau)
    target_sdt = np.tanh(target_sdt / tau)
    return source_sdt, target_sdt


def _torch_dtype_from_name(name):
    if torch is None:
        raise ImportError("PyTorch is required for LDDMM dtype selection.")
    if name in (torch.float64, "float64", "double"):
        return torch.float64
    if name in (torch.float32, "float32", "single"):
        return torch.float32
    raise ValueError(f"Unsupported dtype setting: {name!r}")


def _union_mask(mask_dict, shape=None):
    masks = [(m > 0) for m in mask_dict.values()]
    if len(masks) == 0:
        if shape is None:
            raise ValueError("Cannot build a union mask from an empty mask dictionary.")
        return np.zeros(shape, dtype=np.uint8)
    out = np.zeros(masks[0].shape if shape is None else shape, dtype=bool)
    for m in masks:
        out |= m
    return out.astype(np.uint8)


def _overall_st_mask_from_points(df_points, x_col="x_prealigned", y_col="y_prealigned", shape=None):
    from scipy import ndimage as _ndi
    from skimage import morphology as _morphology

    if shape is None:
        shape = (H0, W0)
    h, w = shape
    if "phys_to_pix_array" in globals():
        xi, yi = phys_to_pix_array(df_points[x_col].to_numpy(float), df_points[y_col].to_numpy(float))
    else:
        _converter = make_phys_to_pix(
            globals().get("xJ", np.arange(w, dtype=float)),
            globals().get("yJ", np.arange(h, dtype=float)),
            h,
            w,
        )
        xi, yi = _converter(df_points[x_col].to_numpy(float), df_points[y_col].to_numpy(float))

    acc = np.zeros((h, w), dtype=np.float32)
    valid = np.isfinite(xi) & np.isfinite(yi)
    if np.any(valid):
        np.add.at(acc, (yi[valid], xi[valid]), 1.0)
    acc = _ndi.gaussian_filter(acc, sigma=3.0)
    if acc.max() > 0:
        acc = acc / acc.max()
    mask = acc >= 0.015
    mask = _morphology.binary_closing(mask, _morphology.disk(8))
    mask = _ndi.binary_fill_holes(mask)
    mask = _morphology.remove_small_objects(mask.astype(bool), min_size=500)
    return mask.astype(np.uint8)


def _overall_he_mask_from_slide(shape=None):
    from scipy import ndimage as _ndi
    from skimage import morphology as _morphology

    if shape is None:
        shape = (H0, W0)
    if "he_mask" in globals() and np.asarray(he_mask).shape == tuple(shape):
        mask = np.asarray(he_mask) > 0
    else:
        mask = _union_mask(he_masks, shape=shape) > 0
    mask = _morphology.binary_closing(mask, _morphology.disk(3))
    mask = _ndi.binary_fill_holes(mask)
    mask = _morphology.remove_small_objects(mask.astype(bool), min_size=500)
    return mask.astype(np.uint8)


def transform_points_source_to_target(xv, v, A, pointsI):
    """Compatibility wrapper around shared lddmm_alignment.map_points_source_to_target."""
    return map_points_source_to_target(xv, v, A, pointsI)
