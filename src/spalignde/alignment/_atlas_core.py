#!/usr/bin/env python3
"""Internal implementation of structure-guided ST-to-Allen-CCF alignment.

The public typed interface lives in :mod:`spalignde.alignment.atlas`. This
module retains the validated research implementation and its coarse-to-fine
stage logic while keeping package users independent of notebook files and
developer-specific paths.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import nrrd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import cv2
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import re
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import morphology, measure
from skimage.morphology import remove_small_objects
from skimage.measure import label as cc_label
import torch
from scipy.ndimage import distance_transform_edt as edt, gaussian_filter, zoom
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects
import time
import torch, numpy as np
from torch.nn.functional import grid_sample
import torch.fft

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)




@dataclass
class STAtlasConfig:
    """Parameters for the ST-to-Atlas alignment tutorial pipeline."""

    st_cluster_csv: Path
    st_counts_csv: Path
    atlas_voxel_csv: Path
    atlas_nrrd: Path
    atlas_slice_z: int = 675
    output_dir: Path = Path("iterative_alignment_outputs")
    levels_csv: Path = Path("banksy_clusters_single_levels.csv")

    n_levels: int = 4
    var_frac: float = 0.8
    min_genes: int = 50
    cluster_col: str = "banksy_cluster_refined"
    stage_labels: list[str] | None = None

    continue_alignment: bool = True
    continue_max_iter: int = 10
    continue_min_pair_gain: int = 1
    device: str | None = None
    dtype: Any | None = None

    prealign_close_ksize: int = 15
    prealign_angle_step_deg: float = 1.0
    prealign_scale_tweak: float = 0.05
    prealign_scale_steps: int = 2

    filter_base_k: int = 20
    filter_detail_area_quantile: float = 0.40
    filter_area_mode: str = "bbox"
    filter_detail_mad_k: float = 1.0
    filter_normal_mad_k: float = 1.2
    filter_apply_grid_thin: bool = True
    filter_grid_size_detail: float = 10
    filter_grid_size_normal: float | None = None
    force_detail_clusters: set[str] = field(default_factory=set)
    force_normal_clusters: set[str] = field(default_factory=set)


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def validate_config(config: STAtlasConfig) -> STAtlasConfig:
    """Validate required inputs and create output directories."""
    required_files = {
        "st_cluster_csv": config.st_cluster_csv,
        "st_counts_csv": config.st_counts_csv,
        "atlas_voxel_csv": config.atlas_voxel_csv,
        "atlas_nrrd": config.atlas_nrrd,
    }
    for name, value in required_files.items():
        path = _as_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
        setattr(config, name, path)

    config.output_dir = _as_path(config.output_dir)
    config.levels_csv = _as_path(config.levels_csv)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.levels_csv.parent.mkdir(parents=True, exist_ok=True)
    return config


def build_refined_st_levels(config: STAtlasConfig):
    """Build coarse-to-fine ST cluster columns for iterative alignment."""
    return build_st_refined_clusters(
        meta_path=config.st_cluster_csv,
        counts_path=config.st_counts_csv,
        out_path=config.levels_csv,
        n_levels=config.n_levels,
        var_frac=config.var_frac,
        min_genes=config.min_genes,
        drop_blank=True,
        cluster_col=config.cluster_col,
    )


def load_atlas_inputs(config: STAtlasConfig, plot: bool = True):
    """Load atlas metadata table and one atlas annotation slice."""
    df2 = pd.read_csv(config.atlas_voxel_csv, index_col=0)
    atlas_info = load_atlas_slice(
        atlas_nrrd_path=str(config.atlas_nrrd),
        atlas_slice_z=config.atlas_slice_z,
        plot=plot,
    )
    return df2, atlas_info


def prealign_st_to_atlas(df_smooth, atlas_info, config: STAtlasConfig):
    """Globally pre-align ST coordinates to the atlas mask by maximizing IoU."""
    global resolution
    sl = atlas_info["sl"]
    xJ = atlas_info["xJ"]
    yJ = atlas_info["yJ"]
    H, W = sl.shape
    resolution_x = (float(xJ[-1]) - float(xJ[0])) / max(W, 1)
    resolution_y = (float(yJ[-1]) - float(yJ[0])) / max(H, 1)
    resolution = (resolution_x + resolution_y) / 2.0
    df_final, prealign_params = align_omics_no_flip_max_iou(
        df_smooth,
        sl,
        xJ,
        yJ,
        close_ksize=config.prealign_close_ksize,
        angle_step_deg=config.prealign_angle_step_deg,
        scale_tweak=config.prealign_scale_tweak,
        scale_steps=config.prealign_scale_steps,
    )
    return df_final, prealign_params


def filter_prealigned_st_points(df_final, label_col: str, config: STAtlasConfig):
    """Filter pre-aligned ST points before mask construction and pair matching."""
    return filter_cluster_for_mask(
        df_final,
        label_col=label_col,
        x_col="x_prealigned",
        y_col="y_prealigned",
        base_k=config.filter_base_k,
        detail_area_quantile=config.filter_detail_area_quantile,
        area_mode=config.filter_area_mode,
        detail_mad_k=config.filter_detail_mad_k,
        normal_mad_k=config.filter_normal_mad_k,
        apply_grid_thin=config.filter_apply_grid_thin,
        grid_size_detail=config.filter_grid_size_detail,
        grid_size_normal=config.filter_grid_size_normal,
        force_detail_clusters=config.force_detail_clusters,
        force_normal_clusters=config.force_normal_clusters,
    )


def make_alignment_context(
    config: STAtlasConfig,
    df1,
    refined_cols,
    df2,
    atlas_info,
    df_smooth,
    df_final,
    df_final_filtered,
    prealign_params,
):
    """Create the shared context used by iterative notebook-derived functions."""
    context = {
        "config": config,
        "df1": df1,
        "refined_cols": refined_cols,
        "df2": df2,
        "atlas_info": atlas_info,
        "sl": atlas_info["sl"],
        "xJ": atlas_info["xJ"],
        "yJ": atlas_info["yJ"],
        "H": atlas_info["H"],
        "W": atlas_info["W"],
        "df_smooth": df_final_filtered.copy(),
        "df_final": df_final,
        "df_final_filtered": df_final_filtered,
        "df_prealign_nofilter": df_final.copy(),
        "prealign_params": prealign_params,
        "STAGE_LABELS": config.stage_labels,
        "CONTINUE_MAX_ITER": int(config.continue_max_iter),
        "CONTINUE_MIN_PAIR_GAIN": int(config.continue_min_pair_gain),
        "CONTINUE_RESULT_DIR": config.output_dir / "continue_alignment",
    }
    if config.device is not None:
        context["device"] = config.device
    if config.dtype is not None:
        context["dtype"] = config.dtype
    return context


def _exec_in_context(source: str, context: dict[str, Any], name: str) -> dict[str, Any]:
    """Execute notebook-derived source with module functions plus context variables."""
    env = {k: v for k, v in globals().items() if k not in {"__builtins__"}}
    env.update(context)
    exec(compile(source, name, "exec"), env)
    context.update({k: v for k, v in env.items() if not k.startswith("__")})
    return context


def run_iterative_multi_level_alignment(context: dict[str, Any]) -> dict[str, Any]:
    """Run the notebook's main coarse-to-fine iterative LDDMM alignment section."""
    return _exec_in_context(ITERATIVE_ALIGNMENT_CELL_SOURCE, context, "iterative_alignment")


def run_continuation_alignment(context: dict[str, Any]) -> dict[str, Any]:
    """Continue LDDMM alignment until matched-pair count stops increasing."""
    return _exec_in_context(CONTINUATION_ALIGNMENT_CELL_SOURCE, context, "continuation_alignment")


def save_alignment_visualization(context: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Save before/after ST-to-Atlas overlay visualizations."""
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = VISUALIZATION_CELL_SOURCE.replace(
        'atlas_result_dir = Path("atlas_result_output")',
        f'atlas_result_dir = Path({str(output_dir)!r})',
    )
    return _exec_in_context(source, context, "alignment_visualization")


def save_alignment_outputs(context: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Save aligned points, matched pairs, stage summaries, and manifest."""
    output_dir = _as_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = OUTPUT_CELL_SOURCE.replace(
        'out_dir = Path("iterative_alignment_outputs")',
        f'out_dir = Path({str(output_dir)!r})',
    )
    return _exec_in_context(source, context, "alignment_outputs")


def run_st_to_atlas_pipeline(config: STAtlasConfig, plot_atlas: bool = True) -> dict[str, Any]:
    """Run the full ST-to-Atlas tutorial pipeline from a config object."""
    config = validate_config(config)
    df1, ks, refined_cols, Z, avg_z = build_refined_st_levels(config)
    df2, atlas_info = load_atlas_inputs(config, plot=plot_atlas)
    df_smooth = df1.copy()
    df_final, prealign_params = prealign_st_to_atlas(df_smooth, atlas_info, config)

    label_col_filter = config.cluster_col
    if label_col_filter not in df_final.columns:
        raise ValueError(f"label column not found after prealignment: {label_col_filter}")
    df_keep, df_removed, filter_stats_df = filter_prealigned_st_points(
        df_final, label_col=label_col_filter, config=config
    )

    context = make_alignment_context(
        config=config,
        df1=df1,
        refined_cols=refined_cols,
        df2=df2,
        atlas_info=atlas_info,
        df_smooth=df_smooth,
        df_final=df_final,
        df_final_filtered=df_keep.copy(),
        prealign_params=prealign_params,
    )
    context.update({
        "ks": ks,
        "Z": Z,
        "avg_z": avg_z,
        "df_keep": df_keep,
        "df_removed": df_removed,
        "filter_stats_df": filter_stats_df,
        "label_col_filter": label_col_filter,
    })

    run_iterative_multi_level_alignment(context)
    if config.continue_alignment:
        run_continuation_alignment(context)
    save_alignment_visualization(context, config.output_dir)
    save_alignment_outputs(context, config.output_dir)
    return context


def load_alignment_result_tables(output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the standard ST-to-Atlas output CSV tables from an output folder."""
    output_dir = _as_path(output_dir)
    paths = {
        "coordinates": output_dir / "coordinates_original_prealign_aligned.csv",
        "final_aligned": output_dir / "final_aligned_all_points.csv",
        "final_filtered": output_dir / "final_filtered_points_for_matching.csv",
        "stage_summary": output_dir / "iterative_alignment_stage_summary.csv",
        "matched_pairs_final": output_dir / "matched_pairs_final_stage.csv",
        "matched_pairs_all": output_dir / "matched_pairs_all_stages.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ST-to-Atlas output files:\n" + "\n".join(missing))
    return {name: pd.read_csv(path) for name, path in paths.items()}


def summarize_alignment_outputs(output_dir: str | Path) -> dict[str, Any]:
    """Return compact metrics for documentation and smoke tests."""
    tables = load_alignment_result_tables(output_dir)
    stage_summary = tables["stage_summary"]
    matched_pairs_final = tables["matched_pairs_final"]
    matched_pairs_all = tables["matched_pairs_all"]
    coordinates = tables["coordinates"]
    final_aligned = tables["final_aligned"]
    final_filtered = tables["final_filtered"]

    summary = {
        "n_original_points": int(len(coordinates)),
        "n_final_aligned_points": int(len(final_aligned)),
        "n_filtered_points_for_matching": int(len(final_filtered)),
        "n_alignment_stages": int(len(stage_summary)),
        "n_final_matched_pairs": int(len(matched_pairs_final)),
        "n_all_stage_matched_pairs": int(len(matched_pairs_all)),
        "stage_pair_counts": stage_summary[["stage", "label_col", "n_pairs"]].to_dict("records"),
    }

    metrics_path = _as_path(output_dir) / "end_to_end_runtime_cuda_metrics.json"
    if metrics_path.is_file():
        with metrics_path.open() as fh:
            metrics = json.load(fh)
        for key in [
            "elapsed_seconds",
            "elapsed_minutes",
            "device_requested",
            "cuda_device_name",
            "peak_cuda_memory_allocated_gib",
        ]:
            if key in metrics:
                summary[key] = metrics[key]
    return summary


def validate_alignment_outputs(output_dir: str | Path) -> dict[str, Any]:
    """Validate standard output consistency and return the computed summary."""
    summary = summarize_alignment_outputs(output_dir)
    if summary["n_original_points"] != summary["n_final_aligned_points"]:
        raise ValueError(
            "Original and final aligned point counts differ: "
            f"{summary['n_original_points']} vs {summary['n_final_aligned_points']}"
        )
    if summary["n_alignment_stages"] < 1:
        raise ValueError("No alignment stages found in stage summary.")
    if summary["n_final_matched_pairs"] < 1:
        raise ValueError("No final matched cluster-to-atlas pairs found.")
    return summary


def transfer_atlas_labels_to_st(
    st_df: pd.DataFrame,
    matched_pairs_df: pd.DataFrame,
    *,
    cluster_col: str = "banksy_cluster_refined",
    unmatched_label: str = "unassigned",
) -> pd.DataFrame:
    """Transfer final cluster-to-atlas assignments back to each ST cell/spot.

    The Atlas pipeline assigns ST clusters to atlas region candidates. This
    helper expands that cluster-level assignment onto the final aligned ST cell
    table so downstream analysis can use one atlas label per ST observation.
    """
    if cluster_col not in st_df.columns:
        raise KeyError(f"ST table missing cluster_col: {cluster_col}")
    required = {"cluster", "candidate_name", "atlas_labels_union"}
    missing = sorted(required - set(matched_pairs_df.columns))
    if missing:
        raise KeyError(f"matched_pairs_df missing required columns: {missing}")

    pairs = matched_pairs_df.copy()
    pairs["cluster"] = pairs["cluster"].astype(str)
    pairs = pairs.drop_duplicates(subset=["cluster"], keep="first")
    pairs = pairs.set_index("cluster", drop=False)

    out = st_df.copy()
    cluster_key = out[cluster_col].astype(str)
    out["atlas_transfer_label"] = cluster_key.map(pairs["candidate_name"]).fillna(unmatched_label)
    out["atlas_transfer_labels_union"] = cluster_key.map(pairs["atlas_labels_union"]).fillna("")
    out["atlas_transfer_pair_type"] = cluster_key.map(pairs.get("pair_type", pd.Series(dtype=object))).fillna("")
    out["atlas_transfer_score"] = cluster_key.map(
        pairs.get("align_score_gated", pairs.get("align_score", pd.Series(dtype=float)))
    )
    out["atlas_transfer_matched"] = out["atlas_transfer_label"].ne(unmatched_label)
    return out


def _labels_from_union(value) -> list[int]:
    """Parse atlas label lists stored as semicolon/comma strings or iterables."""
    if isinstance(value, (list, tuple, np.ndarray, set)):
        vals = value
    else:
        s = str(value).strip().replace(" ", "")
        if s == "" or s.lower() == "nan":
            return []
        sep = ";" if ";" in s else ","
        vals = s.split(sep) if sep in s else [s]
    out = []
    for val in vals:
        if str(val).strip() == "":
            continue
        try:
            out.append(int(float(val)))
        except Exception:
            continue
    return out


def atlas_transfer_color_map(matched_pairs_df: pd.DataFrame, *, unmatched_label: str = "unassigned"):
    """Build a stable candidate-name color map for atlas and transferred ST plots."""
    if "candidate_name" not in matched_pairs_df.columns:
        raise KeyError("matched_pairs_df missing candidate_name")
    labels = [str(x) for x in matched_pairs_df["candidate_name"].dropna().astype(str).unique()]
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    color_map = {label: mcolors.to_rgba(cmap(i), alpha=1.0) for i, label in enumerate(labels)}
    color_map[unmatched_label] = (0.18, 0.18, 0.18, 0.40)
    return color_map


def build_atlas_label_overlay(
    sl: np.ndarray,
    matched_pairs_df: pd.DataFrame,
    color_map: dict[str, tuple[float, float, float, float]] | None = None,
    *,
    unmatched_atlas_rgba=(0.92, 0.92, 0.92, 0.90),
) -> tuple[np.ndarray, dict[str, tuple[float, float, float, float]]]:
    """Create an RGBA atlas image colored by transferred atlas candidate labels."""
    if color_map is None:
        color_map = atlas_transfer_color_map(matched_pairs_df)

    overlay = np.zeros((*sl.shape, 4), dtype=float)
    brain = sl > 0
    matched = np.zeros(sl.shape, dtype=bool)

    for _, row in matched_pairs_df.iterrows():
        label = str(row.get("candidate_name", ""))
        labels = _labels_from_union(row.get("atlas_labels_union", ""))
        if not label or len(labels) == 0:
            continue
        mask = np.isin(sl, labels)
        overlay[mask] = color_map.get(label, (0.5, 0.5, 0.5, 1.0))
        matched |= mask

    overlay[brain & (~matched)] = np.array(unmatched_atlas_rgba, dtype=float)
    return overlay, color_map


def plot_atlas_and_transferred_st(
    *,
    sl: np.ndarray,
    xJ,
    yJ,
    st_df: pd.DataFrame,
    matched_pairs_df: pd.DataFrame,
    label_col: str = "atlas_transfer_label",
    x_col: str = "x_aligned",
    y_col: str = "y_aligned",
    unmatched_label: str = "unassigned",
    output_prefix: str | Path | None = None,
    point_size: float = 1.0,
    figsize=(13, 6),
):
    """Plot matched atlas regions and aligned ST cells with the same colors."""
    for col in [label_col, x_col, y_col]:
        if col not in st_df.columns:
            raise KeyError(f"st_df missing required column: {col}")

    color_map = atlas_transfer_color_map(matched_pairs_df, unmatched_label=unmatched_label)
    overlay, color_map = build_atlas_label_overlay(sl, matched_pairs_df, color_map)

    H, W = sl.shape
    phys_to_pix = make_phys_to_pix(xJ, yJ, H, W)
    x = st_df[x_col].to_numpy(dtype=float)
    y = st_df[y_col].to_numpy(dtype=float)
    xi, yi = phys_to_pix(x, y)
    in_view = (
        np.isfinite(x)
        & np.isfinite(y)
        & (xi >= 0)
        & (xi < W)
        & (yi >= 0)
        & (yi < H)
    )

    labels = st_df[label_col].fillna(unmatched_label).astype(str).to_numpy()
    point_colors = np.array([color_map.get(label, color_map[unmatched_label]) for label in labels])

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True, facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.imshow(sl > 0, cmap="binary", alpha=0.10, origin="lower")
        ax.axis("off")
        ax.set_aspect("equal")

    axes[0].imshow(overlay, origin="lower", interpolation="nearest")
    axes[0].set_title("Atlas regions used for label transfer")

    axes[1].imshow(overlay, origin="lower", interpolation="nearest", alpha=0.25)
    axes[1].scatter(
        xi[in_view],
        yi[in_view],
        s=point_size,
        c=point_colors[in_view],
        edgecolors="none",
        rasterized=True,
    )
    axes[1].set_title("Aligned ST cells with transferred atlas labels")

    legend_labels = [str(x) for x in matched_pairs_df["candidate_name"].dropna().astype(str).unique()]
    handles = [
        Patch(facecolor=color_map[label], edgecolor="none", label=label)
        for label in legend_labels
        if label in color_map
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            markerscale=1.0,
        )

    fig.tight_layout()
    if output_prefix is not None:
        output_prefix = _as_path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(output_prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    return fig, axes


def _hex_to_rgba(hex_value, alpha: float = 1.0):
    """Convert Allen atlas hex colors to matplotlib RGBA tuples."""
    if pd.isna(hex_value):
        return (0.6, 0.6, 0.6, alpha)
    s = str(hex_value).strip().lstrip("#")
    if len(s) != 6:
        return (0.6, 0.6, 0.6, alpha)
    try:
        return tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4)) + (alpha,)
    except Exception:
        return (0.6, 0.6, 0.6, alpha)


def build_atlas_id_metadata(df2: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Build id -> acronym/name/color lookup from the Allen atlas metadata table."""
    meta = {}
    for idx, row in df2.iterrows():
        try:
            atlas_id = int(idx)
        except Exception:
            continue
        acronym = str(row.get("acronym", atlas_id))
        name = str(row.get("name", acronym))
        color_hex = str(row.get("color_hex_triplet", "999999")).strip().lstrip("#")
        meta[atlas_id] = {
            "atlas_id": atlas_id,
            "acronym": acronym,
            "name": name,
            "color_hex_triplet": color_hex,
            "rgba": _hex_to_rgba(color_hex, alpha=1.0),
        }
    return meta


def transfer_all_atlas_labels_to_st(
    st_df: pd.DataFrame,
    *,
    sl: np.ndarray,
    xJ,
    yJ,
    atlas_metadata: pd.DataFrame,
    x_col: str = "x_aligned",
    y_col: str = "y_aligned",
    background_label: str = "outside_atlas",
) -> pd.DataFrame:
    """Transfer atlas labels to every ST cell by sampling the aligned atlas slice.

    Unlike ``transfer_atlas_labels_to_st``, this does not use the matched-pair
    table. It assigns each aligned ST cell the Allen annotation label at the
    cell's final aligned coordinate.
    """
    for col in [x_col, y_col]:
        if col not in st_df.columns:
            raise KeyError(f"st_df missing required coordinate column: {col}")

    H, W = sl.shape
    phys_to_pix = make_phys_to_pix(xJ, yJ, H, W)
    x = st_df[x_col].to_numpy(dtype=float)
    y = st_df[y_col].to_numpy(dtype=float)
    xi, yi = phys_to_pix(x, y)
    in_view = (
        np.isfinite(x)
        & np.isfinite(y)
        & (xi >= 0)
        & (xi < W)
        & (yi >= 0)
        & (yi < H)
    )

    atlas_ids = np.zeros(len(st_df), dtype=np.int64)
    atlas_ids[in_view] = sl[yi[in_view], xi[in_view]].astype(np.int64)

    atlas_meta = build_atlas_id_metadata(atlas_metadata)

    def _meta_value(atlas_id: int, key: str):
        if atlas_id == 0:
            if key == "acronym":
                return background_label
            if key == "name":
                return background_label
            if key == "color_hex_triplet":
                return "999999"
            return np.nan
        item = atlas_meta.get(int(atlas_id))
        if item is None:
            if key == "acronym":
                return f"unknown_{int(atlas_id)}"
            if key == "name":
                return f"unknown_{int(atlas_id)}"
            if key == "color_hex_triplet":
                return "999999"
            return np.nan
        return item[key]

    out = st_df.copy()
    out["atlas_voxel_xi"] = xi
    out["atlas_voxel_yi"] = yi
    out["atlas_voxel_in_view"] = in_view
    out["atlas_label_id"] = atlas_ids
    out["atlas_label_acronym"] = [_meta_value(v, "acronym") for v in atlas_ids]
    out["atlas_label_name"] = [_meta_value(v, "name") for v in atlas_ids]
    out["atlas_label_color_hex"] = [_meta_value(v, "color_hex_triplet") for v in atlas_ids]
    out["atlas_label_transferred"] = in_view & (atlas_ids != 0)
    out["atlas_label"] = atlas_ids
    out["inside_atlas_slice"] = in_view
    out["atlas_region_name"] = out["atlas_label_name"]
    out["atlas_region_acronym"] = out["atlas_label_acronym"]
    bg_mask = out["atlas_label"].astype(int).eq(0)
    out.loc[bg_mask, "atlas_region_name"] = "background"
    out.loc[bg_mask, "atlas_region_acronym"] = "BG"
    return out


def build_benchmark_transfer_table(st_with_atlas: pd.DataFrame) -> pd.DataFrame:
    """Return the benchmarking-compatible transferred-label table columns."""
    required = [
        "cell_id",
        "x",
        "y",
        "x_aligned",
        "y_aligned",
        "atlas_label",
        "inside_atlas_slice",
        "atlas_region_name",
        "atlas_region_acronym",
    ]
    missing = [c for c in required if c not in st_with_atlas.columns]
    if missing:
        raise KeyError(f"st_with_atlas missing columns for benchmark transfer table: {missing}")

    out = pd.DataFrame()
    out["cell_id"] = st_with_atlas["cell_id"].astype(str)
    out["original_x"] = st_with_atlas["x"]
    out["original_y"] = st_with_atlas["y"]
    out["prealign_x"] = st_with_atlas.get("x_prealigned", np.nan)
    out["prealign_y"] = st_with_atlas.get("y_prealigned", np.nan)
    out["x_aligned"] = st_with_atlas["x_aligned"]
    out["y_aligned"] = st_with_atlas["y_aligned"]
    out["atlas_label"] = st_with_atlas["atlas_label"].astype(int)
    out["inside_atlas_slice"] = st_with_atlas["inside_atlas_slice"].astype(bool)
    out["atlas_region_name"] = st_with_atlas["atlas_region_name"].astype(str)
    out["atlas_region_acronym"] = st_with_atlas["atlas_region_acronym"].astype(str)
    return out


def build_white_label_color_map_for_atlas(
    sl: np.ndarray,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Build the white-background color map used by benchmarking label scatter plots."""
    labels = np.array(sorted(int(x) for x in np.unique(sl)))
    rng = np.random.default_rng(seed)
    rows = []
    for label in labels:
        if int(label) == 0:
            r, g, b = 1.0, 1.0, 1.0
        else:
            r, g, b = rng.random(3)
        rows.append(
            {
                "atlas_label": int(label),
                "r": float(r),
                "g": float(g),
                "b": float(b),
                "hex": mcolors.to_hex((r, g, b)),
            }
        )
    return pd.DataFrame(rows)


def load_atlas_label_color_map(path: str | Path | None = None, sl: np.ndarray | None = None) -> dict[str, str]:
    """Load or build atlas_label -> hex color map for label scatter plots."""
    if path is not None and Path(path).expanduser().is_file():
        colors = pd.read_csv(Path(path).expanduser())
    elif sl is not None:
        colors = build_white_label_color_map_for_atlas(sl)
    else:
        raise ValueError("Provide either an existing color-map path or sl.")
    colors = colors.copy()
    colors["atlas_label"] = colors["atlas_label"].astype(str)
    return colors.set_index("atlas_label")["hex"].to_dict()


def plot_atlas_label_scatter(
    ax,
    df: pd.DataFrame,
    title: str,
    *,
    x_col: str = "x_aligned",
    y_col: str = "y_aligned",
    label_col: str = "atlas_label",
    color_map: dict[str, str] | None = None,
    point_size: float = 2.0,
    point_alpha: float = 0.8,
    invert_y: bool = False,
):
    """Scatter aligned cells by transferred atlas_label, matching benchmarking style."""
    plot_df = df.dropna(subset=[x_col, y_col, label_col]).copy()
    plot_df[label_col] = plot_df[label_col].astype(int).astype(str)
    if color_map is None:
        labels = sorted(plot_df[label_col].unique(), key=lambda x: int(x))
        color_map = {"0": "#ffffff"}
        cmap = plt.get_cmap("tab20", max(len(labels), 1))
        for i, lab in enumerate(labels):
            color_map.setdefault(lab, mcolors.to_hex(cmap(i)))

    present = set(plot_df[label_col])
    labels = [lab for lab in color_map if lab in present]
    labels += [lab for lab in plot_df[label_col].value_counts().index if lab not in color_map]

    for lab in labels:
        sub = plot_df[plot_df[label_col] == lab]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=point_size,
            alpha=point_alpha,
            color=color_map.get(lab, "#b3b3b3"),
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    ax.axis("off")
    return ax


def plot_atlas_slice_and_label_scatter(
    *,
    sl: np.ndarray,
    xJ,
    yJ,
    st_df: pd.DataFrame,
    color_map: dict[str, str],
    output_prefix: str | Path | None = None,
    point_size: float = 2.0,
    point_alpha: float = 0.8,
    figsize=(14, 7),
):
    """Plot atlas slice labels and transferred ST labels with the benchmark color map."""
    H, W = sl.shape
    overlay = np.zeros((H, W, 4), dtype=float)
    for atlas_id in np.unique(sl):
        color = color_map.get(str(int(atlas_id)), "#b3b3b3")
        overlay[sl == atlas_id] = mcolors.to_rgba(color, alpha=1.0)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    extent = [float(xJ[0]), float(xJ[-1]), float(yJ[0]), float(yJ[-1])]
    axes[0].imshow(overlay, origin="lower", extent=extent, interpolation="nearest")
    axes[0].set_title("Allen atlas labels")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].axis("off")

    plot_atlas_label_scatter(
        axes[1],
        st_df,
        "Aligned ST transferred atlas labels",
        color_map=color_map,
        point_size=point_size,
        point_alpha=point_alpha,
        invert_y=False,
    )

    if output_prefix is not None:
        output_prefix = _as_path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
        fig.savefig(output_prefix.with_suffix(".pdf"), dpi=300)
    return fig, axes


def build_full_atlas_color_overlay(
    sl: np.ndarray,
    atlas_metadata: pd.DataFrame,
    *,
    background_rgba=(1.0, 1.0, 1.0, 0.0),
) -> np.ndarray:
    """Render a full Allen atlas annotation slice with native atlas colors."""
    overlay = np.zeros((*sl.shape, 4), dtype=float)
    overlay[:] = np.array(background_rgba, dtype=float)
    atlas_meta = build_atlas_id_metadata(atlas_metadata)
    for atlas_id in np.unique(sl):
        atlas_id_int = int(atlas_id)
        if atlas_id_int == 0:
            continue
        color = atlas_meta.get(atlas_id_int, {}).get("rgba", (0.6, 0.6, 0.6, 1.0))
        overlay[sl == atlas_id_int] = color
    return overlay


def plot_atlas_and_all_transferred_st(
    *,
    sl: np.ndarray,
    xJ,
    yJ,
    st_df: pd.DataFrame,
    atlas_metadata: pd.DataFrame,
    x_col: str = "x_aligned",
    y_col: str = "y_aligned",
    label_id_col: str = "atlas_label_id",
    output_prefix: str | Path | None = None,
    point_size: float = 1.0,
    figsize=(13, 6),
    max_legend_labels: int = 18,
):
    """Plot all atlas labels and ST cells transferred from aligned atlas pixels."""
    for col in [x_col, y_col, label_id_col]:
        if col not in st_df.columns:
            raise KeyError(f"st_df missing required column: {col}")

    H, W = sl.shape
    phys_to_pix = make_phys_to_pix(xJ, yJ, H, W)
    x = st_df[x_col].to_numpy(dtype=float)
    y = st_df[y_col].to_numpy(dtype=float)
    xi, yi = phys_to_pix(x, y)
    in_view = (
        np.isfinite(x)
        & np.isfinite(y)
        & (xi >= 0)
        & (xi < W)
        & (yi >= 0)
        & (yi < H)
    )

    atlas_meta = build_atlas_id_metadata(atlas_metadata)
    label_ids = st_df[label_id_col].fillna(0).astype(int).to_numpy()
    point_colors = np.array(
        [atlas_meta.get(int(v), {}).get("rgba", (0.2, 0.2, 0.2, 0.35)) for v in label_ids]
    )
    point_colors[label_ids == 0] = np.array((0.2, 0.2, 0.2, 0.25))

    overlay = build_full_atlas_color_overlay(sl, atlas_metadata)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True, facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.axis("off")
        ax.set_aspect("equal")

    axes[0].imshow(overlay, origin="lower", interpolation="nearest")
    axes[0].set_title("Allen atlas annotation slice")

    axes[1].imshow(overlay, origin="lower", interpolation="nearest", alpha=0.20)
    axes[1].scatter(
        xi[in_view],
        yi[in_view],
        s=point_size,
        c=point_colors[in_view],
        edgecolors="none",
        rasterized=True,
    )
    axes[1].set_title("Aligned ST cells with sampled atlas labels")

    transferred = st_df.loc[st_df.get("atlas_label_transferred", pd.Series(False, index=st_df.index))]
    if len(transferred) > 0:
        top_ids = transferred[label_id_col].astype(int).value_counts().head(max_legend_labels).index.tolist()
        handles = []
        for atlas_id in top_ids:
            item = atlas_meta.get(int(atlas_id))
            if item is None:
                continue
            handles.append(
                Patch(
                    facecolor=item["rgba"],
                    edgecolor="none",
                    label=f"{item['acronym']} ({int((transferred[label_id_col].astype(int) == atlas_id).sum())})",
                )
            )
        if handles:
            fig.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                frameon=False,
                title="Top transferred labels",
            )

    fig.tight_layout()
    if output_prefix is not None:
        output_prefix = _as_path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(output_prefix.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    return fig, axes


# Constants copied from the notebook alignment setup.
W_ALIGN = dict(
    sdf_corr=0.08,
    chamfer_sim=0.06,
    dice=0.18,
    area_sim=0.47,
    thick_sim=0.21,
)

DEFAULT_MASK_PARAMS = dict(
    BIN_THR=0.03,        # slightly lower => fuller smooth region
    SIGMA_FIX=10,       # higher smoothing
    AUTO_SIG=True,
    SIGMA_MIN=1.2,
    SIGMA_MAX=6.0,       # allow broader blur on large clusters
    SIGMA_SCL=1.1,
    CLOSE_R=10,           # stronger closing

    REF_CLOSE_R=3,
    REF_OPEN_R=1,
    REF_SMOOTH_MAX=1.8,  # allow stronger edge smoothing
    KEEP_LARGEST=False,

    MIN_SIZE=180,        # remove tiny islands
    FILL_HOLES=True
)

MASK_PARAMS_THIN = dict(
    BIN_THR=0.11,        # a bit stricter boundary
    SIGMA_FIX=1.0,       # less smoothing to keep detail
    AUTO_SIG=True,
    SIGMA_MIN=0.6,
    SIGMA_MAX=2.2,
    SIGMA_SCL=0.8,
    CLOSE_R=1,

    REF_CLOSE_R=1,
    REF_OPEN_R=0,        # keep thin branches/details
    REF_SMOOTH_MAX=0.55, # less contour smoothing
    KEEP_LARGEST=False,

    MIN_SIZE=200,         # keep small true thin parts
    FILL_HOLES=False
)

# Helper functions extracted from the notebook.
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


def load_atlas_slice(
    atlas_nrrd_path,
    atlas_slice_z,
    flip_ud=True,
    plot=True,
    random_seed=0,
    figsize=(8, 6)
):
    """
    Load one 2D atlas slice from NRRD and optionally plot it.

    Parameters
    ----------
    atlas_nrrd_path : str
        Path to atlas NRRD file.
    atlas_slice_z : int
        Slice index along Z dimension.
    flip_ud : bool, default=True
        Whether to flip the slice vertically so y increases upward.
    plot : bool, default=True
        Whether to display the slice.
    random_seed : int, default=0
        Random seed for slice label coloring.
    figsize : tuple, default=(8, 6)
        Figure size for plotting.

    Returns
    -------
    out : dict
        A dictionary containing:
        - anno : full atlas volume
        - hdr  : NRRD header
        - sl   : 2D slice
        - xJ   : x coordinates in physical space
        - yJ   : y coordinates in physical space
        - dx   : x spacing
        - dy   : y spacing
        - H    : slice height
        - W    : slice width
        - z    : slice index
    """
    anno, hdr = nrrd.read(atlas_nrrd_path)   # (Z, Y, X)
    sl = anno[atlas_slice_z].copy()          # (Y, X)

    if flip_ud:
        sl = np.flipud(sl)

    # read voxel spacing
    sd = hdr.get("space directions", None)
    dx = dy = 10.0
    if sd is not None:
        dy = float(np.linalg.norm(sd[1]))
        dx = float(np.linalg.norm(sd[2]))

    H, W = sl.shape
    xJ = np.linspace(0, (W - 1) * dx, W)
    yJ = np.linspace(0, (H - 1) * dy, H)

    if plot:
        labels, inv = np.unique(sl, return_inverse=True)
        M = len(labels)

        rng = np.random.default_rng(random_seed)
        if labels[0] == 0:
            colors = np.vstack([[0, 0, 0], rng.random((M - 1, 3))])
        else:
            colors = rng.random((M, 3))

        cmap = ListedColormap(colors)

        plt.figure(figsize=figsize)
        plt.imshow(
            inv.reshape(sl.shape),
            cmap=cmap,
            interpolation="nearest",
            origin="lower",
            extent=(xJ[0], xJ[-1], yJ[0], yJ[-1])
        )
        plt.title(f"CCFv3 coronal slice z={atlas_slice_z}")
        plt.axis("equal")
        plt.xlabel("x (um)")
        plt.ylabel("y (um)")
        plt.show()

    return {
        "anno": anno,
        "hdr": hdr,
        "sl": sl,
        "xJ": xJ,
        "yJ": yJ,
        "dx": dx,
        "dy": dy,
        "H": H,
        "W": W,
        "z": atlas_slice_z
    }


def get_ellipse_from_points(x, y, res=25, padding=50, close_ksize=15):
    """Fit a bounding ellipse from scattered points using morphology + cv2.fitEllipse."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    w_pix = int((x_max - x_min) / res) + padding * 2
    h_pix = int((y_max - y_min) / res) + padding * 2
    if w_pix <= 0 or h_pix <= 0:
        return None

    img = np.zeros((h_pix, w_pix), dtype=np.uint8)
    px = ((x - x_min) / res + padding).astype(int)
    py = ((y - y_min) / res + padding).astype(int)

    px = np.clip(px, 0, w_pix - 1)
    py = np.clip(py, 0, h_pix - 1)
    img[py, px] = 255

    k = max(3, int(close_ksize) // 2 * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.dilate(img, kernel, iterations=1)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    (cx_pix, cy_pix), (width, height), angle = cv2.fitEllipse(cnt)

    cx_phys = cx_pix * res + (x_min - padding * res)
    cy_phys = cy_pix * res + (y_min - padding * res)
    width_phys = width * res
    height_phys = height * res

    return (cx_phys, cy_phys), (width_phys, height_phys), angle


def get_ellipse_from_atlas(mask, x_axis, y_axis):
    """Fit ellipse from the atlas boundary mask and map params back to physical coordinates."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    (cx_pix, cy_pix), (w_pix, h_pix), angle = cv2.fitEllipse(cnt)

    x_min, x_max = x_axis[0], x_axis[-1]
    y_min, y_max = y_axis[0], y_axis[-1]
    W_img = len(x_axis)
    H_img = len(y_axis)

    cx = x_min + (x_max - x_min) * (cx_pix / max(W_img - 1, 1))
    cy = y_min + (y_max - y_min) * (cy_pix / max(H_img - 1, 1))

    res_x = (x_max - x_min) / max(W_img - 1, 1)
    res_y = (y_max - y_min) / max(H_img - 1, 1)
    phys_w = w_pix * res_x
    phys_h = h_pix * res_y

    return (cx, cy), (phys_w, phys_h), angle


def transform_points(points, source_center, target_center, scale, angle_deg):
    """Apply centered similarity transform from source_center to target_center."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    pts = np.asarray(points, float)
    shifted = pts - np.array(source_center)
    rotated_scaled = (shifted @ R.T) * scale
    return rotated_scaled + np.array(target_center)


def generate_mask_from_points(x, y, x_grid, y_grid, shape, close_ksize=20):
    """Rasterize aligned points onto atlas grid and create filled outer mask."""
    H, W = shape
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    dx = (x_max - x_min) / max(W - 1, 1)
    dy = (y_max - y_min) / max(H - 1, 1)

    px = np.round((x - x_min) / dx).astype(int)
    py = np.round((y - y_min) / dy).astype(int)

    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px = px[valid]
    py = py[valid]

    mask = np.zeros((H, W), dtype=np.uint8)
    if len(px) > 0:
        mask[py, px] = 1

    k = max(3, int(close_ksize) // 2 * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_final = cv2.dilate(mask_closed, kernel, iterations=1)

    return mask_final


def iou_score(mask_a, mask_b):
    a = (mask_a > 0)
    b = (mask_b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def align_omics_no_flip_max_iou(df, sl, xJ, yJ,
                               close_ksize=15,
                               angle_step_deg=1.0,
                               scale_tweak=0.05,
                               scale_steps=2,
                               angle_range=360.0):
    """Global align with no flip and choose transform by maximum IoU."""
    atlas_params = get_ellipse_from_atlas(sl, xJ, yJ)
    if atlas_params is None:
        raise ValueError('Failed to fit atlas ellipse.')
    (acx, acy), (aw, ah), a_angle = atlas_params

    omics_params = get_ellipse_from_points(df['x'].values, df['y'].values, res=resolution, close_ksize=close_ksize)
    if omics_params is None:
        raise ValueError('Failed to fit omics ellipse.')
    (ocx, ocy), (ow, oh), o_angle = omics_params

    base_angle = a_angle - o_angle
    base_scale = max(aw, ah) / max(ow, oh)

    P = df[['x', 'y']].to_numpy(float)
    atlas_mask = (sl > 0).astype(np.uint8)

    # Build candidate search grid around initial angle and slight scale perturbations
    n_scale = 2 * scale_steps + 1
    scale_factors = base_scale * (1.0 + np.linspace(-scale_tweak, scale_tweak, n_scale))

    # wrap base angle to [-180, 180]
    base_angle = ((base_angle + 180) % 360) - 180
    # full 360-degree sweep so angle is not restricted
    # (centering on base_angle keeps periodic indexing stable but does not restrict range)
    angle_candidates = np.arange(base_angle, base_angle + 360.0, angle_step_deg)

    best_iou = -1.0
    best = None

    for sc in scale_factors:
        for ang in angle_candidates:
            cand_xy = transform_points(P, (ocx, ocy), (acx, acy), sc, ang)
            m_try = generate_mask_from_points(
                cand_xy[:, 0],
                cand_xy[:, 1],
                xJ,
                yJ,
                sl.shape,
                close_ksize=max(12, close_ksize),
            )
            iou = iou_score(m_try, atlas_mask)

            if iou > best_iou:
                best_iou = iou
                best = {
                    'angle_deg': float(ang),
                    'scale': float(sc),
                    'points': cand_xy,
                    'mask': m_try,
                    'iou': float(iou),
                }

    if best is None:
        raise ValueError('Could not evaluate candidates.')

    df_out = df.copy()
    df_out['x_prealigned'] = best['points'][:, 0]
    df_out['y_prealigned'] = best['points'][:, 1]

    # Convert centered transform to standard translation form: x' = s R x + t
    theta_best = np.radians(best['angle_deg'])
    c, s = np.cos(theta_best), np.sin(theta_best)
    R = np.array([[c, -s], [s, c]], dtype=float)
    t = np.array([acx, acy], dtype=float) - best['scale'] * (R @ np.array([ocx, ocy], dtype=float))
    tx, ty = float(t[0]), float(t[1])

    params = {
        'best_angle_deg': float(best['angle_deg']),
        'best_scale': float(best['scale']),
        'tx': tx,
        'ty': ty,
        'base_angle_deg': float(base_angle),
        'base_scale': float(base_scale),
        'atlas_center': (float(acx), float(acy)),
        'omics_center': (float(ocx), float(ocy)),
        'iou': float(best['iou']),
    }

    return df_out, params


def plot_mask_iou_overlay(df_aligned, sl, xJ, yJ, close_ksize=20):
    """Plot atlas/omics masks and print IoU score."""
    omics_mask = generate_mask_from_points(
        df_aligned['x_prealigned'].values,
        df_aligned['y_prealigned'].values,
        xJ,
        yJ,
        sl.shape,
        close_ksize=close_ksize,
    )

    atlas_mask = (sl > 0).astype(int)

    score = iou_score(omics_mask, atlas_mask)
    print(f"Shape Alignment Score (IoU): {score:.3f}")

    plt.figure(figsize=(10, 10))
    extent_val = [xJ[0], xJ[-1], yJ[0], yJ[-1]]

    cmap_atlas = ListedColormap(['none', 'green'])
    cmap_omics = ListedColormap(['none', 'red'])

    plt.imshow(atlas_mask, cmap=cmap_atlas, origin='lower', extent=extent_val, alpha=0.45)
    plt.imshow(omics_mask, cmap=cmap_omics, origin='lower', extent=extent_val, alpha=0.45)

    plt.title(f"Mask Overlay (Green=Atlas, Red=Aligned Omics)\nIoU Score: {score:.3f}")
    plt.xlabel('Physical X')
    plt.ylabel('Physical Y')

    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label='Atlas CCF'),
        Patch(facecolor='red', alpha=0.5, label='Aligned Omics'),
        Patch(facecolor='#AA8800', alpha=0.8, label='Overlap (Yellowish)'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.show()

    return score, omics_mask


def estimate_cluster_shape(coords):
    if len(coords) < 5:
        return {"elongation": np.nan, "pc1": np.nan, "pc2": np.nan, "anisotropy": np.nan}

    pca = PCA(n_components=2)
    pca.fit(coords)
    ev = pca.explained_variance_

    pc1 = float(np.sqrt(ev[0])) if ev[0] > 0 else 0.0
    pc2 = float(np.sqrt(ev[1])) if ev[1] > 0 else 0.0
    elongation = np.inf if pc2 == 0 else pc1 / (pc2 + 1e-12)
    anisotropy = (pc1 - pc2) / (pc1 + pc2 + 1e-12)
    return {"elongation": elongation, "pc1": pc1, "pc2": pc2, "anisotropy": anisotropy}


def robust_bbox_metrics(coords, k=8, core_q=0.85):
    """
    Robust bbox metrics:
    1) remove sparse/outlier points by local KNN scale
    2) compute bbox from IQR (25%-75%) on core points
    """
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return {"bbox_w": np.nan, "bbox_h": np.nan, "bbox_ar": np.nan, "bbox_area": np.nan}

    core = coords
    if len(coords) >= max(k + 1, 20):
        tree = cKDTree(coords)
        d, _ = tree.query(coords, k=k + 1)           # includes self
        local_scale = np.median(d[:, 1:], axis=1)    # larger => sparser/outlier-like
        thr = np.quantile(local_scale, core_q)       # keep dense core
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


def estimate_knn_spacing(coords, k=6):
    if len(coords) < max(k + 1, 3):
        return np.nan
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k + 1)
    nn = dists[:, 1:]
    return float(np.median(nn))


def estimate_thin_score(coords):
    shp = estimate_cluster_shape(coords)
    bb = robust_bbox_metrics(coords)
    sp = estimate_knn_spacing(coords, k=6)

    elong = shp["elongation"] if np.isfinite(shp["elongation"]) else 1.0
    anis = shp["anisotropy"] if np.isfinite(shp["anisotropy"]) else 0.0
    ar = bb["bbox_ar"]

    minor_axis = min(bb["bbox_w"], bb["bbox_h"])
    thickness_to_spacing = np.nan if not np.isfinite(sp) else minor_axis / (sp + 1e-9)

    thin_score = (
        0.40 * np.log1p(max(elong - 1.0, 0.0)) +
        0.30 * max(anis, 0.0) +
        0.20 * np.log1p(max(ar - 1.0, 0.0)) +
        0.10 * (0.0 if not np.isfinite(thickness_to_spacing) else max(0.0, 3.0 - thickness_to_spacing))
    )

    return {
        "elongation": shp["elongation"],
        "anisotropy": shp["anisotropy"],
        "bbox_ar": bb["bbox_ar"],
        "bbox_area": bb["bbox_area"],
        "minor_axis": minor_axis,
        "knn_spacing_med": sp,
        "thickness_to_spacing": thickness_to_spacing,
        "thin_score": float(thin_score),
    }


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
    base_k=10,
    min_points=None,

    # area-only rule
    detail_area_quantile=0.40,   # bottom 40% area -> detail
    area_mode="bbox",            # "bbox" or "n_points"

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

    # ---------- area table ----------
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

    # small area -> detail
    detail_clusters = set(
        area_tbl.loc[area_tbl[gate_col] <= area_thr, "cluster"].astype(str).tolist()
    )

    print(f"[area-only gate] mode={area_mode}, q={detail_area_quantile}, threshold={area_thr:.3f}, n_detail={len(detail_clusters)}")

    keep_list, rm_list, stats = [], [], []
    clusters = df[label_col].dropna().unique()

    for cl in clusters:
        cl_str = str(cl)
        sub = df[df[label_col] == cl].copy()
        n = len(sub)

        cfg = cluster_specific_params.get(cl, {})
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

        # area-only + manual override
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


def make_phys_to_pix(xJ, yJ, H, W):
    """
    Create a physical-to-pixel converter based on atlas canvas.
    """
    CANVAS_XMIN, CANVAS_XMAX = float(xJ[0]), float(xJ[-1])
    CANVAS_YMIN, CANVAS_YMAX = float(yJ[0]), float(yJ[-1])

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
    bin_thr=0.08,
    min_obj_frac=0.0,
    close_radius=1,
    fill_holes=True,
    auto_sigma=True,
    sigma_min=1.0,
    sigma_max=5.0,
    sigma_scale=1.0
):
    acc = np.zeros((h, w), np.float32)

    if auto_sigma and len(y_idx) >= 5:
        pts = np.stack([y_idx, x_idx], axis=1).astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, h - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, w - 1)
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=5)   # includes self
        d_med = np.median(d[:, -1])   # 4th neighbor distance
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
    mask,
    close_r=2,
    open_r=1,
    smooth=1.0,
    keep_largest=False,
    fill_holes=True
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
    return dict(
        n_cc=n_cc,
        largest_frac=float(largest / (total + 1e-12))
    )


def keep_cc_by_cumfrac(mask_u8, cumfrac=0.90, max_k=6, min_area=0, connectivity=2):
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
    dyn_smooth = float(np.clip(
        np.sqrt(raw_area) / 15.0,
        1.0,
        params["REF_SMOOTH_MAX"]
    ))

    mask = refine_edges_morph(
        hard,
        close_r=params["REF_CLOSE_R"],
        open_r=params["REF_OPEN_R"],
        smooth=dyn_smooth,
        keep_largest=params["KEEP_LARGEST"],
        fill_holes=params["FILL_HOLES"]
    )

    mask = remove_small_objects(
        mask.astype(bool),
        min_size=int(params["MIN_SIZE"])
    ).astype(np.uint8)

    return soft, mask


def build_cluster_masks(
    df_smooth,
    sl,
    xJ,
    yJ,
    x_col="x_aligned",
    y_col="y_aligned",
    label_col="banksy_cluster_refined",
    params=None,                      # normal/default params
    params_thin=None,                 # detail params
    shape_type_col="shape_type",
    thin_values=("detail",),          # now use detail
    thin_rule="mode",                 # "mode" (recommended) or "any"
    verbose=True
):
    if params is None:
        params = DEFAULT_MASK_PARAMS.copy()
    if params_thin is None:
        params_thin = params

    thin_values = set(map(str, thin_values))

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

        params_use = params_thin if is_detail else params
        shape_type_used = "detail" if is_detail else "normal"

        soft, mask = build_mask_for_cluster(
            sub=sub,
            x_col=x_col, y_col=y_col,
            h=H, w=W,
            phys_to_pix_array=phys_to_pix_array,
            params=params_use
        )

        if mask.sum() == 0:
            continue

        st_soft[cl_str] = soft
        st_masks[cl_str] = mask
        rows.append({
            "cluster": cl_str,
            "mask_area": int(mask.sum()),
            "n_points": int(len(sub)),
            "shape_type_used": shape_type_used
        })

    mask_df = pd.DataFrame(rows)
    if len(mask_df) > 0:
        mask_df = mask_df.sort_values("mask_area", ascending=False).reset_index(drop=True)

    return {
        "st_soft": st_soft,
        "st_masks": st_masks,
        "mask_df": mask_df,
        "phys_to_pix_array": phys_to_pix_array
    }


def build_atlas_slice_precompute(sl):
    """
    Precompute slice-level label info.
    """
    atlas_labels = np.unique(sl)
    atlas_labels = atlas_labels[atlas_labels != 0]
    atlas_labels = np.sort(atlas_labels)

    label2idx = {int(l): i for i, l in enumerate(atlas_labels)}
    nL = len(atlas_labels)

    sl_flat = sl.ravel()
    mask_nz = sl_flat != 0
    sl_flat_nz = sl_flat[mask_nz]

    labs_sorted = np.sort(sl_flat_nz)
    uniq_ids, counts = np.unique(labs_sorted, return_counts=True)

    pos = np.searchsorted(atlas_labels, uniq_ids)
    atlas_area = np.zeros(nL, dtype=int)
    atlas_area[pos] = counts

    return {
        "atlas_labels": atlas_labels,
        "label2idx": label2idx,
        "nL": nL,
        "atlas_area": atlas_area
    }


def precompute_counts_for_cluster(mc_bool, sl, atlas_labels, nL):
    """
    Precompute cluster area and overlap counts in compressed atlas-label space.
    """
    Ac = int(mc_bool.sum())
    labs = sl[mc_bool]
    labs = labs[labs != 0]

    inter_counts = np.zeros(nL, dtype=int)
    if labs.size > 0:
        labs_sorted = np.sort(labs)
        idx, counts = np.unique(labs_sorted, return_counts=True)
        pos = np.searchsorted(atlas_labels, idx)
        inter_counts[pos] = counts

    return Ac, inter_counts


def dice_of_set(Ac, inter_counts, label_set, label2idx, atlas_area):
    """
    Dice score between cluster mask and a set of Allen leaf labels.
    """
    if not label_set:
        return 0.0

    idxs = [label2idx[int(l)] for l in label_set if int(l) in label2idx]
    if not idxs:
        return 0.0

    idxs = np.asarray(idxs, dtype=int)
    inter = int(inter_counts[idxs].sum())
    areaB = int(atlas_area[idxs].sum())

    return (2.0 * inter / (Ac + areaB)) if (Ac + areaB) > 0 else 0.0


def path_to_tuple(path_str):
    return tuple(int(p) for p in str(path_str).strip("/").split("/") if p)


def build_prefix_label_mapping(df2, atlas_labels, min_prefix_len=2):
    """
    Restrict Allen prefix tree to labels present in current slice.
    """
    df2 = df2.copy()
    df2.index = df2.index.astype(int)

    slice_leaf_labels = set(atlas_labels.tolist())
    labels_df = df2.loc[df2.index.intersection(atlas_labels)]

    path_prefix_to_labels = {}
    for leaf_id, row in labels_df.iterrows():
        pt = path_to_tuple(row["structure_id_path"])
        if len(pt) == 0:
            continue
        for depth in range(1, len(pt) + 1):
            prefix = pt[:depth]
            path_prefix_to_labels.setdefault(prefix, set()).add(int(leaf_id))

    out = {}
    for prefix, labs in path_prefix_to_labels.items():
        inter = labs.intersection(slice_leaf_labels)
        if len(inter) > 0:
            out[prefix] = inter
    path_prefix_to_labels = out

    prefix_meta = {}
    prefixes_by_depth = {}

    for prefix, labset in path_prefix_to_labels.items():
        if len(prefix) < min_prefix_len:
            continue

        node_id = int(prefix[-1])
        if node_id not in df2.index:
            continue

        node_row = df2.loc[node_id]
        depth = len(prefix)

        prefix_meta[prefix] = {
            "node_id": node_id,
            "depth": depth,
            "acronym": node_row.get("acronym", ""),
            "name": node_row.get("name", ""),
            "path": "/" + "/".join(map(str, prefix)) + "/",
            "label_set": set(labset)
        }
        prefixes_by_depth.setdefault(depth, []).append(prefix)

    return {
        "path_prefix_to_labels": path_prefix_to_labels,
        "prefix_meta": prefix_meta,
        "prefixes_by_depth": prefixes_by_depth
    }


def make_prefix_candidates(prefix_info, depth_list=None):
    rows = []

    for prefix, meta in prefix_info["prefix_meta"].items():
        if (depth_list is not None) and (meta["depth"] not in depth_list):
            continue

        rows.append({
            "candidate_type": "prefix",
            "candidate_name": meta["acronym"] if str(meta["acronym"]) != "" else str(meta["node_id"]),
            "node_id": meta["node_id"],
            "depth": meta["depth"],
            "path": meta["path"],
            "n_leaf_labels": len(meta["label_set"]),
            "label_set": set(meta["label_set"]),
            "acronym": meta["acronym"],
            "name": meta["name"],
        })

    cand_df = pd.DataFrame(rows)
    if len(cand_df) > 0:
        cand_df = cand_df.reset_index(drop=True)
    return cand_df


def build_layer_label_mapping(df2, atlas_labels):
    """
    Build cortical layer -> atlas leaf label set mapping
    restricted to labels present in current slice.
    """
    df2 = df2.copy()
    df2.index = df2.index.astype(int)
    labels_df = df2.loc[df2.index.intersection(atlas_labels)].copy()

    name_pat = re.compile(
        r"\blayer\s*(1|2/3|4|5|6a|6b)\b",
        flags=re.IGNORECASE
    )

    suffix_to_layer = {
        "1":   "L1",
        "2/3": "L2/3",
        "4":   "L4",
        "5":   "L5",
        "6a":  "L6a",
        "6b":  "L6b",
    }

    layer_to_labels = {v: set() for v in suffix_to_layer.values()}

    for lab_id, row in labels_df.iterrows():
        name = str(row.get("name", ""))
        m = name_pat.search(name)
        if not m:
            continue

        suf = m.group(1).lower()
        lname = suffix_to_layer[suf]
        layer_to_labels[lname].add(int(lab_id))

    layer_to_labels = {k: v for k, v in layer_to_labels.items() if len(v) > 0}
    return layer_to_labels


def make_layer_candidates(layer_to_labels):
    rows = []
    layer_order = ["L1", "L2/3", "L4", "L5", "L6a", "L6b"]

    for lname in layer_order:
        if lname not in layer_to_labels:
            continue
        label_set = layer_to_labels[lname]
        rows.append({
            "candidate_type": "layer",
            "candidate_name": lname,
            "node_id": np.nan,
            "depth": np.nan,
            "path": "",
            "n_leaf_labels": len(label_set),
            "label_set": set(label_set),
            "acronym": lname,
            "name": lname,
        })

    cand_df = pd.DataFrame(rows)
    if len(cand_df) > 0:
        cand_df = cand_df.reset_index(drop=True)
    return cand_df


def mask_from_label_set(sl, label_set):
    """
    Binary atlas mask for a set of Allen labels.
    """
    if not label_set:
        return np.zeros_like(sl, dtype=np.uint8)
    return np.isin(sl, list(label_set)).astype(np.uint8)


def boundary(mask):
    m = mask.astype(bool)
    if not m.any():
        return m
    se = morphology.disk(1)
    er = morphology.binary_erosion(m, se)
    return m ^ er


def extract_boundary(mask):
    return boundary(mask)


def boundary_iou(mask1, mask2):
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    if not (m1.any() or m2.any()):
        return 0.0

    b1 = boundary(m1)
    b2 = boundary(m2)

    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return float(inter / union) if union > 0 else 0.0


def sdf_corr_band(mask1, mask2, band=20, clip=None):
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    if not (m1.any() and m2.any()):
        return 0.0

    def sdf(m):
        d_in = ndimage.distance_transform_edt(m)
        d_out = ndimage.distance_transform_edt(~m)
        s = d_in - d_out
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
    return float((corr + 1) / 2.0)


def chamfer_similarity(mask1, mask2, d0=25.0):
    b1 = boundary(mask1)
    b2 = boundary(mask2)

    if b1.sum() == 0 or b2.sum() == 0:
        return 0.0, np.inf

    edt1 = ndimage.distance_transform_edt(~b1)
    edt2 = ndimage.distance_transform_edt(~b2)

    d12 = float(edt2[b1].mean())
    d21 = float(edt1[b2].mean())
    avgd = 0.5 * (d12 + d21)

    sim = float(np.exp(-avgd / (d0 + 1e-8)))
    return sim, avgd


def surface_distance_metrics(mask1, mask2):
    b1 = extract_boundary(mask1)
    b2 = extract_boundary(mask2)
    if not (b1.any() and b2.any()):
        return 0.0, 0.0

    dt1 = ndimage.distance_transform_edt(~b1)
    dt2 = ndimage.distance_transform_edt(~b2)

    d12 = dt2[b1]
    d21 = dt1[b2]
    if d12.size == 0 or d21.size == 0:
        return 0.0, 0.0

    all_d = np.concatenate([d12, d21])
    asd = float(all_d.mean())
    hd = float(all_d.max())
    return asd, hd


def depth_similarity(mask1, mask2):
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    if not (m1.any() and m2.any()):
        return 0.0

    prof1 = m1.sum(axis=1).astype(float)
    prof2 = m2.sum(axis=1).astype(float)

    if prof1.sum() == 0 or prof2.sum() == 0:
        return 0.0

    prof1 /= prof1.sum()
    prof2 /= prof2.sum()

    dot = float(np.dot(prof1, prof2))
    nrm = float(np.linalg.norm(prof1) * np.linalg.norm(prof2))
    return dot / nrm if nrm > 0 else 0.0


def polar_signature(mask, n_bins=72):
    m = mask.astype(bool)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.zeros(n_bins, dtype=float)

    cx, cy = xs.mean(), ys.mean()
    dx = xs - cx
    dy = ys - cy
    angles = np.arctan2(dy, dx)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    radii = np.hypot(dx, dy)

    if radii.max() > 0:
        radii = radii / radii.max()

    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    sig = np.zeros(n_bins, dtype=float)
    inds = np.digitize(angles, bins) - 1
    inds = np.clip(inds, 0, n_bins - 1)

    for k in range(n_bins):
        mask_k = (inds == k)
        if np.any(mask_k):
            sig[k] = radii[mask_k].max()

    return sig


def polar_signature_similarity(mask1, mask2, n_bins=72):
    s1 = polar_signature(mask1, n_bins=n_bins)
    s2 = polar_signature(mask2, n_bins=n_bins)
    if np.all(s1 == 0) or np.all(s2 == 0):
        return 0.0

    s1 = s1 / (np.linalg.norm(s1) + 1e-8)
    s2 = s2 / (np.linalg.norm(s2) + 1e-8)
    return float(np.dot(s1, s2))


def shape_feature_metrics(mask1, mask2):
    def get_features(m):
        m = m.astype(np.uint8)
        if not np.any(m):
            return 0.0, np.zeros(7)

        area = np.sum(m)
        contours = measure.find_contours(m, 0.5)
        if not contours:
            return 0.0, np.zeros(7)

        perimeter = sum(len(c) for c in contours)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        import cv2
        moments = cv2.moments(m)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

        return circularity, hu_log

    c1, h1 = get_features(mask1)
    c2, h2 = get_features(mask2)

    circ_sim = 1.0 - abs(c1 - c2) / (max(c1, c2, 1e-8))
    hu_dist = np.linalg.norm(h1 - h2)
    hu_sim = np.exp(-hu_dist * 0.2)

    return float(np.clip(circ_sim, 0, 1)), float(hu_sim)


def thickness(mask, stat="median"):
    m = mask.astype(bool)
    if not m.any():
        return 0.0

    dt = distance_transform_edt(m)
    vals = dt[m]
    if vals.size == 0:
        return 0.0

    if stat == "mean":
        return float(vals.mean())
    elif stat == "p75":
        return float(np.percentile(vals, 75))
    else:
        return float(np.median(vals))


def area_sim_from_masks(mask1, mask2, eps=1e-9):
    a = float(np.sum(mask1)) + eps
    b = float(np.sum(mask2)) + eps
    return float(np.exp(-abs(np.log(a / b))))


def compute_all_metrics(mask_cluster, mask_atlas, dice_region):
    metrics = {}

    metrics["dice"] = float(dice_region)
    metrics["sdf_corr"] = sdf_corr_band(mask_cluster, mask_atlas, band=20, clip=None)
    metrics["boundary_iou"] = boundary_iou(mask_cluster, mask_atlas)

    chamfer_sim, chamfer_avgd = chamfer_similarity(mask_cluster, mask_atlas, d0=25.0)
    metrics["chamfer_sim"] = chamfer_sim
    metrics["chamfer_dist"] = chamfer_avgd

    asd, hd = surface_distance_metrics(mask_cluster, mask_atlas)
    metrics["asd"] = asd
    metrics["hd"] = hd

    metrics["depth_sim"] = depth_similarity(mask_cluster, mask_atlas)
    metrics["polar_sim"] = polar_signature_similarity(mask_cluster, mask_atlas)

    circ_sim, hu_sim = shape_feature_metrics(mask_cluster, mask_atlas)
    metrics["circ_sim"] = circ_sim
    metrics["hu_sim"] = hu_sim

    t_st = thickness(mask_cluster, stat="p75")
    t_at = thickness(mask_atlas, stat="p75")
    metrics["thick_st"] = t_st
    metrics["thick_at"] = t_at
    metrics["thick_sim"] = np.exp(-abs(np.log((t_st+1e-6)/(t_at+1e-6))))

    metrics["area_sim"] = area_sim_from_masks(mask_cluster, mask_atlas)
    metrics["shape_sim"] = (metrics["sdf_corr"] + metrics["hu_sim"] + metrics["circ_sim"]) / 3.0

    return metrics


def _weighted_sum(metrics, W):
    s = 0.0
    for k, w in W.items():
        v = metrics.get(k, 0.0)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        s += w * float(v)
    return float(s)


def _dist_penalty_from_chamfer_dist(cd):
    """
    Soft penalty from chamfer distance.
    """
    if cd is None or (isinstance(cd, float) and np.isnan(cd)):
        return 1.0

    cd = float(cd)
    t = 24.0
    s = 8.0
    return float(1.0 / (1.0 + np.exp((cd - t) / (s + 1e-8))))


def align_score(metrics):
    base = _weighted_sum(metrics, W_ALIGN)
    return float(np.clip(base, 0, 1))


def evaluate_cluster_candidates_generic(
    cluster_name,
    cluster_mask,
    sl,
    atlas_pre,
    candidate_df,
    topk=10,
    min_dice_for_eval=0.05,
    topn_by_dice=None,
    min_area_intersection=20
):
    """
    Evaluate one ST cluster against a generic set of atlas candidates.

    candidate_df must contain:
        candidate_type, candidate_name, label_set
    """
    if len(candidate_df) == 0:
        return pd.DataFrame()

    atlas_labels = atlas_pre["atlas_labels"]
    label2idx = atlas_pre["label2idx"]
    nL = atlas_pre["nL"]
    atlas_area = atlas_pre["atlas_area"]

    mc_bool = cluster_mask.astype(bool)
    Ac, inter_counts = precompute_counts_for_cluster(
        mc_bool=mc_bool,
        sl=sl,
        atlas_labels=atlas_labels,
        nL=nL
    )

    if Ac == 0:
        return pd.DataFrame()

    # -------------------------
    # Step 1: fast prescreen
    # -------------------------
    prescreen_rows = []

    for cand_idx, cand in candidate_df.iterrows():
        label_set = cand["label_set"]

        idxs = [label2idx[int(l)] for l in label_set if int(l) in label2idx]
        if len(idxs) == 0:
            continue

        idxs = np.asarray(idxs, dtype=int)
        inter = int(inter_counts[idxs].sum())
        if inter < min_area_intersection:
            continue

        areaB = int(atlas_area[idxs].sum())
        dice_region = (2.0 * inter / (Ac + areaB)) if (Ac + areaB) > 0 else 0.0

        prescreen_rows.append({
            "candidate_idx": cand_idx,
            "intersection": inter,
            "area_atlas": areaB,
            "dice": dice_region
        })

    if len(prescreen_rows) == 0:
        return pd.DataFrame()

    prescreen_df = pd.DataFrame(prescreen_rows).sort_values(
        ["dice", "intersection"],
        ascending=[False, False]
    ).reset_index(drop=True)

    if topn_by_dice is not None:
        prescreen_df = prescreen_df.head(topn_by_dice).copy()

    prescreen_df = prescreen_df[prescreen_df["dice"] >= min_dice_for_eval].copy()
    if len(prescreen_df) == 0:
        return pd.DataFrame()

    # -------------------------
    # Step 2: full metrics
    # -------------------------
    rows = []

    for _, rr in prescreen_df.iterrows():
        cand = candidate_df.loc[rr["candidate_idx"]]
        label_set = cand["label_set"]
        dice_region = float(rr["dice"])

        atlas_mask = mask_from_label_set(sl, label_set)
        if atlas_mask.sum() == 0:
            continue

        metrics = compute_all_metrics(cluster_mask, atlas_mask, dice_region=dice_region)
        score = align_score(metrics)

        row = {
            "cluster": cluster_name,
            "candidate_type": cand["candidate_type"],
            "candidate_name": cand["candidate_name"],
            "node_id": cand.get("node_id", np.nan),
            "depth": cand.get("depth", np.nan),
            "path": cand.get("path", ""),
            "acronym": cand.get("acronym", ""),
            "name": cand.get("name", ""),
            "n_leaf_labels": cand.get("n_leaf_labels", len(label_set)),

            "area_cluster": int(Ac),
            "area_atlas": int(rr["area_atlas"]),
            "intersection": int(rr["intersection"]),
            "align_score": float(score),
        }
        row.update(metrics)
        rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    res_df = pd.DataFrame(rows).sort_values(
        ["align_score", "dice", "sdf_corr", "chamfer_sim"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    if topk is not None:
        res_df = res_df.head(topk).copy()

    return res_df


def evaluate_all_clusters_generic(
    st_masks,
    sl,
    atlas_pre,
    candidate_df,
    topk_per_cluster=10,
    min_dice_for_eval=0.05,
    topn_by_dice=30,
    min_area_intersection=20,
    verbose=True
):
    all_rows = []

    cluster_names = list(st_masks.keys())
    for i, cl in enumerate(cluster_names, start=1):
        if verbose:
            print(f"[{i}/{len(cluster_names)}] evaluating cluster={cl}")

        res_df = evaluate_cluster_candidates_generic(
            cluster_name=cl,
            cluster_mask=st_masks[cl],
            sl=sl,
            atlas_pre=atlas_pre,
            candidate_df=candidate_df,
            topk=topk_per_cluster,
            min_dice_for_eval=min_dice_for_eval,
            topn_by_dice=topn_by_dice,
            min_area_intersection=min_area_intersection
        )

        if len(res_df) > 0:
            all_rows.append(res_df)

    if len(all_rows) == 0:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["cluster", "align_score"], ascending=[True, False]).reset_index(drop=True)
    return out


def run_dual_pair_pipeline(
    df_smooth,
    df2,
    atlas_info,
    x_col="x_aligned",
    y_col="y_aligned",
    label_col="banksy_cluster_refined",

    # mask controls
    mask_params=None,
    mask_params_thin=None,
    shape_type_col="shape_type",
    thin_values=("detail",),          # changed
    thin_rule="mode",                 # added

    min_prefix_len=2,
    depth_list=None,

    topk_prefix=10,
    topk_layer=6,

    min_dice_prefix=0.05,
    min_dice_layer=0.03,

    topn_prefix_by_dice=30,
    topn_layer_by_dice=None,

    min_area_intersection_prefix=20,
    min_area_intersection_layer=20,

    verbose=True
):
    sl = atlas_info["sl"]
    xJ = atlas_info["xJ"]
    yJ = atlas_info["yJ"]

    mask_result = build_cluster_masks(
        df_smooth=df_smooth,
        sl=sl,
        xJ=xJ,
        yJ=yJ,
        x_col=x_col,
        y_col=y_col,
        label_col=label_col,
        params=mask_params,
        params_thin=mask_params_thin,
        shape_type_col=shape_type_col,
        thin_values=thin_values,
        thin_rule=thin_rule,          # pass through
        verbose=verbose
    )

    st_masks = mask_result["st_masks"]
    st_soft = mask_result["st_soft"]

    atlas_pre = build_atlas_slice_precompute(sl)

    prefix_info = build_prefix_label_mapping(
        df2=df2,
        atlas_labels=atlas_pre["atlas_labels"],
        min_prefix_len=min_prefix_len
    )
    prefix_candidate_df = make_prefix_candidates(prefix_info=prefix_info, depth_list=depth_list)

    layer_to_labels = build_layer_label_mapping(df2=df2, atlas_labels=atlas_pre["atlas_labels"])
    layer_candidate_df = make_layer_candidates(layer_to_labels)

    if verbose:
        print("num prefix candidates:", len(prefix_candidate_df))
        print("num layer candidates:", len(layer_candidate_df))
        if len(layer_to_labels) > 0:
            print("layers in current slice:", sorted(layer_to_labels.keys()))

    pairs_df = evaluate_all_clusters_generic(
        st_masks=st_masks, sl=sl, atlas_pre=atlas_pre, candidate_df=prefix_candidate_df,
        topk_per_cluster=topk_prefix, min_dice_for_eval=min_dice_prefix,
        topn_by_dice=topn_prefix_by_dice, min_area_intersection=min_area_intersection_prefix,
        verbose=verbose
    )

    pairs_layer_df = evaluate_all_clusters_generic(
        st_masks=st_masks, sl=sl, atlas_pre=atlas_pre, candidate_df=layer_candidate_df,
        topk_per_cluster=topk_layer, min_dice_for_eval=min_dice_layer,
        topn_by_dice=topn_layer_by_dice, min_area_intersection=min_area_intersection_layer,
        verbose=verbose
    )

    return {
        "atlas_info": atlas_info,
        "mask_result": mask_result,
        "atlas_pre": atlas_pre,
        "prefix_info": prefix_info,
        "prefix_candidate_df": prefix_candidate_df,
        "layer_to_labels": layer_to_labels,
        "layer_candidate_df": layer_candidate_df,
        "pairs_df": pairs_df,
        "pairs_layer_df": pairs_layer_df,
        "st_masks": st_masks,
        "st_soft": st_soft,
    }


def get_best_pair_per_cluster(pair_df):
    if pair_df is None or len(pair_df) == 0:
        return pd.DataFrame()
    return (
        pair_df.sort_values(["cluster", "align_score"], ascending=[True, False])
               .groupby("cluster", as_index=False)
               .head(1)
               .reset_index(drop=True)
    )


def path_str_to_tuple(path_str):
    return tuple(int(p) for p in str(path_str).strip("/").split("/") if p)


def compute_gate_factor(
    dice,
    sdf_corr,
    asd,
    thick_sim,
    dice_soft=0.42,
    sdf_soft=0.55,
    asd_soft=50.0,
    thick_soft=0.65,
    min_factor=0.45,
    thick_power=1.0,
):
    """
    Compute a soft penalty factor in [min_factor, 1].

    Rules:
    - dice too low       -> penalize
    - sdf_corr too low   -> penalize
    - asd too high       -> penalize
    - thick_sim too low  -> penalize

    Parameters
    ----------
    dice_soft : float
        Soft threshold for dice. If dice < dice_soft, apply penalty.
    sdf_soft : float
        Soft threshold for sdf_corr. If sdf_corr < sdf_soft, apply penalty.
    asd_soft : float
        Soft threshold for asd. If asd > asd_soft, apply penalty.
    thick_soft : float
        Soft threshold for thick_sim. If thick_sim < thick_soft, apply penalty.
    min_factor : float
        Minimum allowed factor to avoid collapsing everything to zero.
    thick_power : float
        Optional extra strength for thickness gate.
    """
    factor = 1.0

    if not pd.isna(dice) and dice < dice_soft:
        factor *= max(float(dice) / float(dice_soft), min_factor)

    if not pd.isna(sdf_corr) and sdf_corr < sdf_soft:
        factor *= max(float(sdf_corr) / float(sdf_soft), min_factor)

    if not pd.isna(asd) and asd > asd_soft:
        factor *= max(float(asd_soft) / float(asd), min_factor)

    if not pd.isna(thick_sim) and thick_sim < thick_soft:
        thick_factor = max(float(thick_sim) / float(thick_soft), min_factor)
        thick_factor = thick_factor ** float(thick_power)
        factor *= thick_factor

    return max(factor, min_factor)


def build_pairs_from_best_df(
    best_df,
    prefix_meta=None,
    layer_to_labels=None,
    score_col="align_score",
    score_thresh=0.4,
    asd_thresh=None,
    use_gate=True,
    gated_score_col="align_score_gated",
    gate_params=None,
):
    """
    Build pair table from best_df with threshold filtering and optional gating.

    Parameters
    ----------
    score_thresh : float
        Minimum score required AFTER gate if use_gate=True,
        otherwise minimum raw align_score.
    asd_thresh : float or None
        Maximum ASD allowed (None = no hard filter)
    use_gate : bool
        Whether to apply soft gate to raw score
    gated_score_col : str
        Name of adjusted score column
    gate_params : dict or None
        Parameters for compute_gate_factor
    """

    if gate_params is None:
        gate_params = dict(
            dice_soft=0.42,
            sdf_soft=0.55,
            asd_soft=50.0,
            thick_soft=0.65,
            min_factor=0.45,
            thick_power=1.0,
        )

    rows = []

    for _, r in best_df.iterrows():

        # -------------------------
        # raw score
        # -------------------------
        score_raw = float(r.get(score_col, np.nan))
        if np.isnan(score_raw):
            continue

        # metrics used by gate
        dice = r.get("dice", np.nan)
        sdf_corr = r.get("sdf_corr", np.nan)
        asd = r.get("asd", np.nan)
        thick_sim = r.get("thick_sim", np.nan)

        # -------------------------
        # hard ASD filter
        # -------------------------
        if asd_thresh is not None and not np.isnan(asd):
            if asd > asd_thresh:
                continue

        # -------------------------
        # gate
        # -------------------------
        gate_factor = 1.0
        if use_gate:
            gate_factor = compute_gate_factor(
                dice=dice,
                sdf_corr=sdf_corr,
                asd=asd,
                thick_sim=thick_sim,
                **gate_params
            )

        score_gated = score_raw * gate_factor

        # threshold on final score used for selection
        score_used = score_gated if use_gate else score_raw
        if score_used < score_thresh:
            continue

        cl = str(r["cluster"])
        candidate_type = r.get("candidate_type", None)

        # ====================================================
        # prefix / leaf pair
        # ====================================================
        if candidate_type == "prefix" or (
            candidate_type is None and pd.notna(r.get("path", np.nan))
        ):

            path_str = r.get("path", None)
            if path_str is None or pd.isna(path_str):
                continue

            prefix = path_str_to_tuple(path_str)
            if prefix_meta is None or prefix not in prefix_meta:
                continue

            meta = prefix_meta[prefix]
            label_set = set(meta["label_set"])

            rows.append({
                "cluster": cl,
                "labels": sorted(label_set),

                "align_score": score_raw,
                gated_score_col: score_gated,
                "gate_factor": gate_factor,

                "pair_type": "leaf",
                "candidate_type": "prefix",
                "candidate_name": r.get("candidate_name", meta.get("acronym", "")),

                "node_id": meta.get("node_id", np.nan),
                "acronym": meta.get("acronym", ""),
                "name": meta.get("name", ""),
                "path": meta.get("path", path_str),

                "dice": dice,
                "sdf_corr": sdf_corr,
                "chamfer_sim": r.get("chamfer_sim", np.nan),
                "chamfer_dist": r.get("chamfer_dist", np.nan),
                "area_sim": r.get("area_sim", np.nan),
                "asd": asd,
                "hd": r.get("hd", np.nan),
                "thick_sim": thick_sim,
            })

        # ====================================================
        # layer pair
        # ====================================================
        elif candidate_type == "layer":

            lname = r.get("candidate_name", None)
            if lname is None or pd.isna(lname):
                continue

            if layer_to_labels is None or lname not in layer_to_labels:
                continue

            label_set = set(layer_to_labels[lname])

            rows.append({
                "cluster": cl,
                "labels": sorted(label_set),

                "align_score": score_raw,
                gated_score_col: score_gated,
                "gate_factor": gate_factor,

                "pair_type": "layer",
                "candidate_type": "layer",
                "candidate_name": lname,

                "node_id": np.nan,
                "acronym": lname,
                "name": lname,
                "path": "",

                "dice": dice,
                "sdf_corr": sdf_corr,
                "chamfer_sim": r.get("chamfer_sim", np.nan),
                "chamfer_dist": r.get("chamfer_dist", np.nan),
                "area_sim": r.get("area_sim", np.nan),
                "asd": asd,
                "hd": r.get("hd", np.nan),
                "thick_sim": thick_sim,
            })

    pairs_df = pd.DataFrame(rows)
    return pairs_df


def select_nonoverlap_pairs_by_score(
    pairs_df,
    score_col="align_score_gated",
    allow_same_cluster_multiple=False
):
    """
    Greedy selection of non-overlapping atlas label sets.
    Higher score wins.
    """
    if pairs_df is None or len(pairs_df) == 0:
        return pd.DataFrame(columns=pairs_df.columns if pairs_df is not None else None)

    df = pairs_df.copy()
    df["cluster"] = df["cluster"].astype(str)

    def _norm_labels(x):
        if isinstance(x, (list, tuple, set, np.ndarray)):
            return sorted(set(int(v) for v in x))
        if pd.isna(x):
            return []
        return [int(x)]

    df["labels"] = df["labels"].apply(_norm_labels)

    sort_cols = [score_col]
    ascending = [False]

    if "dice" in df.columns:
        sort_cols.append("dice")
        ascending.append(False)

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    selected_rows = []
    used_labels = set()
    used_clusters = set()

    for _, row in df.iterrows():
        cl = str(row["cluster"])
        labs = set(row["labels"])

        if len(labs) == 0:
            continue

        if (not allow_same_cluster_multiple) and (cl in used_clusters):
            continue

        if len(labs & used_labels) > 0:
            continue

        selected_rows.append(row.to_dict())
        used_labels.update(labs)
        used_clusters.add(cl)

    selected_df = pd.DataFrame(selected_rows)
    if len(selected_df) > 0:
        selected_df = selected_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    return selected_df


def build_pair_union_df(pairs_df, score_col="align_score_gated"):
    """
    Merge multiple rows per cluster into one union of atlas labels.
    Also keep best score row info for display.
    """
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=["cluster", "atlas_labels_union"])

    pairs_df = pairs_df.copy()
    pairs_df["cluster"] = pairs_df["cluster"].astype(str)

    rows = []
    for cl, sub in pairs_df.groupby("cluster"):
        sub = sub.sort_values(score_col, ascending=False).reset_index(drop=True)

        union_labels = sorted({lab for lst in sub["labels"] for lab in lst})
        best = sub.iloc[0]

        rows.append({
            "cluster": cl,
            "atlas_labels_union": union_labels,
            "pair_type": best.get("pair_type", ""),
            "candidate_name": best.get("candidate_name", ""),
            "align_score": best.get("align_score", np.nan),
            "align_score_gated": best.get("align_score_gated", np.nan),
            "gate_factor": best.get("gate_factor", np.nan),
            "dice": best.get("dice", np.nan),
            "sdf_corr": best.get("sdf_corr", np.nan),
            "chamfer_sim": best.get("chamfer_sim", np.nan),
            "chamfer_dist": best.get("chamfer_dist", np.nan),
            "area_sim": best.get("area_sim", np.nan),
            "asd": best.get("asd", np.nan),
            "hd": best.get("hd", np.nan),
            "thick_sim": best.get("thick_sim", np.nan),
        })

    return pd.DataFrame(rows)


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


def preprocess_st_mask_channel(m, sigma_pre=1.4, thr=0.5, close_r=2, open_r=1, min_area=50):
    m = m.astype(bool)
    if min_area > 0:
        m = remove_small_objects(m, min_size=int(min_area))
    if close_r > 0:
        m = binary_closing(m, disk(int(close_r)))
    if open_r > 0:
        m = binary_opening(m, disk(int(open_r)))
    f = gaussian_filter(m.astype(np.float32), sigma=float(sigma_pre))
    return (f > float(thr)).astype(np.uint8)


def preprocess_he_mask_channel(m, sigma_pre=0.4, thr=0.5, close_r=1, open_r=1, min_area=80):
    m = m.astype(bool)
    if min_area > 0:
        m = remove_small_objects(m, min_size=int(min_area))
    if close_r > 0:
        m = binary_closing(m, disk(int(close_r)))
    if open_r > 0:
        m = binary_opening(m, disk(int(open_r)))
    f = gaussian_filter(m.astype(np.float32), sigma=float(sigma_pre))
    return (f > float(thr)).astype(np.uint8)


def preprocess_onehot_asymmetric(source_onehot, target_onehot, st_cfg=None, he_cfg=None):
    st_cfg = st_cfg or {}
    he_cfg = he_cfg or {}

    source_clean = np.zeros_like(source_onehot, dtype=np.uint8)
    target_clean = np.zeros_like(target_onehot, dtype=np.uint8)

    for c in range(source_onehot.shape[0]):
        source_clean[c] = preprocess_st_mask_channel(source_onehot[c], **st_cfg)
        target_clean[c] = preprocess_he_mask_channel(target_onehot[c], **he_cfg)

    return source_clean, target_clean


def channel_weights_from_area(onehot, power=0.9, w_min=0.25, w_max=4.5, eps=1e-6):
    area = onehot.reshape(onehot.shape[0], -1).sum(axis=1).astype(np.float32)
    w = 1.0 / np.power(area + eps, power)
    w = w / (w.mean() + eps)
    w = np.clip(w, w_min, w_max)
    w = w / (w.mean() + eps)
    return w.astype(np.float32)


def apply_channel_weights(X, w):
    return X * w[:, None, None]


def parse_labels_any(x):
    if isinstance(x, (list, tuple, np.ndarray, set)):
        return [int(v) for v in x]
    s = str(x).strip()
    if s == '' or s.lower() == 'nan':
        return []
    s = s.replace(' ', '')
    for sep in [';', ',']:
        if sep in s:
            return [int(v) for v in s.split(sep) if v != '']
    try:
        return [int(float(s))]
    except Exception:
        return []


def build_pair_onehot_from_pair_df(pair_df, st_masks, sl, add_global_channel=True):
    if pair_df is None or len(pair_df) == 0:
        raise ValueError('pair_df is empty. Build pair_df first.')

    H, W = sl.shape
    rows = []

    for _, r in pair_df.iterrows():
        cl = str(r.get('cluster', '')).strip()
        if cl == '' or cl not in st_masks:
            continue

        labs = parse_labels_any(r.get('atlas_labels_union', []))
        if len(labs) == 0:
            continue

        source_mask = (st_masks[cl] > 0).astype(np.uint8)
        target_mask = np.isin(sl, labs).astype(np.uint8)

        if source_mask.sum() == 0 or target_mask.sum() == 0:
            continue

        rows.append({
            'cluster': cl,
            'labels': labs,
            'source_mask': source_mask,
            'target_mask': target_mask,
        })

    if len(rows) == 0:
        raise ValueError('No valid channels from pair_df. Check cluster ids and atlas_labels_union.')

    C = len(rows) + (1 if add_global_channel else 0)
    source_onehot = np.zeros((C, H, W), dtype=np.uint8)
    target_onehot = np.zeros((C, H, W), dtype=np.uint8)

    for ch, row in enumerate(rows):
        source_onehot[ch] = row['source_mask']
        target_onehot[ch] = row['target_mask']

    if add_global_channel:
        source_onehot[-1] = np.logical_or.reduce([(m > 0) for m in st_masks.values()]).astype(np.uint8)
        target_onehot[-1] = (sl > 0).astype(np.uint8)

    return source_onehot, target_onehot, rows


def affine_from_components(L, T):
    """Build 2D homogeneous affine matrix from linear part and translation.

    Parameters
    ----------
    L : tensor, shape (2,2)
        Linear component.
    T : tensor, shape (2,)
        Translation component in (y, x) order.
    """
    A = torch.eye(3, device=L.device, dtype=L.dtype)
    A[:2, :2] = L
    A[:2, 2] = T
    return A


def clip(I):
    Ic = torch.clone(I)
    Ic[Ic < 0] = 0
    Ic[Ic > 1] = 1
    return Ic


def sample_image_on_coords(grid_coords, values, query_yx, padding_mode="zeros", mode="bilinear", align_corners=True):
    # grid_coords: [y_coords, x_coords]
    # values: (C,H,W)
    # query_yx: (2,*,*) in physical coords, channel order [y, x]
    values = torch.as_tensor(values)
    query_yx = torch.as_tensor(query_yx).clone()

    for dim in range(2):
        query_yx[dim] -= grid_coords[dim][0]
        query_yx[dim] /= (grid_coords[dim][-1] - grid_coords[dim][0])
    query_yx = query_yx * 2.0 - 1.0

    sampling_grid = query_yx.flip(0).permute((1, 2, 0))[None]
    out = grid_sample(
        values[None],
        sampling_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return out[0]


def central_diff(f, dx, dim):
    # f: (..., H, W)
    pad = [0, 0, 0, 0]
    if dim == -1:  # x/cols
        # pad W: left/right replicate
        left = f[..., :1]
        right = f[..., -1:]
        f_pad = torch.cat([left, f, right], dim=-1)
        return (f_pad[..., 2:] - f_pad[..., :-2])/(2.0*dx)
    else:          # y/rows
        top = f[..., :1, :]
        bottom = f[..., -1:, :]
        f_pad = torch.cat([top, f, bottom], dim=-2)
        return (f_pad[..., 2:, :] - f_pad[..., :-2, :])/(2.0*dx)


def jacobian_and_div(v, dv):
    # v: (H,W,2), dv: tensor([dy, dx])
    vy = v[..., 0]
    vx = v[..., 1]
    dy, dx = dv[0], dv[1]
    dvx_dx = central_diff(vx, dx, dim=-1)
    dvx_dy = central_diff(vx, dy, dim=-2)
    dvy_dx = central_diff(vy, dx, dim=-1)
    dvy_dy = central_diff(vy, dy, dim=-2)
    # shape alignment: central_diff keeps (H,W)
    div_v = dvx_dx + dvy_dy
    # (∇v)^T m term components will use these
    return dvx_dx, dvx_dy, dvy_dx, dvy_dy, div_v


def build_kernel_K(xv, a=500.0, p=2.0):
    # xv: [y_coords, x_coords]
    y, x = xv
    dy = (y[1]-y[0]).to(y.device)
    dx = (x[1]-x[0]).to(x.device)
    H, W = len(y), len(x)

    # frequency grid in physical units matching earlier code
    fy = torch.arange(H, device=y.device, dtype=y.dtype)/(H*dy)
    fx = torch.arange(W, device=x.device, dtype=x.dtype)/(W*dx)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')
    # LL like in your code
    LL = (1.0 + 2.0*a**2 * ((1.0 - torch.cos(2.0*np.pi*FY*dy))/(dy**2) +
                             (1.0 - torch.cos(2.0*np.pi*FX*dx))/(dx**2)))**(p*2.0)
    K = 1.0/LL  # scalar kernel
    return K


def apply_K(m, K):
    # m: (H,W,2), K: (H,W). Apply per-channel via FFT
    v = []
    for c in range(2):
        Mf = torch.fft.fftn(m[..., c], dim=(-2, -1))
        vf = Mf * K
        v_c = torch.fft.ifftn(vf, dim=(-2, -1)).real
        v.append(v_c)
    v = torch.stack(v, dim=-1)
    return v


def advect_field(F, v, xv, dt):
    # F: (C,H,W)  (e.g., C=2 for vector m)
    # v: (H,W,2)
    H, W = v.shape[:2]
    Y, X = xv
    XV = torch.stack(torch.meshgrid(Y, X, indexing='ij'), -1)  # (H,W,2)
    back = (XV - v*dt).permute(2,0,1)                          # (2,H,W)
    return sample_image_on_coords(xv, F, back, padding_mode="border")


def geodesic_shooting(m0, xv, nt, K, dv):
    dt = 1.0/nt
    m = m0
    v_list, m_list = [], []
    for t in range(nt):
        v = apply_K(m, K)            # v = K * m
        v_list.append(v)
        m_list.append(m)
        dvx_dx, dvx_dy, dvy_dx, dvy_dy, div_v = jacobian_and_div(v, dv)
        mx, my = m[..., 0], m[..., 1]
        Sx = dvx_dx*mx + dvy_dx*my
        Sy = dvx_dy*mx + dvy_dy*my
        S = torch.stack((Sx, Sy), dim=-1)
        S = S + m*div_v[..., None]
        m_tilde = (m - dt*S).permute(2, 0, 1)
        m_next = advect_field(m_tilde, v, xv, dt).permute(1, 2, 0)
        m = m_next
    return v_list, m_list


def LDDMM_shooting_mixture(
    x_src, source_image, x_tgt, target_image,
    affine_init=None, m0_init=None, velocity_grid=None,
    model_cfg=None, optim_cfg=None, em_cfg=None, intensity_cfg=None,
    device='cpu', dtype=torch.float64, muB=None, muA=None,
    verbose=True, print_every=100,
):
    """Shooting-based LDDMM with config-style parameter groups.

    Parameters
    ----------
    x_src : list[1D tensor/array]
        Source image grid coordinates in physical units: [y_coords, x_coords].
    source_image : tensor/array, shape (C, H_src, W_src)
        Source multi-channel image on ``x_src``.
    x_tgt : list[1D tensor/array]
        Target image grid coordinates in physical units: [y_coords, x_coords].
    target_image : tensor/array, shape (C, H_tgt, W_tgt)
        Target multi-channel image on ``x_tgt``.
    affine_init : tensor/array or None, shape (3,3) or (2,3)
        Initial affine transform mapping source -> target.
        If None, identity affine is used.
    m0_init : tensor/array or None, shape (H_v, W_v, 2)
        Initial momentum for geodesic shooting.
        If None, initialized to zeros.
    velocity_grid : list[1D tensor/array] or None
        Grid for velocity/momentum field [y_coords, x_coords].
        If None, built automatically from source extent and ``model_cfg``.
    model_cfg : dict or None
        Geometric/model settings:
        ``a`` (smoothness scale), ``p`` (operator power),
        ``expand`` (velocity-domain expansion factor), ``nt`` (time steps),
        ``grid_step`` (velocity grid spacing).
    optim_cfg : dict or None
        Optimization schedule settings:
        ``niter``, ``diffeo_start``, ``lrL``, ``lrT``, ``lrM``,
        ``affine_slowdown``, ``grad_clip_m0``, ``lrM_decay``, ``lrM_min``.
    em_cfg : dict or None
        EM update schedule for mixture weights:
        ``update_every`` (iterations), ``start_iter`` (warmup end).
    intensity_cfg : dict or None
        Data-term and regularization scales:
        ``sigmaM`` (match), ``sigmaA``/``sigmaB`` (appearance classes),
        and ``sigmaR`` (deformation regularization).
    device : str
        Torch device (e.g., ``'cpu'`` or ``'cuda:0'``).
    dtype : torch dtype
        Numeric precision used internally.
    muA, muB : tensor/array or None
        Optional fixed appearance means for A/B mixture classes.
        If None, estimated during EM updates.
    draw_every : int
        Reserved display cadence parameter.
    verbose : bool
        Whether to print optimization progress.
    print_every : int
        Iteration interval for log printing.

    Returns
    -------
    out : dict
        Registration results including affine ``A``, velocity ``v``,
        momentum ``m0``, transformed coordinates, and energy history.
    """

    def _merge_cfg(defaults, user_cfg):
        cfg = dict(defaults)
        if user_cfg is not None:
            cfg.update(user_cfg)
        return cfg

    def _to_tensor(x, *, requires_grad=False):
        t = torch.as_tensor(x, device=device, dtype=dtype)
        if requires_grad:
            t = t.detach().clone().requires_grad_(True)
        return t

    model = _merge_cfg(
        {
            'a': 500.0,
            'p': 2.0,
            'expand': 2.0,
            'nt': 5,
            'grid_step': None,
        },
        model_cfg,
    )
    optim = _merge_cfg(
        {
            'niter': 500,
            'diffeo_start': 0,
            'lrL': 2e-8,
            'lrT': 2e-1,
            'lrM': 2e3,
            'affine_slowdown': 10.0,
            'grad_clip_m0': None,
            'lrM_decay': 1.0,
            'lrM_min': 2e3,
        },
        optim_cfg,
    )
    em = _merge_cfg(
        {
            'update_every': 5,
            'start_iter': 50,
        },
        em_cfg,
    )
    intensity = _merge_cfg(
        {
            'sigmaM': 1.0,
            'sigmaB': 2.0,
            'sigmaA': 5.0,
            'sigmaR': 5e5,
        },
        intensity_cfg,
    )

    a = model['a']
    p = model['p']
    expand = model['expand']
    nt = int(model['nt'])
    grid_step = model['grid_step']

    niter = int(optim['niter'])
    diffeo_start = int(optim['diffeo_start'])
    lrL = optim['lrL']
    lrT = optim['lrT']
    lrM = optim['lrM']
    affine_slowdown = optim['affine_slowdown']
    grad_clip_m0 = optim['grad_clip_m0']
    lrM_decay = optim['lrM_decay']
    lrM_min = optim['lrM_min']

    em_update_every = int(em['update_every'])
    em_start = int(em['start_iter'])

    sigmaM = intensity['sigmaM']
    sigmaB = intensity['sigmaB']
    sigmaA = intensity['sigmaA']
    sigmaR = intensity['sigmaR']

    # Tensorize image grids + images
    x_src = [_to_tensor(x) for x in x_src]
    x_tgt = [_to_tensor(x) for x in x_tgt]
    source_image = _to_tensor(source_image)
    target_image = _to_tensor(target_image)

    # Affine init (from affine_init matrix, or identity)
    if affine_init is None:
        L0 = torch.eye(2, device=device, dtype=dtype)
        T0 = torch.zeros(2, device=device, dtype=dtype)
    else:
        A0 = _to_tensor(affine_init)
        if A0.shape == (3, 3):
            L0, T0 = A0[:2, :2], A0[:2, -1]
        elif A0.shape == (2, 3):
            L0, T0 = A0[:2, :2], A0[:2, -1]
        else:
            raise ValueError(f'affine_init must be (3,3) or (2,3), got {tuple(A0.shape)}')

    L = L0.detach().clone().requires_grad_(True)
    T = T0.detach().clone().requires_grad_(True)

    # Velocity grid init
    if velocity_grid is None:
        lo = torch.stack([x[0] for x in x_src])
        hi = torch.stack([x[-1] for x in x_src])
        ctr = 0.5 * (lo + hi)
        rad = 0.5 * (hi - lo) * expand

        if grid_step is None:
            grid_step = float(a) * 0.5

        xv = []
        for c, r in zip(ctr, rad):
            g = torch.arange(c - r, c + r, grid_step, device=device, dtype=dtype)
            if g.numel() < 3:
                g = torch.linspace(c - r, c + r, steps=3, device=device, dtype=dtype)
            xv.append(g)
    else:
        xv = [_to_tensor(g) for g in velocity_grid]

    dv = torch.stack([g[1] - g[0] for g in xv])
    DV = torch.prod(dv)

    # Kernel for v = K * m
    K = build_kernel_K(xv, a=a, p=p)

    # Initial momentum
    H, W = len(xv[0]), len(xv[1])
    if m0_init is None:
        m0 = torch.zeros((H, W, 2), device=device, dtype=dtype, requires_grad=True)
    else:
        m0 = _to_tensor(m0_init, requires_grad=True)

    # Target mesh
    XJ = torch.stack(torch.meshgrid(*x_tgt, indexing='ij'), -1)

    # Mixture weights
    WM = torch.ones(target_image[0].shape, dtype=target_image.dtype, device=target_image.device)*0.5
    WB = torch.ones(target_image[0].shape, dtype=target_image.dtype, device=target_image.device)*0.4
    WA = torch.ones(target_image[0].shape, dtype=target_image.dtype, device=target_image.device)*0.1

    estimate_muA = muA is None
    estimate_muB = muB is None

    Esave = []
    t_start = time.time()

    if verbose:
        print(f"[LDDMM] start: niter={niter}, nt={nt}, device={device}, source_shape={tuple(source_image.shape)}, target_shape={tuple(target_image.shape)}")

    # Optimization loop (GD for L, T, m0)
    for it in range(niter):
        # Compose affine inverse once per iter
        A = affine_from_components(L, T)
        Ai = torch.linalg.inv(A)

        # Shooting: build v(t), m(t) from m0
        v_list, m_list = geodesic_shooting(m0, xv, nt, K, dv)  # lists of (H,W,2)

        # Warp coordinates XJ back to source frame using v(t) and affine inverse
        Xs = (Ai[:2,:2] @ XJ[..., None])[..., 0] + Ai[:2, -1]
        for t in range(nt-1, -1, -1):
            v_t = torch.stack((v_list[t][...,0], v_list[t][...,1]), dim=0)  # (2,H,W)
            Xs = Xs + sample_image_on_coords(xv, -v_t, Xs.permute(2,0,1)).permute(1,2,0)/nt

        # Resample source image
        source_warped = sample_image_on_coords(x_src, source_image, Xs.permute(2,0,1), padding_mode="border")

        # Linear contrast estimation (same as your original)
        B = torch.ones(1 + source_warped.shape[0], source_warped.shape[1]*source_warped.shape[2], device=source_warped.device, dtype=source_warped.dtype)
        B[1:source_warped.shape[0]+1] = source_warped.reshape(source_warped.shape[0], -1)
        with torch.no_grad():
            BB = B @ (B*WM.ravel()).T
            BJ = B @ ((target_image*WM).reshape(target_image.shape[0], -1)).T
            small = 0.1
            coeffs = torch.linalg.solve(BB + small*torch.eye(BB.shape[0], device=BB.device, dtype=BB.dtype), BJ)
        source_warped_fit = ((B.T @ coeffs).T).reshape(target_image.shape)

        # Energies
        EM = torch.sum((source_warped_fit - target_image)**2 * WM) / (2.0 * (sigmaM**2))
        if EM.dim() != 0:
            EM = EM.sum()  # guard: ensure scalar
        # Regularizer: integrate over time using <m,v> = <Lv, v> (avoid FFT autograd)
        dt = 1.0/nt
        ER_terms = [torch.sum(m_list[t] * v_list[t]) * DV * dt for t in range(nt)]
        ER = (0.5/(sigmaR**2)) * torch.stack(ER_terms).sum()

        E = EM + ER
        E_scalar = E
        if E_scalar.dim() != 0:
            E_scalar = E_scalar.sum()
        tosave = [E_scalar.detach().cpu().item(), EM.detach().cpu().item(), ER.detach().cpu().item()]

        Esave.append(tosave)

        if verbose and ((it % print_every == 0) or (it == niter - 1)):
            msg = f"[LDDMM] iter {it+1}/{niter} E={tosave[0]:.6e} EM={tosave[1]:.6e} ER={tosave[2]:.6e}"
            print(msg)

        # Backprop
        E.backward()
        with torch.no_grad():
            # explicit schedules (no boolean-arithmetic expressions)
            diffeo_on = (it >= diffeo_start)
            if diffeo_on:
                affine_scale = 1.0 / affine_slowdown
                m_steps = it - diffeo_start
                lrM_t = max(lrM_min, lrM * (lrM_decay ** m_steps))
            else:
                affine_scale = 1.0
                lrM_t = 0.0

            # affine updates
            L -= (lrL * affine_scale) * L.grad
            T -= (lrT * affine_scale) * T.grad
            L.grad.zero_(); T.grad.zero_()

            # initial momentum update with optional clipping
            if torch.isfinite(m0.grad).all() and lrM_t > 0.0:
                if grad_clip_m0 is not None:
                    gnorm = torch.linalg.vector_norm(m0.grad)
                    if torch.isfinite(gnorm) and gnorm > grad_clip_m0:
                        m0.grad.mul_(grad_clip_m0 / (gnorm + 1e-12))
                m0 -= lrM_t * m0.grad
            m0.grad.zero_()

        # EM updates for mixture weights
        if not it % em_update_every:
            with torch.no_grad():
                if estimate_muA:
                    muA = torch.sum(WA*target_image, dim=(-1,-2))/torch.sum(WA)
                if estimate_muB:
                    muB = torch.sum(WB*target_image, dim=(-1,-2))/torch.sum(WB)

                if it >= em_start:
                    W = torch.stack((WM, WA, WB))
                    pi = torch.sum(W, dim=(1,2))
                    pi += torch.max(pi)*1e-6
                    pi /= torch.sum(pi)

                    WMn = pi[0]* torch.exp(-torch.sum((source_warped_fit - target_image)**2, 0)/2.0/(sigmaM**2)) / (np.sqrt(2.0*np.pi*(sigmaM**2))**target_image.shape[0])
                    WAn = pi[1]* torch.exp(-torch.sum((muA[...,None,None] - target_image)**2, 0)/2.0/(sigmaA**2)) / (np.sqrt(2.0*np.pi*(sigmaA**2))**target_image.shape[0])
                    WBn = pi[2]* torch.exp(-torch.sum((muB[...,None,None] - target_image)**2, 0)/2.0/(sigmaB**2)) / (np.sqrt(2.0*np.pi*(sigmaB**2))**target_image.shape[0])

                    WS = WMn + WAn + WBn
                    WS += torch.max(WS)*1e-6
                    WM, WA, WB = WMn/WS, WAn/WS, WBn/WS

    # Final A and last v/m fields (from m0)
    A = affine_from_components(L, T).clone().detach()
    v_list, m_list = geodesic_shooting(m0.detach(), xv, nt, K, dv)
    v0 = torch.stack(v_list, dim=0).detach()  # (nt,H,W,2)
    m_all = torch.stack(m_list, dim=0).detach()

    elapsed_sec = time.time() - t_start
    final_E = Esave[-1][0] if len(Esave) > 0 else np.nan

    if verbose:
        print(f"[LDDMM] done in {elapsed_sec:.1f}s. finalE={final_E:.6e}")

    return {
        'A': A,
        'm0': m0.detach(),
        'v': v0,
        'm': m_all,
        'xv': xv,
        'WM': WM.clone().detach(),
        'WB': WB.clone().detach(),
        'WA': WA.clone().detach(),
        'Esave': Esave,
        'elapsed_sec': elapsed_sec,
    }


def LDDMM_shooting(
    x_src, source_image, x_tgt, target_image,
    affine_init=None, m0_init=None, velocity_grid=None,
    model_cfg=None, optim_cfg=None, em_cfg=None, intensity_cfg=None,
    device='cpu', dtype=torch.float64, muB=None, muA=None,
    verbose=True, print_every=100,
):
    """Atlas-compatible entry point backed by package-local LDDMM code."""
    loss_mode = str((intensity_cfg or {}).get("loss_mode", "feature"))
    if loss_mode == "mixture":
        return LDDMM_shooting_mixture(
            x_src=x_src,
            source_image=source_image,
            x_tgt=x_tgt,
            target_image=target_image,
            affine_init=affine_init,
            m0_init=m0_init,
            velocity_grid=velocity_grid,
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            em_cfg=em_cfg,
            intensity_cfg=intensity_cfg,
            device=device,
            dtype=dtype,
            muB=muB,
            muA=muA,
            verbose=verbose,
            print_every=print_every,
        )
    if loss_mode not in {"feature", "robust_cluster"}:
        raise ValueError(f"Unknown Atlas LDDMM loss_mode: {loss_mode!r}")

    feature_intensity_cfg = dict(intensity_cfg or {})
    if loss_mode == "feature":
        feature_intensity_cfg.setdefault("energy_type", "feature")
        feature_intensity_cfg.setdefault("feature_loss", "charbonnier")
        feature_intensity_cfg.setdefault("feature_eps", 1e-3)
        feature_intensity_cfg.setdefault("spatial_weight_mode", "none")
    else:
        feature_intensity_cfg.setdefault("energy_type", "robust_cluster")
        feature_intensity_cfg.setdefault("loss_mode", "robust_cluster")
        # Atlas uses pair-mask signed distance channels. The final channel is a
        # global SDF, not a density/tissue channel, so disable density handling.
        feature_intensity_cfg.setdefault("use_density_channel", False)

    from ._slddmm_core import LDDMM_shooting as package_lddmm_shooting

    return package_lddmm_shooting(
        x_src=x_src,
        I=source_image,
        x_tgt=x_tgt,
        J=target_image,
        affine_init=affine_init,
        m0_init=m0_init,
        velocity_grid=velocity_grid,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        em_cfg=em_cfg,
        intensity_cfg=feature_intensity_cfg,
        device=device,
        dtype=dtype,
        verbose=verbose,
        print_every=print_every,
    )


def _normalize_point_transform_inputs(velocity_grid, velocity_field, affine_matrix, points):
    ref = None
    if torch.is_tensor(velocity_field):
        ref = velocity_field
    elif isinstance(velocity_grid, (list, tuple)) and len(velocity_grid) > 0 and torch.is_tensor(velocity_grid[0]):
        ref = velocity_grid[0]
    elif torch.is_tensor(affine_matrix):
        ref = affine_matrix

    device = ref.device if ref is not None else torch.device('cpu')
    dtype = ref.dtype if ref is not None else torch.float32

    def _to_tensor(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    if not isinstance(velocity_grid, (list, tuple)) or len(velocity_grid) != 2:
        raise ValueError('velocity_grid must be [y_coords, x_coords]')

    vg = [_to_tensor(g) for g in velocity_grid]
    vf = _to_tensor(velocity_field)
    A = _to_tensor(affine_matrix)
    pts = _to_tensor(points)

    if vf.ndim != 4 or vf.shape[-1] != 2:
        raise ValueError(f'velocity_field must have shape (nt,H,W,2), got {tuple(vf.shape)}')
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f'points must have shape (N,2), got {tuple(pts.shape)}')

    if A.shape == (2, 3):
        A_h = torch.eye(3, device=device, dtype=dtype)
        A_h[:2, :] = A
        A = A_h
    elif A.shape != (3, 3):
        raise ValueError(f'affine_matrix must be (3,3) or (2,3), got {tuple(A.shape)}')

    return vg, vf, A, pts


def map_points_source_to_target(velocity_grid, velocity_field, affine_matrix, source_points):
    vg, vf, A, pts = _normalize_point_transform_inputs(velocity_grid, velocity_field, affine_matrix, source_points)

    nt = vf.shape[0]
    out_pts = pts.clone()
    for t in range(nt):
        v_t = vf[t].permute(2, 0, 1)  # (2,H,W)
        disp = sample_image_on_coords(vg, v_t, out_pts.T[..., None], padding_mode='border')[..., 0].T
        out_pts = out_pts + disp / nt

    out_pts = (A[:2, :2] @ out_pts.T + A[:2, 2:3]).T
    return out_pts


def map_points_target_to_source(velocity_grid, velocity_field, affine_matrix, target_points):
    vg, vf, A, pts = _normalize_point_transform_inputs(velocity_grid, velocity_field, affine_matrix, target_points)

    A_inv = torch.linalg.inv(A)
    out_pts = (A_inv[:2, :2] @ pts.T + A_inv[:2, 2:3]).T

    nt = vf.shape[0]
    for t in range(nt - 1, -1, -1):
        minus_v_t = (-vf[t]).permute(2, 0, 1)
        disp = sample_image_on_coords(vg, minus_v_t, out_pts.T[..., None], padding_mode='border')[..., 0].T
        out_pts = out_pts + disp / nt

    return out_pts


def transform_points_source_to_target(xv, v, A, points):
    # Backward-compatible wrapper name used by this notebook
    return map_points_source_to_target(xv, v, A, points)

# Notebook-derived executable sections.

ITERATIVE_ALIGNMENT_CELL_SOURCE = '# Iterative multi-level alignment (auto levels from notebook setup)\nimport re\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom scipy.ndimage import zoom\n\n# ------------------------------------------------------------\n# helpers from v4 final-stage input recipe\n# ------------------------------------------------------------\ndef _area_balance_weights(source_onehot, target_onehot, power=0.8, w_min=0.5, w_max=2.5, eps=1e-6):\n    aS = source_onehot.reshape(source_onehot.shape[0], -1).sum(axis=1).astype(np.float64)\n    aT = target_onehot.reshape(target_onehot.shape[0], -1).sum(axis=1).astype(np.float64)\n    area = 0.5 * (aS + aT) + eps\n    w = (np.median(area) / area) ** power\n    w = np.clip(w, w_min, w_max).astype(np.float32)\n    return w\n\n\ndef _build_equalized_sdt(source_onehot, target_onehot, clip_dist=4.0, sigma_sdt=0.9, band=4.0, tau=2.0):\n    source_sdt = onehot_to_sdt(source_onehot, clip_dist=clip_dist, sigma_sdt=sigma_sdt).astype(np.float32)\n    target_sdt = onehot_to_sdt(target_onehot, clip_dist=clip_dist, sigma_sdt=sigma_sdt).astype(np.float32)\n\n    C = source_sdt.shape[0]\n    for c in range(C):\n        m = (np.abs(source_sdt[c]) <= band) | (np.abs(target_sdt[c]) <= band)\n        if m.any():\n            s = np.sqrt(0.5 * (np.mean(source_sdt[c][m] ** 2) + np.mean(target_sdt[c][m] ** 2))) + 1e-6\n            source_sdt[c] /= s\n            target_sdt[c] /= s\n\n    source_sdt = np.tanh(source_sdt / tau)\n    target_sdt = np.tanh(target_sdt / tau)\n    return source_sdt, target_sdt\n\n\n# ------------------------------------------------------------\n# 0) Resolve stage labels from current notebook settings\n#    If n_levels=3 -> [k_small, k_mid, final]\n#    If n_levels=4 -> [k1, k2, k3, final]\n# ------------------------------------------------------------\ndef _resolve_stage_labels(df_ref, refined_cols_var=None, final_col=\'banksy_cluster_refined\'):\n    stage_cols = []\n\n    if \'STAGE_LABELS\' in globals() and STAGE_LABELS is not None and len(STAGE_LABELS) > 0:\n        for c in STAGE_LABELS:\n            if c in df_ref.columns:\n                stage_cols.append(c)\n        if len(stage_cols) > 0:\n            return stage_cols\n\n    if refined_cols_var is not None:\n        for c in refined_cols_var:\n            if c in df_ref.columns:\n                stage_cols.append(c)\n\n    if len(stage_cols) == 0:\n        k_cols = []\n        pat = re.compile(r\'^banksy_cluster_refined_k(\\d+)$\')\n        for c in df_ref.columns:\n            m = pat.match(str(c))\n            if m:\n                k_cols.append((int(m.group(1)), c))\n        k_cols = sorted(k_cols, key=lambda t: t[0])\n        stage_cols = [c for _, c in k_cols]\n\n    if final_col in df_ref.columns and final_col not in stage_cols:\n        stage_cols.append(final_col)\n\n    return stage_cols\n\n\nbase_all = df_prealign_nofilter.copy() if \'df_prealign_nofilter\' in globals() else df_final.copy()\nbase_filtered = df_final_filtered.copy() if \'df_final_filtered\' in globals() else df_smooth.copy()\ndf_pre_iter_all = base_all.copy()  # keep snapshot for before/after visualization\n\nstage_labels = _resolve_stage_labels(base_all, refined_cols if \'refined_cols\' in globals() else None)\nif len(stage_labels) == 0:\n    raise ValueError(\'No stage labels found. Expected refined k-level columns and/or banksy_cluster_refined.\')\n\nprint(\'[iter] stage labels:\', stage_labels)\nprint(\'[iter] total stages:\', len(stage_labels))\n\n# ------------------------------------------------------------\n# 1) Params: base(_2) for non-final stages, v4 for final stage\n# ------------------------------------------------------------\nBASE_W_ALIGN = dict(W_ALIGN) if \'W_ALIGN\' in globals() else dict(\n    sdf_corr=0.08,\n    chamfer_sim=0.06,\n    dice=0.18,\n    area_sim=0.47,\n    thick_sim=0.21,\n)\n\nFINAL_W_ALIGN_V4 = dict(\n    sdf_corr=0.08,\n    chamfer_sim=0.06,\n    dice=0.18,\n    area_sim=0.47,\n    thick_sim=0.21,\n)\n\nBASE_PAIR_CFG = dict(\n    min_prefix_len=2,\n    depth_list=None,\n    topk_prefix=10,\n    topk_layer=6,\n    min_dice_prefix=0.05,\n    min_dice_layer=0.03,\n    topn_prefix_by_dice=30,\n    topn_layer_by_dice=None,\n    min_area_intersection_prefix=20,\n    min_area_intersection_layer=20,\n)\n\nBASE_GATE_PARAMS = dict(\n    dice_soft=0.3,\n    sdf_soft=0.55,\n    asd_soft=50.0,\n    thick_soft=0.65,\n    min_factor=0.45,\n    thick_power=1.2,\n)\n\nBASE_THRESH = dict(\n    leaf_score=0.5,\n    leaf_asd=50,\n    layer_score=0.5,\n    layer_asd=50,\n)\n\nBASE_INPUT_CFG = dict(\n    mode=\'standard_sdt\',\n    st_cfg=dict(sigma_pre=1.4, thr=0.5, close_r=2, open_r=1, min_area=50),\n    he_cfg=dict(sigma_pre=0.4, thr=0.5, close_r=1, open_r=1, min_area=80),\n    clip_dist=60.0,\n    sigma_sdt=1.2,\n    area_power=0.9,\n    w_min=0.25,\n    w_max=4.5,\n    zoom_scale=0.3,\n)\n\n# v4 final-stage input/LDDMM params\nFINAL_INPUT_CFG_V4 = dict(\n    mode=\'equalized_sdt\',\n    st_cfg=dict(sigma_pre=1.4, thr=0.5, close_r=2, open_r=1, min_area=50),\n    he_cfg=dict(sigma_pre=0.4, thr=0.5, close_r=1, open_r=1, min_area=80),\n    clip_dist=4.0,\n    sigma_sdt=0.9,\n    band=4.0,\n    tau=2.0,\n    area_power=0.8,\n    w_min=0.5,\n    w_max=2.5,\n    global_channel_scale=1.6,\n    zoom_scale=0.6,\n)\n\nBASE_MODEL_CFG = dict(nt=5, a=500.0, p=2.0, expand=2.0)\nBASE_OPTIM_CFG = dict(\n    niter=500,\n    diffeo_start=100,\n    lrL=2e-8,\n    lrT=2e-1,\n    lrM=2e3,\n    affine_slowdown=10.0,\n    lrM_decay=0.9995,\n    lrM_min=200.0,\n)\nBASE_EM_CFG = dict(update_every=5, start_iter=50)\nBASE_INTENSITY_CFG = dict(sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=5e5)\n\nFINAL_MODEL_CFG_V4 = dict(nt=5, a=200.0, p=2.0, expand=2.0, grid_step=50)\nFINAL_OPTIM_CFG_V4 = dict(\n    niter=200,\n    diffeo_start=0,\n    lrL=2e-8,\n    lrT=2e-1,\n    lrM=2e3,\n    affine_slowdown=10.0,\n    lrM_decay=0.9995,\n    lrM_min=200.0,\n)\nFINAL_EM_CFG_V4 = dict(update_every=5, start_iter=50)\nFINAL_INTENSITY_CFG_V4 = dict(sigmaM=1.0, sigmaB=2.0, sigmaA=5.0, sigmaR=5e5)\n\n# optional user overrides\nif \'ITER_PAIR_CFG_OVERRIDE\' in globals():\n    BASE_PAIR_CFG.update(ITER_PAIR_CFG_OVERRIDE)\nif \'ITER_GATE_PARAMS_OVERRIDE\' in globals():\n    BASE_GATE_PARAMS.update(ITER_GATE_PARAMS_OVERRIDE)\nif \'ITER_THRESH_OVERRIDE\' in globals():\n    BASE_THRESH.update(ITER_THRESH_OVERRIDE)\nif \'ITER_INPUT_CFG_OVERRIDE\' in globals():\n    BASE_INPUT_CFG.update(ITER_INPUT_CFG_OVERRIDE)\nif \'ITER_MODEL_CFG_OVERRIDE\' in globals():\n    BASE_MODEL_CFG.update(ITER_MODEL_CFG_OVERRIDE)\nif \'ITER_OPTIM_CFG_OVERRIDE\' in globals():\n    BASE_OPTIM_CFG.update(ITER_OPTIM_CFG_OVERRIDE)\nif \'ITER_EM_CFG_OVERRIDE\' in globals():\n    BASE_EM_CFG.update(ITER_EM_CFG_OVERRIDE)\nif \'ITER_INTENSITY_CFG_OVERRIDE\' in globals():\n    BASE_INTENSITY_CFG.update(ITER_INTENSITY_CFG_OVERRIDE)\n\n# dedicated final-stage override hooks\nif \'ITER_FINAL_W_ALIGN_OVERRIDE\' in globals():\n    FINAL_W_ALIGN_V4.update(ITER_FINAL_W_ALIGN_OVERRIDE)\nif \'ITER_FINAL_INPUT_CFG_OVERRIDE\' in globals():\n    FINAL_INPUT_CFG_V4.update(ITER_FINAL_INPUT_CFG_OVERRIDE)\nif \'ITER_FINAL_MODEL_CFG_OVERRIDE\' in globals():\n    FINAL_MODEL_CFG_V4.update(ITER_FINAL_MODEL_CFG_OVERRIDE)\nif \'ITER_FINAL_OPTIM_CFG_OVERRIDE\' in globals():\n    FINAL_OPTIM_CFG_V4.update(ITER_FINAL_OPTIM_CFG_OVERRIDE)\nif \'ITER_FINAL_EM_CFG_OVERRIDE\' in globals():\n    FINAL_EM_CFG_V4.update(ITER_FINAL_EM_CFG_OVERRIDE)\nif \'ITER_FINAL_INTENSITY_CFG_OVERRIDE\' in globals():\n    FINAL_INTENSITY_CFG_V4.update(ITER_FINAL_INTENSITY_CFG_OVERRIDE)\n\nmask_params_use = DEFAULT_MASK_PARAMS if \'DEFAULT_MASK_PARAMS\' in globals() else dict()\nmask_params_thin_use = MASK_PARAMS_THIN if \'MASK_PARAMS_THIN\' in globals() else dict()\n\n# ------------------------------------------------------------\n# 2) Iterative stages\n# ------------------------------------------------------------\ndf_all_iter = base_all.copy()\ndf_filtered_iter = base_filtered.copy()\niter_stage_outputs = []\n\n_prev_W_ALIGN = dict(W_ALIGN) if \'W_ALIGN\' in globals() else None\n\nfor stage_idx, label_col_stage in enumerate(stage_labels, start=1):\n    is_final_stage = (stage_idx == len(stage_labels))\n    stage_tag = f\'stage-{stage_idx}\'\n\n    stage_pair_cfg = dict(BASE_PAIR_CFG)\n    stage_gate_params = dict(BASE_GATE_PARAMS)\n    stage_thresh = dict(BASE_THRESH)\n\n    if is_final_stage:\n        stage_w_align = dict(FINAL_W_ALIGN_V4)\n        stage_input_cfg = dict(FINAL_INPUT_CFG_V4)\n        stage_model_cfg = dict(FINAL_MODEL_CFG_V4)\n        stage_optim_cfg = dict(FINAL_OPTIM_CFG_V4)\n        stage_em_cfg = dict(FINAL_EM_CFG_V4)\n        stage_intensity_cfg = dict(FINAL_INTENSITY_CFG_V4)\n    else:\n        stage_w_align = dict(BASE_W_ALIGN)\n        stage_input_cfg = dict(BASE_INPUT_CFG)\n        stage_model_cfg = dict(BASE_MODEL_CFG)\n        stage_optim_cfg = dict(BASE_OPTIM_CFG)\n        stage_em_cfg = dict(BASE_EM_CFG)\n        stage_intensity_cfg = dict(BASE_INTENSITY_CFG)\n\n    globals()[\'W_ALIGN\'] = stage_w_align\n\n    print(f"\\n[{stage_tag}] label_col={label_col_stage} | final_stage={is_final_stage}")\n\n    # pair matching on filtered subset\n    result_stage = run_dual_pair_pipeline(\n        df_smooth=df_filtered_iter,\n        df2=df2,\n        atlas_info=atlas_info,\n        x_col=\'x_prealigned\',\n        y_col=\'y_prealigned\',\n        label_col=label_col_stage,\n        mask_params=mask_params_use,\n        mask_params_thin=mask_params_thin_use,\n        shape_type_col=\'shape_type\',\n        thin_values=(\'detail\',),\n        thin_rule=\'mode\',\n        verbose=True,\n        **stage_pair_cfg,\n    )\n\n    pairs_df_stage = result_stage[\'pairs_df\']\n    pairs_layer_df_stage = result_stage[\'pairs_layer_df\']\n    st_masks_stage = result_stage[\'st_masks\']\n    sl_stage = result_stage[\'atlas_info\'][\'sl\']\n    xJ_stage = result_stage[\'atlas_info\'][\'xJ\']\n    yJ_stage = result_stage[\'atlas_info\'][\'yJ\']\n\n    prefix_meta_stage = result_stage[\'prefix_info\'][\'prefix_meta\']\n    layer_to_labels_stage = result_stage[\'layer_to_labels\']\n\n    # build gated candidate pairs\n    pairs_leaf_stage = build_pairs_from_best_df(\n        best_df=pairs_df_stage,\n        prefix_meta=prefix_meta_stage,\n        layer_to_labels=layer_to_labels_stage,\n        score_col=\'align_score\',\n        score_thresh=stage_thresh[\'leaf_score\'],\n        asd_thresh=stage_thresh[\'leaf_asd\'],\n        use_gate=True,\n        gated_score_col=\'align_score_gated\',\n        gate_params=stage_gate_params,\n    )\n\n    pairs_layer_stage = build_pairs_from_best_df(\n        best_df=pairs_layer_df_stage,\n        prefix_meta=prefix_meta_stage,\n        layer_to_labels=layer_to_labels_stage,\n        score_col=\'align_score\',\n        score_thresh=stage_thresh[\'layer_score\'],\n        asd_thresh=stage_thresh[\'layer_asd\'],\n        use_gate=True,\n        gated_score_col=\'align_score_gated\',\n        gate_params=stage_gate_params,\n    )\n\n    selected_leaf_stage = select_nonoverlap_pairs_by_score(\n        pairs_leaf_stage,\n        score_col=\'align_score_gated\',\n        allow_same_cluster_multiple=False,\n    )\n    selected_layer_stage = select_nonoverlap_pairs_by_score(\n        pairs_layer_stage,\n        score_col=\'align_score_gated\',\n        allow_same_cluster_multiple=False,\n    )\n\n    # layer-first + lock conflicts + leaf fill (same logic as regular pipeline)\n    layer_df = selected_layer_stage.copy()\n    need_cols = {\'cluster\', \'labels\'}\n    if layer_df.empty or not need_cols.issubset(layer_df.columns):\n        pairs_layer_first = pd.DataFrame(columns=[\'cluster\', \'labels\'])\n    else:\n        layer_df[\'cluster\'] = layer_df[\'cluster\'].astype(str)\n        pairs_layer_first = select_nonoverlap_pairs_by_score(\n            layer_df,\n            score_col=\'align_score_gated\' if \'align_score_gated\' in layer_df.columns else \'align_score\',\n            allow_same_cluster_multiple=False,\n        )\n\n    locked_clusters = set(pairs_layer_first[\'cluster\'].astype(str).tolist()) if len(pairs_layer_first) else set()\n    locked_labels = set()\n    if len(pairs_layer_first):\n        for v in pairs_layer_first[\'labels\'].tolist():\n            if isinstance(v, (list, tuple, set, np.ndarray)):\n                locked_labels |= set(int(x) for x in v)\n\n    leaf_df = selected_leaf_stage.copy()\n    if len(leaf_df):\n        leaf_df[\'cluster\'] = leaf_df[\'cluster\'].astype(str)\n        if len(locked_clusters):\n            leaf_df = leaf_df[~leaf_df[\'cluster\'].isin(locked_clusters)]\n        if len(locked_labels) and \'labels\' in leaf_df.columns:\n            leaf_df = leaf_df[~leaf_df[\'labels\'].apply(lambda x: len(set(int(v) for v in x) & locked_labels) > 0)]\n        pairs_leaf_fill = select_nonoverlap_pairs_by_score(\n            leaf_df,\n            score_col=\'align_score_gated\' if \'align_score_gated\' in leaf_df.columns else \'align_score\',\n            allow_same_cluster_multiple=False,\n        )\n    else:\n        pairs_leaf_fill = pd.DataFrame(columns=[\'cluster\', \'labels\'])\n\n    pairs_combined = pd.concat([pairs_layer_first, pairs_leaf_fill], ignore_index=True, sort=False)\n    pair_df_stage = build_pair_union_df(pairs_combined)\n\n    if pair_df_stage is None or len(pair_df_stage) == 0:\n        print(f"[{stage_tag}] no valid pairs after filtering, skip deformation")\n        iter_stage_outputs.append(dict(\n            stage=stage_idx,\n            label_col=label_col_stage,\n            n_pairs=0,\n            pair_df=pair_df_stage,\n            result=result_stage,\n            lddmm_out=None,\n            is_final_stage=is_final_stage,\n            stage_w_align=stage_w_align,\n        ))\n        continue\n\n    # build LDDMM inputs\n    source_onehot, target_onehot, pair_rows_stage = build_pair_onehot_from_pair_df(\n        pair_df=pair_df_stage,\n        st_masks=st_masks_stage,\n        sl=sl_stage,\n        add_global_channel=True,\n    )\n\n    source_bin, target_bin = preprocess_onehot_asymmetric(\n        source_onehot,\n        target_onehot,\n        st_cfg=stage_input_cfg[\'st_cfg\'],\n        he_cfg=stage_input_cfg[\'he_cfg\'],\n    )\n\n    if stage_input_cfg.get(\'mode\', \'standard_sdt\') == \'equalized_sdt\':\n        source_sdt, target_sdt = _build_equalized_sdt(\n            source_bin,\n            target_bin,\n            clip_dist=stage_input_cfg[\'clip_dist\'],\n            sigma_sdt=stage_input_cfg[\'sigma_sdt\'],\n            band=stage_input_cfg.get(\'band\', 4.0),\n            tau=stage_input_cfg.get(\'tau\', 2.0),\n        )\n        channel_w = _area_balance_weights(\n            source_bin,\n            target_bin,\n            power=stage_input_cfg[\'area_power\'],\n            w_min=stage_input_cfg[\'w_min\'],\n            w_max=stage_input_cfg[\'w_max\'],\n        )\n        if len(channel_w) > 0:\n            channel_w[-1] *= float(stage_input_cfg.get(\'global_channel_scale\', 1.0))\n        source_image_np = source_sdt * channel_w[:, None, None]\n        target_image_np = target_sdt * channel_w[:, None, None]\n    else:\n        source_sdt = onehot_to_sdt(source_bin, clip_dist=stage_input_cfg[\'clip_dist\'], sigma_sdt=stage_input_cfg[\'sigma_sdt\'])\n        target_sdt = onehot_to_sdt(target_bin, clip_dist=stage_input_cfg[\'clip_dist\'], sigma_sdt=stage_input_cfg[\'sigma_sdt\'])\n        channel_w = channel_weights_from_area(\n            np.maximum(source_bin, target_bin),\n            power=stage_input_cfg[\'area_power\'],\n            w_min=stage_input_cfg[\'w_min\'],\n            w_max=stage_input_cfg[\'w_max\'],\n        )\n        source_image_np = apply_channel_weights(source_sdt, channel_w)\n        target_image_np = apply_channel_weights(target_sdt, channel_w)\n\n    source_grid_y = np.asarray(yJ_stage, dtype=np.float64)\n    source_grid_x = np.asarray(xJ_stage, dtype=np.float64)\n    target_grid_y = source_grid_y.copy()\n    target_grid_x = source_grid_x.copy()\n\n    zoom_scale = float(stage_input_cfg[\'zoom_scale\'])\n    if zoom_scale < 1.0:\n        C, H, W = source_image_np.shape\n        H2 = max(16, int(round(H * zoom_scale)))\n        W2 = max(16, int(round(W * zoom_scale)))\n        zy = H2 / H\n        zx = W2 / W\n        source_image_np = zoom(source_image_np, (1, zy, zx), order=1).astype(np.float32)\n        target_image_np = zoom(target_image_np, (1, zy, zx), order=1).astype(np.float32)\n        source_grid_y = np.linspace(float(source_grid_y[0]), float(source_grid_y[-1]), H2, dtype=np.float64)\n        source_grid_x = np.linspace(float(source_grid_x[0]), float(source_grid_x[-1]), W2, dtype=np.float64)\n        target_grid_y = source_grid_y.copy()\n        target_grid_x = source_grid_x.copy()\n\n    device_stage = device if \'device\' in globals() else (\'cuda\' if torch.cuda.is_available() else \'cpu\')\n    dtype_stage = dtype if \'dtype\' in globals() else torch.float64\n\n    source_image_t = torch.from_numpy(source_image_np).to(device=device_stage, dtype=dtype_stage)\n    target_image_t = torch.from_numpy(target_image_np).to(device=device_stage, dtype=dtype_stage)\n\n    source_grid = [\n        torch.as_tensor(source_grid_y, device=device_stage, dtype=dtype_stage),\n        torch.as_tensor(source_grid_x, device=device_stage, dtype=dtype_stage),\n    ]\n    target_grid = [\n        torch.as_tensor(target_grid_y, device=device_stage, dtype=dtype_stage),\n        torch.as_tensor(target_grid_x, device=device_stage, dtype=dtype_stage),\n    ]\n\n    out_stage = LDDMM_shooting(\n        x_src=source_grid,\n        source_image=source_image_t,\n        x_tgt=target_grid,\n        target_image=target_image_t,\n        model_cfg=stage_model_cfg,\n        optim_cfg=stage_optim_cfg,\n        em_cfg=stage_em_cfg,\n        intensity_cfg=stage_intensity_cfg,\n        device=device_stage,\n        dtype=dtype_stage,\n        verbose=True,\n        print_every=100,\n    )\n\n    A_stage = out_stage[\'A\']\n    v_stage = out_stage[\'v\']\n    xv_stage = out_stage[\'xv\']\n\n    # apply deformation to ALL points\n    points_all = df_all_iter[[\'y_prealigned\', \'x_prealigned\']].to_numpy(dtype=np.float64)\n    mapped_all = map_points_source_to_target(xv_stage, v_stage, A_stage, points_all)\n    if torch.is_tensor(mapped_all):\n        mapped_all = mapped_all.detach().cpu().numpy()\n    df_all_iter[\'y_prealigned\'] = mapped_all[:, 0].astype(float)\n    df_all_iter[\'x_prealigned\'] = mapped_all[:, 1].astype(float)\n    df_all_iter[\'y_aligned\'] = df_all_iter[\'y_prealigned\']\n    df_all_iter[\'x_aligned\'] = df_all_iter[\'x_prealigned\']\n\n    # keep filtered subset in sync for next stage pair matching\n    points_f = df_filtered_iter[[\'y_prealigned\', \'x_prealigned\']].to_numpy(dtype=np.float64)\n    mapped_f = map_points_source_to_target(xv_stage, v_stage, A_stage, points_f)\n    if torch.is_tensor(mapped_f):\n        mapped_f = mapped_f.detach().cpu().numpy()\n    df_filtered_iter[\'y_prealigned\'] = mapped_f[:, 0].astype(float)\n    df_filtered_iter[\'x_prealigned\'] = mapped_f[:, 1].astype(float)\n    df_filtered_iter[\'y_aligned\'] = df_filtered_iter[\'y_prealigned\']\n    df_filtered_iter[\'x_aligned\'] = df_filtered_iter[\'x_prealigned\']\n\n    iter_stage_outputs.append(dict(\n        stage=stage_idx,\n        label_col=label_col_stage,\n        n_pairs=int(len(pair_df_stage)),\n        pair_df=pair_df_stage,\n        pairs_leaf=pairs_leaf_stage,\n        pairs_layer=pairs_layer_stage,\n        selected_leaf=selected_leaf_stage,\n        selected_layer=selected_layer_stage,\n        result=result_stage,\n        lddmm_out=out_stage,\n        is_final_stage=is_final_stage,\n        stage_w_align=stage_w_align,\n        stage_input_cfg=stage_input_cfg,\n        stage_model_cfg=stage_model_cfg,\n        stage_optim_cfg=stage_optim_cfg,\n    ))\n\n    print(f"[{stage_tag}] pairs: {len(pair_df_stage)} | all points transformed: {len(df_all_iter)}")\n\n# restore W_ALIGN after loop\nif _prev_W_ALIGN is not None:\n    globals()[\'W_ALIGN\'] = _prev_W_ALIGN\n\n# ------------------------------------------------------------\n# 3) Final outputs used by downstream cells\n# ------------------------------------------------------------\ndf_aligned_all = df_all_iter.copy()\ndf_prealign_nofilter = df_all_iter.copy()\ndf_smooth = df_filtered_iter.copy()\n\n# Collect matched pair tables from every stage for plotting/export.\n# Each stage output already keeps the full matched pair table in stage_out[\'pair_df\'].\niter_pairs_by_stage = {}\niter_pair_summary_rows = []\niter_pairs_all = []\n\nfor stage_out in iter_stage_outputs:\n    stage = int(stage_out.get(\'stage\', len(iter_pairs_by_stage) + 1))\n    label_col = stage_out.get(\'label_col\', \'\')\n    pair_table = stage_out.get(\'pair_df\')\n\n    if pair_table is None:\n        pair_table = pd.DataFrame()\n    else:\n        pair_table = pair_table.copy()\n\n    iter_pairs_by_stage[stage] = pair_table\n    iter_pair_summary_rows.append({\n        \'stage\': stage,\n        \'label_col\': label_col,\n        \'is_final_stage\': bool(stage_out.get(\'is_final_stage\', False)),\n        \'n_pairs\': int(len(pair_table)),\n        \'lddmm_ran\': stage_out.get(\'lddmm_out\') is not None,\n    })\n\n    if len(pair_table) > 0:\n        tmp = pair_table.copy()\n        tmp.insert(0, \'stage\', stage)\n        tmp.insert(1, \'label_col\', label_col)\n        tmp.insert(2, \'is_final_stage\', bool(stage_out.get(\'is_final_stage\', False)))\n        iter_pairs_all.append(tmp)\n\niter_pair_summary_df = pd.DataFrame(iter_pair_summary_rows)\niter_pairs_all_df = pd.concat(iter_pairs_all, ignore_index=True, sort=False) if len(iter_pairs_all) else pd.DataFrame()\n\nfinal_stage_out = next((s for s in reversed(iter_stage_outputs) if s.get(\'pair_df\') is not None and len(s.get(\'pair_df\')) > 0), None)\nif final_stage_out is not None:\n    final_pair_df = final_stage_out[\'pair_df\'].copy()\n    pair_df = final_pair_df.copy()  # make downstream plotting use final-stage pairs\nelse:\n    final_pair_df = pd.DataFrame()\n    pair_df = final_pair_df.copy()\n\nif len(iter_stage_outputs) > 0 and iter_stage_outputs[-1][\'lddmm_out\'] is not None:\n    out = iter_stage_outputs[-1][\'lddmm_out\']\n    A = out[\'A\']\n    v = out[\'v\']\n    xv = out[\'xv\']\n\nprint(\'\\n[iter] done.\')\nprint(\'[iter] stages completed:\', len(iter_stage_outputs))\nprint(\'[iter] final aligned points:\', len(df_aligned_all))\nprint(\'[iter] matched pairs by stage:\')\nprint(iter_pair_summary_df)\n'

CONTINUATION_ALIGNMENT_CELL_SOURCE = '# Continue alignment until matched pairs stop increasing\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport torch\nimport matplotlib.pyplot as plt\nimport matplotlib.colors as mcolors\nfrom scipy.ndimage import zoom\n\nif "df_aligned_all" not in globals() or "df_smooth" not in globals():\n    raise ValueError("Run section 6 first; df_aligned_all and df_smooth are required.")\n\n# User-tunable controls. Override these in a previous cell if needed.\nCONTINUE_LABEL_COL = globals().get("CONTINUE_LABEL_COL", "banksy_cluster_refined")\nCONTINUE_MAX_ITER = int(globals().get("CONTINUE_MAX_ITER", 10))\nCONTINUE_MIN_PAIR_GAIN = int(globals().get("CONTINUE_MIN_PAIR_GAIN", 1))\nCONTINUE_RESULT_DIR = Path(globals().get("CONTINUE_RESULT_DIR", "iterative_alignment_outputs/continue_alignment"))\nCONTINUE_RESULT_DIR.mkdir(parents=True, exist_ok=True)\n\nif CONTINUE_LABEL_COL not in df_aligned_all.columns:\n    raise ValueError(f"CONTINUE_LABEL_COL={CONTINUE_LABEL_COL!r} is not present in df_aligned_all.")\n\n# Reuse final-stage settings from section 6; allow dedicated continuation overrides.\nCONTINUE_W_ALIGN = dict(FINAL_W_ALIGN_V4 if "FINAL_W_ALIGN_V4" in globals() else W_ALIGN)\nCONTINUE_PAIR_CFG = dict(BASE_PAIR_CFG if "BASE_PAIR_CFG" in globals() else {})\nCONTINUE_GATE_PARAMS = dict(BASE_GATE_PARAMS if "BASE_GATE_PARAMS" in globals() else {})\nCONTINUE_THRESH = dict(BASE_THRESH if "BASE_THRESH" in globals() else {})\nCONTINUE_INPUT_CFG = dict(FINAL_INPUT_CFG_V4 if "FINAL_INPUT_CFG_V4" in globals() else BASE_INPUT_CFG)\nCONTINUE_MODEL_CFG = dict(FINAL_MODEL_CFG_V4 if "FINAL_MODEL_CFG_V4" in globals() else BASE_MODEL_CFG)\nCONTINUE_OPTIM_CFG = dict(FINAL_OPTIM_CFG_V4 if "FINAL_OPTIM_CFG_V4" in globals() else BASE_OPTIM_CFG)\nCONTINUE_EM_CFG = dict(FINAL_EM_CFG_V4 if "FINAL_EM_CFG_V4" in globals() else BASE_EM_CFG)\nCONTINUE_INTENSITY_CFG = dict(FINAL_INTENSITY_CFG_V4 if "FINAL_INTENSITY_CFG_V4" in globals() else BASE_INTENSITY_CFG)\n\nfor _name, _cfg in [\n    ("CONTINUE_W_ALIGN_OVERRIDE", CONTINUE_W_ALIGN),\n    ("CONTINUE_PAIR_CFG_OVERRIDE", CONTINUE_PAIR_CFG),\n    ("CONTINUE_GATE_PARAMS_OVERRIDE", CONTINUE_GATE_PARAMS),\n    ("CONTINUE_THRESH_OVERRIDE", CONTINUE_THRESH),\n    ("CONTINUE_INPUT_CFG_OVERRIDE", CONTINUE_INPUT_CFG),\n    ("CONTINUE_MODEL_CFG_OVERRIDE", CONTINUE_MODEL_CFG),\n    ("CONTINUE_OPTIM_CFG_OVERRIDE", CONTINUE_OPTIM_CFG),\n    ("CONTINUE_EM_CFG_OVERRIDE", CONTINUE_EM_CFG),\n    ("CONTINUE_INTENSITY_CFG_OVERRIDE", CONTINUE_INTENSITY_CFG),\n]:\n    if _name in globals():\n        _cfg.update(globals()[_name])\n\nmask_params_use = DEFAULT_MASK_PARAMS if "DEFAULT_MASK_PARAMS" in globals() else dict()\nmask_params_thin_use = MASK_PARAMS_THIN if "MASK_PARAMS_THIN" in globals() else dict()\n\n\ndef _continue_find_pairs(df_filtered_current, label_col, verbose=False):\n    prev_w_align = dict(W_ALIGN) if "W_ALIGN" in globals() else None\n    globals()["W_ALIGN"] = CONTINUE_W_ALIGN\n\n    result = run_dual_pair_pipeline(\n        df_smooth=df_filtered_current,\n        df2=df2,\n        atlas_info=atlas_info,\n        x_col="x_prealigned",\n        y_col="y_prealigned",\n        label_col=label_col,\n        mask_params=mask_params_use,\n        mask_params_thin=mask_params_thin_use,\n        shape_type_col="shape_type",\n        thin_values=("detail",),\n        thin_rule="mode",\n        verbose=verbose,\n        **CONTINUE_PAIR_CFG,\n    )\n\n    prefix_meta = result["prefix_info"]["prefix_meta"]\n    layer_to_labels = result["layer_to_labels"]\n\n    pairs_leaf = build_pairs_from_best_df(\n        best_df=result["pairs_df"],\n        prefix_meta=prefix_meta,\n        layer_to_labels=layer_to_labels,\n        score_col="align_score",\n        score_thresh=CONTINUE_THRESH.get("leaf_score", 0.5),\n        asd_thresh=CONTINUE_THRESH.get("leaf_asd", 50),\n        use_gate=True,\n        gated_score_col="align_score_gated",\n        gate_params=CONTINUE_GATE_PARAMS,\n    )\n    pairs_layer = build_pairs_from_best_df(\n        best_df=result["pairs_layer_df"],\n        prefix_meta=prefix_meta,\n        layer_to_labels=layer_to_labels,\n        score_col="align_score",\n        score_thresh=CONTINUE_THRESH.get("layer_score", 0.5),\n        asd_thresh=CONTINUE_THRESH.get("layer_asd", 50),\n        use_gate=True,\n        gated_score_col="align_score_gated",\n        gate_params=CONTINUE_GATE_PARAMS,\n    )\n\n    selected_leaf = select_nonoverlap_pairs_by_score(\n        pairs_leaf,\n        score_col="align_score_gated",\n        allow_same_cluster_multiple=False,\n    )\n    selected_layer = select_nonoverlap_pairs_by_score(\n        pairs_layer,\n        score_col="align_score_gated",\n        allow_same_cluster_multiple=False,\n    )\n\n    layer_df = selected_layer.copy()\n    if layer_df.empty or not {"cluster", "labels"}.issubset(layer_df.columns):\n        pairs_layer_first = pd.DataFrame(columns=["cluster", "labels"])\n    else:\n        layer_df["cluster"] = layer_df["cluster"].astype(str)\n        pairs_layer_first = select_nonoverlap_pairs_by_score(\n            layer_df,\n            score_col="align_score_gated" if "align_score_gated" in layer_df.columns else "align_score",\n            allow_same_cluster_multiple=False,\n        )\n\n    locked_clusters = set(pairs_layer_first["cluster"].astype(str).tolist()) if len(pairs_layer_first) else set()\n    locked_labels = set()\n    if len(pairs_layer_first):\n        for v in pairs_layer_first["labels"].tolist():\n            if isinstance(v, (list, tuple, set, np.ndarray)):\n                locked_labels |= set(int(x) for x in v)\n\n    leaf_df = selected_leaf.copy()\n    if len(leaf_df):\n        leaf_df["cluster"] = leaf_df["cluster"].astype(str)\n        if len(locked_clusters):\n            leaf_df = leaf_df[~leaf_df["cluster"].isin(locked_clusters)]\n        if len(locked_labels) and "labels" in leaf_df.columns:\n            leaf_df = leaf_df[~leaf_df["labels"].apply(lambda x: len(set(int(v) for v in x) & locked_labels) > 0)]\n        pairs_leaf_fill = select_nonoverlap_pairs_by_score(\n            leaf_df,\n            score_col="align_score_gated" if "align_score_gated" in leaf_df.columns else "align_score",\n            allow_same_cluster_multiple=False,\n        )\n    else:\n        pairs_leaf_fill = pd.DataFrame(columns=["cluster", "labels"])\n\n    pairs_combined = pd.concat([pairs_layer_first, pairs_leaf_fill], ignore_index=True, sort=False)\n    pair_df_current = build_pair_union_df(pairs_combined)\n    if pair_df_current is None:\n        pair_df_current = pd.DataFrame()\n\n    if prev_w_align is not None:\n        globals()["W_ALIGN"] = prev_w_align\n\n    return dict(\n        result=result,\n        pair_df=pair_df_current,\n        pairs_leaf=pairs_leaf,\n        pairs_layer=pairs_layer,\n        selected_leaf=selected_leaf,\n        selected_layer=selected_layer,\n    )\n\n\ndef _continue_run_lddmm(pair_df_current, result_current):\n    source_onehot, target_onehot, pair_rows_current = build_pair_onehot_from_pair_df(\n        pair_df=pair_df_current,\n        st_masks=result_current["st_masks"],\n        sl=result_current["atlas_info"]["sl"],\n        add_global_channel=True,\n    )\n\n    source_bin, target_bin = preprocess_onehot_asymmetric(\n        source_onehot,\n        target_onehot,\n        st_cfg=CONTINUE_INPUT_CFG["st_cfg"],\n        he_cfg=CONTINUE_INPUT_CFG["he_cfg"],\n    )\n\n    if CONTINUE_INPUT_CFG.get("mode", "standard_sdt") == "equalized_sdt":\n        source_sdt, target_sdt = _build_equalized_sdt(\n            source_bin,\n            target_bin,\n            clip_dist=CONTINUE_INPUT_CFG["clip_dist"],\n            sigma_sdt=CONTINUE_INPUT_CFG["sigma_sdt"],\n            band=CONTINUE_INPUT_CFG.get("band", 4.0),\n            tau=CONTINUE_INPUT_CFG.get("tau", 2.0),\n        )\n        channel_w = _area_balance_weights(\n            source_bin,\n            target_bin,\n            power=CONTINUE_INPUT_CFG["area_power"],\n            w_min=CONTINUE_INPUT_CFG["w_min"],\n            w_max=CONTINUE_INPUT_CFG["w_max"],\n        )\n        if len(channel_w) > 0:\n            channel_w[-1] *= float(CONTINUE_INPUT_CFG.get("global_channel_scale", 1.0))\n        source_image_np = source_sdt * channel_w[:, None, None]\n        target_image_np = target_sdt * channel_w[:, None, None]\n    else:\n        source_sdt = onehot_to_sdt(source_bin, clip_dist=CONTINUE_INPUT_CFG["clip_dist"], sigma_sdt=CONTINUE_INPUT_CFG["sigma_sdt"])\n        target_sdt = onehot_to_sdt(target_bin, clip_dist=CONTINUE_INPUT_CFG["clip_dist"], sigma_sdt=CONTINUE_INPUT_CFG["sigma_sdt"])\n        channel_w = channel_weights_from_area(\n            np.maximum(source_bin, target_bin),\n            power=CONTINUE_INPUT_CFG["area_power"],\n            w_min=CONTINUE_INPUT_CFG["w_min"],\n            w_max=CONTINUE_INPUT_CFG["w_max"],\n        )\n        source_image_np = apply_channel_weights(source_sdt, channel_w)\n        target_image_np = apply_channel_weights(target_sdt, channel_w)\n\n    xJ_current = result_current["atlas_info"]["xJ"]\n    yJ_current = result_current["atlas_info"]["yJ"]\n    source_grid_y = np.asarray(yJ_current, dtype=np.float64)\n    source_grid_x = np.asarray(xJ_current, dtype=np.float64)\n    target_grid_y = source_grid_y.copy()\n    target_grid_x = source_grid_x.copy()\n\n    zoom_scale = float(CONTINUE_INPUT_CFG["zoom_scale"])\n    if zoom_scale < 1.0:\n        C, H, W = source_image_np.shape\n        H2 = max(16, int(round(H * zoom_scale)))\n        W2 = max(16, int(round(W * zoom_scale)))\n        zy = H2 / H\n        zx = W2 / W\n        source_image_np = zoom(source_image_np, (1, zy, zx), order=1).astype(np.float32)\n        target_image_np = zoom(target_image_np, (1, zy, zx), order=1).astype(np.float32)\n        source_grid_y = np.linspace(float(source_grid_y[0]), float(source_grid_y[-1]), H2, dtype=np.float64)\n        source_grid_x = np.linspace(float(source_grid_x[0]), float(source_grid_x[-1]), W2, dtype=np.float64)\n        target_grid_y = source_grid_y.copy()\n        target_grid_x = source_grid_x.copy()\n\n    device_current = device if "device" in globals() else ("cuda" if torch.cuda.is_available() else "cpu")\n    dtype_current = dtype if "dtype" in globals() else torch.float64\n\n    source_image_t = torch.from_numpy(source_image_np).to(device=device_current, dtype=dtype_current)\n    target_image_t = torch.from_numpy(target_image_np).to(device=device_current, dtype=dtype_current)\n    source_grid = [\n        torch.as_tensor(source_grid_y, device=device_current, dtype=dtype_current),\n        torch.as_tensor(source_grid_x, device=device_current, dtype=dtype_current),\n    ]\n    target_grid = [\n        torch.as_tensor(target_grid_y, device=device_current, dtype=dtype_current),\n        torch.as_tensor(target_grid_x, device=device_current, dtype=dtype_current),\n    ]\n\n    return LDDMM_shooting(\n        x_src=source_grid,\n        source_image=source_image_t,\n        x_tgt=target_grid,\n        target_image=target_image_t,\n        model_cfg=CONTINUE_MODEL_CFG,\n        optim_cfg=CONTINUE_OPTIM_CFG,\n        em_cfg=CONTINUE_EM_CFG,\n        intensity_cfg=CONTINUE_INTENSITY_CFG,\n        device=device_current,\n        dtype=dtype_current,\n        verbose=True,\n        print_every=100,\n    )\n\n\ndef _continue_map_points(df_current, out_current):\n    mapped_df = df_current.copy()\n    points = mapped_df[["y_prealigned", "x_prealigned"]].to_numpy(dtype=np.float64)\n    mapped = map_points_source_to_target(out_current["xv"], out_current["v"], out_current["A"], points)\n    if torch.is_tensor(mapped):\n        mapped = mapped.detach().cpu().numpy()\n    mapped_df["y_prealigned"] = mapped[:, 0].astype(float)\n    mapped_df["x_prealigned"] = mapped[:, 1].astype(float)\n    mapped_df["y_aligned"] = mapped_df["y_prealigned"]\n    mapped_df["x_aligned"] = mapped_df["x_prealigned"]\n    return mapped_df\n\n\ndef _continue_parse_labels(x):\n    if isinstance(x, (list, tuple, np.ndarray, set)):\n        return [int(v) for v in x]\n    s = str(x).strip().replace(" ", "")\n    if s == "" or s.lower() == "nan":\n        return []\n    for sep in [";", ","]:\n        if sep in s:\n            return [int(v) for v in s.split(sep) if v != ""]\n    try:\n        return [int(float(s))]\n    except Exception:\n        return []\n\n\ndef _continue_build_overlay(pair_table, sl_current, other_rgba=(0.92, 0.92, 0.92, 0.90)):\n    H, W = sl_current.shape\n    overlay = np.zeros((H, W, 4), dtype=float)\n    brain = sl_current > 0\n    if pair_table is None or len(pair_table) == 0:\n        overlay[brain] = np.array(other_rgba, dtype=float)\n        return overlay, {}\n\n    pair_table = pair_table.copy()\n    pair_table["cluster"] = pair_table["cluster"].astype(str)\n    if "atlas_labels_union" in pair_table.columns:\n        pair_table["labels_list"] = pair_table["atlas_labels_union"].apply(_continue_parse_labels)\n    elif "labels" in pair_table.columns:\n        pair_table["labels_list"] = pair_table["labels"].apply(_continue_parse_labels)\n    else:\n        raise KeyError("pair_table needs atlas_labels_union or labels column")\n\n    clusters = pair_table["cluster"].astype(str).unique().tolist()\n    cmap = plt.get_cmap("tab20", max(len(clusters), 1))\n    color_map = {str(cl): np.array(mcolors.to_rgba(cmap(i), alpha=1.0)) for i, cl in enumerate(clusters)}\n\n    matched = np.zeros((H, W), dtype=bool)\n    for _, row in pair_table.iterrows():\n        labs = row["labels_list"]\n        if len(labs) == 0:\n            continue\n        m = np.isin(sl_current, labs)\n        overlay[m] = color_map[str(row["cluster"])]\n        matched |= m\n    overlay[brain & (~matched)] = np.array(other_rgba, dtype=float)\n    return overlay, color_map\n\n\ndef _continue_phys_to_pix(xJ_current, yJ_current, H, W):\n    xmin, xmax = float(xJ_current[0]), float(xJ_current[-1])\n    ymin, ymax = float(yJ_current[0]), float(yJ_current[-1])\n    sx = (W - 1) / (xmax - xmin)\n    sy = (H - 1) / (ymax - ymin)\n\n    def _convert(x, y):\n        xi = np.round((np.asarray(x) - xmin) * sx).astype(int)\n        yi = np.round((np.asarray(y) - ymin) * sy).astype(int)\n        return xi, yi\n\n    return _convert\n\n\ndef _continue_plot_iteration(iter_idx, df_before_iter, df_after_iter, before_pairs, after_pairs, result_before, result_after):\n    sl_current = result_after["atlas_info"]["sl"]\n    xJ_current = result_after["atlas_info"]["xJ"]\n    yJ_current = result_after["atlas_info"]["yJ"]\n    H, W = sl_current.shape\n    phys_to_pix = _continue_phys_to_pix(xJ_current, yJ_current, H, W)\n\n    overlay_before, colors_before = _continue_build_overlay(before_pairs, sl_current)\n    overlay_after, colors_after = _continue_build_overlay(after_pairs, sl_current)\n    other_color = np.array([0.12, 0.12, 0.12, 0.45], dtype=float)\n\n    labels_before = df_before_iter[CONTINUE_LABEL_COL].astype(str).to_numpy()\n    labels_after = df_after_iter[CONTINUE_LABEL_COL].astype(str).to_numpy()\n    colors_b = np.array([colors_before.get(str(cl), other_color) for cl in labels_before])\n    colors_a = np.array([colors_after.get(str(cl), other_color) for cl in labels_after])\n\n    xb = df_before_iter["x_prealigned"].to_numpy(dtype=float)\n    yb = df_before_iter["y_prealigned"].to_numpy(dtype=float)\n    xa = df_after_iter["x_prealigned"].to_numpy(dtype=float)\n    ya = df_after_iter["y_prealigned"].to_numpy(dtype=float)\n    xib, yib = phys_to_pix(xb, yb)\n    xia, yia = phys_to_pix(xa, ya)\n\n    mask_b = np.isfinite(xb) & np.isfinite(yb) & (xib >= 0) & (xib < W) & (yib >= 0) & (yib < H)\n    mask_a = np.isfinite(xa) & np.isfinite(ya) & (xia >= 0) & (xia < W) & (yia >= 0) & (yia < H)\n\n    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True, facecolor="white")\n    panels = [\n        (axes[0], overlay_before, xib, yib, mask_b, colors_b, f"Continue iter {iter_idx} before ({len(before_pairs)} pairs)"),\n        (axes[1], overlay_after, xia, yia, mask_a, colors_a, f"Continue iter {iter_idx} after ({len(after_pairs)} pairs)"),\n    ]\n    for ax, overlay, xi, yi, mask, point_colors, title in panels:\n        ax.set_facecolor("white")\n        ax.imshow(sl_current > 0, cmap="binary", alpha=0.10, origin="lower")\n        ax.imshow(overlay, origin="lower", interpolation="nearest")\n        ax.scatter(xi[mask], yi[mask], s=1, c=point_colors[mask], edgecolors="none", rasterized=True)\n        ax.set_title(title)\n        ax.axis("off")\n        ax.set_aspect("equal")\n\n    plt.tight_layout()\n    png_path = CONTINUE_RESULT_DIR / f"continue_alignment_iter_{iter_idx:02d}_before_after.png"\n    pdf_path = CONTINUE_RESULT_DIR / f"continue_alignment_iter_{iter_idx:02d}_before_after.pdf"\n    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")\n    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")\n    print("saved continuation visualization PNG:", png_path)\n    print("saved continuation visualization PDF:", pdf_path)\n    plt.show()\n\n\ncontinue_df_all = df_aligned_all.copy()\ncontinue_df_filtered = df_smooth.copy()\ncontinue_start_all = continue_df_all.copy()\ncontinue_alignment_outputs = []\ncontinue_pair_count_rows = []\ncontinue_stop_reason = "max_iter_reached"\n\nfor continue_iter in range(1, CONTINUE_MAX_ITER + 1):\n    print(f"\\n[continue-{continue_iter}] finding pairs before refinement")\n    before_match = _continue_find_pairs(continue_df_filtered, CONTINUE_LABEL_COL, verbose=True)\n    before_pairs = before_match["pair_df"]\n    n_before = int(len(before_pairs))\n    print(f"[continue-{continue_iter}] pairs before: {n_before}")\n\n    if n_before == 0:\n        continue_stop_reason = "no_pairs_before_refinement"\n        continue_pair_count_rows.append(dict(iteration=continue_iter, n_pairs_before=0, n_pairs_after=0, pair_gain=0, lddmm_ran=False, stop=True))\n        break\n\n    out_continue = _continue_run_lddmm(before_pairs, before_match["result"])\n    before_all_for_plot = continue_df_all.copy()\n    before_filtered_for_plot = continue_df_filtered.copy()\n\n    after_df_all = _continue_map_points(continue_df_all, out_continue)\n    after_df_filtered = _continue_map_points(continue_df_filtered, out_continue)\n\n    print(f"[continue-{continue_iter}] finding pairs after refinement")\n    after_match = _continue_find_pairs(after_df_filtered, CONTINUE_LABEL_COL, verbose=True)\n    after_pairs = after_match["pair_df"]\n    n_after = int(len(after_pairs))\n    pair_gain = n_after - n_before\n    print(f"[continue-{continue_iter}] pairs after: {n_after} | gain: {pair_gain}")\n\n    _continue_plot_iteration(\n        continue_iter,\n        before_all_for_plot,\n        after_df_all,\n        before_pairs,\n        after_pairs,\n        before_match["result"],\n        after_match["result"],\n    )\n\n    continue_alignment_outputs.append(dict(\n        iteration=continue_iter,\n        label_col=CONTINUE_LABEL_COL,\n        n_pairs_before=n_before,\n        n_pairs_after=n_after,\n        pair_gain=pair_gain,\n        before_pair_df=before_pairs.copy(),\n        after_pair_df=after_pairs.copy(),\n        before_match=before_match,\n        after_match=after_match,\n        lddmm_out=out_continue,\n    ))\n    continue_pair_count_rows.append(dict(\n        iteration=continue_iter,\n        n_pairs_before=n_before,\n        n_pairs_after=n_after,\n        pair_gain=pair_gain,\n        lddmm_ran=True,\n        stop=pair_gain < CONTINUE_MIN_PAIR_GAIN,\n    ))\n\n    continue_df_all = after_df_all\n    continue_df_filtered = after_df_filtered\n\n    if pair_gain < CONTINUE_MIN_PAIR_GAIN:\n        continue_stop_reason = "pair_count_not_increased"\n        break\n\ncontinue_pair_count_df = pd.DataFrame(continue_pair_count_rows)\n\n# Publish final continuation result to the same globals used by downstream visualization/export cells.\ndf_aligned_all = continue_df_all.copy()\ndf_prealign_nofilter = continue_df_all.copy()\ndf_smooth = continue_df_filtered.copy()\n\nif len(continue_alignment_outputs) > 0:\n    final_continue = continue_alignment_outputs[-1]\n    final_pair_df = final_continue["after_pair_df"].copy()\n    pair_df = final_pair_df.copy()\n    out = final_continue["lddmm_out"]\n    A = out["A"]\n    v = out["v"]\n    xv = out["xv"]\n\ncontinue_pair_count_path = CONTINUE_RESULT_DIR / "continue_alignment_pair_counts.csv"\ncontinue_pair_count_df.to_csv(continue_pair_count_path, index=False)\nprint("\\n[continue] stop reason:", continue_stop_reason)\nprint("[continue] saved pair counts:", continue_pair_count_path)\ndisplay(continue_pair_count_df)\n\nif len(continue_pair_count_df) > 0:\n    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")\n    ax.plot(continue_pair_count_df["iteration"], continue_pair_count_df["n_pairs_before"], marker="o", label="before")\n    ax.plot(continue_pair_count_df["iteration"], continue_pair_count_df["n_pairs_after"], marker="o", label="after")\n    ax.set_xlabel("Continuation iteration")\n    ax.set_ylabel("Matched pairs")\n    ax.set_title("Matched pairs across continuation iterations")\n    ax.set_xticks(continue_pair_count_df["iteration"].astype(int).tolist())\n    ax.grid(True, alpha=0.25)\n    ax.legend(frameon=False)\n    plt.tight_layout()\n    trend_png = CONTINUE_RESULT_DIR / "continue_alignment_pair_counts.png"\n    trend_pdf = CONTINUE_RESULT_DIR / "continue_alignment_pair_counts.pdf"\n    fig.savefig(trend_png, dpi=300, bbox_inches="tight", facecolor="white")\n    fig.savefig(trend_pdf, dpi=300, bbox_inches="tight", facecolor="white")\n    print("saved continuation pair-count plot PNG:", trend_png)\n    print("saved continuation pair-count plot PDF:", trend_pdf)\n    plt.show()\n'

VISUALIZATION_CELL_SOURCE = 'from pathlib import Path\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport matplotlib.colors as mcolors\n\n\ndef save_rasterized_scatter_pdf_png(\n    fig,\n    png_path,\n    pdf_path,\n    dpi=300,\n    bbox_inches="tight",\n    pad_inches=0,\n    facecolor="white",\n):\n    """Save PNG plus rasterized PDF, matching FF/figure/save_rasterized_scatter.py."""\n    fig.savefig(\n        png_path,\n        dpi=dpi,\n        bbox_inches=bbox_inches,\n        pad_inches=pad_inches,\n        facecolor=facecolor,\n    )\n    fig.savefig(\n        pdf_path,\n        dpi=dpi,\n        bbox_inches=bbox_inches,\n        pad_inches=pad_inches,\n        facecolor=facecolor,\n    )\n\n# -----------------------------\n# Before/after tables\n# -----------------------------\nif "df_pre_iter_all" in globals():\n    df_before = df_pre_iter_all.copy()\nelif "df_final" in globals():\n    df_before = df_final.copy()\nelse:\n    raise ValueError("Need df_pre_iter_all or df_final for BEFORE coordinates.")\n\nif "df_aligned_all" not in globals():\n    raise ValueError("Run the iterative multi-level alignment cell first; df_aligned_all is missing.")\ndf_after = df_aligned_all.copy()\n\n# Prefer index alignment; fall back to row order only when indexes do not overlap.\ncommon_idx = df_before.index.intersection(df_after.index)\nif len(common_idx) > 0:\n    df_before = df_before.loc[common_idx].copy()\n    df_after = df_after.loc[common_idx].copy()\nelse:\n    n = min(len(df_before), len(df_after))\n    df_before = df_before.iloc[:n].copy()\n    df_after = df_after.iloc[:n].copy()\n\n# -----------------------------\n# Pair table and label column\n# -----------------------------\n# Prefer final continuation pairs when section 7 has been run.\nif "continue_alignment_outputs" in globals() and len(continue_alignment_outputs) > 0:\n    last_continue = continue_alignment_outputs[-1]\n    pair_vis_df = last_continue.get("after_pair_df", pd.DataFrame()).copy()\n    label_col_vis = last_continue.get("label_col", globals().get("CONTINUE_LABEL_COL", "banksy_cluster_refined"))\nelif "pair_df" in globals():\n    pair_vis_df = pair_df.copy()\n    label_col_vis = globals().get("CONTINUE_LABEL_COL", "banksy_cluster_refined")\nelif "iter_stage_outputs" in globals() and len(iter_stage_outputs) > 0:\n    last_stage = next((s for s in reversed(iter_stage_outputs) if s.get("pair_df") is not None), iter_stage_outputs[-1])\n    pair_vis_df = last_stage["pair_df"].copy()\n    label_col_vis = last_stage.get("label_col", "banksy_cluster_refined")\nelse:\n    raise ValueError("Need continue_alignment_outputs, pair_df, or iter_stage_outputs for atlas overlay colors.")\n\nif pair_vis_df is None or len(pair_vis_df) == 0:\n    print("No matched pair table found; plotting gray atlas background and points only.")\n    pair_vis_df = None\nelse:\n    pair_vis_df["cluster"] = pair_vis_df["cluster"].astype(str)\n\nif label_col_vis not in df_after.columns:\n    if "banksy_cluster_refined" in df_after.columns:\n        label_col_vis = "banksy_cluster_refined"\n    else:\n        label_col_vis = next((c for c in df_after.columns if c.startswith("banksy_cluster_refined")), None)\nif label_col_vis is None:\n    raise ValueError("Could not find a cluster label column in df_after.")\n\n# -----------------------------\n# Helpers\n# -----------------------------\ndef parse_labels_any(x):\n    if isinstance(x, (list, tuple, np.ndarray, set)):\n        return [int(v) for v in x]\n    s = str(x).strip().replace(" ", "")\n    if s == "" or s.lower() == "nan":\n        return []\n    for sep in [";", ","]:\n        if sep in s:\n            return [int(v) for v in s.split(sep) if v != ""]\n    try:\n        return [int(float(s))]\n    except Exception:\n        return []\n\n\ndef build_pair_overlay(pair_table, sl, other_rgba=(0.92, 0.92, 0.92, 0.90)):\n    H, W = sl.shape\n    overlay = np.zeros((H, W, 4), dtype=float)\n    brain = sl > 0\n\n    if pair_table is None or len(pair_table) == 0:\n        overlay[brain] = np.array(other_rgba, dtype=float)\n        return overlay, {}\n\n    pair_table = pair_table.copy()\n    pair_table["cluster"] = pair_table["cluster"].astype(str)\n\n    if "atlas_labels_union" in pair_table.columns:\n        pair_table["labels_list"] = pair_table["atlas_labels_union"].apply(parse_labels_any)\n    elif "labels" in pair_table.columns:\n        pair_table["labels_list"] = pair_table["labels"].apply(parse_labels_any)\n    else:\n        raise KeyError("pair_table needs atlas_labels_union or labels column")\n\n    clusters = pair_table["cluster"].astype(str).unique().tolist()\n    cmap = plt.get_cmap("tab20", max(len(clusters), 1))\n    color_map = {str(cl): np.array(mcolors.to_rgba(cmap(i), alpha=1.0)) for i, cl in enumerate(clusters)}\n\n    matched = np.zeros((H, W), dtype=bool)\n    for _, r in pair_table.iterrows():\n        labs = r["labels_list"]\n        if len(labs) == 0:\n            continue\n        m = np.isin(sl, labs)\n        overlay[m] = color_map[str(r["cluster"])]\n        matched |= m\n\n    overlay[brain & (~matched)] = np.array(other_rgba, dtype=float)\n    return overlay, color_map\n\n\ndef get_phys_to_pix_converter():\n    H, W = sl.shape\n    if "phys_to_pix_array" in globals():\n        return phys_to_pix_array\n    if "make_phys_to_pix" in globals():\n        return make_phys_to_pix(xJ, yJ, H, W)\n\n    # Fallback equivalent to make_phys_to_pix.\n    xmin, xmax = float(xJ[0]), float(xJ[-1])\n    ymin, ymax = float(yJ[0]), float(yJ[-1])\n    sx = (W - 1) / (xmax - xmin)\n    sy = (H - 1) / (ymax - ymin)\n\n    def _phys_to_pix_array(x, y):\n        xi = np.round((np.asarray(x) - xmin) * sx).astype(int)\n        yi = np.round((np.asarray(y) - ymin) * sy).astype(int)\n        return xi, yi\n\n    return _phys_to_pix_array\n\n# -----------------------------\n# Build overlay and point colors\n# -----------------------------\noverlay_atlas, cluster_color_map = build_pair_overlay(pair_vis_df, sl)\nOTHER_COLOR = np.array([0.12, 0.12, 0.12, 0.45], dtype=float)\n\nst_clusters = df_after[label_col_vis].astype(str).values\npoint_colors = np.zeros((len(st_clusters), 4), dtype=float)\nfor i, cl in enumerate(st_clusters):\n    point_colors[i] = cluster_color_map.get(str(cl), OTHER_COLOR)\n\n# -----------------------------\n# Coordinates\n# -----------------------------\nH, W = sl.shape\nphys_to_pix = get_phys_to_pix_converter()\n\nbefore_x_col = "x_prealigned" if "x_prealigned" in df_before.columns else "x"\nbefore_y_col = "y_prealigned" if "y_prealigned" in df_before.columns else "y"\nafter_x_col = "x_aligned" if "x_aligned" in df_after.columns else "x_prealigned"\nafter_y_col = "y_aligned" if "y_aligned" in df_after.columns else "y_prealigned"\n\nx_before = df_before[before_x_col].to_numpy(dtype=float)\ny_before = df_before[before_y_col].to_numpy(dtype=float)\nxi_before, yi_before = phys_to_pix(x_before, y_before)\n\nx_after = df_after[after_x_col].to_numpy(dtype=float)\ny_after = df_after[after_y_col].to_numpy(dtype=float)\nxi_after, yi_after = phys_to_pix(x_after, y_after)\n\nmask_before = (\n    np.isfinite(x_before) & np.isfinite(y_before) &\n    (xi_before >= 0) & (xi_before < W) &\n    (yi_before >= 0) & (yi_before < H)\n)\nmask_after = (\n    np.isfinite(x_after) & np.isfinite(y_after) &\n    (xi_after >= 0) & (xi_after < W) &\n    (yi_after >= 0) & (yi_after < H)\n)\n\nprint(f"label column: {label_col_vis}")\nprint(f"points in view before/after: {mask_before.sum()} / {mask_after.sum()} of {len(df_after)}")\nprint(f"matched pairs shown: {0 if pair_vis_df is None else len(pair_vis_df)}")\n\n# -----------------------------\n# Plot and save\n# -----------------------------\natlas_result_dir = Path("atlas_result_output")\natlas_result_dir.mkdir(parents=True, exist_ok=True)\n\nresult_prefix = "iterative_lddmm_before_after_all_points"\nresult_png_path = atlas_result_dir / f"{result_prefix}.png"\nresult_pdf_path = atlas_result_dir / f"{result_prefix}.pdf"\n\nfig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True, facecolor="white")\n\nfor ax, xi, yi, mask, title in [\n    (axes[0], xi_before, yi_before, mask_before, "Before iterative LDDMM"),\n    (axes[1], xi_after, yi_after, mask_after, "After iterative LDDMM"),\n]:\n    ax.set_facecolor("white")\n    ax.imshow(sl > 0, cmap="binary", alpha=0.10, origin="lower")\n    ax.imshow(overlay_atlas, origin="lower", interpolation="nearest")\n    ax.scatter(\n        xi[mask],\n        yi[mask],\n        s=1,\n        c=point_colors[mask],\n        edgecolors="none",\n        rasterized=True,\n    )\n    ax.set_title(f"{title} (all points)")\n    ax.axis("off")\n    ax.set_aspect("equal")\n\nplt.tight_layout()\nsave_rasterized_scatter_pdf_png(fig, result_png_path, result_pdf_path, dpi=300)\nprint("saved result visualization PNG:", result_png_path)\nprint("saved result visualization PDF:", result_pdf_path)\nplt.show()\n'

OUTPUT_CELL_SOURCE = 'from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\n\nif "df_aligned_all" not in globals():\n    raise ValueError("Run the iterative multi-level alignment cell first; df_aligned_all is missing.")\n\nout_dir = Path("iterative_alignment_outputs")\nout_dir.mkdir(parents=True, exist_ok=True)\n\n\ndef _csv_safe_value(x):\n    if isinstance(x, (list, tuple, set, np.ndarray)):\n        return ";".join(str(int(v)) if isinstance(v, (int, np.integer)) or str(v).replace(".", "", 1).isdigit() else str(v) for v in list(x))\n    return x\n\n\ndef _make_csv_safe(df):\n    df = df.copy()\n    for col in df.columns:\n        if df[col].dtype == "object":\n            df[col] = df[col].map(_csv_safe_value)\n    return df\n\n# Final aligned point table: all cells/points after the last iterative transform.\nfinal_aligned_path = out_dir / "final_aligned_all_points.csv"\ndf_aligned_all.to_csv(final_aligned_path, index=True)\n\n# Filtered point table used for final-stage pair discovery, useful for debugging masks.\nfiltered_aligned_path = out_dir / "final_filtered_points_for_matching.csv"\nif "df_smooth" in globals():\n    df_smooth.to_csv(filtered_aligned_path, index=True)\nelse:\n    filtered_aligned_path = None\n\n# Pair tables: one CSV per stage and one combined CSV.\nif "iter_pairs_by_stage" not in globals():\n    iter_pairs_by_stage = {}\n    if "iter_stage_outputs" in globals():\n        for stage_out in iter_stage_outputs:\n            stage = int(stage_out.get("stage", len(iter_pairs_by_stage) + 1))\n            iter_pairs_by_stage[stage] = stage_out.get("pair_df", pd.DataFrame()).copy()\n\npair_paths = []\nfor stage, pair_table in iter_pairs_by_stage.items():\n    if pair_table is None:\n        pair_table = pd.DataFrame()\n    label_col = ""\n    is_final_stage = False\n    if "iter_stage_outputs" in globals():\n        match = next((s for s in iter_stage_outputs if int(s.get("stage", -1)) == int(stage)), None)\n        if match is not None:\n            label_col = match.get("label_col", "")\n            is_final_stage = bool(match.get("is_final_stage", False))\n\n    stage_df = pair_table.copy()\n    if len(stage_df) > 0:\n        stage_df.insert(0, "stage", int(stage))\n        stage_df.insert(1, "label_col", label_col)\n        stage_df.insert(2, "is_final_stage", is_final_stage)\n\n    stage_path = out_dir / f"matched_pairs_stage_{int(stage):02d}.csv"\n    _make_csv_safe(stage_df).to_csv(stage_path, index=False)\n    pair_paths.append(stage_path)\n\nif "iter_pairs_all_df" in globals() and len(iter_pairs_all_df) > 0:\n    combined_pairs_df = iter_pairs_all_df.copy()\nelif len(pair_paths) > 0:\n    combined_rows = []\n    for stage, pair_table in iter_pairs_by_stage.items():\n        if pair_table is not None and len(pair_table) > 0:\n            tmp = pair_table.copy()\n            tmp.insert(0, "stage", int(stage))\n            combined_rows.append(tmp)\n    combined_pairs_df = pd.concat(combined_rows, ignore_index=True, sort=False) if len(combined_rows) else pd.DataFrame()\nelse:\n    combined_pairs_df = pd.DataFrame()\n\ncombined_pairs_path = out_dir / "matched_pairs_all_stages.csv"\n_make_csv_safe(combined_pairs_df).to_csv(combined_pairs_path, index=False)\n\nif "final_pair_df" in globals():\n    final_pairs_df = final_pair_df.copy()\nelif "pair_df" in globals():\n    final_pairs_df = pair_df.copy()\nelse:\n    final_pairs_df = pd.DataFrame()\n\nfinal_pairs_path = out_dir / "matched_pairs_final_stage.csv"\n_make_csv_safe(final_pairs_df).to_csv(final_pairs_path, index=False)\n\nif "iter_pair_summary_df" in globals():\n    summary_df = iter_pair_summary_df.copy()\nelif "iter_stage_outputs" in globals():\n    summary_df = pd.DataFrame([\n        {\n            "stage": s.get("stage"),\n            "label_col": s.get("label_col"),\n            "is_final_stage": bool(s.get("is_final_stage", False)),\n            "n_pairs": 0 if s.get("pair_df") is None else len(s.get("pair_df")),\n            "lddmm_ran": s.get("lddmm_out") is not None,\n        }\n        for s in iter_stage_outputs\n    ])\nelse:\n    summary_df = pd.DataFrame()\n\nsummary_path = out_dir / "iterative_alignment_stage_summary.csv"\nsummary_df.to_csv(summary_path, index=False)\n\nmanifest = {\n    "final_aligned_all_points": str(final_aligned_path),\n    "final_filtered_points_for_matching": None if filtered_aligned_path is None else str(filtered_aligned_path),\n    "matched_pairs_final_stage": str(final_pairs_path),\n    "matched_pairs_all_stages": str(combined_pairs_path),\n    "stage_summary": str(summary_path),\n    "matched_pairs_each_stage": [str(x) for x in pair_paths],\n}\nmanifest_path = out_dir / "manifest.json"\nmanifest_path.write_text(json.dumps(manifest, indent=2))\n\nprint("saved final alignment outputs to:", out_dir.resolve())\nprint("final aligned points:", final_aligned_path)\nprint("final matched pairs:", final_pairs_path)\nprint("all-stage matched pairs:", combined_pairs_path)\nprint("stage summary:", summary_path)\n\nsummary_df\n'
