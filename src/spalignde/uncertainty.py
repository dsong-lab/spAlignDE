"""Cross-sample subsampling-stability utilities for the MERFISH benchmark.

The functions here reproduce the analysis described for Figure 2E and the
corresponding Supplementary Methods.  Each of ten independently subsampled
datasets has its own structure construction, prealignment and S-LDDMM
transformation.  Transformation variability is evaluated by mapping the same
replicate-1 prealigned query points through all ten transformations.
"""

from __future__ import annotations

import json
import inspect
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


from .alignment import _slddmm_core as lddmm


def set_plot_style() -> None:
    """Apply compact publication-style matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _repeat_from_path(path: Path) -> int:
    match = re.search(r"rep(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse repeat id from {path}")
    return int(match.group(1))


def discover_lddmm_inputs(input_dir: str | Path) -> list[dict]:
    """Find prepared replicate LDDMM input files."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("lddmm_input_rep*.npz"), key=_repeat_from_path)
    if not files:
        raise RuntimeError(f"No lddmm_input_rep*.npz files found in {input_dir}")
    return [{"repeat": _repeat_from_path(path), "path": str(path)} for path in files]


def load_repeat_input(path: str | Path) -> dict:
    """Load one replicate input and normalize legacy/current key names."""
    dat = np.load(path, allow_pickle=True)
    out = {key: dat[key] for key in dat.files}
    out["source_image"] = out.get("source_image", out.get("I"))
    out["target_image"] = out.get("target_image", out.get("J"))
    out["source_grid_x"] = out.get("source_grid_x", out.get("XI"))
    out["source_grid_y"] = out.get("source_grid_y", out.get("YI"))
    out["target_grid_x"] = out.get("target_grid_x", out.get("XJ"))
    out["target_grid_y"] = out.get("target_grid_y", out.get("YJ"))
    return out


def _as_numpy_1d(x) -> np.ndarray:
    return lddmm.as_numpy_1d(x)


def _string_scalar(x) -> str:
    arr = np.asarray(x)
    return str(arr.item() if arr.ndim == 0 else arr.reshape(-1)[0])


def _float_scalar(x) -> float:
    arr = np.asarray(x)
    return float(arr.item() if arr.ndim == 0 else arr.reshape(-1)[0])


def run_or_load_alignments(
    manifest: list[dict],
    *,
    output_dir: str | Path,
    model_cfg: dict | None = None,
    optim_cfg: dict | None = None,
    em_cfg: dict | None = None,
    intensity_cfg: dict | None = None,
    device: str | None = None,
    force: bool = False,
    save_transforms: bool = True,
    require_transforms: bool = False,
    verbose: bool = True,
    print_every: int = 100,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Run LDDMM for missing replicates and cache mapped source coordinates."""
    output_dir = Path(output_dir)
    aligned_dir = output_dir / "aligned_points"
    transform_dir = output_dir / "transforms"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    transform_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if model_cfg is None:
        model_cfg = {"a": 300, "p": 2.0, "expand": 2.0, "nt": 3, "grid_step": 100}
    if optim_cfg is None:
        optim_cfg = {"niter": 500, "lrM": 4e3, "lrM_min": 4e3, "grad_clip_m0": 1e3}

    requested_config = {
        "model_cfg": model_cfg,
        "optim_cfg": optim_cfg,
        "em_cfg": em_cfg,
        "intensity_cfg": intensity_cfg,
        "device": device,
    }
    config_path = output_dir / "run_config.json"
    if not force and config_path.exists():
        with open(config_path, encoding="utf-8") as handle:
            cached_config = json.load(handle)
        if cached_config != requested_config:
            force = True
            if verbose:
                print("Cached transformations use different parameters; rerunning all replicates.")

    summary_path = output_dir / "lddmm_repeat_summary.csv"
    prior_summary = pd.read_csv(summary_path) if summary_path.exists() and not force else pd.DataFrame()
    prior_by_repeat = {
        int(row["repeat"]): row.to_dict()
        for _, row in prior_summary.iterrows()
        if "repeat" in prior_summary.columns
    }

    aligned_by_repeat: dict[int, pd.DataFrame] = {}
    rows = []
    for item in manifest:
        rep = int(item["repeat"])
        path = Path(item["path"])
        out_csv = aligned_dir / f"aligned_points_rep{rep:02d}.csv"
        transform_npz = transform_dir / f"lddmm_transform_rep{rep:02d}.npz"
        dat = load_repeat_input(path)

        cached_alignment_ok = out_csv.exists() and not force
        cached_transform_ok = (not save_transforms) or transform_npz.exists() or (not require_transforms)
        if cached_alignment_ok and cached_transform_ok:
            df = pd.read_csv(out_csv)
            aligned_by_repeat[rep] = df
            row = prior_by_repeat.get(
                rep,
                {
                    "repeat": rep,
                    "input_path": str(path),
                    "status": "cached",
                    "n_source_points": int(len(df)),
                    "finalE": np.nan,
                    "elapsed_sec": np.nan,
                },
            )
            row["status"] = "cached"
            rows.append(row)
            continue

        print(f"\n=== Running LDDMM repeat {rep:02d} ===")
        call_kwargs = {
            "model_cfg": model_cfg,
            "optim_cfg": optim_cfg,
            "device": device,
            "verbose": verbose,
            "print_every": print_every,
        }
        supported = inspect.signature(lddmm.run_lddmm_pipeline_source_target).parameters
        if "em_cfg" in supported:
            call_kwargs["em_cfg"] = em_cfg
        if "intensity_cfg" in supported:
            call_kwargs["intensity_cfg"] = intensity_cfg

        res = lddmm.run_lddmm_pipeline_source_target(
            dat["source_grid_y"],
            dat["source_grid_x"],
            dat["source_image"],
            dat["target_grid_y"],
            dat["target_grid_x"],
            dat["target_image"],
            **call_kwargs,
        )

        source_points_yx = np.column_stack([dat["y_src_prealign"], dat["x_src_prealign"]])
        mapped_yx = lddmm.map_points_source_to_target(res["xv"], res["v"], res["A"], source_points_yx)
        mapped_xy = np.column_stack([_as_numpy_1d(mapped_yx[:, 1]), _as_numpy_1d(mapped_yx[:, 0])])

        df = pd.DataFrame(
            {
                "repeat": rep,
                "source_cell_id": np.asarray(dat["src_cell_numeric_ids"]).astype(str),
                "x_raw": np.asarray(dat["x_src_raw"], dtype=float),
                "y_raw": np.asarray(dat["y_src_raw"], dtype=float),
                "x_prealign": np.asarray(dat["x_src_prealign"], dtype=float),
                "y_prealign": np.asarray(dat["y_src_prealign"], dtype=float),
                "x_lddmm": mapped_xy[:, 0],
                "y_lddmm": mapped_xy[:, 1],
                "source_label": np.asarray(dat["src_labels"]).astype(str),
            }
        )
        df.to_csv(out_csv, index=False)
        aligned_by_repeat[rep] = df

        if save_transforms:
            xv = res["xv"]
            v = res["v"]
            A = res["A"]
            np.savez_compressed(
                transform_npz,
                repeat=np.asarray(rep, dtype=np.int32),
                xv_y=_as_numpy_1d(xv[0]),
                xv_x=_as_numpy_1d(xv[1]),
                v=np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, dtype=np.float32),
                A=np.asarray(A.detach().cpu() if torch.is_tensor(A) else A, dtype=np.float32),
            )

        final_e = float(np.asarray(res["Esave"])[-1][0]) if len(res.get("Esave", [])) else np.nan
        rows.append(
            {
                "repeat": rep,
                "input_path": str(path),
                "status": "ran",
                "device": device,
                "n_source_points": int(len(df)),
                "n_target_points": int(len(dat["x_tgt"])),
                "prealign_method": _string_scalar(dat.get("prealign_method", "unknown")),
                "prealign_weighted_centroid_rmse": _float_scalar(
                    dat.get("prealign_weighted_centroid_rmse", np.nan)
                ),
                "finalE": final_e,
                "elapsed_sec": float(res.get("elapsed_sec", np.nan)),
            }
        )

    summary = pd.DataFrame(rows).sort_values("repeat").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(requested_config, handle, indent=2)
    return aligned_by_repeat, summary


def compute_point_uncertainty(
    aligned_by_repeat: dict[int, pd.DataFrame],
    *,
    min_repeats: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean mapped coordinates and coordinate spread by source cell."""
    all_aligned = pd.concat(aligned_by_repeat.values(), ignore_index=True)
    all_aligned["source_cell_id"] = all_aligned["source_cell_id"].astype(str)

    grouped = all_aligned.groupby("source_cell_id", sort=False)
    base = grouped.agg(
        n_repeats=("repeat", "nunique"),
        x_raw=("x_raw", "mean"),
        y_raw=("y_raw", "mean"),
        x_prealign=("x_prealign", "mean"),
        y_prealign=("y_prealign", "mean"),
        x_mean=("x_lddmm", "mean"),
        y_mean=("y_lddmm", "mean"),
        x_std=("x_lddmm", "std"),
        y_std=("y_lddmm", "std"),
        x_var=("x_lddmm", "var"),
        y_var=("y_lddmm", "var"),
        source_label=("source_label", lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0]),
    ).reset_index()

    base = base.loc[base["n_repeats"] >= int(min_repeats)].copy()
    merged = all_aligned.merge(base[["source_cell_id", "x_mean", "y_mean"]], on="source_cell_id", how="inner")
    merged["dist_to_mean"] = np.hypot(
        merged["x_lddmm"] - merged["x_mean"],
        merged["y_lddmm"] - merged["y_mean"],
    )

    dist_stats = (
        merged.groupby("source_cell_id", sort=False)["dist_to_mean"]
        .agg(dist_mean="mean", dist_std="std", dist_max="max")
        .reset_index()
    )
    out = base.merge(dist_stats, on="source_cell_id", how="left")
    out["x_std"] = out["x_std"].fillna(0.0)
    out["y_std"] = out["y_std"].fillna(0.0)
    out["x_var"] = out["x_var"].fillna(0.0)
    out["y_var"] = out["y_var"].fillna(0.0)
    out["std_total"] = np.sqrt(out["x_var"] + out["y_var"])
    out["dist_std"] = out["dist_std"].fillna(0.0)
    out["uncertainty_rank"] = out["std_total"].rank(method="first", ascending=False).astype(int)
    out["uncertainty_percentile"] = out["std_total"].rank(pct=True)
    out = out.sort_values("uncertainty_rank").reset_index(drop=True)
    return out, all_aligned


def save_uncertainty_outputs(
    uncertainty_df: pd.DataFrame,
    all_aligned: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write coordinate uncertainty tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_path = output_dir / "point_uncertainty_summary.csv"
    aligned_path = output_dir / "aligned_points_all_repeats.csv"
    uncertainty_df.to_csv(uncertainty_path, index=False)
    all_aligned.to_csv(aligned_path, index=False)
    return {"uncertainty": str(uncertainty_path), "aligned_points": str(aligned_path)}


def load_target_xy(input_path: str | Path) -> np.ndarray:
    dat = load_repeat_input(input_path)
    return np.column_stack([np.asarray(dat["x_tgt"], dtype=float), np.asarray(dat["y_tgt"], dtype=float)])


def load_saved_transforms(transform_dir: str | Path) -> dict[int, dict]:
    """Load cached LDDMM transforms saved by ``run_or_load_alignments``."""
    transform_dir = Path(transform_dir)
    files = sorted(transform_dir.glob("lddmm_transform_rep*.npz"), key=_repeat_from_path)
    if not files:
        raise RuntimeError(f"No lddmm_transform_rep*.npz files found in {transform_dir}")

    out = {}
    for path in files:
        dat = np.load(path)
        rep = int(np.asarray(dat["repeat"]).item())
        out[rep] = {
            "xv": [dat["xv_y"], dat["xv_x"]],
            "v": dat["v"],
            "A": dat["A"],
            "path": str(path),
        }
    return out


def map_reference_points_through_transforms(
    transforms_by_repeat: dict[int, dict],
    *,
    reference_input_path: str | Path,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Tutorial-equivalent mapping of one source point set through every repeat.

    This mirrors ``evaluate_lddmm_repeats`` in the robustness tutorial: choose
    one selected repeat's prealigned source coordinates, then map that exact
    point set through each repeat LDDMM transform.
    """
    dat = load_repeat_input(reference_input_path)
    x_src_new = np.asarray(dat["x_src_prealign"], dtype=float)
    y_src_new = np.asarray(dat["y_src_prealign"], dtype=float)
    src_points_yx = np.column_stack([y_src_new, x_src_new])

    src_lddmm_xy_by_repeat = {}
    for rep in sorted(transforms_by_repeat):
        tr = transforms_by_repeat[rep]
        mapped_yx = lddmm.map_points_source_to_target(tr["xv"], tr["v"], tr["A"], src_points_yx)
        src_lddmm_xy_by_repeat[rep] = np.column_stack(
            [_as_numpy_1d(mapped_yx[:, 1]), _as_numpy_1d(mapped_yx[:, 0])]
        )
    return src_lddmm_xy_by_repeat, x_src_new, y_src_new


def compute_repeat_point_variance(
    src_lddmm_xy_by_repeat: dict[int, np.ndarray],
    x_src_new,
    y_src_new,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute pointwise coordinate/distance variance across repeat transforms.

    This intentionally matches the robustness tutorial helper of the same name.
    It expects every repeat array to contain the same source point set in the
    same order.
    """
    if len(src_lddmm_xy_by_repeat) < 2:
        raise RuntimeError("Need src_lddmm_xy_by_repeat with >=2 repeats.")
    reps = sorted(src_lddmm_xy_by_repeat.keys())
    arr = np.stack([np.asarray(src_lddmm_xy_by_repeat[r], dtype=float) for r in reps], axis=0)
    mean_xy = arr.mean(axis=0)
    dist_to_mean = np.linalg.norm(arr - mean_xy[None, :, :], axis=2)
    var_xy = arr.var(axis=0, ddof=1)

    point_var_df = pd.DataFrame(
        {
            "point_idx": np.arange(arr.shape[1]),
            "x_src": np.asarray(x_src_new),
            "y_src": np.asarray(y_src_new),
            "x_mean": mean_xy[:, 0],
            "y_mean": mean_xy[:, 1],
            "var_x": var_xy[:, 0],
            "var_y": var_xy[:, 1],
            "dist_mean": dist_to_mean.mean(axis=0),
            "dist_var": dist_to_mean.var(axis=0, ddof=1),
            "dist_std": np.sqrt(dist_to_mean.var(axis=0, ddof=1)),
        }
    )
    point_var_df["var_total"] = point_var_df["var_x"] + point_var_df["var_y"]
    point_var_df["std_x"] = np.sqrt(point_var_df["var_x"])
    point_var_df["std_y"] = np.sqrt(point_var_df["var_y"])
    point_var_df["std_total"] = np.sqrt(point_var_df["var_total"])
    return point_var_df, dist_to_mean


def plot_distance_variance_map(
    point_var_df: pd.DataFrame,
    src_lddmm_xy_by_repeat: dict[int, np.ndarray],
    tgt_xy,
    *,
    repeat: int = 1,
    high_percentile: float = 95.0,
    vmax_percentile: float = 99.0,
):
    """Plot the Figure 2E metric and outline its high-variability region."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Ellipse

    if repeat not in src_lddmm_xy_by_repeat:
        repeat = sorted(src_lddmm_xy_by_repeat.keys())[0]
    xy = np.asarray(src_lddmm_xy_by_repeat[repeat])
    target_xy = np.asarray(tgt_xy, dtype=float)
    vals = point_var_df["dist_var"].to_numpy(dtype=float)
    vmax = float(np.nanpercentile(vals, vmax_percentile))
    threshold = float(np.nanpercentile(vals, high_percentile))
    high = vals >= threshold

    def high_variability_ellipse(points: np.ndarray, padding: float = 1.18):
        points = np.asarray(points, dtype=float)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 3:
            return None
        center = np.nanmedian(points, axis=0)
        spread = points - center
        covariance = np.cov(spread.T)
        if not np.isfinite(covariance).all():
            return None
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        vectors = vectors[:, order]
        rotated = spread @ vectors
        half_width, half_height = np.nanpercentile(np.abs(rotated), 97.5, axis=0)
        angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
        return center, 2 * half_width * padding, 2 * half_height * padding, angle

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 5.1), constrained_layout=True)
    ax.scatter(
        target_xy[:, 0],
        target_xy[:, 1],
        s=0.45,
        alpha=0.18,
        c="#B8B8B8",
        linewidths=0,
        rasterized=True,
        label="S2R2 reference",
    )
    norm = Normalize(vmin=float(np.nanmin(vals)), vmax=vmax, clip=True)
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=vals,
        s=1.6,
        cmap="viridis",
        norm=norm,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
        label=f"S2R3 query (replicate {repeat})",
    )

    ellipse = high_variability_ellipse(xy[high])
    if ellipse is not None:
        center, width, height, angle = ellipse
        ax.add_patch(
            Ellipse(
                xy=center,
                width=width,
                height=height,
                angle=angle,
                facecolor="none",
                edgecolor="#2F3A45",
                linewidth=1.0,
                linestyle=(0, (3.2, 2.2)),
                label=f"top {100 - high_percentile:g}% variability",
                zorder=5,
            )
        )

    all_xy = np.vstack([xy, target_xy])
    (xmin, ymin), (xmax, ymax) = np.nanmin(all_xy, axis=0), np.nanmax(all_xy, axis=0)
    pad = 0.025 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_title("Subsampling-based transformation variability", fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.04)
    cbar.set_label("Distance variance across replicates")
    ax.text(
        0.01,
        0.01,
        f"95th percentile = {threshold:,.1f}; n = {int(high.sum()):,}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#2F3A45",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    return fig


def plot_all_replicate_overlays(
    aligned_by_repeat: dict[int, pd.DataFrame],
    target_xy: np.ndarray,
    *,
    ncols: int = 5,
    s_src: float = 0.35,
    s_tgt: float = 0.25,
    alpha_src: float = 0.20,
    alpha_tgt: float = 0.08,
):
    """Small-multiple QC overlay for every replicate alignment."""
    reps = sorted(aligned_by_repeat)
    ncols = min(int(ncols), max(len(reps), 1))
    nrows = int(np.ceil(len(reps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    tgt = np.asarray(target_xy, dtype=float)

    for i, rep in enumerate(reps):
        ax = axes[i]
        src = aligned_by_repeat[rep][["x_lddmm", "y_lddmm"]].to_numpy(dtype=float)
        ax.scatter(tgt[:, 0], tgt[:, 1], s=s_tgt, color="#9E9E9E", alpha=alpha_tgt, linewidths=0)
        ax.scatter(src[:, 0], src[:, 1], s=s_src, color="#2F7F73", alpha=alpha_src, linewidths=0)
        ax.set_title(f"rep {rep:02d}", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(reps), len(axes)):
        axes[j].axis("off")
    return fig


def plot_uncertainty_map(
    uncertainty_df: pd.DataFrame,
    target_xy: np.ndarray | None = None,
    *,
    value_col: str = "std_total",
    clip_percentile: float = 99.0,
    title: str = "Spatial map of alignment uncertainty",
):
    """Map coordinate uncertainty on mean aligned source coordinates."""
    vals = uncertainty_df[value_col].to_numpy(dtype=float)
    vmax = np.nanpercentile(vals, clip_percentile) if clip_percentile is not None else np.nanmax(vals)
    vals_plot = np.clip(vals, None, vmax)

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.4), constrained_layout=True)
    if target_xy is not None:
        tgt = np.asarray(target_xy, dtype=float)
        ax.scatter(tgt[:, 0], tgt[:, 1], s=0.55, color="#B8B8B8", alpha=0.12, label="target")
    sc = ax.scatter(
        uncertainty_df["x_mean"],
        uncertainty_df["y_mean"],
        c=vals_plot,
        s=1.6,
        cmap="magma",
        alpha=0.90,
        linewidths=0,
        label="source mean",
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(value_col)
    return fig


def plot_uncertainty_distribution(
    uncertainty_df: pd.DataFrame,
    *,
    value_col: str = "dist_var",
    high_percentile: float = 95.0,
):
    """Show the pointwise transformation-variability distribution."""
    vals = uncertainty_df[value_col].to_numpy(dtype=float)
    cutoff = float(np.nanpercentile(vals, high_percentile))
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 2.8), constrained_layout=True)
    ax.hist(vals, bins=60, color="#4C8C84", edgecolor="white", linewidth=0.3)
    ax.axvline(cutoff, color="#C44E52", linewidth=1.2, label=f"{high_percentile:.0f}th percentile")
    label = "Distance variance across replicates" if value_col == "dist_var" else value_col
    ax.set_xlabel(label)
    ax.set_ylabel("Fixed query points")
    ax.set_title("Distribution of pointwise transformation variability")
    ax.legend()
    return fig, cutoff


def write_brief_report(
    *,
    output_dir: str | Path,
    summary: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    high_percentile: float = 95.0,
) -> Path:
    """Write the paper-aligned machine-readable benchmark summary."""
    output_dir = Path(output_dir)
    cutoff = float(np.nanpercentile(uncertainty_df["dist_var"], high_percentile))
    high = uncertainty_df.loc[uncertainty_df["dist_var"] >= cutoff]
    count_key = "n_points_with_uncertainty" if "point_idx" in uncertainty_df.columns else "n_cells_with_uncertainty"
    report = {
        "n_repeats": int(summary["repeat"].nunique()),
        count_key: int(len(uncertainty_df)),
        "primary_metric": "dist_var",
        "median_dist_var": float(uncertainty_df["dist_var"].median()),
        "p95_dist_var": cutoff,
        "max_dist_var": float(uncertainty_df["dist_var"].max()),
        "median_std_total": float(uncertainty_df["std_total"].median()),
        "p95_std_total": float(np.nanpercentile(uncertainty_df["std_total"], high_percentile)),
        "max_std_total": float(uncertainty_df["std_total"].max()),
        "n_high_variability_points": int(len(high)),
        "interpretation": (
            "Most fixed query points have low transformation variability across the ten "
            "80% subsampling replicates. The upper tail is concentrated where reference-query "
            "overlap is weak or incomplete. This is an empirical stability measure, not a "
            "calibrated probability or confidence interval."
        ),
    }
    path = output_dir / "uncertainty_report.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path
