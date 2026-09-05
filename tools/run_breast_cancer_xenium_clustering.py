#!/usr/bin/env python
"""Check and cluster the Xenium breast cancer replicates.

The workflow intentionally uses only Xenium ``Gene Expression`` features,
performs light cell-level QC, normalizes all 313 panel genes, and compares a
range of BANKSY lambda values before and after Harmony integration.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import harmonypy as hm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from banksy.initialize_banksy import initialize_banksy
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

import spAlignDE


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


SAMPLE_COLORS = {"Rep1": "#0F4D92", "Rep2": "#E28E2C"}
RANDOM_STATE = 1000


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def lambda_key(value: float) -> str:
    return f"lambda_{value:g}".replace(".", "p")


def validate_lambdas(values: list[float]) -> list[float]:
    result = sorted(set(float(value) for value in values))
    if not result or any(value < 0 or value > 1 for value in result):
        raise ValueError("BANKSY lambda values must be in [0, 1].")
    return result


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_xenium_sample(
    rep: int,
    data_dir: Path,
    source_dir: Path,
    *,
    min_counts: int,
    min_genes: int,
    max_control_fraction: float,
    max_cells: int | None,
    random_state: int,
) -> tuple[ad.AnnData, dict[str, Any], pd.DataFrame]:
    sample_id = f"Rep{rep}"
    matrix_path = data_dir / f"Xenium_FFPE_Human_Breast_Cancer_Rep{rep}_cell_feature_matrix.h5"
    cells_path = source_dir / f"Xenium_FFPE_Human_Breast_Cancer_Rep{rep}_cells.csv.gz"
    if not matrix_path.exists() or not cells_path.exists():
        raise FileNotFoundError(f"Missing input for {sample_id}: {matrix_path} or {cells_path}")

    cells = pd.read_csv(cells_path)
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.rename(columns={"x_centroid": "x", "y_centroid": "y"})

    with h5py.File(matrix_path, "r") as handle:
        matrix = handle["matrix"]
        barcodes = decode(matrix["barcodes"][:])
        feature_names = decode(matrix["features/name"][:])
        feature_types = decode(matrix["features/feature_type"][:])
        shape = tuple(int(value) for value in matrix["shape"][:])
        genes_by_cells = sp.csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=shape,
        )

    csv_ids = cells["cell_id"].to_numpy()
    csv_rows_source = len(cells)
    barcodes_exact_and_ordered = np.array_equal(barcodes, csv_ids)
    duplicate_ids = cells.loc[cells["cell_id"].duplicated(keep=False), "cell_id"].unique()
    if len(duplicate_ids):
        raise ValueError(
            f"{sample_id}: cells CSV contains duplicate cell_id values; examples: "
            f"{duplicate_ids[:5].tolist()}."
        )
    csv_index = pd.Index(csv_ids)
    h5_index = pd.Index(barcodes)
    missing_barcodes = h5_index[~h5_index.isin(csv_index)]
    if len(missing_barcodes):
        raise ValueError(
            f"{sample_id}: cells CSV is missing {len(missing_barcodes)} H5 barcodes; "
            f"examples: {missing_barcodes[:5].tolist()}."
        )

    # A cells table may contain metadata-only rows with no matching expression
    # column. Match by cell_id and reorder to the H5 matrix in that case.
    cells = cells.set_index("cell_id", drop=False).loc[barcodes].reset_index(drop=True)

    gene_mask = feature_types == "Gene Expression"
    if int(gene_mask.sum()) != 313:
        raise ValueError(f"{sample_id}: expected 313 Gene Expression features, found {gene_mask.sum()}.")
    gene_names = feature_names[gene_mask]
    gene_counts = genes_by_cells[gene_mask, :].T.tocsr().astype(np.float32)
    control_counts = np.asarray(genes_by_cells[~gene_mask, :].sum(axis=0)).ravel()
    total_gene_counts = np.asarray(gene_counts.sum(axis=1)).ravel()
    detected_genes = np.asarray((gene_counts > 0).sum(axis=1)).ravel()

    csv_transcripts = cells["transcript_counts"].to_numpy()
    if not np.array_equal(total_gene_counts.astype(csv_transcripts.dtype), csv_transcripts):
        corr = float(np.corrcoef(total_gene_counts, csv_transcripts)[0, 1])
        raise ValueError(f"{sample_id}: H5 gene sums do not match cells CSV transcript_counts (r={corr:.4f}).")

    control_fraction = control_counts / np.maximum(total_gene_counts + control_counts, 1)
    keep = (
        (total_gene_counts >= min_counts)
        & (detected_genes >= min_genes)
        & (control_fraction <= max_control_fraction)
    )

    rng = np.random.default_rng(random_state + rep)
    kept_positions = np.flatnonzero(keep)
    if max_cells is not None and len(kept_positions) > max_cells:
        kept_positions = np.sort(rng.choice(kept_positions, size=max_cells, replace=False))

    obs = cells.iloc[kept_positions].copy()
    obs["sample_id"] = sample_id
    obs["n_genes_by_counts"] = detected_genes[kept_positions]
    obs["gene_counts_from_h5"] = total_gene_counts[kept_positions]
    obs["control_counts_from_h5"] = control_counts[kept_positions]
    obs["control_fraction"] = control_fraction[kept_positions]
    obs.index = pd.Index([f"{sample_id}_{cell_id}" for cell_id in obs["cell_id"]], name="spot_uid")

    var = pd.DataFrame({"feature_type": "Gene Expression"}, index=pd.Index(gene_names, name="gene"))
    counts = gene_counts[kept_positions, :].tocsr()
    adata = ad.AnnData(X=counts.copy(), obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float64)

    q = [0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1]
    input_check = {
        "sample_id": sample_id,
        "matrix_path": str(matrix_path),
        "cells_path": str(cells_path),
        "n_cells_h5": int(shape[1]),
        "n_cells_csv_source": int(csv_rows_source),
        "n_cells_csv_used": int(len(cells)),
        "n_cells_csv_ignored": int(csv_rows_source - len(cells)),
        "barcodes_exact_and_ordered": bool(barcodes_exact_and_ordered),
        "h5_barcodes_present_once": True,
        "gene_sums_equal_csv_transcript_counts": True,
        "n_features_total": int(shape[0]),
        "n_gene_expression_features": int(gene_mask.sum()),
        "feature_type_counts": {
            str(key): int(value) for key, value in zip(*np.unique(feature_types, return_counts=True))
        },
        "gene_count_quantiles": {str(key): float(value) for key, value in zip(q, np.quantile(total_gene_counts, q))},
        "detected_gene_quantiles": {str(key): float(value) for key, value in zip(q, np.quantile(detected_genes, q))},
        "n_cells_after_qc": int(len(kept_positions)),
        "n_cells_removed": int(len(cells) - int(keep.sum())),
        "n_cells_subsampled": int(len(kept_positions)),
    }
    qc_row = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "n_cells_before": len(cells),
                "n_cells_after_qc": int(keep.sum()),
                "n_cells_used": len(kept_positions),
                "n_removed_qc": int((~keep).sum()),
                "pct_removed_qc": float((~keep).mean()),
                "removed_low_counts": int((total_gene_counts < min_counts).sum()),
                "removed_low_detected_genes": int((detected_genes < min_genes).sum()),
                "removed_high_control_fraction": int((control_fraction > max_control_fraction).sum()),
            }
        ]
    )
    return adata, input_check, qc_row


def zscore_float32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    values -= mean
    values /= std
    return np.nan_to_num(values, copy=False)


def vectorized_banksy_blocks(adata: ad.AnnData, num_neighbours: int) -> dict[str, np.ndarray]:
    """Build the three z-scored BANKSY blocks in the original cell order."""
    n_obs, n_vars = adata.shape
    blocks = {
        "self": np.empty((n_obs, n_vars), dtype=np.float32),
        "nbr0": np.empty((n_obs, n_vars), dtype=np.float32),
        "nbr1": np.empty((n_obs, n_vars), dtype=np.float32),
    }

    sample_values = adata.obs["sample_id"].astype(str).to_numpy()
    for sample_id in sorted(np.unique(sample_values)):
        positions = np.flatnonzero(sample_values == sample_id)
        sample = adata[positions, :].copy()
        log(f"Initializing BANKSY spatial weights for {sample_id} ({sample.n_obs:,} cells)")
        banksy_dict = initialize_banksy(
            sample,
            coord_keys=("x", "y", "spatial"),
            num_neighbours=num_neighbours,
            nbr_weight_decay="scaled_gaussian",
            max_m=1,
            plt_edge_hist=False,
            plt_nbr_weights=False,
            plt_agf_angles=False,
            plt_theta=False,
        )
        weights = banksy_dict["scaled_gaussian"]["weights"]
        expression = sample.X.toarray().astype(np.float32, copy=False)

        nbr0 = np.asarray(weights[0] @ expression)
        weight1 = weights[1].tocsr()
        adjacency = sp.csr_matrix(
            (np.ones(weight1.nnz, dtype=np.float32), weight1.indices.copy(), weight1.indptr.copy()),
            shape=weight1.shape,
        )
        degree = np.diff(adjacency.indptr).astype(np.float32)
        degree[degree == 0] = 1.0
        local_mean = np.asarray(adjacency @ expression) / degree[:, None]
        weighted = np.asarray(weight1 @ expression)
        weight_sum = np.asarray(weight1.sum(axis=1)).ravel()
        nbr1 = np.abs(weighted - weight_sum[:, None] * local_mean)

        blocks["self"][positions, :] = zscore_float32(expression)
        blocks["nbr0"][positions, :] = zscore_float32(nbr0)
        blocks["nbr1"][positions, :] = zscore_float32(nbr1)
        del sample, banksy_dict, weights, expression, nbr0, weight1, adjacency, local_mean, weighted, nbr1
        gc.collect()
    return blocks


def banksy_representation(blocks: dict[str, np.ndarray], lambda_value: float) -> np.ndarray:
    self_weight = np.sqrt(1.0 - lambda_value)
    nbr0_weight = np.sqrt(lambda_value * 2.0 / 3.0)
    nbr1_weight = np.sqrt(lambda_value * 1.0 / 3.0)
    active: list[np.ndarray] = []
    if self_weight > 0:
        active.append((blocks["self"] * self_weight).astype(np.float32, copy=False))
    if nbr0_weight > 0:
        active.append((blocks["nbr0"] * nbr0_weight).astype(np.float32, copy=False))
    if nbr1_weight > 0:
        active.append((blocks["nbr1"] * nbr1_weight).astype(np.float32, copy=False))
    if len(active) == 1:
        return np.ascontiguousarray(active[0])
    return np.ascontiguousarray(np.concatenate(active, axis=1), dtype=np.float32)


def remove_neighbor_graph(adata: ad.AnnData, key: str) -> None:
    for suffix in ("distances", "connectivities"):
        adata.obsp.pop(f"{key}_{suffix}", None)
    adata.uns.pop(key, None)


def batch_metrics(
    embedding: np.ndarray,
    sample_codes: np.ndarray,
    *,
    random_state: int,
    max_cells: int = 40000,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    if len(sample_codes) > max_cells:
        selected = np.sort(rng.choice(len(sample_codes), size=max_cells, replace=False))
    else:
        selected = np.arange(len(sample_codes))
    x = np.asarray(embedding[selected], dtype=np.float32)
    y = sample_codes[selected]

    neighbors = NearestNeighbors(n_neighbors=31, metric="euclidean", n_jobs=-1).fit(x)
    neighbor_ids = neighbors.kneighbors(return_distance=False)[:, 1:]
    same_batch = float(np.mean(y[neighbor_ids] == y[:, None]))
    p_batch1 = (y[neighbor_ids] == 1).mean(axis=1)
    ilisi = float(np.mean(1.0 / (p_batch1**2 + (1.0 - p_batch1) ** 2)))

    silhouette_n = min(5000, len(y))
    silhouette = float(
        silhouette_score(x, y, sample_size=silhouette_n, random_state=random_state, metric="euclidean")
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    predicted = cross_val_predict(
        LogisticRegression(max_iter=500, class_weight="balanced"),
        x,
        y,
        cv=cv,
        n_jobs=1,
    )
    balanced_accuracy = float(balanced_accuracy_score(y, predicted))
    expected_same = float(np.sum((np.bincount(y) / len(y)) ** 2))
    return {
        "same_batch_knn30": same_batch,
        "random_same_batch_expectation": expected_same,
        "mean_ilisi": ilisi,
        "batch_silhouette": silhouette,
        "batch_logistic_balanced_accuracy": balanced_accuracy,
        "metric_n_cells": int(len(y)),
    }


def cluster_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    sample_codes: np.ndarray,
    *,
    random_state: int,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    selected = np.arange(len(labels))
    if len(selected) > 10000:
        selected = np.sort(rng.choice(selected, size=10000, replace=False))
    silhouette = float(
        silhouette_score(embedding[selected], labels[selected], sample_size=min(5000, len(selected)), random_state=random_state)
    )
    table = pd.crosstab(labels, sample_codes).to_numpy(dtype=float)
    fractions = table / np.maximum(table.sum(axis=1, keepdims=True), 1)
    entropy = -(fractions * np.log2(np.maximum(fractions, 1e-12))).sum(axis=1)
    weighted_entropy = float(np.average(entropy, weights=table.sum(axis=1)))
    return {
        "n_clusters": int(len(np.unique(labels))),
        "cluster_silhouette": silhouette,
        "weighted_cluster_batch_entropy": weighted_entropy,
    }


def compute_umap_and_leiden(
    obs: pd.DataFrame,
    pca_scores: np.ndarray,
    harmony_scores: np.ndarray,
    *,
    n_neighbors: int,
    resolution: float,
    random_state: int,
) -> tuple[ad.AnnData, np.ndarray, np.ndarray]:
    result = ad.AnnData(obs=obs.copy())
    result.obsm["X_pca_banksy"] = np.asarray(pca_scores, dtype=np.float32)
    result.obsm["X_harmony_banksy"] = np.asarray(harmony_scores, dtype=np.float32)

    sc.pp.neighbors(
        result,
        use_rep="X_pca_banksy",
        n_neighbors=n_neighbors,
        random_state=random_state,
        key_added="neighbors_before_harmony",
    )
    sc.tl.umap(result, neighbors_key="neighbors_before_harmony", random_state=random_state, min_dist=0.35)
    before_umap = result.obsm["X_umap"].copy()
    remove_neighbor_graph(result, "neighbors_before_harmony")

    sc.pp.neighbors(
        result,
        use_rep="X_harmony_banksy",
        n_neighbors=n_neighbors,
        random_state=random_state,
        key_added="neighbors_after_harmony",
    )
    sc.tl.umap(result, neighbors_key="neighbors_after_harmony", random_state=random_state, min_dist=0.35)
    after_umap = result.obsm["X_umap"].copy()
    sc.tl.leiden(
        result,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        random_state=random_state,
        directed=False,
        neighbors_key="neighbors_after_harmony",
        key_added="leiden_harmony",
    )
    result.obsm["X_umap_before_harmony"] = before_umap
    result.obsm["X_umap_after_harmony"] = after_umap
    result.obsm.pop("X_umap", None)
    remove_neighbor_graph(result, "neighbors_after_harmony")
    return result, before_umap, after_umap


def scatter_samples(ax: plt.Axes, coords: np.ndarray, samples: np.ndarray, title: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    colors = pd.Series(samples[order]).map(SAMPLE_COLORS).to_numpy()
    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=colors,
        s=0.22,
        alpha=0.42,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")


def save_figure(fig: plt.Figure, stem: Path, *, dpi: int = 300) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lambda_before_after(
    before: np.ndarray,
    after: np.ndarray,
    samples: np.ndarray,
    lambda_value: float,
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
    out_dir: Path,
    seed: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    scatter_samples(
        axes[0],
        before,
        samples,
        f"Before Harmony | lambda={lambda_value:g}\nsame-batch NN={metrics_before['same_batch_knn30']:.3f}",
        seed,
    )
    scatter_samples(
        axes[1],
        after,
        samples,
        f"After Harmony | lambda={lambda_value:g}\nsame-batch NN={metrics_after['same_batch_knn30']:.3f}",
        seed,
    )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=4)
        for label, color in SAMPLE_COLORS.items()
    ]
    axes[1].legend(handles=handles, loc="best", markerscale=1.2)
    save_figure(fig, out_dir / f"umap_before_after_{lambda_key(lambda_value)}")


def plot_cluster_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    lambda_value: float,
    out_dir: Path,
    seed: int,
) -> None:
    levels = sorted(np.unique(labels), key=lambda value: (len(str(value)), str(value)))
    palette = plt.get_cmap("tab20", max(len(levels), 1))
    color_map = {level: palette(index) for index, level in enumerate(levels)}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    fig, ax = plt.subplots(figsize=(3.8, 3.45), constrained_layout=True)
    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=[color_map[value] for value in labels[order]],
        s=0.22,
        alpha=0.48,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(f"After Harmony Leiden | lambda={lambda_value:g}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[level], label=str(level), markersize=3)
        for level in levels
    ]
    ax.legend(handles=handles, title="Leiden", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)
    save_figure(fig, out_dir / f"umap_clusters_{lambda_key(lambda_value)}")


def plot_summary_grid(records: list[dict[str, Any]], samples: np.ndarray, out_dir: Path, seed: int) -> None:
    fig, axes = plt.subplots(len(records), 2, figsize=(7.2, 2.65 * len(records)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, record in enumerate(records):
        value = record["lambda"]
        scatter_samples(
            axes[row, 0],
            record["before_umap"],
            samples,
            f"lambda={value:g} | before Harmony",
            seed + row,
        )
        scatter_samples(
            axes[row, 1],
            record["after_umap"],
            samples,
            f"lambda={value:g} | after Harmony",
            seed + row,
        )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=4)
        for label, color in SAMPLE_COLORS.items()
    ]
    axes[0, 1].legend(handles=handles, loc="best")
    save_figure(fig, out_dir / "umap_batch_before_after_all_lambdas", dpi=320)


def plot_metric_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    before = metrics[metrics["stage"] == "before_harmony"].sort_values("lambda")
    after = metrics[metrics["stage"] == "after_harmony"].sort_values("lambda")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    for frame, label, color, marker in [
        (before, "Before Harmony", "#767676", "o"),
        (after, "After Harmony", "#0F4D92", "s"),
    ]:
        axes[0].plot(frame["lambda"], frame["same_batch_knn30"], marker=marker, color=color, label=label)
        axes[1].plot(frame["lambda"], frame["mean_ilisi"], marker=marker, color=color, label=label)
        axes[2].plot(
            frame["lambda"],
            frame["batch_logistic_balanced_accuracy"],
            marker=marker,
            color=color,
            label=label,
        )
    axes[0].axhline(
        metrics["random_same_batch_expectation"].iloc[0], color="#B64342", linestyle="--", linewidth=0.9
    )
    axes[2].axhline(0.5, color="#B64342", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("Same-batch 30-NN")
    axes[1].set_ylabel("Mean iLISI")
    axes[2].set_ylabel("Batch prediction\nbalanced accuracy")
    for ax in axes:
        ax.set_xlabel("BANKSY lambda")
        ax.set_xticks(sorted(metrics["lambda"].unique()))
    axes[1].legend(loc="best")
    save_figure(fig, out_dir / "batch_metrics_by_lambda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the two Xenium cell-feature H5 files.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the matching official Xenium cells CSV files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.2, 0.5, 0.8, 1.0])
    parser.add_argument("--min-counts", type=int, default=20)
    parser.add_argument("--min-genes", type=int, default=10)
    parser.add_argument("--max-control-fraction", type=float, default=0.05)
    parser.add_argument("--normalize-target", type=float, default=1e4)
    parser.add_argument("--banksy-neighbors", type=int, default=30)
    parser.add_argument("--graph-neighbors", type=int, default=50)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--harmony-theta", type=float, default=4.0)
    parser.add_argument("--harmony-max-iter", type=int, default=30)
    parser.add_argument("--leiden-resolution", type=float, default=0.3)
    parser.add_argument("--max-cells-per-sample", type=int)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lambdas = validate_lambdas(args.lambdas)
    seed_controls = spAlignDE.set_random_seed(
        args.seed,
        deterministic_torch=True,
    )
    sc.settings.seed = args.seed

    config = {
        "input_data_dir": str(args.data_dir),
        "official_cells_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "lambdas": lambdas,
        "min_counts": args.min_counts,
        "min_genes": args.min_genes,
        "max_control_fraction": args.max_control_fraction,
        "normalize_total_target": args.normalize_target,
        "hvg_selection": False,
        "features_used": "Gene Expression only",
        "banksy_neighbors": args.banksy_neighbors,
        "banksy_max_m": 1,
        "banksy_decay": "scaled_gaussian",
        "pca_components": args.pca_components,
        "harmony_theta": args.harmony_theta,
        "harmony_max_iter": args.harmony_max_iter,
        "graph_neighbors": args.graph_neighbors,
        "leiden_resolution": args.leiden_resolution,
        "leiden_flavor": "igraph",
        "leiden_n_iterations": 2,
        "random_state": args.seed,
        "seed_controls": seed_controls,
        "max_cells_per_sample": args.max_cells_per_sample,
    }
    write_json(args.output_dir / "run_config.json", config)

    samples: list[ad.AnnData] = []
    input_checks: list[dict[str, Any]] = []
    qc_rows: list[pd.DataFrame] = []
    for rep in (1, 2):
        log(f"Reading and auditing Xenium Rep{rep}")
        sample, input_check, qc_row = read_xenium_sample(
            rep,
            args.data_dir,
            args.source_dir,
            min_counts=args.min_counts,
            min_genes=args.min_genes,
            max_control_fraction=args.max_control_fraction,
            max_cells=args.max_cells_per_sample,
            random_state=args.seed,
        )
        samples.append(sample)
        input_checks.append(input_check)
        qc_rows.append(qc_row)

    write_json(args.output_dir / "input_check.json", input_checks)
    pd.concat(qc_rows, ignore_index=True).to_csv(args.output_dir / "qc_summary.csv", index=False)

    joint = ad.concat(samples, join="inner", merge="same", index_unique=None)
    joint.obs["sample_id"] = pd.Categorical(joint.obs["sample_id"], categories=["Rep1", "Rep2"])
    joint.obsm["spatial"] = joint.obs[["x", "y"]].to_numpy(dtype=np.float64)
    if joint.n_vars != 313:
        raise RuntimeError(f"Expected 313 shared panel genes, found {joint.n_vars}.")
    pd.DataFrame({"gene": joint.var_names.astype(str), "feature_type": "Gene Expression"}).to_csv(
        args.output_dir / "genes_used.csv", index=False
    )

    log(f"Normalizing {joint.n_obs:,} cells x {joint.n_vars} genes with target_sum={args.normalize_target:g}")
    sc.pp.normalize_total(joint, target_sum=args.normalize_target)
    sc.pp.log1p(joint)
    joint.uns["preprocessing"] = config
    preprocessed_path = args.output_dir / "xenium_rep1_rep2_preprocessed_313genes.h5ad"
    joint.write_h5ad(preprocessed_path, compression="gzip")
    log(f"Wrote {preprocessed_path}")

    blocks = vectorized_banksy_blocks(joint, args.banksy_neighbors)
    samples_array = joint.obs["sample_id"].astype(str).to_numpy()
    sample_codes = pd.Categorical(samples_array, categories=["Rep1", "Rep2"]).codes
    metric_rows: list[dict[str, Any]] = []
    figure_records: list[dict[str, Any]] = []

    for lambda_value in lambdas:
        key = lambda_key(lambda_value)
        lambda_dir = args.output_dir / key
        lambda_dir.mkdir(parents=True, exist_ok=True)
        log(f"Running BANKSY lambda={lambda_value:g}")
        representation = banksy_representation(blocks, lambda_value)
        pca = PCA(n_components=args.pca_components, svd_solver="randomized", random_state=args.seed)
        pca_scores = pca.fit_transform(representation).astype(np.float32)
        del representation
        gc.collect()

        log(f"Running Harmony theta={args.harmony_theta:g} for lambda={lambda_value:g}")
        harmony = hm.run_harmony(
            pca_scores,
            joint.obs,
            "sample_id",
            theta=args.harmony_theta,
            max_iter_harmony=args.harmony_max_iter,
            random_state=args.seed,
            verbose=False,
        )
        harmony_scores = np.asarray(harmony.Z_corr, dtype=np.float32)
        if harmony_scores.shape != pca_scores.shape:
            if harmony_scores.T.shape == pca_scores.shape:
                harmony_scores = harmony_scores.T.copy()
            else:
                raise RuntimeError(
                    f"Unexpected Harmony shape {harmony_scores.shape}; expected {pca_scores.shape}."
                )

        before_metrics = batch_metrics(pca_scores, sample_codes, random_state=args.seed)
        after_metrics = batch_metrics(harmony_scores, sample_codes, random_state=args.seed)

        log(f"Computing UMAP and Leiden for lambda={lambda_value:g}")
        result, before_umap, after_umap = compute_umap_and_leiden(
            joint.obs,
            pca_scores,
            harmony_scores,
            n_neighbors=args.graph_neighbors,
            resolution=args.leiden_resolution,
            random_state=args.seed,
        )
        labels = result.obs["leiden_harmony"].astype(str).to_numpy()
        after_cluster_metrics = cluster_metrics(
            harmony_scores, labels, sample_codes, random_state=args.seed
        )
        for stage, stage_metrics in [("before_harmony", before_metrics), ("after_harmony", after_metrics)]:
            metric_rows.append(
                {
                    "lambda": lambda_value,
                    "stage": stage,
                    "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
                    **stage_metrics,
                    **(after_cluster_metrics if stage == "after_harmony" else {}),
                }
            )

        cluster_table = pd.crosstab(result.obs["leiden_harmony"], result.obs["sample_id"])
        cluster_table.to_csv(lambda_dir / "cluster_by_sample_counts.csv")

        export = result.obs[
            [
                "sample_id",
                "cell_id",
                "x",
                "y",
                "gene_counts_from_h5",
                "n_genes_by_counts",
                "control_fraction",
                "leiden_harmony",
            ]
        ].copy()
        export["umap_before_1"] = before_umap[:, 0]
        export["umap_before_2"] = before_umap[:, 1]
        export["umap_after_1"] = after_umap[:, 0]
        export["umap_after_2"] = after_umap[:, 1]
        export.to_csv(lambda_dir / "umap_and_clusters.csv.gz", compression="gzip")

        result.uns["lambda"] = lambda_value
        result.uns["harmony_theta"] = args.harmony_theta
        result.uns["leiden_resolution"] = args.leiden_resolution
        result.uns["pca_explained_variance_ratio"] = pca.explained_variance_ratio_
        result.write_h5ad(lambda_dir / "embeddings_and_clusters.h5ad", compression="gzip")

        plot_lambda_before_after(
            before_umap,
            after_umap,
            samples_array,
            lambda_value,
            before_metrics,
            after_metrics,
            lambda_dir,
            args.seed,
        )
        plot_cluster_umap(after_umap, labels, lambda_value, lambda_dir, args.seed)
        figure_records.append(
            {
                "lambda": lambda_value,
                "before_umap": before_umap,
                "after_umap": after_umap,
            }
        )
        log(
            f"lambda={lambda_value:g}: same-batch NN {before_metrics['same_batch_knn30']:.3f} -> "
            f"{after_metrics['same_batch_knn30']:.3f}; iLISI {before_metrics['mean_ilisi']:.3f} -> "
            f"{after_metrics['mean_ilisi']:.3f}; clusters={after_cluster_metrics['n_clusters']}"
        )
        del result, pca_scores, harmony_scores, harmony, before_umap, after_umap, labels, pca
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "batch_and_cluster_metrics.csv", index=False)
    plot_summary_grid(figure_records, samples_array, args.output_dir, args.seed)
    plot_metric_summary(metrics, args.output_dir)
    log(f"Completed all lambda values. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
