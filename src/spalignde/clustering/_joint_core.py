"""Reusable joint clustering utilities for spAlignDE documentation.

This module is a documentation-ready extraction of the joint BANKSY clustering
workflow. It supports two input styles:

- CSV folder/input list: one metadata CSV and one count CSV per sample.
- AnnData input: one combined AnnData object or ``.h5ad`` with all samples.

The core workflow is:

1. Normalize inputs into per-sample metadata/count tables.
2. Align genes across samples.
3. Build one joint AnnData object.
4. Compute BANKSY features per sample.
5. Integrate samples with joint PCA and Harmony.
6. Cluster with Leiden on the Harmony BANKSY representation.
7. Export raw joint clustering results.
8. Optionally run boundary-aware refinement as a separate post-processing step.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import harmonypy as hm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from banksy.embed_banksy import generate_banksy_matrix
from banksy.initialize_banksy import initialize_banksy
from banksy_utils.umap_pca import pca_umap
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import binary_closing, disk, remove_small_holes
from sklearn.neighbors import NearestNeighbors


@dataclass
class PipelineParams:
    """Parameters for the joint BANKSY clustering workflow."""

    num_neighbours: int = 30
    lambda_list: Tuple[float, ...] = (0.8,)
    pca_dims: Tuple[int, ...] = (20,)
    resolutions: Tuple[float, ...] = (1.4,)
    partition_seed: int = 1000
    decay: str = "scaled_gaussian"
    harmony_max_iter: int = 30
    harmony_theta: float = 2.0
    umap_n_neighbors: int = 30
    snn_k: int = 50


def discover_sample_csv_pairs(data_dir: str | os.PathLike[str]) -> List[Dict[str, str]]:
    """Discover paired CSV inputs in a folder.

    Expected file names are ``cell_metadata_<sample_id>.csv`` and
    ``cell_by_gene_<sample_id>.csv``.
    """
    data_path = Path(data_dir).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    metas: Dict[str, str] = {}
    counts: Dict[str, str] = {}
    for csv_path in data_path.glob("*.csv"):
        name = csv_path.name
        if name.startswith("cell_metadata_"):
            sample_id = name[len("cell_metadata_") : -4]
            if sample_id:
                metas[sample_id] = str(csv_path)
        elif name.startswith("cell_by_gene_"):
            sample_id = name[len("cell_by_gene_") : -4]
            if sample_id:
                counts[sample_id] = str(csv_path)

    sample_ids = sorted(set(metas).intersection(counts))
    if not sample_ids:
        found = sorted(p.name for p in data_path.glob("*.csv"))
        raise FileNotFoundError(
            "No matched CSV pairs found. Expected files named "
            "'cell_metadata_<sample_id>.csv' and 'cell_by_gene_<sample_id>.csv'. "
            f"CSV files found: {found[:10]}"
        )

    return [
        {"sample_id": sid, "metadata_csv": metas[sid], "counts_csv": counts[sid]}
        for sid in sample_ids
    ]


def validate_and_load_sample(
    metadata_csv: str | os.PathLike[str],
    counts_csv: str | os.PathLike[str],
    sample_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load one sample's metadata/count CSV pair and validate required fields."""
    meta = pd.read_csv(metadata_csv).copy()
    counts = pd.read_csv(counts_csv).copy()

    required_meta_cols = {"cell_id", "x", "y"}
    missing_meta = required_meta_cols.difference(meta.columns)
    if missing_meta:
        raise ValueError(f"[{sample_id}] metadata missing columns: {sorted(missing_meta)}")
    if "cell_id" not in counts.columns:
        raise ValueError(f"[{sample_id}] counts CSV must contain 'cell_id'.")

    meta["cell_id"] = meta["cell_id"].astype(str)
    counts["cell_id"] = counts["cell_id"].astype(str)

    if meta["cell_id"].duplicated().any():
        raise ValueError(f"[{sample_id}] metadata contains duplicated cell_id values.")
    if counts["cell_id"].duplicated().any():
        raise ValueError(f"[{sample_id}] counts contains duplicated cell_id values.")

    for col in ("x", "y"):
        if not pd.api.types.is_numeric_dtype(meta[col]):
            raise TypeError(f"[{sample_id}] metadata column '{col}' must be numeric.")

    gene_cols = [c for c in counts.columns if c != "cell_id"]
    if not gene_cols:
        raise ValueError(f"[{sample_id}] counts CSV has no gene columns.")
    bad_cols = [c for c in gene_cols if not pd.api.types.is_numeric_dtype(counts[c])]
    if bad_cols:
        raise TypeError(f"[{sample_id}] non-numeric gene columns: {bad_cols[:10]}")
    if counts[gene_cols].isna().any().any():
        raise ValueError(f"[{sample_id}] counts contain NA values.")
    if (counts[gene_cols] < 0).any().any():
        raise ValueError(f"[{sample_id}] counts contain negative values.")

    meta = meta.set_index("cell_id")
    counts = counts.set_index("cell_id")
    common_ids = meta.index.intersection(counts.index)
    if len(common_ids) == 0:
        raise ValueError(f"[{sample_id}] no overlapping cell_id values.")

    meta_aligned = meta.loc[common_ids].copy()
    counts_aligned = counts.loc[common_ids, gene_cols].copy()
    meta_aligned["cell_id"] = meta_aligned.index.astype(str)
    meta_aligned["sample_id"] = str(sample_id)
    return _set_spot_uid(meta_aligned, counts_aligned, sample_id)


def load_csv_sample_tables(
    samples: Iterable[Mapping[str, str]],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load and validate many CSV sample definitions."""
    sample_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        meta, counts = validate_and_load_sample(
            sample["metadata_csv"],
            sample["counts_csv"],
            sample_id,
        )
        sample_tables[sample_id] = {"meta": meta, "counts": counts}
    return sample_tables


def sample_tables_from_anndata(
    adata: ad.AnnData,
    *,
    sample_key: str = "sample_id",
    x_key: str = "x",
    y_key: str = "y",
    cell_id_key: str = "cell_id",
    layer: Optional[str] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Convert one combined AnnData object into per-sample metadata/count tables.

    Required ``adata.obs`` columns are ``sample_key``, ``x_key``, and ``y_key``.
    If ``cell_id_key`` is absent, ``adata.obs_names`` are used as cell IDs.
    """
    required = {sample_key, x_key, y_key}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"AnnData obs missing required columns: {sorted(missing)}")

    obs = adata.obs.copy()
    obs[sample_key] = obs[sample_key].astype(str)
    if cell_id_key not in obs.columns:
        obs[cell_id_key] = adata.obs_names.astype(str)
    obs[cell_id_key] = obs[cell_id_key].astype(str)

    for col in (x_key, y_key):
        if not pd.api.types.is_numeric_dtype(obs[col]):
            raise TypeError(f"AnnData obs column '{col}' must be numeric.")

    x_matrix = adata.layers[layer] if layer is not None else adata.X
    x_array = x_matrix.toarray() if sp.issparse(x_matrix) else np.asarray(x_matrix)
    counts_df = pd.DataFrame(
        x_array,
        index=adata.obs_names.astype(str),
        columns=adata.var_names.astype(str),
    )

    tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sample_id in sorted(obs[sample_key].unique()):
        mask = obs[sample_key] == sample_id
        meta = obs.loc[mask, :].copy()
        meta = meta.rename(columns={sample_key: "sample_id", x_key: "x", y_key: "y", cell_id_key: "cell_id"})
        counts = counts_df.loc[mask.values, :].copy()
        meta, counts = _set_spot_uid(meta, counts, str(sample_id))
        tables[str(sample_id)] = {"meta": meta, "counts": counts}

    return tables


def load_h5ad_sample_tables(
    h5ad_path: str | os.PathLike[str],
    **kwargs: Any,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load a combined ``.h5ad`` and convert it to per-sample tables."""
    return sample_tables_from_anndata(sc.read_h5ad(h5ad_path), **kwargs)


def _set_spot_uid(
    meta: pd.DataFrame,
    counts: pd.DataFrame,
    sample_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Set globally unique spot IDs as indices for metadata and counts."""
    meta = meta.copy()
    counts = counts.copy()
    meta["cell_id"] = meta["cell_id"].astype(str)
    if meta["cell_id"].str.startswith(f"{sample_id}_").all():
        spot_uid = meta["cell_id"].tolist()
    else:
        spot_uid = [f"{sample_id}_{cid}" for cid in meta["cell_id"]]
    meta.index = pd.Index(spot_uid, name="spot_uid")
    counts.index = pd.Index(spot_uid, name="spot_uid")
    return meta, counts


def align_genes_across_samples(
    sample_tables: Mapping[str, Mapping[str, pd.DataFrame]],
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """Align count matrices to the union of genes across all samples."""
    all_genes = sorted({gene for sid in sample_tables for gene in sample_tables[sid]["counts"].columns})
    aligned_counts: Dict[str, pd.DataFrame] = {}
    for sid, tables in sample_tables.items():
        counts = tables["counts"]
        aligned = pd.DataFrame(0.0, index=counts.index, columns=all_genes)
        aligned.loc[:, counts.columns] = counts.values
        aligned_counts[sid] = aligned
    return all_genes, aligned_counts


def build_raw_counts(aligned_counts: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Return float count matrices with stable sample ordering."""
    return {sid: aligned_counts[sid].astype(float).copy() for sid in sorted(aligned_counts)}


def build_joint_adata(
    sample_tables: Mapping[str, Mapping[str, pd.DataFrame]],
    raw_counts: Mapping[str, pd.DataFrame],
) -> ad.AnnData:
    """Build one joint AnnData from per-sample metadata and count matrices."""
    obs = pd.concat([sample_tables[sid]["meta"] for sid in sorted(sample_tables)], axis=0)
    x_df = pd.concat([raw_counts[sid] for sid in sorted(raw_counts)], axis=0)
    obs = obs.loc[x_df.index].copy()

    adata = ad.AnnData(X=sp.csr_matrix(x_df.values))
    adata.obs_names = x_df.index.astype(str)
    adata.var_names = x_df.columns.astype(str)
    adata.obs = obs
    adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
    return adata


def run_banksy_per_sample(
    adata: ad.AnnData,
    *,
    num_neighbours: int,
    lambda_list: Sequence[float],
    coord_keys: Tuple[str, str, str] = ("x", "y", "spatial"),
    nbr_weight_decay: str = "scaled_gaussian",
) -> Dict[str, Any]:
    """Compute BANKSY features independently for each sample."""
    banksy_results: Dict[str, Any] = {}
    for sample_id in sorted(adata.obs["sample_id"].astype(str).unique()):
        adata_s = adata[adata.obs["sample_id"].astype(str) == sample_id].copy()
        banksy_dict = initialize_banksy(
            adata_s,
            coord_keys=coord_keys,
            num_neighbours=num_neighbours,
            nbr_weight_decay=nbr_weight_decay,
            max_m=1,
            plt_edge_hist=False,
            plt_nbr_weights=False,
            plt_agf_angles=False,
            plt_theta=False,
        )
        banksy_dict, _ = generate_banksy_matrix(
            adata_s,
            banksy_dict,
            lambda_list=list(lambda_list),
            max_m=1,
            verbose=False,
        )
        banksy_results[sample_id] = banksy_dict
    return banksy_results


def integrate_joint_with_harmony(
    banksy_results: Mapping[str, Any],
    *,
    selected_lambda: float,
    decay: str,
    pca_dims: Sequence[int],
    random_state: int = 1000,
    harmony_max_iter: int = 30,
    harmony_theta: float = 2.0,
) -> ad.AnnData:
    """Concatenate BANKSY outputs, run joint PCA, then Harmony correction."""
    parts = []
    for sample_id in sorted(banksy_results):
        sample_adata = banksy_results[sample_id][decay][selected_lambda]["adata"].copy()
        sample_adata.obs["sample_id"] = sample_id
        sample_adata.obs["cell_id"] = sample_adata.obs["cell_id"].astype(str)
        sample_adata.obs_names = [f"{sample_id}_{cid}" for cid in sample_adata.obs["cell_id"]]
        parts.append(sample_adata)

    banksy_adata = ad.concat(parts, join="outer")
    tmp = {decay: {selected_lambda: {"adata": banksy_adata}}}
    # pyBANKSY's PCA helper delegates to randomized sklearn PCA without
    # exposing random_state. Seed immediately before that PCA call.
    np.random.seed(random_state)
    sc.settings.seed = random_state
    pca_umap(tmp, pca_dims=list(pca_dims), add_umap=False, plt_remaining_var=False)
    banksy_adata = tmp[decay][selected_lambda]["adata"]

    pca_key = f"reduced_pc_{list(pca_dims)[0]}"
    harmony = hm.run_harmony(
        banksy_adata.obsm[pca_key],
        banksy_adata.obs,
        "sample_id",
        max_iter_harmony=harmony_max_iter,
        theta=harmony_theta,
        random_state=random_state,
        verbose=False,
    )
    banksy_adata.obsm["Harmony_BANKSY"] = harmony.Z_corr
    return banksy_adata


def compute_pre_post_harmony_umap(
    banksy_adata: ad.AnnData,
    *,
    pca_dim: int,
    n_neighbors: int,
    random_state: int = 1000,
) -> ad.AnnData:
    """Compute UMAPs before and after Harmony integration."""
    sc.pp.neighbors(
        banksy_adata,
        use_rep=f"reduced_pc_{pca_dim}",
        n_neighbors=n_neighbors,
        random_state=random_state,
        key_added="neighbors_pca",
    )
    sc.tl.umap(banksy_adata, neighbors_key="neighbors_pca", random_state=random_state)
    banksy_adata.obsm["X_umap_pca"] = banksy_adata.obsm["X_umap"].copy()

    sc.pp.neighbors(
        banksy_adata,
        use_rep="Harmony_BANKSY",
        n_neighbors=n_neighbors,
        random_state=random_state,
        key_added="neighbors_harmony",
    )
    sc.tl.umap(banksy_adata, neighbors_key="neighbors_harmony", random_state=random_state)
    banksy_adata.obsm["X_umap_harmony"] = banksy_adata.obsm["X_umap"].copy()
    return banksy_adata


def run_harmony_snn_leiden(
    banksy_adata: ad.AnnData,
    *,
    k: int,
    resolution: float,
    random_state: int = 1000,
    flavor: str = "leidenalg",
    n_iterations: int = -1,
    key_added: str = "leiden_harmony",
) -> ad.AnnData:
    """Build an SNN graph on Harmony BANKSY features and run Leiden clustering."""
    x = banksy_adata.obsm["Harmony_BANKSY"]
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x)
    dist, idx = nn.kneighbors(x)
    n = x.shape[0]
    nbr_sets = [set(row) for row in idx]

    rows: List[int] = []
    cols: List[int] = []
    weights: List[float] = []
    for i in range(n):
        order = np.argsort(dist[i])
        for rank, (j, d) in enumerate(zip(idx[i][order], dist[i][order]), start=1):
            if i == j:
                continue
            shared = len(nbr_sets[i].intersection(nbr_sets[j]))
            if rank <= 3 or shared >= 5:
                rows.append(i)
                cols.append(int(j))
                weights.append(1.0 / (1.0 + float(d)))

    graph = sp.coo_matrix((weights, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T).tocsr()
    banksy_adata.obsp["connectivities"] = graph
    banksy_adata.obsp["distances"] = graph
    banksy_adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": k, "method": "banksy_snn"},
    }
    sc.tl.leiden(
        banksy_adata,
        resolution=resolution,
        n_iterations=n_iterations,
        random_state=random_state,
        directed=False,
        flavor=flavor,
        key_added=key_added,
    )
    return banksy_adata


def get_spatial_xy(adata_s: ad.AnnData, x_col: str = "x", y_col: str = "y") -> np.ndarray:
    """Return spatial coordinates from obs columns or ``obsm['spatial']``."""
    if {x_col, y_col}.issubset(adata_s.obs.columns):
        return adata_s.obs[[x_col, y_col]].to_numpy().astype(float)
    if "spatial" in adata_s.obsm:
        return np.asarray(adata_s.obsm["spatial"])[:, :2].astype(float)
    raise KeyError("No coordinates found. Need x/y in obs or obsm['spatial'].")


def estimate_boundary_mask(
    xy: np.ndarray,
    *,
    grid_size: int = 256,
    closing_radius: int = 2,
    hole_area: int = 64,
    boundary_dist_px: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate whether each point is near the tissue boundary."""
    x = xy[:, 0]
    y = xy[:, 1]
    eps = 1e-8

    gx = ((x - x.min()) / (x.max() - x.min() + eps) * (grid_size - 1)).astype(int)
    gy = ((y - y.min()) / (y.max() - y.min() + eps) * (grid_size - 1)).astype(int)

    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[gy, gx] = True
    mask = binary_closing(mask, footprint=disk(closing_radius))
    mask = remove_small_holes(mask, area_threshold=hole_area)

    dist = ndi.distance_transform_edt(mask)
    dist_pts = dist[gy, gx]
    return dist_pts <= boundary_dist_px, dist_pts


def weighted_vote(labels: np.ndarray, distances: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    """Distance-weighted label vote used by boundary-aware refinement."""
    labels = np.asarray(labels, dtype=object)
    distances = np.asarray(distances, dtype=float)
    weights = 1.0 / (distances + 1e-6)

    score: Dict[str, float] = {}
    for label, weight in zip(labels, weights):
        key = str(label)
        score[key] = score.get(key, 0.0) + float(weight)

    best_label = max(score, key=score.get)
    total = sum(score.values())
    best_frac = score[best_label] / total if total > 0 else 0.0
    return best_label, float(best_frac), score


def refine_labels_boundary_aware(
    adata_s: ad.AnnData,
    *,
    raw_label_col: str,
    x_col: str = "x",
    y_col: str = "y",
    k_interior: int = 150,
    k_boundary: int = 8,
    boundary_dist_px: int = 3,
    protected_labels: Iterable[str] = ("L1", "Layer1", "layer1", "1"),
    min_same_frac_keep: float = 0.8,
    min_new_frac_interior: float = 0.2,
    min_new_frac_boundary: float = 0.8,
    min_new_frac_protected: float = 0.9,
    only_change_if_disagree: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """Refine labels with the same boundary-aware rule used by cluster_single/single_clustering.py."""
    xy = get_spatial_xy(adata_s, x_col=x_col, y_col=y_col)
    raw = adata_s.obs[raw_label_col].astype(str).to_numpy()
    n = len(raw)

    is_boundary, _ = estimate_boundary_mask(xy, boundary_dist_px=boundary_dist_px)
    tree = cKDTree(xy)
    k_max = max(k_interior, k_boundary) + 1
    d_all, idx_all = tree.query(xy, k=k_max)

    refined = raw.copy()
    protected_set = set(map(str, protected_labels))
    n_changed = 0
    n_boundary_changed = 0
    n_protected_changed = 0

    for i in range(n):
        current = raw[i]
        use_k = k_boundary if is_boundary[i] else k_interior

        nbr_idx = idx_all[i, 1 : use_k + 1]
        nbr_dist = d_all[i, 1 : use_k + 1]
        nbr_labels = raw[nbr_idx]

        best_label, best_frac, score = weighted_vote(nbr_labels, nbr_dist)
        total_score = sum(score.values()) if score else 0.0
        current_frac = score.get(str(current), 0.0) / total_score if total_score > 0 else 0.0

        if current in protected_set:
            change_thr = min_new_frac_protected
        elif is_boundary[i]:
            change_thr = min_new_frac_boundary
        else:
            change_thr = min_new_frac_interior

        weak_current = current_frac < min_same_frac_keep if only_change_if_disagree else True

        if (best_label != current) and weak_current and (best_frac >= change_thr):
            refined[i] = best_label
            n_changed += 1
            if is_boundary[i]:
                n_boundary_changed += 1
            if current in protected_set:
                n_protected_changed += 1

    stats = {
        "n_cells": int(n),
        "n_changed": int(n_changed),
        "pct_changed": float(n_changed / max(n, 1)),
        "n_boundary": int(is_boundary.sum()),
        "n_boundary_changed": int(n_boundary_changed),
        "n_protected_changed": int(n_protected_changed),
    }
    return refined.tolist(), stats


def run_boundary_refinement_by_sample(
    banksy_adata: ad.AnnData,
    *,
    raw_label_col: str = "leiden_harmony",
    refined_label_col: str = "leiden_harmony_refined",
    sample_id_col: str = "sample_id",
    x_col: str = "x",
    y_col: str = "y",
    min_cells_per_sample: int = 1,
    k_interior: int = 150,
    k_boundary: int = 8,
    boundary_dist_px: int = 3,
    protected_labels: Iterable[str] = ("L1", "Layer1", "layer1", "1"),
    min_same_frac_keep: float = 0.8,
    min_new_frac_interior: float = 0.2,
    min_new_frac_boundary: float = 0.8,
    min_new_frac_protected: float = 0.9,
    only_change_if_disagree: bool = True,
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Run optional refinement after joint clustering.

    This is intentionally separate from joint clustering and uses the same
    boundary-aware method as the public single-sample clustering workflow.
    """
    if raw_label_col not in banksy_adata.obs.columns:
        raise KeyError(f"Missing raw label column: {raw_label_col}")

    banksy_adata.obs[refined_label_col] = banksy_adata.obs[raw_label_col].astype(str)
    refine_stats: Dict[str, Dict[str, Any]] = {}

    if sample_id_col in banksy_adata.obs.columns:
        sample_ids = sorted(banksy_adata.obs[sample_id_col].astype(str).unique())
        masks = [banksy_adata.obs[sample_id_col].astype(str) == sid for sid in sample_ids]
    else:
        sample_ids = ["__all__"]
        masks = [slice(None)]

    for sample_id, mask in zip(sample_ids, masks):
        sample_adata = banksy_adata[mask].copy()
        n_cells = int(sample_adata.n_obs)

        if n_cells >= min_cells_per_sample:
            refined_labels, stats = refine_labels_boundary_aware(
                sample_adata,
                raw_label_col=raw_label_col,
                x_col=x_col,
                y_col=y_col,
                k_interior=k_interior,
                k_boundary=k_boundary,
                boundary_dist_px=boundary_dist_px,
                protected_labels=protected_labels,
                min_same_frac_keep=min_same_frac_keep,
                min_new_frac_interior=min_new_frac_interior,
                min_new_frac_boundary=min_new_frac_boundary,
                min_new_frac_protected=min_new_frac_protected,
                only_change_if_disagree=only_change_if_disagree,
            )
        else:
            refined_labels = sample_adata.obs[raw_label_col].astype(str).tolist()
            stats = {
                "n_cells": n_cells,
                "n_changed": 0,
                "pct_changed": 0.0,
                "n_boundary": 0,
                "n_boundary_changed": 0,
                "n_protected_changed": 0,
            }

        banksy_adata.obs.loc[mask, refined_label_col] = pd.Categorical(refined_labels).astype(str)
        refine_stats[str(sample_id)] = stats

    banksy_adata.obs[refined_label_col] = pd.Categorical(banksy_adata.obs[refined_label_col])
    return banksy_adata, pd.DataFrame(refine_stats).T


def plot_spatial_clusters_per_sample(
    banksy_adata: ad.AnnData,
    *,
    label_col: str = "leiden_harmony_refined",
    palette: str = "tab20",
    figsize: Tuple[int, int] = (6, 6),
    point_size: int = 1,
    alpha: float = 0.4,
) -> None:
    """Plot spatial cluster maps for each sample."""
    all_labels = sorted(banksy_adata.obs[label_col].astype(str).unique())
    label_to_code = {label: i for i, label in enumerate(all_labels)}
    for sample_id in sorted(banksy_adata.obs["sample_id"].astype(str).unique()):
        subset = banksy_adata[banksy_adata.obs["sample_id"].astype(str) == sample_id]
        labels = subset.obs[label_col].astype(str).map(label_to_code)
        plt.figure(figsize=figsize)
        plt.scatter(
            subset.obs["x"],
            subset.obs["y"],
            c=labels,
            s=point_size,
            cmap=palette,
            alpha=alpha,
            vmin=0,
            vmax=max(label_to_code.values()),
        )
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.axis("off")
        plt.title(f"BANKSY clusters | sample_id={sample_id} | label={label_col}")
        plt.show()


def plot_umap_before_after_harmony(
    banksy_adata: ad.AnnData,
    *,
    point_size: int = 3,
    alpha: float = 0.5,
) -> None:
    """Plot UMAPs before and after Harmony, colored by sample ID."""
    sample_ids = banksy_adata.obs["sample_id"].astype(str).to_numpy()
    levels = np.unique(sample_ids)
    colors = sns.color_palette("tab10", n_colors=len(levels))
    color_map = dict(zip(levels, colors))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in [
        (axes[0], "X_umap_pca", "Before Harmony"),
        (axes[1], "X_umap_harmony", "After Harmony"),
    ]:
        embedding = banksy_adata.obsm[key]
        for level in levels:
            mask = sample_ids == level
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=point_size, alpha=alpha, color=color_map[level], label=level)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(title="sample_id", frameon=False, markerscale=3)
    plt.tight_layout()
    plt.show()


def export_cluster_results(
    banksy_adata: ad.AnnData,
    final_output_csv: str | os.PathLike[str],
    *,
    include_refined: bool = True,
) -> pd.DataFrame:
    """Export one joint CSV and one per-sample CSV."""
    export_cols = ["sample_id", "cell_id", "x", "y", "leiden_harmony"]
    if include_refined and "leiden_harmony_refined" in banksy_adata.obs.columns:
        export_cols.append("leiden_harmony_refined")
    final_df = banksy_adata.obs[export_cols].copy().reset_index().rename(columns={"index": "spot_uid"})

    final_output_path = Path(final_output_csv)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(final_output_path, index=False)
    for sample_id in sorted(final_df["sample_id"].astype(str).unique()):
        sample_df = final_df[final_df["sample_id"].astype(str) == sample_id]
        sample_df.to_csv(final_output_path.parent / f"banksy_clusters_{sample_id}.csv", index=False)
    return final_df


def run_joint_clustering_pipeline(
    *,
    samples: Optional[Iterable[Mapping[str, str]]] = None,
    data_dir: Optional[str | os.PathLike[str]] = None,
    adata: Optional[ad.AnnData] = None,
    h5ad_path: Optional[str | os.PathLike[str]] = None,
    output_csv: str | os.PathLike[str],
    output_h5ad: Optional[str | os.PathLike[str]] = None,
    params: Optional[PipelineParams] = None,
    run_refinement: bool = False,
    sample_key: str = "sample_id",
    x_key: str = "x",
    y_key: str = "y",
    cell_id_key: str = "cell_id",
    layer: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete joint clustering pipeline.

    Provide exactly one input source: ``samples``, ``data_dir``, ``adata``, or
    ``h5ad_path``.
    """
    params = params or PipelineParams()
    input_count = sum(source is not None for source in [samples, data_dir, adata, h5ad_path])
    if input_count != 1:
        raise ValueError("Provide exactly one input source: samples, data_dir, adata, or h5ad_path.")

    if data_dir is not None:
        sample_tables = load_csv_sample_tables(discover_sample_csv_pairs(data_dir))
    elif samples is not None:
        sample_tables = load_csv_sample_tables(samples)
    elif adata is not None:
        sample_tables = sample_tables_from_anndata(
            adata,
            sample_key=sample_key,
            x_key=x_key,
            y_key=y_key,
            cell_id_key=cell_id_key,
            layer=layer,
        )
    else:
        sample_tables = load_h5ad_sample_tables(
            h5ad_path,
            sample_key=sample_key,
            x_key=x_key,
            y_key=y_key,
            cell_id_key=cell_id_key,
            layer=layer,
        )

    if len(sample_tables) < 2:
        raise ValueError("Joint clustering requires at least two samples.")

    all_genes, aligned_counts = align_genes_across_samples(sample_tables)
    raw_counts = build_raw_counts(aligned_counts)
    joint_adata = build_joint_adata(sample_tables, raw_counts)
    banksy_results = run_banksy_per_sample(
        joint_adata,
        num_neighbours=params.num_neighbours,
        lambda_list=params.lambda_list,
        nbr_weight_decay=params.decay,
    )
    banksy_adata = integrate_joint_with_harmony(
        banksy_results,
        selected_lambda=params.lambda_list[0],
        decay=params.decay,
        pca_dims=params.pca_dims,
        random_state=params.partition_seed,
        harmony_max_iter=params.harmony_max_iter,
        harmony_theta=params.harmony_theta,
    )
    banksy_adata = compute_pre_post_harmony_umap(
        banksy_adata,
        pca_dim=params.pca_dims[0],
        n_neighbors=params.umap_n_neighbors,
        random_state=params.partition_seed,
    )
    banksy_adata = run_harmony_snn_leiden(
        banksy_adata,
        k=params.snn_k,
        resolution=params.resolutions[0],
        random_state=params.partition_seed,
    )

    refine_stats = None
    if run_refinement:
        banksy_adata, refine_stats = run_boundary_refinement_by_sample(banksy_adata)

    final_df = export_cluster_results(banksy_adata, output_csv, include_refined=run_refinement)
    if output_h5ad is not None:
        Path(output_h5ad).parent.mkdir(parents=True, exist_ok=True)
        banksy_adata.write_h5ad(output_h5ad)

    return {
        "sample_tables": sample_tables,
        "all_genes": all_genes,
        "joint_adata": joint_adata,
        "banksy_results": banksy_results,
        "banksy_adata": banksy_adata,
        "refine_stats": refine_stats,
        "final_df": final_df,
        "output_csv": str(output_csv),
        "output_h5ad": str(output_h5ad) if output_h5ad is not None else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run spAlignDE joint BANKSY clustering.")
    parser.add_argument("--input-mode", choices=["csv", "h5ad"], required=True)
    parser.add_argument("--data-dir", help="CSV folder for --input-mode csv.")
    parser.add_argument("--h5ad", help="Combined AnnData file for --input-mode h5ad.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-prefix", default="banksy_clusters_joint")
    parser.add_argument("--resolution", type=float, default=1.4)
    parser.add_argument("--lambda-value", type=float, default=0.8)
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--num-neighbours", type=int, default=30)
    parser.add_argument("--snn-k", type=int, default=50)
    parser.add_argument("--refine", action="store_true", help="Run optional boundary-aware label refinement.")
    parser.add_argument("--write-h5ad", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Command-line entry point."""
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    output_csv = out_dir / f"{args.output_prefix}.csv"
    output_h5ad = out_dir / f"{args.output_prefix}.h5ad" if args.write_h5ad else None
    params = PipelineParams(
        num_neighbours=args.num_neighbours,
        lambda_list=(args.lambda_value,),
        pca_dims=(args.pca_dim,),
        resolutions=(args.resolution,),
        snn_k=args.snn_k,
    )

    if args.input_mode == "csv":
        if not args.data_dir:
            raise ValueError("--data-dir is required for --input-mode csv.")
        result = run_joint_clustering_pipeline(
            data_dir=args.data_dir,
            output_csv=output_csv,
            output_h5ad=output_h5ad,
            params=params,
            run_refinement=args.refine,
        )
    else:
        if not args.h5ad:
            raise ValueError("--h5ad is required for --input-mode h5ad.")
        result = run_joint_clustering_pipeline(
            h5ad_path=args.h5ad,
            output_csv=output_csv,
            output_h5ad=output_h5ad,
            params=params,
            run_refinement=args.refine,
        )

    print(f"Saved joint clustering CSV: {output_csv}")
    if output_h5ad is not None:
        print(f"Saved clustered AnnData: {output_h5ad}")
    for sample_id in sorted(result["final_df"]["sample_id"].astype(str).unique()):
        print(f"Saved per-sample CSV: {out_dir / f'banksy_clusters_{sample_id}.csv'}")
    return result


if __name__ == "__main__":
    main()
