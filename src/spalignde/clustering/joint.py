"""Public AnnData interface for joint spatial clustering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from ..io import spatial_coordinates, validate_cross_sample_anndata
from ..random import set_random_seed


@dataclass(frozen=True)
class JointClusteringConfig:
    """Parameters for joint BANKSY/Harmony/Leiden structure discovery.

    Increasing ``banksy_lambda`` gives spatial-neighborhood information more
    influence; increasing ``resolution`` usually produces more clusters.
    ``harmony_theta`` controls sample correction and ``snn_neighbors`` controls
    shared-graph connectivity. Select settings from spatial coherence and
    cross-sample coverage, not cluster integer values.
    """

    num_neighbors: int = 30
    banksy_lambda: float = 0.8
    pca_dim: int = 20
    resolution: float = 1.4
    snn_neighbors: int = 50
    harmony_theta: float = 2.0
    harmony_max_iter: int = 30
    random_state: int = 1000
    leiden_flavor: str = "leidenalg"
    leiden_n_iterations: int = -1
    decay: str = "scaled_gaussian"
    refine_boundaries: bool = True
    compute_umap: bool = False


def _observation_keys(
    adata: ad.AnnData,
    *,
    sample_key: str,
    cell_id_key: str,
) -> pd.MultiIndex:
    if cell_id_key in adata.obs:
        cell_ids = adata.obs[cell_id_key].astype(str)
    else:
        cell_ids = pd.Series(
            adata.obs_names.astype(str),
            index=adata.obs_names,
            dtype="object",
        )
    return pd.MultiIndex.from_arrays(
        [adata.obs[sample_key].astype(str), cell_ids],
        names=["sample_id", "cell_id"],
    )


def _map_result_labels(
    banksy_adata: ad.AnnData,
    *,
    result_key: str,
    input_keys: pd.MultiIndex,
) -> pd.Categorical:
    result_keys = pd.MultiIndex.from_arrays(
        [
            banksy_adata.obs["sample_id"].astype(str),
            banksy_adata.obs["cell_id"].astype(str),
        ],
        names=["sample_id", "cell_id"],
    )
    labels = pd.Series(
        banksy_adata.obs[result_key].astype(str).to_numpy(),
        index=result_keys,
    )
    mapped = labels.reindex(input_keys)
    if mapped.isna().any():
        n_missing = int(mapped.isna().sum())
        raise RuntimeError(
            f"Could not map {n_missing} clustered observations back to the input AnnData"
        )
    return pd.Categorical(mapped.to_numpy())


def cluster_joint(
    adata: ad.AnnData,
    *,
    config: JointClusteringConfig | None = None,
    sample_key: str = "sample_id",
    spatial_key: str = "spatial",
    cluster_key: str = "cluster",
    cell_id_key: str = "cell_id",
    layer: str | None = None,
    copy: bool = True,
    return_details: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, dict[str, Any]]:
    """Identify shared spatial structures and add them to ``adata.obs``.

    BANKSY features are calculated separately within each sample. The resulting
    representations are concatenated, reduced by joint PCA, corrected with
    Harmony using the sample identifier and clustered with Leiden on a
    shared-nearest-neighbor graph. Optional boundary-aware refinement is applied
    within each sample.

    The returned AnnData preserves the input expression matrix and original
    ``adata.obsm[spatial_key]``. Raw Leiden labels are written to
    ``adata.obs[f"{cluster_key}_raw"]``. When boundary refinement is enabled,
    refined labels are also written to
    ``adata.obs[f"{cluster_key}_refined"]``. The selected final labels are
    always written to ``adata.obs[cluster_key]``.
    """
    config = config or JointClusteringConfig()
    validate_cross_sample_anndata(
        adata,
        sample_key=sample_key,
        spatial_key=spatial_key,
        cluster_key=cluster_key,
        require_cluster=False,
    )
    seed_controls = set_random_seed(config.random_state)

    from . import _joint_core as core

    source = adata.to_memory() if adata.isbacked else adata
    work = source.copy()
    xy = spatial_coordinates(work, spatial_key=spatial_key)
    # The public AnnData contract stores the authoritative coordinates in
    # ``obsm[spatial_key]``.  Overwrite (rather than append and rename)
    # conventional ``obs['x']``/``obs['y']`` columns so inputs that already
    # contain them cannot create duplicate column names inside AnnData.
    work.obs["x"] = xy[:, 0]
    work.obs["y"] = xy[:, 1]
    if cell_id_key not in work.obs:
        work.obs[cell_id_key] = work.obs_names.astype(str)

    sample_tables = core.sample_tables_from_anndata(
        work,
        sample_key=sample_key,
        x_key="x",
        y_key="y",
        cell_id_key=cell_id_key,
        layer=layer,
    )
    _, aligned_counts = core.align_genes_across_samples(sample_tables)
    joint_adata = core.build_joint_adata(
        sample_tables,
        core.build_raw_counts(aligned_counts),
    )
    banksy_results = core.run_banksy_per_sample(
        joint_adata,
        num_neighbours=config.num_neighbors,
        lambda_list=(config.banksy_lambda,),
        nbr_weight_decay=config.decay,
    )
    banksy_adata = core.integrate_joint_with_harmony(
        banksy_results,
        selected_lambda=config.banksy_lambda,
        decay=config.decay,
        pca_dims=(config.pca_dim,),
        random_state=config.random_state,
        harmony_max_iter=config.harmony_max_iter,
        harmony_theta=config.harmony_theta,
    )
    if config.compute_umap:
        banksy_adata = core.compute_pre_post_harmony_umap(
            banksy_adata,
            pca_dim=config.pca_dim,
            n_neighbors=config.num_neighbors,
            random_state=config.random_state,
        )
    banksy_adata = core.run_harmony_snn_leiden(
        banksy_adata,
        k=config.snn_neighbors,
        resolution=config.resolution,
        random_state=config.random_state,
        flavor=config.leiden_flavor,
        n_iterations=config.leiden_n_iterations,
        key_added="_spalignde_cluster_raw",
    )

    refine_stats = None
    selected_key = "_spalignde_cluster_raw"
    if config.refine_boundaries:
        banksy_adata, refine_stats = core.run_boundary_refinement_by_sample(
            banksy_adata,
            raw_label_col="_spalignde_cluster_raw",
            refined_label_col="_spalignde_cluster_refined",
        )
        selected_key = "_spalignde_cluster_refined"

    input_keys = _observation_keys(
        source,
        sample_key=sample_key,
        cell_id_key=cell_id_key,
    )
    raw_cluster_key = f"{cluster_key}_raw"
    refined_cluster_key = f"{cluster_key}_refined"
    raw_labels = _map_result_labels(
        banksy_adata,
        result_key="_spalignde_cluster_raw",
        input_keys=input_keys,
    )
    final_labels = raw_labels
    if config.refine_boundaries:
        final_labels = _map_result_labels(
            banksy_adata,
            result_key="_spalignde_cluster_refined",
            input_keys=input_keys,
        )

    output = source.copy() if copy else source
    output.obs[raw_cluster_key] = raw_labels
    if config.refine_boundaries:
        output.obs[refined_cluster_key] = final_labels
    output.obs[cluster_key] = final_labels
    output.uns.pop("spalignde", None)
    output.uns.setdefault("spAlignDE", {})
    output.uns["spAlignDE"]["joint_clustering"] = {
        **asdict(config),
        "sample_key": sample_key,
        "spatial_key": spatial_key,
        "cluster_key": cluster_key,
        "raw_cluster_key": raw_cluster_key,
        "refined_cluster_key": (
            refined_cluster_key if config.refine_boundaries else None
        ),
        "n_clusters": int(output.obs[cluster_key].nunique()),
        "seed_controls": seed_controls,
    }

    details = {
        "banksy_adata": banksy_adata,
        "refinement_stats": refine_stats,
        "config": config,
    }
    return (output, details) if return_details else output


def plot_joint_cluster_refinement(
    adata: ad.AnnData,
    *,
    sample_key: str = "sample_id",
    spatial_key: str = "spatial",
    raw_cluster_key: str = "cluster_raw",
    refined_cluster_key: str = "cluster_refined",
    samples: list[str] | tuple[str, ...] | None = None,
    point_size: float = 0.5,
    alpha: float = 0.8,
    palette: str = "turbo",
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, Any]:
    """Plot raw and boundary-refined joint clusters for every sample.

    Cluster identities use one shared color mapping across all panels, so the
    same label has the same color in both samples and both refinement stages.
    """
    import matplotlib.pyplot as plt

    required_obs = (sample_key, raw_cluster_key, refined_cluster_key)
    missing = [key for key in required_obs if key not in adata.obs]
    if missing:
        raise KeyError(f"Missing required adata.obs columns: {missing}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{spatial_key!r}]")

    sample_values = adata.obs[sample_key].astype(str)
    if samples is None:
        samples = tuple(pd.unique(sample_values))
    else:
        samples = tuple(str(sample) for sample in samples)
    if not samples:
        raise ValueError("At least one sample is required")

    label_columns = (raw_cluster_key, refined_cluster_key)
    all_labels = pd.concat(
        [adata.obs[key].astype(str) for key in label_columns],
        ignore_index=True,
    ).unique()

    def label_sort_key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    label_order = sorted(all_labels, key=label_sort_key)
    color_values = plt.get_cmap(palette)(
        np.linspace(0.02, 0.98, max(len(label_order), 2))
    )
    label_to_color = dict(zip(label_order, color_values, strict=False))

    if figsize is None:
        figsize = (5.5 * len(samples), 9.5)
    fig, axes = plt.subplots(
        2,
        len(samples),
        figsize=figsize,
        squeeze=False,
    )
    xy = np.asarray(adata.obsm[spatial_key])
    row_titles = ("Raw joint clusters", "Boundary-refined clusters")

    for row, (label_key, row_title) in enumerate(zip(label_columns, row_titles)):
        labels = adata.obs[label_key].astype(str)
        colors = np.asarray([label_to_color[label] for label in labels])
        for column, sample in enumerate(samples):
            axis = axes[row, column]
            mask = sample_values.eq(sample).to_numpy()
            if not mask.any():
                raise ValueError(f"Sample {sample!r} is absent from adata.obs")
            axis.scatter(
                xy[mask, 0],
                xy[mask, 1],
                c=colors[mask],
                s=point_size,
                alpha=alpha,
                linewidths=0,
                rasterized=True,
            )
            axis.set_title(f"{sample}: {row_title}")
            axis.set_aspect("equal")
            axis.axis("off")

    fig.tight_layout()
    return fig, axes
