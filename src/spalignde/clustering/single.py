"""Public AnnData interface for single-sample spatial clustering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from ..io import spatial_coordinates, validate_single_sample_anndata


@dataclass(frozen=True)
class SingleClusteringConfig:
    """Parameters for single-sample BANKSY clustering and refinement.

    Increasing ``banksy_lambda`` gives spatial-neighborhood information more
    influence; increasing ``resolution`` usually produces more clusters.
    Boundary-refinement neighborhoods and support thresholds should be tuned
    only after inspecting raw spatial labels, especially for thin structures.
    """

    num_neighbors: int = 30
    banksy_lambda: float = 0.8
    resolution: float = 1.2
    pca_dim: int = 20
    max_m: int = 1
    decay: str = "scaled_gaussian"
    random_state: int = 1234
    refine_boundaries: bool = True
    k_interior: int = 150
    k_boundary: int = 8
    boundary_distance_pixels: int = 3
    protected_labels: tuple[str, ...] = ("L1", "Layer1", "layer1", "1")
    min_same_fraction: float = 0.8
    min_new_fraction_interior: float = 0.2
    min_new_fraction_boundary: float = 0.8
    min_new_fraction_protected: float = 0.9


def _select_banksy_labels(
    results: pd.DataFrame,
    *,
    banksy_lambda: float,
    n_obs: int,
) -> tuple[pd.Categorical, Any]:
    if results.empty:
        raise RuntimeError("BANKSY returned no clustering results")
    if "lambda_param" not in results:
        raise RuntimeError("BANKSY results do not contain 'lambda_param'")
    selected = results[np.isclose(results["lambda_param"], banksy_lambda)]
    if selected.empty:
        available = sorted(results["lambda_param"].dropna().unique().tolist())
        raise ValueError(
            f"No BANKSY result found for lambda={banksy_lambda}; "
            f"available values: {available}"
        )

    selected_index = selected.index[0]
    labels = results.loc[selected_index, "labels"]
    if hasattr(labels, "dense"):
        labels = labels.dense
    labels = np.asarray(labels).reshape(-1)
    if labels.size != n_obs:
        raise RuntimeError(
            f"BANKSY returned {labels.size} labels for {n_obs} observations"
        )
    return pd.Categorical(labels.astype(str)), selected_index


def cluster_single(
    adata: ad.AnnData,
    *,
    config: SingleClusteringConfig | None = None,
    spatial_key: str = "spatial",
    cluster_key: str = "cluster",
    copy: bool = True,
    banksy_output_dir: str | Path | None = None,
    return_details: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, dict[str, Any]]:
    """Cluster one spatial sample and add labels to ``adata.obs``.

    The input expression matrix and canonical coordinates are preserved. Raw
    BANKSY labels are stored in ``<cluster_key>_raw``. When boundary-aware
    refinement is enabled, refined labels are stored in
    ``<cluster_key>_refined`` and selected as the final ``cluster_key``.
    """
    config = config or SingleClusteringConfig()
    validate_single_sample_anndata(adata, spatial_key=spatial_key)
    if config.num_neighbors < 1:
        raise ValueError("num_neighbors must be positive")
    if config.pca_dim < 1:
        raise ValueError("pca_dim must be positive")
    if adata.n_obs <= 50:
        raise ValueError(
            "Single-sample BANKSY requires more than 50 observations for its "
            "Leiden nearest-neighbor graph"
        )
    if adata.n_vars < config.pca_dim:
        raise ValueError(
            f"pca_dim={config.pca_dim} exceeds the {adata.n_vars} input genes"
        )
    if config.max_m < 0:
        raise ValueError("max_m must be non-negative")
    if config.k_interior < 1 or config.k_boundary < 1:
        raise ValueError("k_interior and k_boundary must be positive")

    source = adata.to_memory() if adata.isbacked else adata
    work = source.copy()
    xy = spatial_coordinates(work, spatial_key=spatial_key)
    work.obs["x"] = xy[:, 0]
    work.obs["y"] = xy[:, 1]

    try:
        from banksy.initialize_banksy import initialize_banksy
        from banksy.run_banksy import run_banksy_multiparam
    except ImportError as error:
        raise ImportError(
            "Single clustering requires the optional clustering dependencies. "
            "Install spAlignDE with the 'clustering' extra."
        ) from error

    def run_banksy(output_dir: Path) -> pd.DataFrame:
        output_dir.mkdir(parents=True, exist_ok=True)
        banksy_dict = initialize_banksy(
            work,
            coord_keys=("x", "y", spatial_key),
            num_neighbours=config.num_neighbors,
            nbr_weight_decay=config.decay,
            max_m=config.max_m,
            plt_edge_hist=False,
            plt_nbr_weights=False,
            plt_agf_angles=False,
            plt_theta=False,
        )
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("turbo")
        color_list = [cmap(value)[:3] for value in np.linspace(0.02, 0.98, 256)]
        return run_banksy_multiparam(
            work,
            banksy_dict,
            lambda_list=[config.banksy_lambda],
            resolutions=[config.resolution],
            color_list=color_list,
            max_m=config.max_m,
            filepath=str(output_dir),
            key=("x", "y"),
            annotation_key=None,
            pca_dims=[config.pca_dim],
            partition_seed=config.random_state,
        )

    if banksy_output_dir is None:
        with TemporaryDirectory(prefix="spAlignDE_banksy_") as temporary_dir:
            results = run_banksy(Path(temporary_dir))
    else:
        results = run_banksy(Path(banksy_output_dir).expanduser())

    raw_labels, selected_parameters = _select_banksy_labels(
        results,
        banksy_lambda=config.banksy_lambda,
        n_obs=work.n_obs,
    )
    raw_cluster_key = f"{cluster_key}_raw"
    refined_cluster_key = f"{cluster_key}_refined"
    work.obs[raw_cluster_key] = raw_labels

    refinement_stats = None
    final_labels = raw_labels
    if config.refine_boundaries:
        from ._joint_core import refine_labels_boundary_aware

        refined, refinement_stats = refine_labels_boundary_aware(
            work,
            raw_label_col=raw_cluster_key,
            x_col="x",
            y_col="y",
            k_interior=min(config.k_interior, max(work.n_obs - 1, 1)),
            k_boundary=min(config.k_boundary, max(work.n_obs - 1, 1)),
            boundary_dist_px=config.boundary_distance_pixels,
            protected_labels=config.protected_labels,
            min_same_frac_keep=config.min_same_fraction,
            min_new_frac_interior=config.min_new_fraction_interior,
            min_new_frac_boundary=config.min_new_fraction_boundary,
            min_new_frac_protected=config.min_new_fraction_protected,
            only_change_if_disagree=True,
        )
        final_labels = pd.Categorical(refined)
        work.obs[refined_cluster_key] = final_labels
    work.obs[cluster_key] = final_labels

    output = source.copy() if copy else source
    output.obs[raw_cluster_key] = work.obs[raw_cluster_key].copy()
    if config.refine_boundaries:
        output.obs[refined_cluster_key] = work.obs[refined_cluster_key].copy()
    output.obs[cluster_key] = work.obs[cluster_key].copy()
    output.uns.pop("spalignde", None)
    output.uns.setdefault("spAlignDE", {})
    config_metadata = asdict(config)
    config_metadata["protected_labels"] = list(config.protected_labels)
    output.uns["spAlignDE"]["single_clustering"] = {
        **config_metadata,
        "spatial_key": spatial_key,
        "cluster_key": cluster_key,
        "raw_cluster_key": raw_cluster_key,
        "refined_cluster_key": (
            refined_cluster_key if config.refine_boundaries else None
        ),
        "n_clusters": int(output.obs[cluster_key].nunique()),
        "selected_parameters": str(selected_parameters),
    }

    details = {
        "banksy_results": results,
        "refinement_stats": refinement_stats,
        "selected_parameters": selected_parameters,
        "config": config,
    }
    return (output, details) if return_details else output


def plot_single_cluster_refinement(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    raw_cluster_key: str = "cluster_raw",
    refined_cluster_key: str = "cluster_refined",
    point_size: float = 0.5,
    alpha: float = 0.85,
    palette: str = "turbo",
    invert_y: bool = True,
    figsize: tuple[float, float] = (11.0, 5.0),
) -> tuple[Any, Any]:
    """Plot raw and boundary-refined labels using one shared color mapping."""
    import matplotlib.pyplot as plt

    missing = [
        key
        for key in (raw_cluster_key, refined_cluster_key)
        if key not in adata.obs
    ]
    if missing:
        raise KeyError(f"Missing required adata.obs columns: {missing}")
    validate_single_sample_anndata(adata, spatial_key=spatial_key)
    xy = np.asarray(adata.obsm[spatial_key])

    all_labels = pd.concat(
        [
            adata.obs[raw_cluster_key].astype(str),
            adata.obs[refined_cluster_key].astype(str),
        ],
        ignore_index=True,
    ).unique()

    def sort_key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    label_order = sorted(all_labels, key=sort_key)
    colors = plt.get_cmap(palette)(
        np.linspace(0.02, 0.98, max(len(label_order), 2))
    )
    color_map = dict(zip(label_order, colors, strict=False))

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for axis, key, title in zip(
        axes,
        (raw_cluster_key, refined_cluster_key),
        ("Raw BANKSY clusters", "Boundary-refined clusters"),
        strict=False,
    ):
        point_colors = np.asarray(
            [color_map[label] for label in adata.obs[key].astype(str)]
        )
        axis.scatter(
            xy[:, 0],
            xy[:, 1],
            c=point_colors,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.axis("off")
        if invert_y:
            axis.invert_yaxis()
    return fig, axes


__all__ = [
    "SingleClusteringConfig",
    "cluster_single",
    "plot_single_cluster_refinement",
]
