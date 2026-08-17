"""AnnData-native cross-sample spatial transcriptomics alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..io import (
    initialize_output_coordinates,
    spatial_coordinates,
    validate_cross_sample_anndata,
    validate_sample_selection,
)
from . import _prealignment_core as pre_core
from . import _slddmm_core as lddmm_core


_METADATA_KEY = "spAlignDE"


def _metadata(adata: ad.AnnData, *, create: bool = False) -> dict[str, Any]:
    """Return the spAlignDE metadata namespace."""
    if create:
        adata.uns.pop("spalignde", None)
        return adata.uns.setdefault(_METADATA_KEY, {})
    return adata.uns[_METADATA_KEY]


@dataclass(frozen=True)
class PrealignmentConfig:
    """Weighted shared-cluster-centroid similarity initialization."""

    allow_scaling: bool = True
    allow_reflection: bool = False
    use_cluster_size_weights: bool = True
    min_cluster_size: int = 10


@dataclass(frozen=True)
class ManualPrealignmentConfig:
    """Explicit source-to-reference similarity transformation.

    Coordinates follow the row-vector convention
    ``aligned = scale * source @ R.T + t``.
    """

    scale: float = 1.0
    theta_deg: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0


@dataclass(frozen=True)
class RasterizationConfig:
    """Cross-sample continuous-field construction.

    ``grid_spacing`` is expressed in the current coordinate units;
    ``blur_sigma`` is expressed in raster pixels. ``cluster_weight`` and
    ``density_weight`` control the relative matching evidence rather than the
    numeric range of the original expression matrix.
    """

    grid_spacing: float = 30.0
    grid_expand: float = 1.05
    blur_sigma: float = 1.0
    cluster_weight: float = 1.0
    density_weight: float = 1.0


@dataclass(frozen=True)
class SLDDMMConfig:
    """Shooting-based LDDMM model and optimization settings.

    ``kernel_scale`` is the legacy parameter ``a``: larger values produce a
    smoother, more global deformation. ``velocity_grid_spacing`` is legacy
    ``grid_step``: smaller values create a denser and more expensive velocity
    field. ``time_steps`` controls flow-integration accuracy, while
    ``iterations`` and ``momentum_lr`` control optimizer duration and step
    size. Distance-valued settings use the coordinates supplied to the
    alignment and must be reconsidered after coordinate rescaling.

    ``sigma_regularization`` appears in the denominator of the deformation
    penalty. Larger values therefore weaken regularization and allow stronger
    warps; smaller values strengthen regularization.

    ``restore_best_checkpoint`` restores the lowest-energy optimizer state at
    the end of a run. It defaults to ``False`` because EM intensity updates can
    change the objective scale; set it to ``True`` only when checkpoint
    restoration is explicitly desired.
    """

    kernel_scale: float = 300.0
    kernel_power: float = 2.0
    velocity_expand: float = 2.0
    time_steps: int = 3
    velocity_grid_spacing: float = 100.0
    iterations: int = 500
    diffeomorphic_start: int = 0
    affine_linear_lr: float = 2e-8
    affine_translation_lr: float = 2e-1
    momentum_lr: float = 2e3
    minimum_momentum_lr: float = 2e3
    affine_slowdown: float = 10.0
    momentum_gradient_clip: float | None = None
    momentum_lr_decay: float = 1.0
    restore_best_checkpoint: bool = False
    em_update_every: int = 5
    em_start: int = 50
    sigma_match: float = 1.0
    sigma_unmatched_a: float = 5.0
    sigma_unmatched_b: float = 2.0
    sigma_regularization: float = 5e5
    dtype: str = "float32"


@dataclass
class PrealignmentResult:
    """Output of shared-structure-centroid pre-alignment."""

    adata: ad.AnnData
    query_sample: str
    reference_sample: str
    params: dict[str, Any]
    centroid_matches: pd.DataFrame
    query_centroids: np.ndarray
    reference_centroids: np.ndarray
    transformed_query_centroids: np.ndarray


@dataclass
class CrossSampleFields:
    """Continuous cluster-composition and density fields on a shared grid."""

    query_sample: str
    reference_sample: str
    shared_clusters: list[str]
    grid_x: np.ndarray
    grid_y: np.ndarray
    query_raw_raster: np.ndarray
    reference_raw_raster: np.ndarray
    query_image: np.ndarray
    reference_image: np.ndarray
    query_composition: np.ndarray
    reference_composition: np.ndarray
    query_density: np.ndarray
    reference_density: np.ndarray
    channel_weights: np.ndarray
    query_preview: np.ndarray
    reference_preview: np.ndarray


@dataclass
class CrossSampleAlignmentResult:
    """Final aligned AnnData plus diagnostics and the learned transformation."""

    adata: ad.AnnData
    query_sample: str
    reference_sample: str
    prealignment: PrealignmentResult | None
    fields: CrossSampleFields
    transform: dict[str, Any]
    metrics: dict[str, Any]
    cluster_performance: pd.DataFrame


def _manual_transform(
    config: ManualPrealignmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(
        [
            config.scale,
            config.theta_deg,
            config.translation_x,
            config.translation_y,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("Manual pre-alignment parameters must be finite")
    if config.scale <= 0:
        raise ValueError("ManualPrealignmentConfig.scale must be positive")

    theta = np.deg2rad(float(config.theta_deg))
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    translation = np.array(
        [config.translation_x, config.translation_y],
        dtype=float,
    )
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = float(config.scale) * rotation
    matrix[:2, 2] = translation
    return rotation, translation, matrix


def apply_similarity_transform(
    points: np.ndarray,
    config: ManualPrealignmentConfig,
) -> np.ndarray:
    """Apply an explicit manual similarity transform to ``(x, y)`` points."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2)")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")
    rotation, translation, _ = _manual_transform(config)
    return float(config.scale) * points @ rotation.T + translation


def _pair_tables(
    adata: ad.AnnData,
    *,
    query_sample: str,
    reference_sample: str,
    sample_key: str,
    cluster_key: str,
    spatial_key: str,
    use_prealigned_query: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    sample = adata.obs[sample_key].astype(str)
    query_mask = (sample == str(query_sample)).to_numpy()
    reference_mask = (sample == str(reference_sample)).to_numpy()
    raw_xy = spatial_coordinates(adata, spatial_key=spatial_key)

    query = adata.obs.loc[query_mask].copy()
    reference = adata.obs.loc[reference_mask].copy()
    query[cluster_key] = query[cluster_key].astype(str)
    reference[cluster_key] = reference[cluster_key].astype(str)

    if use_prealigned_query:
        required = {"x_prealigned", "y_prealigned"}
        missing = required.difference(adata.obs.columns)
        if missing:
            raise ValueError(
                "Pre-aligned coordinates are missing. Run prealign_cross_sample first."
            )
        query_xy = adata.obs.loc[
            query_mask, ["x_prealigned", "y_prealigned"]
        ].to_numpy(dtype=float)
    else:
        query_xy = raw_xy[query_mask]
    reference_xy = raw_xy[reference_mask]

    query["x"], query["y"] = query_xy[:, 0], query_xy[:, 1]
    reference["x"], reference["y"] = reference_xy[:, 0], reference_xy[:, 1]
    return query, reference, query_xy, reference_xy


def prealign_cross_sample(
    adata: ad.AnnData,
    *,
    query_sample: str,
    reference_sample: str,
    config: PrealignmentConfig | None = None,
    sample_key: str = "sample_id",
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    copy: bool = True,
    verbose: bool = True,
) -> PrealignmentResult:
    """Estimate and apply query-to-reference global similarity pre-alignment."""
    config = config or PrealignmentConfig()
    validate_cross_sample_anndata(
        adata,
        sample_key=sample_key,
        spatial_key=spatial_key,
        cluster_key=cluster_key,
    )
    validate_sample_selection(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
    )

    query, reference, query_xy, _ = _pair_tables(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        use_prealigned_query=False,
    )
    (
        x_pre,
        y_pre,
        params,
        centroid_matches,
        query_centroids,
        reference_centroids,
        transformed_query_centroids,
    ) = pre_core.run_cluster_correspondence_prealign(
        query,
        reference,
        query_xy[:, 0],
        query_xy[:, 1],
        source_sample_id=str(query_sample),
        target_sample_id=str(reference_sample),
        cluster_col=cluster_key,
        allow_scaling=config.allow_scaling,
        allow_reflection=config.allow_reflection,
        use_cluster_size_weights=config.use_cluster_size_weights,
        min_cluster_size=config.min_cluster_size,
        out_dir=None,
        verbose=verbose,
    )

    output = adata.copy() if copy else adata
    initialize_output_coordinates(output, spatial_key=spatial_key)
    query_mask = (
        output.obs[sample_key].astype(str) == str(query_sample)
    ).to_numpy()
    output.obs.loc[query_mask, "x_prealigned"] = np.asarray(x_pre)
    output.obs.loc[query_mask, "y_prealigned"] = np.asarray(y_pre)
    output.obs.loc[query_mask, "x_aligned"] = np.asarray(x_pre)
    output.obs.loc[query_mask, "y_aligned"] = np.asarray(y_pre)

    _metadata(output, create=True)["prealignment"] = {
        **params,
        "sample_key": sample_key,
        "cluster_key": cluster_key,
        "spatial_key": spatial_key,
    }
    return PrealignmentResult(
        adata=output,
        query_sample=str(query_sample),
        reference_sample=str(reference_sample),
        params=params,
        centroid_matches=centroid_matches,
        query_centroids=query_centroids,
        reference_centroids=reference_centroids,
        transformed_query_centroids=transformed_query_centroids,
    )


def prealign_cross_sample_manual(
    adata: ad.AnnData,
    *,
    query_sample: str,
    reference_sample: str,
    config: ManualPrealignmentConfig,
    sample_key: str = "sample_id",
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    copy: bool = True,
    verbose: bool = True,
) -> PrealignmentResult:
    """Apply a user-specified query-to-reference similarity pre-alignment."""
    validate_cross_sample_anndata(
        adata,
        sample_key=sample_key,
        spatial_key=spatial_key,
        cluster_key=cluster_key,
    )
    validate_sample_selection(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
    )
    query, reference, query_xy, _ = _pair_tables(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        use_prealigned_query=False,
    )
    transformed_query = apply_similarity_transform(query_xy, config)
    rotation, translation, matrix = _manual_transform(config)

    query_centroid_table = pre_core.compute_cluster_centroids(
        query,
        cluster_key,
    )
    reference_centroid_table = pre_core.compute_cluster_centroids(
        reference,
        cluster_key,
    )
    shared = query_centroid_table.index.intersection(
        reference_centroid_table.index
    )
    query_centroids = query_centroid_table.loc[
        shared, ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)
    reference_centroids = reference_centroid_table.loc[
        shared, ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)
    transformed_query_centroids = apply_similarity_transform(
        query_centroids,
        config,
    )
    residuals = np.linalg.norm(
        transformed_query_centroids - reference_centroids,
        axis=1,
    )
    query_sizes = query_centroid_table.loc[shared, "n_cells"].to_numpy(
        dtype=float
    )
    reference_sizes = reference_centroid_table.loc[
        shared, "n_cells"
    ].to_numpy(dtype=float)
    weights = np.minimum(query_sizes, reference_sizes)
    if len(shared):
        weighted_rmse = float(
            np.sqrt(np.average(residuals**2, weights=weights))
        )
        unweighted_rmse = float(np.sqrt(np.mean(residuals**2)))
    else:
        weighted_rmse = float("nan")
        unweighted_rmse = float("nan")

    centroid_matches = pd.DataFrame(
        {
            "cluster": shared.astype(str),
            "source_centroid_x": query_centroids[:, 0],
            "source_centroid_y": query_centroids[:, 1],
            "target_centroid_x": reference_centroids[:, 0],
            "target_centroid_y": reference_centroids[:, 1],
            "source_n_cells": query_sizes.astype(int),
            "target_n_cells": reference_sizes.astype(int),
            "weight": weights,
            "source_aligned_centroid_x": transformed_query_centroids[:, 0],
            "source_aligned_centroid_y": transformed_query_centroids[:, 1],
            "residual_distance": residuals,
        }
    )
    params = {
        "prealign_method": "manual_similarity",
        "transform_convention": "aligned = scale * source @ R.T + t",
        "matrix_convention": "[x_new, y_new, 1]^T = matrix @ [x, y, 1]^T",
        "source_sample": str(query_sample),
        "target_sample": str(reference_sample),
        "cluster_col": cluster_key,
        "scale": float(config.scale),
        "theta_deg": float(config.theta_deg),
        "translation_x": float(config.translation_x),
        "translation_y": float(config.translation_y),
        "det_R": float(np.linalg.det(rotation)),
        "R": rotation.tolist(),
        "t": translation.tolist(),
        "matrix": matrix.tolist(),
        "n_shared_clusters": int(len(shared)),
        "weighted_centroid_rmse": weighted_rmse,
        "unweighted_centroid_rmse": unweighted_rmse,
    }

    output = adata.copy() if copy else adata
    initialize_output_coordinates(output, spatial_key=spatial_key)
    query_mask = (
        output.obs[sample_key].astype(str) == str(query_sample)
    ).to_numpy()
    output.obs.loc[query_mask, "x_prealigned"] = transformed_query[:, 0]
    output.obs.loc[query_mask, "y_prealigned"] = transformed_query[:, 1]
    output.obs.loc[query_mask, "x_aligned"] = transformed_query[:, 0]
    output.obs.loc[query_mask, "y_aligned"] = transformed_query[:, 1]
    _metadata(output, create=True)["prealignment"] = {
        **params,
        "sample_key": sample_key,
        "cluster_key": cluster_key,
        "spatial_key": spatial_key,
    }
    if verbose:
        print(
            "Using manual similarity pre-alignment: "
            f"scale={config.scale:.6g}, theta={config.theta_deg:.3f}, "
            f"tx={config.translation_x:.3f}, ty={config.translation_y:.3f}"
        )
    return PrealignmentResult(
        adata=output,
        query_sample=str(query_sample),
        reference_sample=str(reference_sample),
        params=params,
        centroid_matches=centroid_matches,
        query_centroids=query_centroids,
        reference_centroids=reference_centroids,
        transformed_query_centroids=transformed_query_centroids,
    )


def _natural_cluster_order(values: set[str]) -> list[str]:
    def key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted(values, key=key)


def rasterize_cross_sample(
    adata: ad.AnnData,
    *,
    query_sample: str,
    reference_sample: str,
    config: RasterizationConfig | None = None,
    sample_key: str = "sample_id",
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
) -> CrossSampleFields:
    """Construct shared-cluster composition and tissue-density fields."""
    config = config or RasterizationConfig()
    validate_cross_sample_anndata(
        adata,
        sample_key=sample_key,
        spatial_key=spatial_key,
        cluster_key=cluster_key,
    )
    validate_sample_selection(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
    )
    query, reference, query_xy, reference_xy = _pair_tables(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        use_prealigned_query=True,
    )

    query_labels = query[cluster_key].astype(str).to_numpy()
    reference_labels = reference[cluster_key].astype(str).to_numpy()
    shared_clusters = _natural_cluster_order(
        set(query_labels).intersection(reference_labels)
    )
    if not shared_clusters:
        raise ValueError("Query and reference have no shared cluster labels")

    (
        grid_x,
        grid_y,
        _,
        _,
        query_raw,
        reference_raw,
        query_image,
        reference_image,
        query_comp,
        reference_comp,
        query_density,
        reference_density,
        channel_weights,
        query_preview,
        reference_preview,
    ) = lddmm_core.prepare_rasterization_and_multichannel(
        query_xy[:, 0],
        query_xy[:, 1],
        reference_xy[:, 0],
        reference_xy[:, 1],
        query_labels,
        reference_labels,
        shared_clusters,
        dx=config.grid_spacing,
        expand=config.grid_expand,
        blur_sigma=config.blur_sigma,
        cluster_w=config.cluster_weight,
        density_w=config.density_weight,
    )
    return CrossSampleFields(
        query_sample=str(query_sample),
        reference_sample=str(reference_sample),
        shared_clusters=shared_clusters,
        grid_x=grid_x,
        grid_y=grid_y,
        query_raw_raster=query_raw,
        reference_raw_raster=reference_raw,
        query_image=query_image,
        reference_image=reference_image,
        query_composition=query_comp,
        reference_composition=reference_comp,
        query_density=query_density,
        reference_density=reference_density,
        channel_weights=channel_weights,
        query_preview=query_preview,
        reference_preview=reference_preview,
    )


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float32", "torch.float32", "single"}:
        return torch.float32
    if normalized in {"float64", "torch.float64", "double"}:
        return torch.float64
    raise ValueError("SLDDMMConfig.dtype must be 'float32' or 'float64'")


def run_slddmm_alignment(
    adata: ad.AnnData,
    fields: CrossSampleFields,
    *,
    config: SLDDMMConfig | None = None,
    sample_key: str = "sample_id",
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    device: str | None = None,
    copy: bool = True,
    verbose: bool = True,
    print_every: int = 100,
    prealignment: PrealignmentResult | None = None,
) -> CrossSampleAlignmentResult:
    """Estimate S-LDDMM refinement and transform the original query locations."""
    config = config or SLDDMMConfig()
    validate_cross_sample_anndata(
        adata,
        sample_key=sample_key,
        spatial_key=spatial_key,
        cluster_key=cluster_key,
    )
    query_sample = fields.query_sample
    reference_sample = fields.reference_sample
    validate_sample_selection(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
    )
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_cfg = {
        "a": config.kernel_scale,
        "p": config.kernel_power,
        "expand": config.velocity_expand,
        "nt": config.time_steps,
        "grid_step": config.velocity_grid_spacing,
    }
    optim_cfg = {
        "niter": config.iterations,
        "diffeo_start": config.diffeomorphic_start,
        "lrL": config.affine_linear_lr,
        "lrT": config.affine_translation_lr,
        "lrM": config.momentum_lr,
        "lrM_min": config.minimum_momentum_lr,
        "affine_slowdown": config.affine_slowdown,
        "grad_clip_m0": config.momentum_gradient_clip,
        "lrM_decay": config.momentum_lr_decay,
        "restore_best": config.restore_best_checkpoint,
    }
    em_cfg = {
        "update_every": config.em_update_every,
        "start_iter": config.em_start,
    }
    intensity_cfg = {
        "sigmaM": config.sigma_match,
        "sigmaA": config.sigma_unmatched_a,
        "sigmaB": config.sigma_unmatched_b,
        "sigmaR": config.sigma_regularization,
    }
    transform = lddmm_core.LDDMM_shooting(
        [fields.grid_y, fields.grid_x],
        fields.query_image,
        [fields.grid_y, fields.grid_x],
        fields.reference_image,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        em_cfg=em_cfg,
        intensity_cfg=intensity_cfg,
        device=device,
        dtype=_torch_dtype(config.dtype),
        verbose=verbose,
        print_every=print_every,
    )

    query, reference, query_xy, reference_xy = _pair_tables(
        adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        use_prealigned_query=True,
    )
    query_points_yx = np.column_stack([query_xy[:, 1], query_xy[:, 0]])
    aligned_yx = lddmm_core.map_points_source_to_target(
        transform["xv"],
        transform["v"],
        transform["A"],
        query_points_yx,
    )
    aligned_xy = aligned_yx.detach().cpu().numpy()[:, [1, 0]]

    output = adata.copy() if copy else adata
    if "x_prealigned" not in output.obs:
        initialize_output_coordinates(output, spatial_key=spatial_key)
    query_mask = (
        output.obs[sample_key].astype(str) == str(query_sample)
    ).to_numpy()
    output.obs.loc[query_mask, "x_aligned"] = aligned_xy[:, 0]
    output.obs.loc[query_mask, "y_aligned"] = aligned_xy[:, 1]

    metric_result = lddmm_core.compute_alignment_metrics(
        query,
        reference,
        query_xy[:, 0],
        query_xy[:, 1],
        aligned_xy[:, 0],
        aligned_xy[:, 1],
        reference_xy[:, 0],
        reference_xy[:, 1],
        cluster_col=cluster_key,
    )
    metrics = {
        "nearest_label_agreement_prealigned": float(metric_result["overall_pre"]),
        "nearest_label_agreement_aligned": float(metric_result["overall_ldd"]),
        "nearest_label_agreement_gain": float(
            metric_result["overall_ldd"] - metric_result["overall_pre"]
        ),
        "elapsed_seconds": float(transform["elapsed_sec"]),
        "n_shared_clusters": int(len(fields.shared_clusters)),
        "device": str(device),
    }
    package_metadata = _metadata(output, create=True)
    slddmm_metadata = {
        key: value
        for key, value in asdict(config).items()
        if value is not None
    }
    package_metadata["cross_sample_alignment"] = {
        "query_sample": str(query_sample),
        "reference_sample": str(reference_sample),
        "sample_key": sample_key,
        "cluster_key": cluster_key,
        "spatial_key": spatial_key,
        "rasterization": {
            "grid_shape": [int(len(fields.grid_y)), int(len(fields.grid_x))],
            "shared_clusters": list(fields.shared_clusters),
        },
        "slddmm": slddmm_metadata,
        "metrics": metrics,
    }
    return CrossSampleAlignmentResult(
        adata=output,
        query_sample=str(query_sample),
        reference_sample=str(reference_sample),
        prealignment=prealignment,
        fields=fields,
        transform=transform,
        metrics=metrics,
        cluster_performance=metric_result["cluster_perf"],
    )


def align_cross_sample(
    adata: ad.AnnData,
    *,
    query_sample: str,
    reference_sample: str,
    prealignment_config: PrealignmentConfig | None = None,
    manual_prealignment_config: ManualPrealignmentConfig | None = None,
    rasterization_config: RasterizationConfig | None = None,
    slddmm_config: SLDDMMConfig | None = None,
    sample_key: str = "sample_id",
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    device: str | None = None,
    copy: bool = True,
    verbose: bool = True,
    print_every: int = 100,
    return_result: bool = False,
) -> ad.AnnData | CrossSampleAlignmentResult:
    """Run pre-alignment, field construction and S-LDDMM for one sample pair."""
    if (
        prealignment_config is not None
        and manual_prealignment_config is not None
    ):
        raise ValueError(
            "Provide either prealignment_config or "
            "manual_prealignment_config, not both"
        )
    if manual_prealignment_config is None:
        prealignment = prealign_cross_sample(
            adata,
            query_sample=query_sample,
            reference_sample=reference_sample,
            config=prealignment_config,
            sample_key=sample_key,
            cluster_key=cluster_key,
            spatial_key=spatial_key,
            copy=copy,
            verbose=verbose,
        )
    else:
        prealignment = prealign_cross_sample_manual(
            adata,
            query_sample=query_sample,
            reference_sample=reference_sample,
            config=manual_prealignment_config,
            sample_key=sample_key,
            cluster_key=cluster_key,
            spatial_key=spatial_key,
            copy=copy,
            verbose=verbose,
        )
    fields = rasterize_cross_sample(
        prealignment.adata,
        query_sample=query_sample,
        reference_sample=reference_sample,
        config=rasterization_config,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
    )
    result = run_slddmm_alignment(
        prealignment.adata,
        fields,
        config=slddmm_config,
        sample_key=sample_key,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        device=device,
        copy=False,
        verbose=verbose,
        print_every=print_every,
        prealignment=prealignment,
    )
    return result if return_result else result.adata


def plot_rasterized_fields(
    fields: CrossSampleFields,
    *,
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """Plot cluster-composition previews and density channels."""
    lddmm_core.plot_rasterization_preview(
        fields.query_preview,
        fields.reference_preview,
        fields.query_density,
        fields.reference_density,
        figsize=figsize,
    )


def plot_prealignment_result(
    prealignment: PrealignmentResult,
    *,
    point_size: float = 1.0,
    point_alpha: float = 0.08,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Plot raw/pre-aligned overlays and shared-centroid correspondences."""
    adata = prealignment.adata
    metadata = _metadata(adata)["prealignment"]
    sample_key = metadata["sample_key"]
    spatial_key = metadata["spatial_key"]
    sample = adata.obs[sample_key].astype(str)
    query_mask = (sample == prealignment.query_sample).to_numpy()
    reference_mask = (sample == prealignment.reference_sample).to_numpy()
    raw = spatial_coordinates(adata, spatial_key=spatial_key)
    query_pre = adata.obs.loc[
        query_mask, ["x_prealigned", "y_prealigned"]
    ].to_numpy(dtype=float)
    method_label = {
        "manual_similarity": "manual similarity",
        "cluster_correspondence": "cluster-correspondence",
        "shared_cluster_weighted_procrustes": "cluster-correspondence",
    }.get(metadata.get("prealign_method"), "global")

    pre_core.plot_cluster_correspondence_prealign_before_after(
        raw[query_mask, 0],
        raw[query_mask, 1],
        query_pre[:, 0],
        query_pre[:, 1],
        raw[reference_mask, 0],
        raw[reference_mask, 1],
        prealignment.query_centroids,
        prealignment.reference_centroids,
        prealignment.transformed_query_centroids,
        source_sample_id=prealignment.query_sample,
        target_sample_id=prealignment.reference_sample,
        method_label=method_label,
        point_size=point_size,
        point_alpha=point_alpha,
        figsize=figsize,
    )


def plot_alignment_result(
    result: CrossSampleAlignmentResult,
    *,
    point_size: float = 1.0,
    alpha: float = 0.10,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[Any, Any]:
    """Plot overlays without clusters and return the figure and axes."""
    adata = result.adata
    metadata = _metadata(adata)["cross_sample_alignment"]
    sample_key = metadata["sample_key"]
    spatial_key = metadata["spatial_key"]
    sample = adata.obs[sample_key].astype(str)
    query_mask = (sample == result.query_sample).to_numpy()
    reference_mask = (sample == result.reference_sample).to_numpy()
    raw = spatial_coordinates(adata, spatial_key=spatial_key)
    query_pre = adata.obs.loc[
        query_mask, ["x_prealigned", "y_prealigned"]
    ].to_numpy(dtype=float)
    query_aligned = adata.obs.loc[
        query_mask, ["x_aligned", "y_aligned"]
    ].to_numpy(dtype=float)
    return lddmm_core.plot_alignment_overlays(
        query_pre,
        query_aligned,
        raw[reference_mask],
        point_size=point_size,
        alpha=alpha,
        figsize=figsize,
    )


def plot_cluster_alignment_result(
    result: CrossSampleAlignmentResult,
    *,
    point_size: float = 1.0,
    target_alpha: float = 0.35,
    source_alpha: float = 0.55,
    figsize: tuple[float, float] = (16, 7),
) -> tuple[Any, Any]:
    """Plot cluster-colored overlays and return the figure and axes."""
    adata = result.adata
    metadata = _metadata(adata)["cross_sample_alignment"]
    sample_key = metadata["sample_key"]
    cluster_key = metadata["cluster_key"]
    spatial_key = metadata["spatial_key"]
    sample = adata.obs[sample_key].astype(str)
    query_mask = (sample == result.query_sample).to_numpy()
    reference_mask = (sample == result.reference_sample).to_numpy()
    raw = spatial_coordinates(adata, spatial_key=spatial_key)
    query_pre = adata.obs.loc[
        query_mask, ["x_prealigned", "y_prealigned"]
    ].to_numpy(dtype=float)
    query_aligned = adata.obs.loc[
        query_mask, ["x_aligned", "y_aligned"]
    ].to_numpy(dtype=float)
    query = adata.obs.loc[query_mask].copy()
    reference = adata.obs.loc[reference_mask].copy()
    reference_xy = raw[reference_mask]
    return lddmm_core.plot_cluster_overlay_before_after(
        query,
        reference,
        query_pre[:, 0],
        query_pre[:, 1],
        query_aligned[:, 0],
        query_aligned[:, 1],
        reference_xy[:, 0],
        reference_xy[:, 1],
        cluster_col=cluster_key,
        point_size=point_size,
        target_alpha=target_alpha,
        source_alpha=source_alpha,
        figsize=figsize,
    )
