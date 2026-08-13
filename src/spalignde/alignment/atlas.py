"""Typed public API for structure-guided ST-to-Allen-CCF alignment."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage

from ..io import spatial_coordinates, validate_single_sample_anndata


@dataclass(frozen=True)
class STAtlasAlignmentConfig:
    """Parameters for automatic coarse-to-fine ST-to-Allen-CCF alignment.

    Tune slice/orientation and whole-tissue pre-alignment before structure
    pairing. More ``n_levels`` add intermediate deformation stages. Pairing
    weights are global, must be non-negative, and must sum to one; area and
    thickness weights are useful for separating narrow structures from broad
    neighbors without anatomy-specific overrides.
    """

    n_levels: int = 3
    minimum_coarse_structures: int = 7
    variance_fraction: float = 0.8
    min_genes: int = 50
    drop_blank_genes: bool = True
    stage_iterations: tuple[int, ...] = (100, 500, 100)
    restore_best_checkpoint: bool = False
    continue_alignment: bool = True
    continue_max_iterations: int = 10
    continue_min_pair_gain: int = 1
    continuation_iterations: int = 200
    continuation_kernel_scale: float = 200.0
    continuation_velocity_grid_spacing: float = 50.0
    continuation_restore_best_checkpoint: bool = False
    device: str | None = None
    prealign_close_kernel: int = 15
    prealign_angle_step_degrees: float = 1.0
    prealign_scale_tweak: float = 0.05
    prealign_scale_steps: int = 2
    filter_base_neighbors: int = 20
    filter_detail_area_quantile: float = 0.40
    filter_detail_mad_multiplier: float = 1.0
    filter_normal_mad_multiplier: float = 1.2
    filter_grid_thinning: bool = True
    filter_detail_grid_size: float = 10.0
    pairing_weight_sdf: float = 0.05
    pairing_weight_chamfer: float = 0.05
    pairing_weight_dice: float = 0.20
    pairing_weight_area: float = 0.50
    pairing_weight_thickness: float = 0.20
    pairing_dice_soft: float = 0.25
    pairing_sdf_soft: float = 0.55
    pairing_asd_soft: float = 50.0
    pairing_thickness_soft: float = 0.65
    pairing_min_factor: float = 0.45
    pairing_thickness_power: float = 1.2
    pairing_score_threshold: float = 0.50
    pairing_max_asd: float = 50.0


@dataclass(frozen=True)
class AllenCCFReference:
    """One annotated Allen CCF slice and its structure-hierarchy table."""

    annotation: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    structures: pd.DataFrame
    slice_index: int
    voxel_size_x: float
    voxel_size_y: float
    annotation_path: Path
    structure_table_path: Path

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.annotation.shape)


@dataclass
class STAtlasAlignmentResult:
    """Outputs from automatic ST-to-Allen-CCF alignment."""

    adata: ad.AnnData
    atlas: AllenCCFReference
    matched_pairs: pd.DataFrame
    stage_summary: pd.DataFrame
    prealignment_parameters: dict[str, Any]
    hierarchy_columns: tuple[str, ...]
    output_dir: Path | None = None
    context: dict[str, Any] | None = None


def _anndata_safe_metadata(value: Any) -> Any:
    """Convert nested runtime values to types supported by AnnData ``uns``."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            str(key): _anndata_safe_metadata(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_anndata_safe_metadata(item) for item in value]
    return value


def load_allen_ccf_reference(
    annotation_path: str | Path,
    structure_table_path: str | Path,
    *,
    slice_index: int = 675,
    flip_vertical: bool = True,
) -> AllenCCFReference:
    """Load one 2D annotated Allen CCF slice and its hierarchy metadata."""
    annotation_path = Path(annotation_path).expanduser().resolve()
    structure_table_path = Path(structure_table_path).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Allen annotation NRRD not found: {annotation_path}")
    if not structure_table_path.is_file():
        raise FileNotFoundError(
            f"Allen structure table not found: {structure_table_path}"
        )

    from . import _atlas_core as core

    atlas_info = core.load_atlas_slice(
        str(annotation_path),
        int(slice_index),
        flip_ud=flip_vertical,
        plot=False,
    )
    structures = pd.read_csv(structure_table_path, index_col=0)
    return AllenCCFReference(
        annotation=np.asarray(atlas_info["sl"]),
        x_coordinates=np.asarray(atlas_info["xJ"], dtype=float),
        y_coordinates=np.asarray(atlas_info["yJ"], dtype=float),
        structures=structures,
        slice_index=int(slice_index),
        voxel_size_x=float(atlas_info["dx"]),
        voxel_size_y=float(atlas_info["dy"]),
        annotation_path=annotation_path,
        structure_table_path=structure_table_path,
    )


def _expression_matrix(adata: ad.AnnData, layer: str | None) -> Any:
    matrix = adata.X if layer is None else adata.layers[layer]
    if sp.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float64, copy=True)
        matrix.data = np.log1p(matrix.data)
        return matrix
    return np.log1p(np.asarray(matrix, dtype=np.float64))


def _column_variance(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_square = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        return np.maximum(mean_square - mean**2, 0.0)
    return np.asarray(matrix.var(axis=0), dtype=float)


def _build_hierarchy_tables(
    adata: ad.AnnData,
    *,
    config: STAtlasAlignmentConfig,
    cluster_key: str,
    spatial_key: str,
    layer: str | None,
) -> tuple[pd.DataFrame, tuple[str, ...], list[int], np.ndarray, pd.DataFrame]:
    validate_single_sample_anndata(
        adata,
        spatial_key=spatial_key,
        cluster_key=cluster_key,
        require_cluster=True,
    )
    if not 0 < config.variance_fraction <= 1:
        raise ValueError("variance_fraction must lie in (0, 1]")
    if config.n_levels < 1:
        raise ValueError("n_levels must be positive")
    if config.minimum_coarse_structures < 2:
        raise ValueError("minimum_coarse_structures must be at least 2")

    source = adata.to_memory() if adata.isbacked else adata
    gene_names = np.asarray(source.var_names.astype(str))
    keep_genes = np.ones(source.n_vars, dtype=bool)
    if config.drop_blank_genes:
        keep_genes = np.asarray(
            [not name.lower().startswith("blank") for name in gene_names]
        )
    if not keep_genes.any():
        raise ValueError("No genes remain after removing blank controls")

    matrix = _expression_matrix(source, layer)
    matrix = matrix[:, keep_genes]
    gene_names = gene_names[keep_genes]
    variances = _column_variance(matrix)
    order = np.argsort(variances)[::-1]
    total_variance = float(variances.sum())
    if total_variance > 0:
        cumulative = np.cumsum(variances[order]) / total_variance
        n_selected = int(np.searchsorted(cumulative, config.variance_fraction) + 1)
    else:
        n_selected = len(order)
    n_selected = min(max(n_selected, config.min_genes), len(order))
    selected = order[:n_selected]
    selected_matrix = matrix[:, selected]

    labels = source.obs[cluster_key].astype(str).to_numpy()
    base_clusters = sorted(pd.unique(labels), key=str)
    if len(base_clusters) < 2:
        raise ValueError("ST-to-atlas alignment requires at least two ST clusters")
    averages = []
    for cluster in base_clusters:
        cluster_matrix = selected_matrix[labels == cluster]
        averages.append(np.asarray(cluster_matrix.mean(axis=0)).ravel())
    average = np.vstack(averages)
    ddof = 1 if average.shape[0] > 1 else 0
    scale = average.std(axis=0, ddof=ddof)
    average_z = (average - average.mean(axis=0)) / (scale + 1e-8)
    average_z[~np.isfinite(average_z)] = 0.0
    hierarchy_linkage = linkage(average_z, method="ward", metric="euclidean")

    n_base = len(base_clusters)
    # Do not create a one-structure hierarchy stage for small toy inputs; the
    # coarsest meaningful structural partition contains at least two groups.
    n_levels = min(config.n_levels, max(n_base - 1, 1))
    if n_levels == 1:
        levels = [n_base]
    else:
        # Atlas alignment needs enough coarse structures to establish multiple
        # anatomical anchors. Start at seven when the finest partition allows
        # it, then space the remaining levels evenly through the final labels.
        max_first = n_base - (n_levels - 1)
        first_level = min(config.minimum_coarse_structures, max_first)
        first_level = max(1, first_level)
        levels = [
            int(round(first_level + index * (n_base - first_level) / (n_levels - 1)))
            for index in range(n_levels)
        ]
        for index in range(1, len(levels)):
            if levels[index] <= levels[index - 1]:
                levels[index] = levels[index - 1] + 1
        levels[-1] = n_base
    dataframe = source.obs.copy()
    xy = spatial_coordinates(source, spatial_key=spatial_key)
    dataframe["x"] = xy[:, 0]
    dataframe["y"] = xy[:, 1]
    dataframe[cluster_key] = labels
    hierarchy_columns: list[str] = []
    for n_clusters in [value for value in levels if value < len(base_clusters)]:
        column = f"{cluster_key}_level_k{n_clusters}"
        cluster_to_level = dict(
            zip(
                base_clusters,
                fcluster(
                    hierarchy_linkage,
                    t=n_clusters,
                    criterion="maxclust",
                ),
                strict=False,
            )
        )
        dataframe[column] = pd.Categorical(
            [str(cluster_to_level[label]) for label in labels]
        )
        hierarchy_columns.append(column)

    average_z_frame = pd.DataFrame(
        average_z,
        index=pd.Index(base_clusters, name=cluster_key),
        columns=gene_names[selected],
    )
    return (
        dataframe,
        tuple(hierarchy_columns),
        levels,
        hierarchy_linkage,
        average_z_frame,
    )


def build_st_cluster_hierarchy(
    adata: ad.AnnData,
    *,
    config: STAtlasAlignmentConfig | None = None,
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    layer: str | None = None,
    copy: bool = True,
) -> tuple[ad.AnnData, tuple[str, ...]]:
    """Add coarse-to-fine expression-based ST structure levels to AnnData."""
    config = config or STAtlasAlignmentConfig()
    table, hierarchy_columns, _, _, _ = _build_hierarchy_tables(
        adata,
        config=config,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        layer=layer,
    )
    source = adata.to_memory() if adata.isbacked else adata
    output = source.copy() if copy else source
    for column in hierarchy_columns:
        output.obs[column] = table[column].copy()
    output.uns.pop("spalignde", None)
    output.uns.setdefault("spAlignDE", {})
    output.uns["spAlignDE"]["st_atlas_hierarchy"] = {
        "cluster_key": cluster_key,
        "hierarchy_columns": list(hierarchy_columns),
        "n_levels": config.n_levels,
        "variance_fraction": config.variance_fraction,
        "min_genes": config.min_genes,
    }
    return output, hierarchy_columns


def _atlas_info(reference: AllenCCFReference) -> dict[str, Any]:
    height, width = reference.annotation.shape
    return {
        "sl": reference.annotation,
        "xJ": reference.x_coordinates,
        "yJ": reference.y_coordinates,
        "dx": reference.voxel_size_x,
        "dy": reference.voxel_size_y,
        "H": height,
        "W": width,
        "z": reference.slice_index,
    }


def _pairing_parameters(
    config: STAtlasAlignmentConfig,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Build and validate structure-pairing score, gate and threshold settings."""
    weights = {
        "sdf_corr": float(config.pairing_weight_sdf),
        "chamfer_sim": float(config.pairing_weight_chamfer),
        "dice": float(config.pairing_weight_dice),
        "area_sim": float(config.pairing_weight_area),
        "thick_sim": float(config.pairing_weight_thickness),
    }
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Atlas pairing weights must be finite and non-negative")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-8):
        raise ValueError("Atlas pairing weights must sum to 1.0")

    gate = {
        "dice_soft": float(config.pairing_dice_soft),
        "sdf_soft": float(config.pairing_sdf_soft),
        "asd_soft": float(config.pairing_asd_soft),
        "thick_soft": float(config.pairing_thickness_soft),
        "min_factor": float(config.pairing_min_factor),
        "thick_power": float(config.pairing_thickness_power),
    }
    thresholds = {
        "leaf_score": float(config.pairing_score_threshold),
        "leaf_asd": float(config.pairing_max_asd),
        "layer_score": float(config.pairing_score_threshold),
        "layer_asd": float(config.pairing_max_asd),
    }
    numeric = {**gate, **thresholds}
    if any(not np.isfinite(value) or value < 0 for value in numeric.values()):
        raise ValueError("Atlas pairing gates and thresholds must be finite and non-negative")
    if not 0 < gate["min_factor"] <= 1:
        raise ValueError("pairing_min_factor must lie in (0, 1]")
    return weights, gate, thresholds


@contextmanager
def _temporary_core_pairing_weights(core, weights: Mapping[str, float]):
    """Synchronize and restore the legacy module-global pairing weights."""
    previous = dict(core.W_ALIGN)
    core.W_ALIGN = dict(weights)
    try:
        yield
    finally:
        core.W_ALIGN = previous


def _optimizer_overrides(
    config: STAtlasAlignmentConfig,
) -> dict[str, Any]:
    """Validate and expose the optimizer schedule for automatic Atlas alignment."""
    stage_iterations = tuple(int(value) for value in config.stage_iterations)
    if len(stage_iterations) != int(config.n_levels):
        raise ValueError(
            "stage_iterations must contain one value per coarse-to-fine stage "
            f"({config.n_levels} values required, found {len(stage_iterations)})"
        )
    if any(value < 1 for value in stage_iterations):
        raise ValueError("stage_iterations values must be positive integers")

    continuation_iterations = int(config.continuation_iterations)
    if continuation_iterations < 1:
        raise ValueError("continuation_iterations must be a positive integer")

    return {
        "STAGE_ITERATIONS_SCHEDULE": list(stage_iterations),
        "ITER_OPTIM_CFG_OVERRIDE": {
            "restore_best": bool(config.restore_best_checkpoint),
        },
        "ITER_FINAL_OPTIM_CFG_OVERRIDE": {
            "restore_best": bool(config.restore_best_checkpoint),
        },
        "CONTINUE_OPTIM_CFG_OVERRIDE": {
            "restore_best": bool(config.continuation_restore_best_checkpoint),
            "niter": continuation_iterations,
        },
    }


def _core_config(
    config: STAtlasAlignmentConfig,
    *,
    atlas: AllenCCFReference,
    cluster_key: str,
    hierarchy_columns: tuple[str, ...],
    output_dir: Path,
):
    from . import _atlas_core as core

    return core.STAtlasConfig(
        st_cluster_csv=Path("unused.csv"),
        st_counts_csv=Path("unused.csv"),
        atlas_voxel_csv=atlas.structure_table_path,
        atlas_nrrd=atlas.annotation_path,
        atlas_slice_z=atlas.slice_index,
        output_dir=output_dir,
        levels_csv=output_dir / "st_cluster_hierarchy.csv",
        n_levels=config.n_levels,
        var_frac=config.variance_fraction,
        min_genes=config.min_genes,
        cluster_col=cluster_key,
        stage_labels=[*hierarchy_columns, cluster_key],
        continue_alignment=config.continue_alignment,
        continue_max_iter=config.continue_max_iterations,
        continue_min_pair_gain=config.continue_min_pair_gain,
        device=config.device,
        prealign_close_ksize=config.prealign_close_kernel,
        prealign_angle_step_deg=config.prealign_angle_step_degrees,
        prealign_scale_tweak=config.prealign_scale_tweak,
        prealign_scale_steps=config.prealign_scale_steps,
        filter_base_k=config.filter_base_neighbors,
        filter_detail_area_quantile=config.filter_detail_area_quantile,
        filter_detail_mad_k=config.filter_detail_mad_multiplier,
        filter_normal_mad_k=config.filter_normal_mad_multiplier,
        filter_apply_grid_thin=config.filter_grid_thinning,
        filter_grid_size_detail=config.filter_detail_grid_size,
    )


def align_st_to_allen_atlas(
    adata: ad.AnnData,
    atlas: AllenCCFReference,
    *,
    config: STAtlasAlignmentConfig | None = None,
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    layer: str | None = None,
    output_dir: str | Path = "st_to_atlas_output",
    copy: bool = True,
    save_outputs: bool = True,
) -> STAtlasAlignmentResult:
    """Align one clustered ST sample to an annotated Allen CCF slice.

    The workflow builds expression-based coarse-to-fine ST structure levels,
    estimates whole-tissue IoU pre-alignment, discovers non-overlapping
    cluster-to-atlas structure pairs, performs iterative S-LDDMM, and samples
    final Allen labels at every aligned ST coordinate.
    """
    config = config or STAtlasAlignmentConfig()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    table, hierarchy_columns, levels, hierarchy_linkage, average_z = (
        _build_hierarchy_tables(
            adata,
            config=config,
            cluster_key=cluster_key,
            spatial_key=spatial_key,
            layer=layer,
        )
    )

    from . import _atlas_core as core

    core_config = _core_config(
        config,
        atlas=atlas,
        cluster_key=cluster_key,
        hierarchy_columns=hierarchy_columns,
        output_dir=output_path,
    )
    atlas_info = _atlas_info(atlas)
    prealigned, prealignment_parameters = core.prealign_st_to_atlas(
        table.copy(),
        atlas_info,
        core_config,
    )
    filtered, removed, filter_stats = core.filter_prealigned_st_points(
        prealigned,
        label_col=cluster_key,
        config=core_config,
    )
    context = core.make_alignment_context(
        config=core_config,
        df1=table,
        refined_cols=list(hierarchy_columns),
        df2=atlas.structures,
        atlas_info=atlas_info,
        df_smooth=table,
        df_final=prealigned,
        df_final_filtered=filtered.copy(),
        prealign_params=prealignment_parameters,
    )
    context.update(
        {
            "ks": levels,
            "Z": hierarchy_linkage,
            "avg_z": average_z,
            "df_keep": filtered,
            "df_removed": removed,
            "filter_stats_df": filter_stats,
            "label_col_filter": cluster_key,
        }
    )
    # Shape-balanced atlas pairing score. Area and thickness carry enough
    # weight to distinguish narrow laminar structures from broader neighbors;
    # ASD remains an independent QC gate rather than a score component.
    atlas_pairing_weights, atlas_pairing_gate, atlas_pairing_thresholds = (
        _pairing_parameters(config)
    )
    context["W_ALIGN"] = dict(atlas_pairing_weights)
    context["ITER_FINAL_W_ALIGN_OVERRIDE"] = dict(atlas_pairing_weights)
    context["CONTINUE_W_ALIGN_OVERRIDE"] = dict(atlas_pairing_weights)
    context["ITER_GATE_PARAMS_OVERRIDE"] = dict(atlas_pairing_gate)
    context["CONTINUE_GATE_PARAMS_OVERRIDE"] = dict(atlas_pairing_gate)
    context["ITER_THRESH_OVERRIDE"] = dict(atlas_pairing_thresholds)
    context["CONTINUE_THRESH_OVERRIDE"] = dict(atlas_pairing_thresholds)
    context["CONTINUE_MODEL_CFG_OVERRIDE"] = {
        "a": float(config.continuation_kernel_scale),
        "grid_step": float(config.continuation_velocity_grid_spacing),
    }
    context.update(_optimizer_overrides(config))
    context["CONTINUE_LABEL_COL"] = cluster_key
    with _temporary_core_pairing_weights(core, atlas_pairing_weights):
        core.run_iterative_multi_level_alignment(context)
        if config.continue_alignment:
            context["continue_start_filtered"] = context["df_smooth"].copy()
            core.run_continuation_alignment(context)

    aligned_table = context["df_aligned_all"].copy()
    labelled_table = core.transfer_all_atlas_labels_to_st(
        aligned_table,
        sl=atlas.annotation,
        xJ=atlas.x_coordinates,
        yJ=atlas.y_coordinates,
        atlas_metadata=atlas.structures,
    )
    source = adata.to_memory() if adata.isbacked else adata
    output = source.copy() if copy else source
    prealigned = prealigned.reindex(output.obs_names)
    labelled_table = labelled_table.reindex(output.obs_names)
    for column in hierarchy_columns:
        output.obs[column] = table.reindex(output.obs_names)[column].copy()
    for column in ("x_prealigned", "y_prealigned"):
        output.obs[column] = prealigned[column].to_numpy(dtype=float)
    for column in ("x_aligned", "y_aligned"):
        output.obs[column] = labelled_table[column].to_numpy(dtype=float)
    atlas_columns = [
        "atlas_voxel_xi",
        "atlas_voxel_yi",
        "atlas_voxel_in_view",
        "atlas_label_id",
        "atlas_label_acronym",
        "atlas_label_name",
        "atlas_label_color_hex",
        "atlas_label_transferred",
        "atlas_label",
        "inside_atlas_slice",
        "atlas_region_name",
        "atlas_region_acronym",
    ]
    for column in atlas_columns:
        output.obs[column] = labelled_table[column].to_numpy()

    final_pairs = context.get("final_pair_df")
    if final_pairs is None:
        final_pairs = context.get("pair_df", pd.DataFrame())
    final_pairs = final_pairs.copy()
    stage_summary = pd.DataFrame(
        [
            {
                "stage": item.get("stage"),
                "label_col": item.get("label_col"),
                "is_final_stage": bool(item.get("is_final_stage", False)),
                "n_pairs": int(item.get("n_pairs", 0)),
                "lddmm_ran": item.get("lddmm_out") is not None,
            }
            for item in context.get("iter_stage_outputs", [])
        ]
    )
    output.uns.pop("spalignde", None)
    output.uns.setdefault("spAlignDE", {})
    config_metadata = _anndata_safe_metadata({
        key: ("auto" if key == "device" and value is None else value)
        for key, value in asdict(config).items()
    })
    output.uns["spAlignDE"]["st_to_allen_atlas"] = {
        **config_metadata,
        "cluster_key": cluster_key,
        "spatial_key": spatial_key,
        "slice_index": atlas.slice_index,
        "hierarchy_columns": list(hierarchy_columns),
        "n_matched_pairs": int(len(final_pairs)),
        "output_dir": str(output_path),
        "pairing_weights": atlas_pairing_weights,
        "pairing_gate": atlas_pairing_gate,
        "pairing_thresholds": {
            "minimum_gated_score": config.pairing_score_threshold,
            "maximum_asd_pixels": config.pairing_max_asd,
        },
        "prealignment_parameters": _anndata_safe_metadata(
            prealignment_parameters
        ),
    }

    if save_outputs:
        core.save_alignment_visualization(context, output_path)
        core.save_alignment_outputs(context, output_path)
        core.build_white_label_color_map_for_atlas(
            atlas.annotation,
            seed=0,
        ).to_csv(
            output_path
            / (
                f"atlas_z{atlas.slice_index}_white_label_color_map_"
                "for_transfer_labels.csv"
            ),
            index=False,
        )
        table.loc[:, list(hierarchy_columns) + [cluster_key, "x", "y"]].to_csv(
            output_path / "st_cluster_hierarchy.csv"
        )
        output.write_h5ad(output_path / "st_to_allen_atlas_aligned.h5ad")

    return STAtlasAlignmentResult(
        adata=output,
        atlas=atlas,
        matched_pairs=final_pairs,
        stage_summary=stage_summary,
        prealignment_parameters=dict(prealignment_parameters),
        hierarchy_columns=hierarchy_columns,
        output_dir=output_path if save_outputs else None,
        context=context,
    )


def load_st_atlas_alignment(
    adata: ad.AnnData,
    atlas: AllenCCFReference,
    output_dir: str | Path,
    *,
    copy: bool = True,
) -> STAtlasAlignmentResult:
    """Attach a previously saved Atlas run to its original AnnData input."""
    output_path = Path(output_dir).expanduser().resolve()
    aligned_path = output_path / "st_to_allen_atlas_aligned.h5ad"
    if aligned_path.is_file():
        aligned_adata = ad.read_h5ad(aligned_path)
        if not aligned_adata.obs_names.equals(adata.obs_names):
            raise ValueError("Saved Atlas result does not match input observation names")
    else:
        final_path = output_path / "final_aligned_all_points.csv"
        coordinate_path = output_path / "coordinates_original_prealign_aligned.csv"
        if not final_path.is_file() or not coordinate_path.is_file():
            raise FileNotFoundError(
                "Saved Atlas result requires st_to_allen_atlas_aligned.h5ad or "
                "the final-aligned and coordinate-summary CSV files"
            )
        final = pd.read_csv(final_path, index_col=0)
        coordinates = pd.read_csv(coordinate_path, index_col=0)
        source = adata.to_memory() if adata.isbacked else adata
        aligned_adata = source.copy() if copy else source
        final = final.reindex(aligned_adata.obs_names)
        coordinates = coordinates.reindex(aligned_adata.obs_names)
        aligned_adata.obs["x_prealigned"] = coordinates["prealign_x"].to_numpy()
        aligned_adata.obs["y_prealigned"] = coordinates["prealign_y"].to_numpy()
        aligned_adata.obs["x_aligned"] = coordinates["aligned_x"].to_numpy()
        aligned_adata.obs["y_aligned"] = coordinates["aligned_y"].to_numpy()
        transferred = _attach_atlas_labels(aligned_adata, atlas)
        aligned_adata = transferred

    pair_path = output_path / "matched_pairs_final_stage.csv"
    summary_path = output_path / "iterative_alignment_stage_summary.csv"
    matched_pairs = pd.read_csv(pair_path) if pair_path.is_file() else pd.DataFrame()
    stage_summary = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    metadata = aligned_adata.uns.get("spAlignDE", {}).get(
        "st_to_allen_atlas", {}
    )
    return STAtlasAlignmentResult(
        adata=aligned_adata,
        atlas=atlas,
        matched_pairs=matched_pairs,
        stage_summary=stage_summary,
        prealignment_parameters=metadata.get("prealignment_parameters", {}),
        hierarchy_columns=tuple(metadata.get("hierarchy_columns", ())),
        output_dir=output_path,
        context=None,
    )


def _attach_atlas_labels(
    adata: ad.AnnData,
    atlas: AllenCCFReference,
) -> ad.AnnData:
    from . import _atlas_core as core

    table = adata.obs.copy()
    labelled = core.transfer_all_atlas_labels_to_st(
        table,
        sl=atlas.annotation,
        xJ=atlas.x_coordinates,
        yJ=atlas.y_coordinates,
        atlas_metadata=atlas.structures,
    )
    for column in labelled.columns.difference(table.columns):
        adata.obs[column] = labelled[column].to_numpy()
    adata.uns.setdefault("spAlignDE", {})
    return adata


def plot_st_atlas_alignment(
    result: STAtlasAlignmentResult,
    *,
    cluster_key: str | None = None,
    structure_color_map: Mapping[str, Any] | None = None,
    point_size: float = 1.0,
    alpha: float | None = None,
    figsize: tuple[float, float] = (14.0, 7.0),
) -> tuple[Any, Any]:
    """Plot the original matched-structure before/after alignment view.

    Matched ST clusters and Allen regions share colors assigned by Allen
    structure name. The default mapping preserves the clear colors used by the
    validated paper notebook and is stable when clustering changes the final
    pair-table order. Unmatched atlas regions are light gray and unmatched ST
    observations are dark gray.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    from . import _atlas_core as core

    adata = result.adata
    required = ("x_prealigned", "y_prealigned", "x_aligned", "y_aligned")
    missing = [column for column in required if column not in adata.obs]
    if missing:
        raise KeyError(f"Missing alignment coordinate columns: {missing}")

    if cluster_key is None:
        metadata = adata.uns.get("spAlignDE", {}).get("st_to_allen_atlas", {})
        preferred = metadata.get("cluster_key", "cluster")
        cluster_key = next(
            (
                key
                for key in (
                    preferred,
                    "cluster",
                    "cluster_refined",
                    "banksy_cluster_refined",
                )
                if key in adata.obs
            ),
            None,
        )
    if cluster_key is None or cluster_key not in adata.obs:
        raise KeyError("Could not find the final ST cluster labels for plotting")

    annotation = result.atlas.annotation
    height, width = annotation.shape
    pair_table = result.matched_pairs.copy()
    overlay = np.zeros((height, width, 4), dtype=float)
    brain = annotation > 0
    other_region_color = np.array((0.92, 0.92, 0.92, 0.90), dtype=float)

    cluster_colors: dict[str, np.ndarray] = {}
    if pair_table.empty:
        overlay[brain] = other_region_color
    else:
        if "cluster" not in pair_table:
            raise KeyError("The matched-pair table requires a 'cluster' column")
        label_column = next(
            (
                column
                for column in ("atlas_labels_union", "labels")
                if column in pair_table
            ),
            None,
        )
        if label_column is None:
            raise KeyError(
                "The matched-pair table requires 'atlas_labels_union' or 'labels'"
            )
        pair_table["cluster"] = pair_table["cluster"].astype(str)
        clusters = pair_table["cluster"].unique().tolist()
        palette = load_atlas_structure_color_map()
        if structure_color_map is not None:
            palette.update({str(key): value for key, value in structure_color_map.items()})

        candidate_by_cluster = {}
        if "candidate_name" in pair_table:
            candidate_by_cluster = (
                pair_table.drop_duplicates("cluster")
                .set_index("cluster")["candidate_name"]
                .astype(str)
                .to_dict()
            )
        unknown_names = sorted(
            {
                candidate_by_cluster.get(cluster, cluster)
                for cluster in clusters
                if candidate_by_cluster.get(cluster, cluster) not in palette
            }
        )
        color_cycle = plt.get_cmap("tab20", max(len(unknown_names), 1))
        fallback_colors = {
            name: mcolors.to_hex(color_cycle(index))
            for index, name in enumerate(unknown_names)
        }
        for cluster in clusters:
            structure_name = candidate_by_cluster.get(cluster, cluster)
            color = palette.get(structure_name)
            if color is None:
                color = fallback_colors[structure_name]
            cluster_colors[cluster] = np.asarray(
                mcolors.to_rgba(color, alpha=1.0)
            )
        matched = np.zeros((height, width), dtype=bool)
        for _, row in pair_table.iterrows():
            labels = core.parse_labels_any(row[label_column])
            if not labels:
                continue
            mask = np.isin(annotation, labels)
            overlay[mask] = cluster_colors[str(row["cluster"])]
            matched |= mask
        overlay[brain & ~matched] = other_region_color

    other_point_color = np.array((0.12, 0.12, 0.12, 0.45), dtype=float)
    point_colors = np.asarray(
        [
            cluster_colors.get(str(cluster), other_point_color)
            for cluster in adata.obs[cluster_key].astype(str)
        ]
    )
    if alpha is not None:
        point_colors = point_colors.copy()
        point_colors[:, 3] *= float(alpha)

    physical_to_pixel = core.make_phys_to_pix(
        result.atlas.x_coordinates,
        result.atlas.y_coordinates,
        height,
        width,
    )
    panels = (
        ("x_prealigned", "y_prealigned", "Before iterative LDDMM (all points)"),
        ("x_aligned", "y_aligned", "After iterative LDDMM (all points)"),
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    for axis, (x_key, y_key, title) in zip(axes, panels, strict=False):
        x = adata.obs[x_key].to_numpy(dtype=float)
        y = adata.obs[y_key].to_numpy(dtype=float)
        x_pixel, y_pixel = physical_to_pixel(x, y)
        in_view = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x_pixel >= 0)
            & (x_pixel < width)
            & (y_pixel >= 0)
            & (y_pixel < height)
        )
        axis.set_facecolor("white")
        axis.imshow(
            brain,
            cmap="binary",
            alpha=0.10,
            origin="lower",
        )
        axis.imshow(overlay, origin="lower", interpolation="nearest")
        axis.scatter(
            x_pixel[in_view],
            y_pixel[in_view],
            s=point_size,
            c=point_colors[in_view],
            edgecolors="none",
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.axis("off")
    fig.tight_layout()
    return fig, axes


_DEFAULT_ATLAS_STRUCTURE_COLOR_MAP = {
    # Exact structure colors used by the validated S2R1 paper figure.
    "L6a": "#1f77b4",
    "fiber tracts": "#aec7e8",
    "PAL": "#ffbb78",
    "PVZ": "#2ca02c",
    "EPI": "#d62728",
    "CA3sp": "#ff9896",
    "DG-sg": "#c5b0d5",
    "L2/3": "#8c564b",
    "CA1sp": "#e377c2",
    "L5": "#f7b6d2",
    "DORsm": "#c7c7c7",
    "L4": "#bcbd22",
    "LZ": "#17becf",
    "STRd": "#9edae5",
    # Additional structures selected by the three-level fresh package run.
    "TH": "#7f7f7f",
    "cst": "#ff7f0e",
    "ccb": "#98df8a",
    "ARH": "#9467bd",
    "PVi": "#c49c94",
    "sAMY": "#dbdb8d",
}


def load_atlas_structure_color_map(
    path: str | Path | None = None,
) -> dict[str, str]:
    """Load a structure-name palette or return the validated paper palette.

    A user-supplied CSV must contain ``candidate_name`` plus either ``hex`` or
    ``color``. The built-in palette keeps structure identities stable across
    runs even when the matched-pair table or cluster numbering changes.
    """
    if path is None:
        return dict(_DEFAULT_ATLAS_STRUCTURE_COLOR_MAP)

    palette_path = Path(path).expanduser()
    if not palette_path.is_file():
        raise FileNotFoundError(f"Atlas structure color map not found: {palette_path}")
    table = pd.read_csv(palette_path)
    if "candidate_name" not in table:
        raise KeyError("Atlas structure color map requires 'candidate_name'")
    color_column = next(
        (column for column in ("hex", "color") if column in table),
        None,
    )
    if color_column is None:
        raise KeyError("Atlas structure color map requires 'hex' or 'color'")
    return dict(
        zip(
            table["candidate_name"].astype(str),
            table[color_column].astype(str),
            strict=False,
        )
    )


def load_atlas_label_color_map(
    path: str | Path | None = None,
    *,
    atlas: AllenCCFReference | None = None,
    seed: int = 0,
) -> dict[str, str]:
    """Load the fixed atlas-label palette, or build it reproducibly."""
    from . import _atlas_core as core

    if path is not None and Path(path).expanduser().is_file():
        return core.load_atlas_label_color_map(path=path)
    if atlas is None:
        if path is not None:
            raise FileNotFoundError(f"Atlas label color map not found: {path}")
        raise ValueError("Provide an existing color-map path or an Allen atlas")
    colors = core.build_white_label_color_map_for_atlas(
        atlas.annotation,
        seed=seed,
    )
    colors["atlas_label"] = colors["atlas_label"].astype(str)
    return colors.set_index("atlas_label")["hex"].to_dict()


def plot_atlas_label_transfer(
    result: STAtlasAlignmentResult,
    *,
    color_map: dict[str, str] | None = None,
    point_size: float = 2.0,
    point_alpha: float = 0.8,
    figsize: tuple[float, float] = (14.0, 7.0),
) -> tuple[Any, Any]:
    """Reproduce the validated Allen/ST transferred-label comparison."""
    from . import _atlas_core as core

    if color_map is None:
        palette_path = None
        if result.output_dir is not None:
            candidate = result.output_dir / (
                f"atlas_z{result.atlas.slice_index}_white_label_color_map_"
                "for_transfer_labels.csv"
            )
            if candidate.is_file():
                palette_path = candidate
        color_map = load_atlas_label_color_map(
            palette_path,
            atlas=result.atlas,
        )
    return core.plot_atlas_slice_and_label_scatter(
        sl=result.atlas.annotation,
        xJ=result.atlas.x_coordinates,
        yJ=result.atlas.y_coordinates,
        st_df=result.adata.obs,
        color_map=color_map,
        output_prefix=None,
        point_size=point_size,
        point_alpha=point_alpha,
        figsize=figsize,
    )


__all__ = [
    "AllenCCFReference",
    "STAtlasAlignmentConfig",
    "STAtlasAlignmentResult",
    "align_st_to_allen_atlas",
    "build_st_cluster_hierarchy",
    "load_allen_ccf_reference",
    "load_atlas_label_color_map",
    "load_atlas_structure_color_map",
    "load_st_atlas_alignment",
    "plot_atlas_label_transfer",
    "plot_st_atlas_alignment",
]
