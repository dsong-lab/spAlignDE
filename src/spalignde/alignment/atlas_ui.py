"""Allen CCF alignment driven by correspondences exported from the UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom

from ..io import spatial_coordinates, validate_single_sample_anndata
from .atlas import (
    AllenCCFReference,
    STAtlasAlignmentConfig,
    STAtlasAlignmentResult,
    _anndata_safe_metadata,
    _atlas_info,
    _core_config,
)


@dataclass
class UIAtlasPairing:
    """Validated many-to-many ST/Allen correspondences exported by the UI.

    ``deformation_groups`` contains one unioned source/target mask channel per
    ``group_id``. ``matched_pairs`` expands each group back to one row per ST
    cluster for plotting and provenance. Individual rows in ``raw`` must not be
    treated as independent LDDMM channels.
    """

    raw: pd.DataFrame
    deformation_groups: pd.DataFrame
    matched_pairs: pd.DataFrame
    atlas_slice_index: int | None
    st_side: Literal["left", "right"]
    source: str


@dataclass(frozen=True)
class UIAtlasAlignmentConfig:
    """Parameters for one UI-curated ST-to-Allen S-LDDMM alignment.

    ``kernel_scale`` and ``velocity_grid_spacing`` are the public names for the
    legacy S-LDDMM parameters ``a`` and ``grid_step``. They are measured in the
    physical coordinate system of the pre-aligned Atlas canvas.
    """

    device: str | None = None
    dtype: Literal["float32", "float64"] = "float64"

    # Global initialization. ``provided`` accepts coordinates produced by a
    # separate manual pre-alignment through the two named AnnData obs columns.
    prealignment_mode: Literal["mask", "provided"] = "mask"
    provided_prealigned_x_key: str = "x_prealigned"
    provided_prealigned_y_key: str = "y_prealigned"

    # Whole-tissue mask initialization and point filtering.
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

    # Pair-mask preprocessing used by the validated UI-based example.
    source_sigma_pre: float = 1.4
    source_threshold: float = 0.5
    source_close_radius: int = 2
    source_open_radius: int = 1
    source_minimum_area: int = 50
    target_sigma_pre: float = 0.4
    target_threshold: float = 0.5
    target_close_radius: int = 1
    target_open_radius: int = 1
    target_minimum_area: int = 80
    signed_distance_clip: float = 4.0
    signed_distance_sigma: float = 0.9
    signed_distance_boundary_band: float = 4.0
    signed_distance_scale: float = 2.0
    area_weight_power: float = 0.8
    minimum_channel_weight: float = 0.5
    maximum_channel_weight: float = 2.5
    global_channel_scale: float = 1.6
    raster_zoom_scale: float = 0.6

    # S-LDDMM model and optimizer.
    kernel_scale: float = 200.0
    operator_power: float = 2.0
    time_steps: int = 5
    velocity_domain_expansion: float = 2.0
    velocity_grid_spacing: float = 50.0
    iterations: int = 500
    diffeomorphic_start_iteration: int = 0
    affine_matrix_learning_rate: float = 2e-8
    translation_learning_rate: float = 2e-1
    momentum_learning_rate: float = 2e3
    affine_slowdown: float = 10.0
    momentum_learning_rate_decay: float = 0.9995
    minimum_momentum_learning_rate: float = 200.0
    restore_best_checkpoint: bool = False
    em_update_every: int = 5
    em_start_iteration: int = 50
    match_scale: float = 1.0
    background_scale: float = 2.0
    artifact_scale: float = 5.0
    regularization_scale: float = 5e5
    landmark_scale: float = 2e1
    print_every: int = 100
    verbose: bool = True


def _parse_id_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = value
    elif pd.isna(value):
        return []
    else:
        text = str(value).strip().strip("[]").replace('"', "").replace("'", "")
        if not text:
            return []
        values = text.replace(";", ",").split(",")
    parsed: list[int] = []
    for item in values:
        text = str(item).strip()
        if text:
            parsed.append(int(float(text)))
    return sorted(set(parsed))


def _ids_for_group(group: pd.DataFrame, side: str) -> list[int]:
    """Prefer original IDs retained by the UI over a custom display ID."""
    for suffix in ("raw_selected_ids", "included_region_ids"):
        column = f"{side}_{suffix}"
        if column in group:
            values = sorted(
                {
                    item
                    for value in group[column].dropna()
                    for item in _parse_id_list(value)
                }
            )
            if values:
                return values
    column = f"{side}_selected_id"
    return sorted(
        {
            item
            for value in group[column].dropna()
            for item in _parse_id_list(value)
        }
    )


def _infer_st_side(frame: pd.DataFrame, requested: str) -> Literal["left", "right"]:
    if requested in {"left", "right"}:
        return requested  # type: ignore[return-value]
    if requested != "auto":
        raise ValueError("st_side must be 'auto', 'left', or 'right'")
    kinds: dict[str, set[str]] = {}
    for side in ("left", "right"):
        column = f"{side}_dataset_kind"
        kinds[side] = (
            set(frame[column].dropna().astype(str).str.lower())
            if column in frame
            else set()
        )
    if "st" in kinds["left"] and "atlas" in kinds["right"]:
        return "left"
    if "st" in kinds["right"] and "atlas" in kinds["left"]:
        return "right"
    raise ValueError(
        "Could not infer which UI panel contains ST. Pass st_side='left' or "
        "st_side='right', or retain the *_dataset_kind columns in the export."
    )


def _atlas_slice_from_export(frame: pd.DataFrame, atlas_side: str) -> int | None:
    values: set[int] = set()
    for column in ("atlas_z_slice", f"{atlas_side}_atlas_z_slice"):
        if column in frame:
            values.update(int(value) for value in frame[column].dropna().astype(int))
    if len(values) > 1:
        raise ValueError(f"UI pairing export contains multiple Allen slices: {sorted(values)}")
    return next(iter(values)) if values else None


def load_ui_atlas_pairing(
    pairs: pd.DataFrame | str | Path,
    *,
    st_side: Literal["auto", "left", "right"] = "auto",
    expected_atlas_slice: int | None = None,
) -> UIAtlasPairing:
    """Load a UI export and reconstruct its many-to-many pairing groups.

    The function supports raw selections and custom regions. When the export
    retains ``*_raw_selected_ids`` or ``*_included_region_ids``, those source
    IDs take precedence over the UI-only custom display ID.
    """
    if isinstance(pairs, pd.DataFrame):
        frame = pairs.copy()
        source = "dataframe"
    else:
        path = Path(pairs).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"UI pairing CSV not found: {path}")
        frame = pd.read_csv(path)
        source = str(path)
    if frame.empty:
        raise ValueError("UI pairing export is empty")
    if "group_id" not in frame:
        raise KeyError("UI pairing export is missing 'group_id'")

    resolved_st_side = _infer_st_side(frame, st_side)
    atlas_side = "right" if resolved_st_side == "left" else "left"
    required = {
        f"{resolved_st_side}_selected_id",
        f"{atlas_side}_selected_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"UI pairing export is missing columns: {missing}")

    slice_index = _atlas_slice_from_export(frame, atlas_side)
    if (
        expected_atlas_slice is not None
        and slice_index is not None
        and int(slice_index) != int(expected_atlas_slice)
    ):
        raise ValueError(
            "UI pairing slice does not match the loaded Allen reference: "
            f"export={slice_index}, reference={expected_atlas_slice}"
        )

    group_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for group_id, group in frame.groupby("group_id", sort=False):
        st_ids = _ids_for_group(group, resolved_st_side)
        atlas_ids = _ids_for_group(group, atlas_side)
        if not st_ids or not atlas_ids:
            continue
        atlas_name_column = f"{atlas_side}_selected_name"
        st_name_column = f"{resolved_st_side}_selected_name"
        atlas_names = (
            group[atlas_name_column].dropna().astype(str).drop_duplicates().tolist()
            if atlas_name_column in group
            else []
        )
        st_names = (
            group[st_name_column].dropna().astype(str).drop_duplicates().tolist()
            if st_name_column in group
            else []
        )
        common = {
            "group_id": str(group_id),
            "atlas_labels_union": atlas_ids,
            "candidate_name": ";".join(atlas_names),
            "align_score": np.nan,
            "align_score_gated": np.nan,
            "gate_factor": np.nan,
        }
        group_rows.append(
            {
                **common,
                "cluster": str(group_id),
                "st_cluster_ids": st_ids,
                "pair_type": "ui_manual_group",
                "st_candidate_name": ";".join(st_names),
            }
        )
        for st_id in st_ids:
            cluster_rows.append(
                {
                    **common,
                    "cluster": str(st_id),
                    "st_cluster_ids": [st_id],
                    "pair_type": "ui_manual_cluster",
                    "st_candidate_name": f"cluster_{st_id}",
                }
            )
    deformation_groups = pd.DataFrame(group_rows)
    matched_pairs = pd.DataFrame(cluster_rows)
    if deformation_groups.empty:
        raise ValueError("No valid ST-to-Allen pairing groups were found in the UI export")
    return UIAtlasPairing(
        raw=frame,
        deformation_groups=deformation_groups,
        matched_pairs=matched_pairs,
        atlas_slice_index=slice_index,
        st_side=resolved_st_side,
        source=source,
    )


def _validate_alignment_config(config: UIAtlasAlignmentConfig) -> None:
    positive = {
        "kernel_scale": config.kernel_scale,
        "operator_power": config.operator_power,
        "time_steps": config.time_steps,
        "velocity_domain_expansion": config.velocity_domain_expansion,
        "velocity_grid_spacing": config.velocity_grid_spacing,
        "iterations": config.iterations,
        "momentum_learning_rate": config.momentum_learning_rate,
        "regularization_scale": config.regularization_scale,
        "raster_zoom_scale": config.raster_zoom_scale,
        "signed_distance_scale": config.signed_distance_scale,
    }
    invalid = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"UI Atlas alignment parameters must be positive: {invalid}")
    if config.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if config.prealignment_mode not in {"mask", "provided"}:
        raise ValueError("prealignment_mode must be 'mask' or 'provided'")
    if not config.provided_prealigned_x_key or not config.provided_prealigned_y_key:
        raise ValueError("provided pre-alignment coordinate keys must be non-empty")


def _base_atlas_config(config: UIAtlasAlignmentConfig) -> STAtlasAlignmentConfig:
    return STAtlasAlignmentConfig(
        n_levels=1,
        minimum_coarse_structures=2,
        continue_alignment=False,
        device=config.device,
        prealign_close_kernel=config.prealign_close_kernel,
        prealign_angle_step_degrees=config.prealign_angle_step_degrees,
        prealign_scale_tweak=config.prealign_scale_tweak,
        prealign_scale_steps=config.prealign_scale_steps,
        filter_base_neighbors=config.filter_base_neighbors,
        filter_detail_area_quantile=config.filter_detail_area_quantile,
        filter_detail_mad_multiplier=config.filter_detail_mad_multiplier,
        filter_normal_mad_multiplier=config.filter_normal_mad_multiplier,
        filter_grid_thinning=config.filter_grid_thinning,
        filter_detail_grid_size=config.filter_detail_grid_size,
    )


def _st_table(adata: ad.AnnData, cluster_key: str, spatial_key: str) -> pd.DataFrame:
    validate_single_sample_anndata(
        adata,
        spatial_key=spatial_key,
        cluster_key=cluster_key,
        require_cluster=True,
    )
    table = adata.obs.copy()
    coordinates = spatial_coordinates(adata, spatial_key=spatial_key)
    table["x"] = coordinates[:, 0]
    table["y"] = coordinates[:, 1]
    table[cluster_key] = table[cluster_key].astype(str)
    return table


def _initialize_st_coordinates(
    table: pd.DataFrame,
    atlas_info: dict[str, Any],
    core_config: Any,
    config: UIAtlasAlignmentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return mask-derived or externally provided pre-aligned coordinates."""
    from . import _atlas_core as core

    if config.prealignment_mode == "mask":
        return core.prealign_st_to_atlas(table.copy(), atlas_info, core_config)

    required = (
        config.provided_prealigned_x_key,
        config.provided_prealigned_y_key,
    )
    missing = [column for column in required if column not in table]
    if missing:
        raise KeyError(
            "prealignment_mode='provided' requires manual/pre-aligned "
            f"coordinates in AnnData obs columns: {missing}"
        )
    provided = table.loc[:, list(required)].to_numpy(dtype=float)
    if provided.shape != (len(table), 2) or not np.isfinite(provided).all():
        raise ValueError("Provided pre-aligned coordinates must be finite n_obs × 2 values")
    initialized = table.copy()
    initialized["x_prealigned"] = provided[:, 0]
    initialized["y_prealigned"] = provided[:, 1]
    return initialized, {
        "method": "provided_coordinates",
        "x_key": config.provided_prealigned_x_key,
        "y_key": config.provided_prealigned_y_key,
    }


def _group_source_masks(
    groups: pd.DataFrame,
    st_masks: dict[str, np.ndarray],
    atlas_annotation: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    available_st = set(str(key) for key in st_masks)
    available_atlas = set(int(value) for value in np.unique(atlas_annotation))
    masks: dict[str, np.ndarray] = {}
    rows: list[pd.Series] = []
    for _, row in groups.iterrows():
        requested_st = [str(value) for value in _parse_id_list(row["st_cluster_ids"])]
        present_st = [value for value in requested_st if value in available_st]
        requested_atlas = _parse_id_list(row["atlas_labels_union"])
        present_atlas = [value for value in requested_atlas if value in available_atlas]
        if not present_st or not present_atlas:
            continue
        mask = np.logical_or.reduce([st_masks[value] > 0 for value in present_st])
        if not mask.any():
            continue
        valid = row.copy()
        valid["st_cluster_ids_present"] = [int(value) for value in present_st]
        valid["st_cluster_ids_missing"] = sorted(
            set(int(value) for value in requested_st).difference(
                int(value) for value in present_st
            )
        )
        valid["atlas_labels_present"] = present_atlas
        valid["atlas_labels_missing"] = sorted(set(requested_atlas).difference(present_atlas))
        valid["atlas_labels_union"] = present_atlas
        masks[str(row["cluster"])] = mask.astype(np.uint8)
        rows.append(valid)
    valid_groups = pd.DataFrame(rows)
    if valid_groups.empty:
        raise ValueError(
            "No UI groups have both usable ST cluster masks and Allen labels in this slice"
        )
    return masks, valid_groups


def _area_weights(
    source: np.ndarray,
    target: np.ndarray,
    config: UIAtlasAlignmentConfig,
) -> np.ndarray:
    source_area = source.reshape(source.shape[0], -1).sum(axis=1).astype(float)
    target_area = target.reshape(target.shape[0], -1).sum(axis=1).astype(float)
    area = 0.5 * (source_area + target_area) + 1e-6
    weights = (np.median(area) / area) ** config.area_weight_power
    return np.clip(
        weights,
        config.minimum_channel_weight,
        config.maximum_channel_weight,
    ).astype(np.float32)


def _equalized_signed_distance(
    source: np.ndarray,
    target: np.ndarray,
    config: UIAtlasAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    from . import _atlas_core as core

    source_sdf = core.onehot_to_sdt(
        source,
        clip_dist=config.signed_distance_clip,
        sigma_sdt=config.signed_distance_sigma,
    ).astype(np.float32)
    target_sdf = core.onehot_to_sdt(
        target,
        clip_dist=config.signed_distance_clip,
        sigma_sdt=config.signed_distance_sigma,
    ).astype(np.float32)
    for channel in range(source_sdf.shape[0]):
        boundary = (
            np.abs(source_sdf[channel]) <= config.signed_distance_boundary_band
        ) | (
            np.abs(target_sdf[channel]) <= config.signed_distance_boundary_band
        )
        if boundary.any():
            scale = np.sqrt(
                0.5
                * (
                    np.mean(source_sdf[channel][boundary] ** 2)
                    + np.mean(target_sdf[channel][boundary] ** 2)
                )
            ) + 1e-6
            source_sdf[channel] /= scale
            target_sdf[channel] /= scale
    return (
        np.tanh(source_sdf / config.signed_distance_scale),
        np.tanh(target_sdf / config.signed_distance_scale),
    )


def _lddmm_images(
    groups: pd.DataFrame,
    st_masks: dict[str, np.ndarray],
    atlas: AllenCCFReference,
    config: UIAtlasAlignmentConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from . import _atlas_core as core

    source_onehot, target_onehot, pair_rows = core.build_pair_onehot_from_pair_df(
        pair_df=groups,
        st_masks=st_masks,
        sl=atlas.annotation,
        add_global_channel=True,
    )
    source_binary, target_binary = core.preprocess_onehot_asymmetric(
        source_onehot,
        target_onehot,
        st_cfg={
            "sigma_pre": config.source_sigma_pre,
            "thr": config.source_threshold,
            "close_r": config.source_close_radius,
            "open_r": config.source_open_radius,
            "min_area": config.source_minimum_area,
        },
        he_cfg={
            "sigma_pre": config.target_sigma_pre,
            "thr": config.target_threshold,
            "close_r": config.target_close_radius,
            "open_r": config.target_open_radius,
            "min_area": config.target_minimum_area,
        },
    )
    source_sdf, target_sdf = _equalized_signed_distance(
        source_binary,
        target_binary,
        config,
    )
    weights = _area_weights(source_binary, target_binary, config)
    weights[-1] *= config.global_channel_scale
    return (
        source_sdf * weights[:, None, None],
        target_sdf * weights[:, None, None],
        pair_rows,
    )


def _run_lddmm(
    source_image: np.ndarray,
    target_image: np.ndarray,
    atlas: AllenCCFReference,
    config: UIAtlasAlignmentConfig,
) -> dict[str, Any]:
    from . import _atlas_core as core

    device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    source_y = np.asarray(atlas.y_coordinates, dtype=float)
    source_x = np.asarray(atlas.x_coordinates, dtype=float)
    target_y = source_y.copy()
    target_x = source_x.copy()
    scale = float(config.raster_zoom_scale)
    if scale != 1.0:
        _, height, width = source_image.shape
        height_out = max(16, int(round(height * scale)))
        width_out = max(16, int(round(width * scale)))
        factors = (1, height_out / height, width_out / width)
        source_image = zoom(source_image, factors, order=1).astype(np.float32)
        target_image = zoom(target_image, factors, order=1).astype(np.float32)
        source_y = np.linspace(source_y[0], source_y[-1], height_out)
        source_x = np.linspace(source_x[0], source_x[-1], width_out)
        target_y = source_y.copy()
        target_x = source_x.copy()
    source_grid = [
        torch.as_tensor(source_y, device=device, dtype=dtype),
        torch.as_tensor(source_x, device=device, dtype=dtype),
    ]
    target_grid = [
        torch.as_tensor(target_y, device=device, dtype=dtype),
        torch.as_tensor(target_x, device=device, dtype=dtype),
    ]
    result = core.LDDMM_shooting(
        x_src=source_grid,
        source_image=torch.as_tensor(source_image, device=device, dtype=dtype),
        x_tgt=target_grid,
        target_image=torch.as_tensor(target_image, device=device, dtype=dtype),
        model_cfg={
            "a": config.kernel_scale,
            "p": config.operator_power,
            "nt": config.time_steps,
            "expand": config.velocity_domain_expansion,
            "grid_step": config.velocity_grid_spacing,
        },
        optim_cfg={
            "niter": config.iterations,
            "diffeo_start": config.diffeomorphic_start_iteration,
            "lrL": config.affine_matrix_learning_rate,
            "lrT": config.translation_learning_rate,
            "lrM": config.momentum_learning_rate,
            "affine_slowdown": config.affine_slowdown,
            "lrM_decay": config.momentum_learning_rate_decay,
            "lrM_min": config.minimum_momentum_learning_rate,
            # The validated UI-based run reports the transformation after the
            # requested final iteration. Keep this explicit because the shared
            # package core otherwise restores its lowest-energy checkpoint.
            "restore_best": config.restore_best_checkpoint,
        },
        em_cfg={
            "update_every": config.em_update_every,
            "start_iter": config.em_start_iteration,
        },
        intensity_cfg={
            "loss_mode": "feature",
            "sigmaM": config.match_scale,
            "sigmaB": config.background_scale,
            "sigmaA": config.artifact_scale,
            "sigmaR": config.regularization_scale,
            "sigmaP": config.landmark_scale,
        },
        device=device,
        dtype=dtype,
        verbose=config.verbose,
        print_every=config.print_every,
    )
    result["device"] = device
    return result


def _map_points(table: pd.DataFrame, lddmm: dict[str, Any]) -> pd.DataFrame:
    from . import _atlas_core as core

    result = table.copy()
    points = result[["y_prealigned", "x_prealigned"]].to_numpy(dtype=float)
    mapped = core.map_points_source_to_target(
        lddmm["xv"],
        lddmm["v"],
        lddmm["A"],
        points,
    )
    if torch.is_tensor(mapped):
        mapped = mapped.detach().cpu().numpy()
    result["y_aligned"] = mapped[:, 0]
    result["x_aligned"] = mapped[:, 1]
    return result


def _safe_pair_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include="object"):
        result[column] = result[column].map(
            lambda value: ";".join(str(item) for item in value)
            if isinstance(value, (list, tuple, set, np.ndarray))
            else value
        )
    return result


def _coordinate_summary(
    original: pd.DataFrame,
    prealigned: pd.DataFrame,
    aligned: pd.DataFrame,
    cluster_key: str,
) -> pd.DataFrame:
    output = pd.DataFrame(index=aligned.index)
    output.index.name = aligned.index.name or "cell_id"
    output["original_x"] = original.reindex(output.index)["x"]
    output["original_y"] = original.reindex(output.index)["y"]
    output[f"original_{cluster_key}"] = original.reindex(output.index)[cluster_key]
    output["prealign_x"] = prealigned.reindex(output.index)["x_prealigned"]
    output["prealign_y"] = prealigned.reindex(output.index)["y_prealigned"]
    output["aligned_x"] = aligned.reindex(output.index)["x_aligned"]
    output["aligned_y"] = aligned.reindex(output.index)["y_aligned"]
    return output


def align_st_to_allen_atlas_from_ui_pairs(
    adata: ad.AnnData,
    atlas: AllenCCFReference,
    ui_pairs: UIAtlasPairing | pd.DataFrame | str | Path,
    *,
    config: UIAtlasAlignmentConfig | None = None,
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    st_side: Literal["auto", "left", "right"] = "auto",
    output_dir: str | Path = "st_to_atlas_ui_output",
    copy: bool = True,
    save_outputs: bool = True,
) -> STAtlasAlignmentResult:
    """Align clustered ST to Allen CCF using a UI-exported pairing CSV.

    Whole-tissue mask pre-alignment is computed from AnnData by default; set
    ``prealignment_mode='provided'`` to continue from separately prepared
    manual pre-alignment columns. Automatic pair discovery, candidate scoring
    and pair matching are always skipped: each accepted UI ``group_id`` becomes
    one unioned signed-distance channel used by a single S-LDDMM stage. Point
    filtering, mask construction and processing, S-LDDMM input construction,
    deformation and label transfer still run in both pre-alignment modes.
    """
    config = config or UIAtlasAlignmentConfig()
    _validate_alignment_config(config)
    pairing = (
        ui_pairs
        if isinstance(ui_pairs, UIAtlasPairing)
        else load_ui_atlas_pairing(
            ui_pairs,
            st_side=st_side,
            expected_atlas_slice=atlas.slice_index,
        )
    )
    if (
        pairing.atlas_slice_index is not None
        and pairing.atlas_slice_index != atlas.slice_index
    ):
        raise ValueError(
            f"UI export uses slice {pairing.atlas_slice_index}, but Atlas reference "
            f"uses slice {atlas.slice_index}"
        )
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    table = _st_table(adata, cluster_key, spatial_key)

    from . import _atlas_core as core

    base_config = _base_atlas_config(config)
    core_config = _core_config(
        base_config,
        atlas=atlas,
        cluster_key=cluster_key,
        hierarchy_columns=(),
        output_dir=output_path,
    )
    core_config.dtype = torch.float64 if config.dtype == "float64" else torch.float32
    atlas_info = _atlas_info(atlas)
    prealigned, prealignment_parameters = _initialize_st_coordinates(
        table,
        atlas_info,
        core_config,
        config,
    )
    filtered, removed, filter_stats = core.filter_prealigned_st_points(
        prealigned,
        label_col=cluster_key,
        config=core_config,
    )
    mask_result = core.build_cluster_masks(
        df_smooth=filtered,
        sl=atlas.annotation,
        xJ=atlas.x_coordinates,
        yJ=atlas.y_coordinates,
        x_col="x_prealigned",
        y_col="y_prealigned",
        label_col=cluster_key,
        params=core.DEFAULT_MASK_PARAMS,
        params_thin=core.MASK_PARAMS_THIN,
        shape_type_col="shape_type",
        thin_values=("detail",),
        thin_rule="mode",
        verbose=config.verbose,
    )
    group_masks, valid_groups = _group_source_masks(
        pairing.deformation_groups,
        mask_result["st_masks"],
        atlas.annotation,
    )
    valid_group_ids = set(valid_groups["group_id"].astype(str))
    matched_pairs = pairing.matched_pairs[
        pairing.matched_pairs["group_id"].astype(str).isin(valid_group_ids)
    ].copy()
    valid_atlas_labels = valid_groups.set_index(
        valid_groups["group_id"].astype(str)
    )["atlas_labels_union"].to_dict()
    matched_pairs["atlas_labels_union"] = matched_pairs["group_id"].astype(str).map(
        valid_atlas_labels
    )
    source_image, target_image, pair_rows = _lddmm_images(
        valid_groups,
        group_masks,
        atlas,
        config,
    )
    lddmm = _run_lddmm(source_image, target_image, atlas, config)
    lddmm["manual_pair_rows"] = pair_rows
    aligned = _map_points(prealigned, lddmm)
    filtered_aligned = _map_points(filtered, lddmm)
    labelled = core.transfer_all_atlas_labels_to_st(
        aligned,
        sl=atlas.annotation,
        xJ=atlas.x_coordinates,
        yJ=atlas.y_coordinates,
        atlas_metadata=atlas.structures,
    )

    source_adata = adata.to_memory() if adata.isbacked else adata
    output = source_adata.copy() if copy else source_adata
    prealigned_ordered = prealigned.reindex(output.obs_names)
    labelled_ordered = labelled.reindex(output.obs_names)
    for column in ("x_prealigned", "y_prealigned"):
        output.obs[column] = prealigned_ordered[column].to_numpy(dtype=float)
    for column in ("x_aligned", "y_aligned"):
        output.obs[column] = labelled_ordered[column].to_numpy(dtype=float)
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
        if column in labelled_ordered:
            output.obs[column] = labelled_ordered[column].to_numpy()

    stage_summary = pd.DataFrame(
        [
            {
                "stage": 1,
                "label_col": cluster_key,
                "is_final_stage": True,
                "n_pairs": int(len(matched_pairs)),
                "n_manual_groups": int(len(valid_groups)),
                "pair_source": "ui_export",
                "pair_matching_ran": False,
                "mask_construction_ran": True,
                "lddmm_input_built": True,
                "lddmm_ran": True,
            }
        ]
    )
    iter_pairs_all = matched_pairs.copy()
    iter_pairs_all.insert(0, "stage", 1)
    context = {
        "config": core_config,
        "df1": table,
        "df2": atlas.structures,
        "atlas_info": atlas_info,
        "sl": atlas.annotation,
        "xJ": atlas.x_coordinates,
        "yJ": atlas.y_coordinates,
        "H": atlas.annotation.shape[0],
        "W": atlas.annotation.shape[1],
        "df_final": prealigned,
        "df_final_filtered": filtered,
        "df_removed": removed,
        "df_pre_iter_all": prealigned.copy(),
        "df_prealign_nofilter": prealigned.copy(),
        "df_smooth": filtered_aligned,
        "df_aligned_all": aligned,
        "prealign_params": prealignment_parameters,
        "pair_df": matched_pairs,
        "final_pair_df": matched_pairs,
        "iter_stage_outputs": [
            {
                "stage": 1,
                "label_col": cluster_key,
                "n_pairs": len(matched_pairs),
                "pair_df": matched_pairs,
                "grouped_pair_df": valid_groups,
                "result": {"mask_result": mask_result, "atlas_info": atlas_info},
                "lddmm_out": lddmm,
                "is_final_stage": True,
                "stage_w_align": {},
            }
        ],
        "iter_pairs_by_stage": {1: matched_pairs},
        "iter_pairs_all_df": iter_pairs_all,
        "iter_pair_summary_df": stage_summary,
        "CONTINUE_LABEL_COL": cluster_key,
        "A": lddmm["A"],
        "v": lddmm["v"],
        "xv": lddmm["xv"],
    }

    output.uns.pop("spalignde", None)
    output.uns.setdefault("spAlignDE", {})
    output.uns["spAlignDE"]["st_to_allen_atlas"] = {
        "pairing_mode": "ui_curated",
        "pair_source": "ui_export",
        "pair_matching_ran": False,
        "prealignment_mode": config.prealignment_mode,
        "mask_construction_ran": True,
        "lddmm_input_built": True,
        "cluster_key": cluster_key,
        "spatial_key": spatial_key,
        "slice_index": atlas.slice_index,
        "ui_pairing_source": pairing.source,
        "ui_st_side": pairing.st_side,
        "n_ui_rows": int(len(pairing.raw)),
        "n_ui_groups": int(len(valid_groups)),
        "n_matched_pairs": int(len(matched_pairs)),
        "prealignment_parameters": _anndata_safe_metadata(prealignment_parameters),
        "ui_alignment_config": _anndata_safe_metadata(asdict(config)),
        "output_dir": str(output_path),
    }

    if save_outputs:
        core.save_alignment_visualization(context, output_path)
        core.save_alignment_outputs(context, output_path)
        pairing.raw.to_csv(output_path / "ui_pairs_raw.csv", index=False)
        _safe_pair_table(pairing.deformation_groups).to_csv(
            output_path / "manual_pairs_grouped_for_lddmm.csv", index=False
        )
        _safe_pair_table(valid_groups).to_csv(
            output_path / "manual_pairs_grouped_valid_for_lddmm.csv", index=False
        )
        _safe_pair_table(matched_pairs).to_csv(
            output_path / "manual_pairs_per_st_cluster.csv", index=False
        )
        filter_stats.to_csv(output_path / "manual_filter_stats.csv", index=False)
        _coordinate_summary(table, prealigned, aligned, cluster_key).to_csv(
            output_path / "coordinates_original_prealign_aligned.csv"
        )
        output.write_h5ad(output_path / "st_to_allen_atlas_ui_aligned.h5ad")
        core.build_white_label_color_map_for_atlas(atlas.annotation, seed=0).to_csv(
            output_path
            / f"atlas_z{atlas.slice_index}_white_label_color_map_for_transfer_labels.csv",
            index=False,
        )

    return STAtlasAlignmentResult(
        adata=output,
        atlas=atlas,
        matched_pairs=matched_pairs,
        stage_summary=stage_summary,
        prealignment_parameters=dict(prealignment_parameters),
        hierarchy_columns=(),
        output_dir=output_path if save_outputs else None,
        context=context,
    )


__all__ = [
    "UIAtlasAlignmentConfig",
    "UIAtlasPairing",
    "align_st_to_allen_atlas_from_ui_pairs",
    "load_ui_atlas_pairing",
]
