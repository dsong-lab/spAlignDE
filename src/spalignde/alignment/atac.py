"""Structure-guided alignment of a partial spatial ATAC assay to ST."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage import morphology
from skimage.measure import label as connected_components

from ..io import spatial_coordinates, validate_single_sample_anndata
from .cross_sample import ManualPrealignmentConfig, apply_similarity_transform


@dataclass(frozen=True)
class ATACSTPrealignmentConfig:
    """Manual whole-tissue initialization and ST reference selection.

    Tune the two similarity transforms before choosing the partial-reference
    crop. ``raster_scale`` defines the downstream canvas units and therefore
    changes the interpretation of all distance-valued alignment parameters.
    """

    st_transform: ManualPrealignmentConfig = ManualPrealignmentConfig(
        scale=1.0,
        theta_deg=-125.0,
        translation_x=0.0,
        translation_y=0.0,
    )
    atac_transform: ManualPrealignmentConfig = ManualPrealignmentConfig(
        scale=1.6,
        theta_deg=-90.0,
        translation_x=-2600.0,
        translation_y=-3300.0,
    )
    reference_crop_axis: str = "x"
    reference_crop_side: str = "left"
    reference_crop_quantile: float = 0.5
    raster_scale: float = 0.25
    canvas_padding: int = 10


@dataclass
class ATACSTPrealignmentResult:
    """Pre-aligned ATAC query and fixed, cropped ST reference."""

    atac: ad.AnnData
    st_reference: ad.AnnData
    params: dict[str, Any]
    canvas_shape_hw: tuple[int, int]


@dataclass(frozen=True)
class ATACSTAlignmentConfig:
    """Automatic structure pairing and S-LDDMM settings for ATAC-to-ST.

    Pairing weights define one global geometric rule and must be finite,
    non-negative, and sum to one. Channel-area settings balance narrow and
    broad masks without region-specific weights.
    ``kernel_scale`` and ``velocity_grid_spacing`` are legacy ``a`` and
    ``grid_step`` in the cropped raster-canvas coordinate system.
    """

    sdf_weight: float = 0.35
    chamfer_weight: float = 0.25
    area_weight: float = 0.30
    dice_weight: float = 0.10
    chamfer_gate_center: float = 16.0
    chamfer_gate_scale: float = 6.0
    pair_score_threshold: float = 0.25
    pair_dice_threshold: float = 0.01
    maximum_pairs: int = 20
    minimum_atac_points: int = 80
    slender_quantile: float = 0.85
    density_neighbors: int = 5
    density_mad_multiplier: float = 3.0
    sdt_smoothing: float = 1.0
    sdt_clip_distance: float = 25.0
    channel_area_power: float = 0.90
    channel_weight_temperature: float = 0.88
    channel_uniform_mix: float = 0.08
    time_steps: int = 8
    iterations: int = 500
    diffeomorphic_start: int = 20
    kernel_scale: float = 100.0
    kernel_power: float = 2.0
    velocity_expand: float = 2.0
    velocity_grid_spacing: float = 40.0
    affine_linear_lr: float = 2e-11
    affine_translation_lr: float = 2e-5
    momentum_lr: float = 1e3
    momentum_gradient_clip: float = 1.0
    rollback_on_energy_rise: bool = True
    rollback_factor: float = 0.5
    rollback_patience: int = 6
    minimum_energy_improvement: float = 1e-5
    minimum_affine_determinant: float = 1e-4
    deformation_regularization: float = 1e6
    matching_scale: float = 0.5
    device: str | None = None
    dtype: str = "float32"


@dataclass
class ATACSTAlignmentResult:
    """Aligned ATAC AnnData, fixed ST reference, and pairing diagnostics."""

    atac: ad.AnnData
    st_reference: ad.AnnData
    matched_pairs: pd.DataFrame
    prealignment_parameters: dict[str, Any]
    output_dir: Path | None = None
    context: dict[str, Any] | None = None


_ST_SLENDER_MASK = {
    "threshold": 0.08,
    "sigma": 1.8,
    "sigma_min": 1.0,
    "sigma_max": 5.0,
    "sigma_scale": 1.0,
    "close_radius": 2,
    "refine_close_radius": 2,
    "refine_open_radius": 1,
    "maximum_smoothing": 1.2,
    "minimum_size": 100,
}

_ST_NORMAL_MASK = {**_ST_SLENDER_MASK, "threshold": 0.20}

_ATAC_SLENDER_MASK = {
    "threshold": 0.04,
    "sigma": 1.0,
    "sigma_min": 0.8,
    "sigma_max": 4.0,
    "sigma_scale": 0.8,
    "close_radius": 1,
    "refine_close_radius": 1,
    "refine_open_radius": 0,
    "maximum_smoothing": 2.0,
    "minimum_size": 50,
}

_ATAC_NORMAL_MASK = {
    "threshold": 0.06,
    "sigma": 2.6,
    "sigma_min": 1.5,
    "sigma_max": 10.0,
    "sigma_scale": 1.4,
    "close_radius": 3,
    "refine_close_radius": 3,
    "refine_open_radius": 1,
    "maximum_smoothing": 6.0,
    "minimum_size": 120,
}


def _metadata_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _metadata_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_metadata_ready(payload), indent=2), encoding="utf-8")


def _crop_reference(
    coordinates: np.ndarray,
    *,
    axis: str,
    side: str,
    quantile: float,
) -> tuple[np.ndarray, float]:
    axis = axis.strip().lower()
    side = side.strip().lower()
    if axis not in {"x", "y"}:
        raise ValueError("reference_crop_axis must be 'x' or 'y'")
    if not 0.0 < quantile < 1.0:
        raise ValueError("reference_crop_quantile must lie strictly between 0 and 1")
    values = coordinates[:, 0 if axis == "x" else 1]
    split = float(np.quantile(values, quantile))
    if (axis, side) in {("x", "left"), ("y", "bottom")}:
        keep = values <= split
    elif (axis, side) in {("x", "right"), ("y", "top")}:
        keep = values >= split
    else:
        valid = "left/right" if axis == "x" else "bottom/top"
        raise ValueError(f"reference_crop_side must be {valid} for axis={axis!r}")
    return keep, split


def prealign_atac_to_st(
    atac: ad.AnnData,
    st_reference: ad.AnnData,
    *,
    config: ATACSTPrealignmentConfig | None = None,
    atac_cluster_key: str = "cluster",
    st_cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    copy: bool = True,
) -> ATACSTPrealignmentResult:
    """Place a partial ATAC query and the matching ST half brain on one grid."""
    config = config or ATACSTPrealignmentConfig()
    validate_single_sample_anndata(
        atac,
        spatial_key=spatial_key,
        cluster_key=atac_cluster_key,
        require_cluster=True,
    )
    validate_single_sample_anndata(
        st_reference,
        spatial_key=spatial_key,
        cluster_key=st_cluster_key,
        require_cluster=True,
    )
    if config.raster_scale <= 0:
        raise ValueError("raster_scale must be positive")
    if config.canvas_padding < 0:
        raise ValueError("canvas_padding cannot be negative")

    atac_source = atac.to_memory() if atac.isbacked else atac
    st_source = st_reference.to_memory() if st_reference.isbacked else st_reference
    atac_output = atac_source.copy() if copy else atac_source
    st_output = st_source.copy() if copy else st_source

    atac_rough = apply_similarity_transform(
        spatial_coordinates(atac_output, spatial_key=spatial_key),
        config.atac_transform,
    )
    st_rough_all = apply_similarity_transform(
        spatial_coordinates(st_output, spatial_key=spatial_key),
        config.st_transform,
    )
    keep, split = _crop_reference(
        st_rough_all,
        axis=config.reference_crop_axis,
        side=config.reference_crop_side,
        quantile=config.reference_crop_quantile,
    )
    st_output = st_output[keep].copy()
    st_rough = st_rough_all[keep]

    scaled_reference = st_rough * float(config.raster_scale)
    scaled_atac = atac_rough * float(config.raster_scale)
    minimum = scaled_reference.min(axis=0)
    maximum = scaled_reference.max(axis=0)
    padding = float(config.canvas_padding)
    reference_canvas = scaled_reference - minimum + padding
    atac_canvas = scaled_atac - minimum + padding
    width = int(np.ceil(maximum[0] - minimum[0])) + 1 + 2 * config.canvas_padding
    height = int(np.ceil(maximum[1] - minimum[1])) + 1 + 2 * config.canvas_padding

    for output, coordinates in (
        (atac_output, atac_canvas),
        (st_output, reference_canvas),
    ):
        output.obs["x_prealigned"] = coordinates[:, 0]
        output.obs["y_prealigned"] = coordinates[:, 1]
        output.obs["x_aligned"] = coordinates[:, 0]
        output.obs["y_aligned"] = coordinates[:, 1]
        output.uns.pop("spalignde", None)

    params = {
        "method": "manual_similarity_with_reference_half_crop",
        "atac_transform": asdict(config.atac_transform),
        "st_transform": asdict(config.st_transform),
        "reference_crop_axis": config.reference_crop_axis,
        "reference_crop_side": config.reference_crop_side,
        "reference_crop_quantile": float(config.reference_crop_quantile),
        "reference_crop_split_before_raster_scale": split,
        "reference_observations_before_crop": int(st_reference.n_obs),
        "reference_observations_after_crop": int(st_output.n_obs),
        "raster_scale": float(config.raster_scale),
        "canvas_padding": int(config.canvas_padding),
        "canvas_offset_xy": minimum.tolist(),
        "canvas_shape_hw": [height, width],
        "atac_cluster_key": atac_cluster_key,
        "st_cluster_key": st_cluster_key,
        "spatial_key": spatial_key,
    }
    atac_output.uns.setdefault("spAlignDE", {}).setdefault(
        "atac_to_st", {}
    )["prealignment"] = _metadata_ready(params)
    st_output.uns.setdefault("spAlignDE", {}).setdefault(
        "atac_to_st_reference", {}
    )["preprocessing"] = _metadata_ready(params)
    return ATACSTPrealignmentResult(
        atac=atac_output,
        st_reference=st_output,
        params=params,
        canvas_shape_hw=(height, width),
    )


def _density_statistics(
    frame: pd.DataFrame,
    *,
    label_key: str,
    x_key: str,
    y_key: str,
    neighbors: int,
    mad_multiplier: float,
) -> pd.DataFrame:
    rows = []
    for label_value, subset in frame.groupby(label_key, observed=True):
        points = subset[[x_key, y_key]].to_numpy(float)
        n_points = len(points)
        k = min(max(int(neighbors), 1), max(n_points - 1, 1))
        if n_points <= 2:
            distances = np.zeros(n_points, dtype=float)
        else:
            queried, _ = cKDTree(points).query(points, k=k + 1)
            distances = queried[:, -1]
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        if mad <= 1e-12:
            threshold = max(median, float(np.quantile(distances, 0.75)))
        else:
            threshold = median + mad_multiplier * 1.4826 * mad
        removed = int(np.sum(distances > threshold))
        rows.append(
            {
                "cluster": str(label_value),
                "observations": int(n_points),
                "removed": removed,
                "removed_fraction": float(removed / max(n_points, 1)),
                "neighbors": int(k),
                "distance_median": median,
                "distance_mad": mad,
                "distance_threshold": float(threshold),
            }
        )
    return pd.DataFrame(rows)


def _component_statistics(mask: np.ndarray) -> tuple[int, float]:
    labels = connected_components(mask.astype(bool), connectivity=2)
    n_components = int(labels.max())
    if n_components == 0:
        return 0, 0.0
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return n_components, float(counts.max() / max(counts.sum(), 1))


def _keep_components(
    mask: np.ndarray,
    *,
    cumulative_fraction: float,
    maximum_components: int,
    minimum_area: int,
) -> np.ndarray:
    labels = connected_components(mask.astype(bool), connectivity=2)
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return mask.astype(np.uint8)
    counts[0] = 0
    identifiers = np.flatnonzero(counts >= int(minimum_area))
    identifiers = identifiers[identifiers != 0]
    if not len(identifiers):
        return np.zeros_like(mask, dtype=np.uint8)
    identifiers = identifiers[np.argsort(counts[identifiers])[::-1]]
    areas = counts[identifiers]
    cumulative = np.cumsum(areas) / max(float(areas.sum()), 1.0)
    keep_count = int(np.searchsorted(cumulative, cumulative_fraction) + 1)
    keep_count = int(np.clip(keep_count, 1, maximum_components))
    return np.isin(labels, identifiers[:keep_count]).astype(np.uint8)


def _cluster_mask(
    points: np.ndarray,
    *,
    shape: tuple[int, int],
    parameters: dict[str, float],
    cleanup_largest_fraction: float,
    cleanup_cumulative_fraction: float,
    cleanup_maximum_components: int,
) -> np.ndarray:
    height, width = shape
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    sigma = float(parameters["sigma"])
    if len(points) >= 5:
        distances, _ = cKDTree(np.stack([y, x], axis=1)).query(
            np.stack([y, x], axis=1),
            k=5,
        )
        sigma = float(
            np.clip(
                parameters["sigma_scale"] * np.median(distances[:, -1]),
                parameters["sigma_min"],
                parameters["sigma_max"],
            )
        )
    accumulator = np.zeros(shape, dtype=np.float32)
    np.add.at(accumulator, (y, x), 1.0)
    accumulator = ndimage.gaussian_filter(accumulator, sigma=sigma)
    if accumulator.max() > 0:
        accumulator /= accumulator.max()
    mask = accumulator >= float(parameters["threshold"])
    close_radius = int(parameters["close_radius"])
    if close_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(close_radius))
    mask = ndimage.binary_fill_holes(mask)
    raw_area = int(mask.sum())
    if raw_area == 0:
        return mask.astype(np.uint8)
    refine_close = int(parameters["refine_close_radius"])
    refine_open = int(parameters["refine_open_radius"])
    if refine_close > 0:
        mask = morphology.binary_closing(mask, morphology.disk(refine_close))
    if refine_open > 0:
        mask = morphology.binary_opening(mask, morphology.disk(refine_open))
    smoothing = int(
        np.clip(
            np.sqrt(raw_area) / 15.0,
            1,
            float(parameters["maximum_smoothing"]),
        )
    )
    mask = ndimage.gaussian_filter(mask.astype(np.float32), smoothing) >= 0.5
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(
        mask,
        min_size=int(parameters["minimum_size"]),
    )
    n_components, largest_fraction = _component_statistics(mask)
    if n_components >= 3 or largest_fraction < cleanup_largest_fraction:
        minimum_size = int(max(200, 0.002 * mask.sum()))
        mask = morphology.remove_small_objects(mask, min_size=minimum_size)
        mask = _keep_components(
            mask,
            cumulative_fraction=cleanup_cumulative_fraction,
            maximum_components=cleanup_maximum_components,
            minimum_area=int(max(80, 0.001 * mask.sum())),
        )
    return mask.astype(np.uint8)


def _rasterize_structures(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    shape: tuple[int, int],
    config: ATACSTAlignmentConfig,
    role: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    frame = adata.obs[[cluster_key, "x_prealigned", "y_prealigned"]].copy()
    frame[cluster_key] = frame[cluster_key].astype(str)
    stats = _density_statistics(
        frame,
        label_key=cluster_key,
        x_key="x_prealigned",
        y_key="y_prealigned",
        neighbors=config.density_neighbors,
        mad_multiplier=config.density_mad_multiplier,
    )
    slender_cutoff = float(
        stats["removed_fraction"].quantile(config.slender_quantile)
    )
    stats["shape_type"] = np.where(
        stats["removed_fraction"] >= slender_cutoff,
        "slender",
        "normal",
    )
    shape_types = dict(zip(stats["cluster"], stats["shape_type"], strict=False))
    if role == "st":
        minimum_points = 30
        slender_parameters = _ST_SLENDER_MASK
        normal_parameters = _ST_NORMAL_MASK
        largest_fraction = 0.80
        cumulative_fraction = 0.95
    elif role == "atac":
        minimum_points = config.minimum_atac_points
        slender_parameters = _ATAC_SLENDER_MASK
        normal_parameters = _ATAC_NORMAL_MASK
        largest_fraction = 0.75
        cumulative_fraction = 0.99
    else:
        raise ValueError("role must be 'atac' or 'st'")

    masks: dict[str, np.ndarray] = {}
    mask_areas: dict[str, int] = {}
    for label_value, subset in frame.groupby(cluster_key, observed=True):
        label_text = str(label_value)
        if len(subset) < minimum_points:
            continue
        parameters = (
            slender_parameters
            if shape_types.get(label_text) == "slender"
            else normal_parameters
        )
        mask = _cluster_mask(
            subset[["x_prealigned", "y_prealigned"]].to_numpy(float),
            shape=shape,
            parameters=parameters,
            cleanup_largest_fraction=largest_fraction,
            cleanup_cumulative_fraction=cumulative_fraction,
            cleanup_maximum_components=2,
        )
        if mask.any():
            masks[label_text] = mask
            mask_areas[label_text] = int(mask.sum())
    stats["mask_area"] = stats["cluster"].map(mask_areas)
    return masks, stats


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(first.sum() + second.sum())
    if denominator == 0:
        return 0.0
    return float(2.0 * np.logical_and(first, second).sum() / denominator)


def _area_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_area = float(first.sum()) + 1e-9
    second_area = float(second.sum()) + 1e-9
    return float(np.exp(-abs(np.log(first_area / second_area))))


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    eroded = ndimage.binary_erosion(binary, structure=np.ones((3, 3), bool))
    return binary ^ eroded


def _chamfer(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    first_boundary = _boundary(first)
    second_boundary = _boundary(second)
    if not first_boundary.any() or not second_boundary.any():
        return 0.0, float("inf")
    distance_first = distance_transform_edt(~first_boundary)
    distance_second = distance_transform_edt(~second_boundary)
    distance = 0.5 * (
        float(distance_second[first_boundary].mean())
        + float(distance_first[second_boundary].mean())
    )
    return float(np.exp(-distance / 30.0)), distance


def _sdf_correlation(first: np.ndarray, second: np.ndarray) -> float:
    def signed_distance(mask: np.ndarray) -> np.ndarray:
        binary = mask.astype(bool)
        return distance_transform_edt(binary) - distance_transform_edt(~binary)

    first_sdf = signed_distance(first)
    second_sdf = signed_distance(second)
    band = (np.abs(first_sdf) <= 20) | (np.abs(second_sdf) <= 20)
    first_values = first_sdf[band]
    second_values = second_sdf[band]
    if (
        first_values.size < 10
        or first_values.std() == 0
        or second_values.std() == 0
    ):
        return 0.0
    correlation = float(np.corrcoef(first_values, second_values)[0, 1])
    return float((correlation + 1.0) / 2.0)


def _atac_pairing_weights(config: ATACSTAlignmentConfig) -> dict[str, float]:
    """Return the normalized ATAC--ST composite-score weights."""
    weights = {
        "sdf_corr": float(config.sdf_weight),
        "chamfer_sim": float(config.chamfer_weight),
        "area_sim": float(config.area_weight),
        "dice": float(config.dice_weight),
    }
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("ATAC pairing weights must be finite and non-negative")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-8):
        raise ValueError("ATAC pairing weights must sum to 1.0")
    return weights


def _pair_structures(
    st_masks: dict[str, np.ndarray],
    atac_masks: dict[str, np.ndarray],
    config: ATACSTAlignmentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = _atac_pairing_weights(config)
    rows = []
    for st_label, st_mask in st_masks.items():
        for atac_label, atac_mask in atac_masks.items():
            dice = _dice(st_mask, atac_mask)
            area_similarity = _area_similarity(st_mask, atac_mask)
            chamfer_similarity, chamfer_distance = _chamfer(st_mask, atac_mask)
            sdf_correlation = _sdf_correlation(st_mask, atac_mask)
            base_score = (
                weights["sdf_corr"] * sdf_correlation
                + weights["chamfer_sim"] * chamfer_similarity
                + weights["area_sim"] * area_similarity
                + weights["dice"] * dice
            )
            if np.isfinite(chamfer_distance):
                exponent = np.clip(
                    (chamfer_distance - config.chamfer_gate_center)
                    / config.chamfer_gate_scale,
                    -60.0,
                    60.0,
                )
                distance_gate = 1.0 / (1.0 + np.exp(exponent))
            else:
                distance_gate = 0.0
            score = float(np.clip(base_score * (0.5 + 0.5 * distance_gate), 0, 1))
            rows.append(
                {
                    "st_structure": st_label,
                    "atac_structure": atac_label,
                    "align_score": score,
                    "base_score": float(base_score),
                    "distance_gate": float(distance_gate),
                    "sdf_corr": sdf_correlation,
                    "chamfer_sim": chamfer_similarity,
                    "chamfer_distance": chamfer_distance,
                    "area_sim": area_similarity,
                    "dice": dice,
                    "st_area": int(st_mask.sum()),
                    "atac_area": int(atac_mask.sum()),
                }
            )
    scores = pd.DataFrame(rows).sort_values(
        "align_score", ascending=False
    ).reset_index(drop=True)
    accepted = scores[
        (scores["align_score"] >= config.pair_score_threshold)
        & (scores["dice"] >= config.pair_dice_threshold)
    ]
    selected_rows = []
    used_st: set[str] = set()
    used_atac: set[str] = set()
    for _, row in accepted.iterrows():
        st_label = str(row["st_structure"])
        atac_label = str(row["atac_structure"])
        if st_label in used_st or atac_label in used_atac:
            continue
        selected_rows.append(row)
        used_st.add(st_label)
        used_atac.add(atac_label)
        if len(selected_rows) >= config.maximum_pairs:
            break
    matched = pd.DataFrame(selected_rows).reset_index(drop=True)
    return scores, matched


def _onehot_to_sdt(
    onehot: np.ndarray,
    *,
    clip_distance: float,
    smoothing: float,
) -> np.ndarray:
    result = np.zeros_like(onehot, dtype=np.float32)
    for channel, mask in enumerate(onehot > 0):
        inside = distance_transform_edt(mask)
        outside = distance_transform_edt(~mask)
        field = outside - inside
        if smoothing > 0:
            field = ndimage.gaussian_filter(field, smoothing)
        result[channel] = np.clip(
            field, -clip_distance, clip_distance
        ) / clip_distance
    return result


def _channel_weights(
    target_onehot: np.ndarray,
    config: ATACSTAlignmentConfig,
) -> np.ndarray:
    area = target_onehot.reshape(target_onehot.shape[0], -1).sum(axis=1)
    weights = 1.0 / np.power(area + 1e-6, config.channel_area_power)
    weights /= weights.mean() + 1e-6
    weights = np.clip(weights, 0.25, 4.5)
    weights /= weights.mean() + 1e-6
    weights = np.power(weights, config.channel_weight_temperature)
    weights /= weights.mean() + 1e-6
    weights = (
        (1.0 - config.channel_uniform_mix) * weights
        + config.channel_uniform_mix
    )
    return (weights / (weights.mean() + 1e-6)).astype(np.float32)


def align_atac_to_st(
    prealigned: ATACSTPrealignmentResult,
    *,
    config: ATACSTAlignmentConfig | None = None,
    atac_cluster_key: str = "cluster",
    st_cluster_key: str = "cluster",
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> ATACSTAlignmentResult:
    """Pair independently inferred ATAC/ST structures and run S-LDDMM."""
    import torch

    from . import _slddmm_core as lddmm

    config = config or ATACSTAlignmentConfig()
    atac = prealigned.atac.copy()
    st_reference = prealigned.st_reference.copy()
    shape = prealigned.canvas_shape_hw
    atac_masks, atac_mask_stats = _rasterize_structures(
        atac,
        cluster_key=atac_cluster_key,
        shape=shape,
        config=config,
        role="atac",
    )
    st_masks, st_mask_stats = _rasterize_structures(
        st_reference,
        cluster_key=st_cluster_key,
        shape=shape,
        config=config,
        role="st",
    )
    pair_scores, matched = _pair_structures(st_masks, atac_masks, config)
    if matched.empty:
        raise RuntimeError(
            "No ATAC/ST structure pair passed the global score and Dice gates. "
            "Inspect the manual pre-alignment before changing pairing weights."
        )

    source_onehot = np.stack(
        [atac_masks[str(value)] for value in matched["atac_structure"]]
    ).astype(np.uint8)
    target_onehot = np.stack(
        [st_masks[str(value)] for value in matched["st_structure"]]
    ).astype(np.uint8)
    source_sdt = _onehot_to_sdt(
        source_onehot,
        clip_distance=config.sdt_clip_distance,
        smoothing=config.sdt_smoothing,
    )
    target_sdt = _onehot_to_sdt(
        target_onehot,
        clip_distance=config.sdt_clip_distance,
        smoothing=config.sdt_smoothing,
    )
    weights = _channel_weights(target_onehot, config)
    source_image = source_sdt * weights[:, None, None]
    target_image = target_sdt * weights[:, None, None]

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    height, width = shape
    grid_y = np.arange(height, dtype=float)
    grid_x = np.arange(width, dtype=float)
    result = lddmm.LDDMM_shooting(
        [grid_y, grid_x],
        source_image,
        [grid_y, grid_x],
        target_image,
        model_cfg={
            "nt": config.time_steps,
            "a": config.kernel_scale,
            "p": config.kernel_power,
            "expand": config.velocity_expand,
            "grid_step": config.velocity_grid_spacing,
        },
        optim_cfg={
            "niter": config.iterations,
            "diffeo_start": config.diffeomorphic_start,
            "lrL": config.affine_linear_lr,
            "lrT": config.affine_translation_lr,
            "lrM": config.momentum_lr,
            "lrM_decay": 1.0,
            "lrM_min": config.momentum_lr,
            "grad_clip_m0": config.momentum_gradient_clip,
            "rollback_on_rise": config.rollback_on_energy_rise,
            "rollback_factor": config.rollback_factor,
            "rollback_patience": config.rollback_patience,
            "minimum_energy_improvement": config.minimum_energy_improvement,
            "minimum_affine_determinant": config.minimum_affine_determinant,
            "restore_best": True,
        },
        em_cfg={"update_every": 5, "start_iter": 50},
        intensity_cfg={
            "sigmaM": config.matching_scale,
            "sigmaR": config.deformation_regularization,
            "sigmaA": 5.0,
            "sigmaB": 2.0,
        },
        device=device,
        dtype=dtype,
        verbose=verbose,
        print_every=100,
    )
    prealigned_xy = atac.obs[["x_prealigned", "y_prealigned"]].to_numpy(float)
    mapped = lddmm.map_points_source_to_target(
        result["xv"],
        result["v"],
        result["A"],
        np.stack([prealigned_xy[:, 1], prealigned_xy[:, 0]], axis=1),
    )
    if torch.is_tensor(mapped):
        mapped = mapped.detach().cpu().numpy()
    atac.obs["x_aligned"] = mapped[:, 1].astype(float)
    atac.obs["y_aligned"] = mapped[:, 0].astype(float)
    alignment_metadata = {
        "atac_cluster_key": atac_cluster_key,
        "st_cluster_key": st_cluster_key,
        "matched_pairs": int(len(matched)),
        "canvas_shape_hw": list(shape),
        "parameters": asdict(config),
        "pairing_weights": {
            "sdf_corr": config.sdf_weight,
            "chamfer_sim": config.chamfer_weight,
            "area_sim": config.area_weight,
            "dice": config.dice_weight,
        },
    }
    atac.uns.setdefault("spAlignDE", {}).setdefault(
        "atac_to_st", {}
    )["alignment"] = _metadata_ready(alignment_metadata)

    destination = None
    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        atac.write_h5ad(destination / "atac_to_st_aligned.h5ad")
        st_reference.write_h5ad(destination / "st_reference_analysis_frame.h5ad")
        matched.to_csv(destination / "matched_structure_pairs.csv", index=False)
        pair_scores.to_csv(destination / "candidate_structure_pairs.csv", index=False)
        atac_mask_stats.to_csv(destination / "atac_mask_summary.csv", index=False)
        st_mask_stats.to_csv(destination / "st_mask_summary.csv", index=False)
        _write_json(
            destination / "alignment_manifest.json",
            {
                "atac_observations": int(atac.n_obs),
                "st_observations": int(st_reference.n_obs),
                "prealignment": prealigned.params,
                "alignment": alignment_metadata,
            },
        )
    return ATACSTAlignmentResult(
        atac=atac,
        st_reference=st_reference,
        matched_pairs=matched,
        prealignment_parameters=dict(prealigned.params),
        output_dir=destination,
        context={
            "atac_masks": atac_masks,
            "st_masks": st_masks,
            "atac_mask_stats": atac_mask_stats,
            "st_mask_stats": st_mask_stats,
            "pair_scores": pair_scores,
            "source_onehot": source_onehot,
            "target_onehot": target_onehot,
            "source_sdt": source_sdt,
            "target_sdt": target_sdt,
            "channel_weights": weights,
            "lddmm_output": result,
        },
    )


def _label_colors(labels: pd.Series, palette: str = "turbo") -> np.ndarray:
    text = labels.astype(str)
    unique = sorted(pd.unique(text), key=str)
    colors = plt.get_cmap(palette)(
        np.linspace(0.02, 0.98, max(len(unique), 2))
    )
    mapping = dict(zip(unique, colors, strict=False))
    return np.asarray([mapping[value] for value in text])


def plot_atac_st_prealignment(
    result: ATACSTPrealignmentResult,
    *,
    atac_cluster_key: str = "cluster",
    st_cluster_key: str = "cluster",
    atac_point_size: float = 3.0,
    st_point_size: float = 0.5,
    figsize: tuple[float, float] = (14.0, 4.6),
) -> tuple[Any, np.ndarray]:
    """Show the cropped ST reference, ATAC initialization, and their overlay."""
    atac_xy = result.atac.obs[["x_prealigned", "y_prealigned"]].to_numpy()
    st_xy = result.st_reference.obs[["x_prealigned", "y_prealigned"]].to_numpy()
    atac_colors = _label_colors(result.atac.obs[atac_cluster_key])
    st_colors = _label_colors(result.st_reference.obs[st_cluster_key], "tab20")
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    axes[0].scatter(st_xy[:, 0], st_xy[:, 1], c=st_colors, s=st_point_size,
                    alpha=0.85, edgecolors="none", rasterized=True)
    axes[0].set_title("Fixed ST half-brain structures")
    axes[1].scatter(atac_xy[:, 0], atac_xy[:, 1], c=atac_colors,
                    s=atac_point_size, alpha=0.90, edgecolors="none", rasterized=True)
    axes[1].set_title("Pre-aligned ATAC structures")
    axes[2].scatter(st_xy[:, 0], st_xy[:, 1], color="#BDBDBD", s=st_point_size,
                    alpha=0.35, edgecolors="none", rasterized=True)
    axes[2].scatter(atac_xy[:, 0], atac_xy[:, 1], c=atac_colors,
                    s=atac_point_size, alpha=0.80, edgecolors="none", rasterized=True)
    axes[2].set_title("Global initialization")
    for axis in axes:
        axis.set_aspect("equal")
        axis.axis("off")
    return fig, axes


def plot_atac_st_alignment(
    result: ATACSTAlignmentResult,
    *,
    atac_cluster_key: str = "cluster",
    atac_point_size: float = 3.0,
    st_point_size: float = 0.5,
    color_by_cluster: bool = True,
    figsize: tuple[float, float] = (11.5, 5.2),
) -> tuple[Any, np.ndarray]:
    """Compare global ATAC initialization and final S-LDDMM alignment."""
    st_xy = result.st_reference.obs[["x_prealigned", "y_prealigned"]].to_numpy()
    before = result.atac.obs[["x_prealigned", "y_prealigned"]].to_numpy()
    after = result.atac.obs[["x_aligned", "y_aligned"]].to_numpy()
    colors = (
        _label_colors(result.atac.obs[atac_cluster_key])
        if color_by_cluster
        else np.repeat(np.asarray([[0.0, 0.4, 0.8, 0.75]]), result.atac.n_obs, axis=0)
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True,
                             sharex=True, sharey=True)
    for axis, coordinates, title in zip(
        axes,
        (before, after),
        ("Global pre-alignment", "Structure-guided S-LDDMM"),
        strict=False,
    ):
        axis.scatter(st_xy[:, 0], st_xy[:, 1], color="#D0D0D0", s=st_point_size,
                     alpha=0.55, edgecolors="none", rasterized=True)
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=colors,
                     s=atac_point_size, alpha=0.85, edgecolors="none", rasterized=True)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.axis("off")
    return fig, axes


def plot_atac_st_matched_structures(
    result: ATACSTAlignmentResult,
    *,
    atac_cluster_key: str = "cluster",
    st_cluster_key: str = "cluster",
    atac_point_size: float = 3.0,
    st_point_size: float = 0.5,
    figsize: tuple[float, float] = (16.0, 4.6),
) -> tuple[Any, np.ndarray]:
    """Use a shared color for every accepted ST/ATAC structure pair."""
    palette = plt.get_cmap("tab20")(
        np.linspace(0.0, 1.0, max(len(result.matched_pairs), 2))
    )
    st_map = {
        str(row.st_structure): palette[index]
        for index, row in result.matched_pairs.iterrows()
    }
    atac_map = {
        str(row.atac_structure): palette[index]
        for index, row in result.matched_pairs.iterrows()
    }
    other = np.asarray([0.82, 0.82, 0.82, 0.55])
    st_colors = np.asarray(
        [st_map.get(value, other) for value in result.st_reference.obs[st_cluster_key].astype(str)]
    )
    atac_colors = np.asarray(
        [atac_map.get(value, other) for value in result.atac.obs[atac_cluster_key].astype(str)]
    )
    st_xy = result.st_reference.obs[["x_prealigned", "y_prealigned"]].to_numpy()
    before = result.atac.obs[["x_prealigned", "y_prealigned"]].to_numpy()
    after = result.atac.obs[["x_aligned", "y_aligned"]].to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    axes[0].scatter(st_xy[:, 0], st_xy[:, 1], c=st_colors, s=st_point_size,
                    alpha=0.85, edgecolors="none", rasterized=True)
    axes[0].set_title("Matched ST structures")
    axes[1].scatter(before[:, 0], before[:, 1], c=atac_colors, s=atac_point_size,
                    alpha=0.90, edgecolors="none", rasterized=True)
    axes[1].set_title("Matched ATAC structures")
    axes[2].scatter(st_xy[:, 0], st_xy[:, 1], c=st_colors, s=st_point_size,
                    alpha=0.65, edgecolors="none", rasterized=True)
    axes[2].scatter(before[:, 0], before[:, 1], c=atac_colors, s=atac_point_size,
                    alpha=0.80, edgecolors="none", rasterized=True)
    axes[2].set_title("Paired overlay before")
    axes[3].scatter(st_xy[:, 0], st_xy[:, 1], c=st_colors, s=st_point_size,
                    alpha=0.65, edgecolors="none", rasterized=True)
    axes[3].scatter(after[:, 0], after[:, 1], c=atac_colors, s=atac_point_size,
                    alpha=0.80, edgecolors="none", rasterized=True)
    axes[3].set_title("Paired overlay after")
    for axis in axes:
        axis.set_aspect("equal")
        axis.axis("off")
    return fig, axes


__all__ = [
    "ATACSTAlignmentConfig",
    "ATACSTAlignmentResult",
    "ATACSTPrealignmentConfig",
    "ATACSTPrealignmentResult",
    "align_atac_to_st",
    "plot_atac_st_alignment",
    "plot_atac_st_matched_structures",
    "plot_atac_st_prealignment",
    "prealign_atac_to_st",
]
