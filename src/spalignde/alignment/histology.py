"""Public ST-to-histology workflow built around image-derived structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from typing import Any
import warnings
import xml.etree.ElementTree as ET

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom

from ..io import spatial_coordinates, validate_single_sample_anndata
from .cross_sample import ManualPrealignmentConfig, apply_similarity_transform


Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class HistologyFeatureConfig:
    """HIPT feature-extraction settings for one high-resolution image.

    Physical pixel size must be resolved before extraction. Shifted tiles
    reduce tile-boundary artifacts by averaging 16 views at greater runtime.
    """

    tile_multiple: int = 224
    jpeg_quality: int = 95
    device: str | None = None
    shifted_tiles: bool = True
    hipt_dir: str | Path | None = None
    extractor_dir: str | Path | None = None
    extractor_python: str | Path | None = None
    target_microns_per_pixel: float | None = 0.5
    source_microns_per_pixel: float | None = None


@dataclass
class HistologyFeatureResult:
    """Prepared image and its image-only HIPT feature field."""

    source_image_path: Path
    prepared_image_path: Path
    feature_path: Path
    image_size_wh: tuple[int, int]
    feature_grid_shape_hw: tuple[int, int]
    output_dir: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class HistologyClusteringConfig:
    """Deterministic H&E feature-clustering profile used in the paper.

    ``image_clusters`` controls initial granularity, ``rgb_weight`` controls
    stain/color influence, and ``coordinate_weight`` promotes spatial
    compactness. Cleanup sizes are measured in feature-grid pixels. Symmetry
    settings should be reconsidered for partial or asymmetric tissues.
    """

    background_clusters: int = 2
    image_clusters: int = 30
    merged_clusters: int = 30
    kmeans_iterations: int = 11
    initialization_samples: int = 50_000
    assignment_chunk: int = 50_000
    border_trim_pixels: int = 256
    rgb_weight: float = 0.25
    coordinate_weight: float = 0.05
    random_state: int = 0
    cleanup_min_size: int = 250
    symmetry_axis: str = "ud"
    symmetry_max_merges: int = 2
    symmetry_min_score_gain: float = 0.02
    symmetry_min_reflected_dice: float = 0.15
    symmetry_max_centroid_distance: float = 0.15
    symmetry_min_feature_cosine: float = 0.30
    cpu_threads: int | None = None


@dataclass
class HistologyClusteringResult:
    """Image-derived tissue mask and cleaned spatial-structure labels."""

    image_path: Path
    feature_path: Path
    labels_raw: np.ndarray
    labels_filled: np.ndarray
    labels_merged: np.ndarray
    labels_clean: np.ndarray
    tissue_mask: np.ndarray
    summary: dict[str, Any]
    output_dir: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.labels_clean.shape)

    @property
    def n_structures(self) -> int:
        labels = self.labels_clean[self.labels_clean >= 0]
        return int(len(np.unique(labels)))


@dataclass(frozen=True)
class STHistologyStructureConfig:
    """Expression-based coarse-to-fine ST structures for histology matching."""

    n_levels: int = 5
    variance_fraction: float = 0.75
    min_genes: int = 50
    drop_blank_genes: bool = True


@dataclass(frozen=True)
class HistologyPrealignmentConfig:
    """Global ST-to-image initialization settings.

    Inspect the mask-overlap result and use manual initialization when tissue
    coverage, tears, cropping or orientation make automatic IoU unreliable.
    Distance and translation values use histology feature-grid coordinates.
    """

    method: str = "mask_overlap"
    manual: ManualPrealignmentConfig = ManualPrealignmentConfig()
    close_kernel: int = 15
    angle_step_degrees: float = 1.0
    scale_tweak: float = 0.05
    scale_steps: int = 2
    search_max_points: int = 50_000
    minimum_recommended_iou: float = 0.20
    random_state: int = 0


@dataclass
class HistologyPrealignmentResult:
    """AnnData with global ST-to-histology coordinates."""

    adata: ad.AnnData
    histology: HistologyClusteringResult
    params: dict[str, Any]


@dataclass(frozen=True)
class STHistologyAlignmentConfig:
    """Structure pairing and S-LDDMM settings for ST-to-histology alignment.

    Pairing weights must be finite, non-negative, and sum to one. Pair gates
    should be tuned from candidate-mask diagnostics. The S-LDDMM
    ``kernel_scale``/``velocity_grid_spacing`` pair is the public equivalent of
    legacy ``a``/``grid_step`` and uses original feature-grid coordinate units.
    ``zoom_scale`` changes image sampling density while retaining axes that
    span the original feature-grid extent. ``restore_best_checkpoint`` defaults
    to ``False`` so the returned transform is the final optimizer iterate.
    """

    pairing_weight_sdf: float = 0.20
    pairing_weight_chamfer: float = 0.40
    pairing_weight_dice: float = 0.15
    pairing_weight_area: float = 0.25
    pairing_weight_thickness: float = 0.00
    pair_score_threshold: float = 0.40
    pair_asd_threshold: float = 30.0
    maximum_pairs: int = 30
    minimum_intersection: int = 20
    zoom_scale: float = 0.60
    global_shape_weight: float = 1.60
    time_steps: int = 5
    kernel_scale: float = 60.0
    kernel_power: float = 2.0
    velocity_expand: float = 2.0
    velocity_grid_spacing: float = 6.0
    iterations: int = 300
    affine_linear_lr: float = 5e-9
    affine_translation_lr: float = 5e-2
    momentum_lr: float = 2e3
    momentum_lr_decay: float = 0.9995
    minimum_momentum_lr: float = 200.0
    restore_best_checkpoint: bool = False
    device: str | None = None
    dtype: str = "float32"


@dataclass
class STHistologyAlignmentResult:
    """Final aligned AnnData and structure-level quality-control objects."""

    adata: ad.AnnData
    histology: HistologyClusteringResult
    matched_pairs: pd.DataFrame
    prealignment_parameters: dict[str, Any]
    output_dir: Path | None = None
    context: dict[str, Any] | None = None


def _file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _anndata_ready(value: Any) -> Any:
    """Convert nested metadata to values supported by AnnData ``uns``."""
    if value is None:
        return "not_applicable"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _anndata_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_anndata_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")


def prepare_histology_image(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    tile_multiple: int = 224,
    jpeg_quality: int = 95,
    target_microns_per_pixel: float | None = 0.5,
    source_microns_per_pixel: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Convert, optionally rescale, and pad one high-resolution image."""
    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Histology image not found: {image_path}")
    if tile_multiple < 1:
        raise ValueError("tile_multiple must be positive")
    if target_microns_per_pixel is not None and target_microns_per_pixel <= 0:
        raise ValueError("target_microns_per_pixel must be positive")
    if source_microns_per_pixel is not None and source_microns_per_pixel <= 0:
        raise ValueError("source_microns_per_pixel must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = output_dir / "he.jpg"
    inferred_pixel_sizes: list[float] = []
    if source_microns_per_pixel is None and image_path.suffix.lower() in {
        ".tif", ".tiff", ".btf", ".ome.tif", ".ome.tiff"
    }:
        try:
            import tifffile

            with tifffile.TiffFile(image_path) as tif:
                ome_metadata = tif.ome_metadata
            if ome_metadata:
                root = ET.fromstring(ome_metadata)
                pixels = next(
                    (
                        element
                        for element in root.iter()
                        if element.tag.rsplit("}", 1)[-1] == "Pixels"
                    ),
                    None,
                )
                if pixels is not None:
                    for key in ("PhysicalSizeX", "PhysicalSizeY"):
                        value = pixels.attrib.get(key)
                        if value is not None and float(value) > 0:
                            inferred_pixel_sizes.append(float(value))
        except (ImportError, OSError, ValueError, ET.ParseError):
            inferred_pixel_sizes = []
    resolved_source_mpp = (
        float(source_microns_per_pixel)
        if source_microns_per_pixel is not None
        else (
            float(np.mean(inferred_pixel_sizes))
            if inferred_pixel_sizes
            else None
        )
    )
    resize_scale = 1.0
    if target_microns_per_pixel is not None and resolved_source_mpp is not None:
        resize_scale = resolved_source_mpp / float(target_microns_per_pixel)
    with Image.open(image_path) as image:
        width, height = image.size
        resized_width = max(1, int(round(width * resize_scale)))
        resized_height = max(1, int(round(height * resize_scale)))
        pad_right = (-resized_width) % int(tile_multiple)
        pad_bottom = (-resized_height) % int(tile_multiple)
        byte_copy = (
            resize_scale == 1.0
            and pad_right == 0
            and pad_bottom == 0
            and image.format in {"JPEG", "JPG"}
        )
        if byte_copy:
            if image_path != prepared:
                shutil.copyfile(image_path, prepared)
            mode = "byte_copy_no_padding"
        else:
            rgb = image.convert("RGB")
            if (resized_width, resized_height) != (width, height):
                rgb = rgb.resize(
                    (resized_width, resized_height),
                    Image.Resampling.LANCZOS,
                )
            canvas = Image.new(
                "RGB",
                (resized_width + pad_right, resized_height + pad_bottom),
                color=(255, 255, 255),
            )
            canvas.paste(rgb, (0, 0))
            canvas.save(prepared, format="JPEG", quality=int(jpeg_quality))
            mode = "rescale_pad_and_jpeg_encode"
    padded_width = resized_width + pad_right
    padded_height = resized_height + pad_bottom
    manifest = {
        "source_image": image_path.name,
        "prepared_image": prepared.name,
        "preparation_mode": mode,
        "tile_multiple_px": int(tile_multiple),
        "original_size_wh": [int(width), int(height)],
        "resized_size_wh": [int(resized_width), int(resized_height)],
        "padded_size_wh": [int(padded_width), int(padded_height)],
        "source_microns_per_pixel": resolved_source_mpp,
        "requested_target_microns_per_pixel": target_microns_per_pixel,
        "output_microns_per_pixel": (
            target_microns_per_pixel
            if resolved_source_mpp is not None
            else None
        ),
        "resize_scale": float(resize_scale),
        "pad_right_px": int(pad_right),
        "pad_bottom_px": int(pad_bottom),
        "feature_grid_shape_hw": [
            int(padded_height // 16),
            int(padded_width // 16),
        ],
        "source_sha256": _file_sha256(image_path),
        "prepared_sha256": _file_sha256(prepared),
    }
    _write_json(output_dir / "histology_image_preparation.json", manifest)
    return prepared, manifest


def _resolve_hipt_assets(
    config: HistologyFeatureConfig,
) -> tuple[Path, Path, Path]:
    configured = (
        config.hipt_dir
        or config.extractor_dir
        or os.environ.get("SPALIGNDE_HIPT_DIR")
    )
    if configured is None:
        raise FileNotFoundError(
            "HIPT assets were not found. Set SPALIGNDE_HIPT_DIR to a directory "
            "containing an official HIPT clone (or its HIPT_4K directory). "
            "The only dataset input is the high-resolution image."
        )
    directory = Path(configured).expanduser().resolve()
    source_directories = (directory / "HIPT_4K", directory)
    for source_dir in source_directories:
        required_source = (
            source_dir / "hipt_4k.py",
            source_dir / "hipt_model_utils.py",
            source_dir / "vision_transformer.py",
            source_dir / "vision_transformer4k.py",
        )
        if not all(path.is_file() for path in required_source):
            continue
        for checkpoint_dir in (
            source_dir / "Checkpoints",
            source_dir / "checkpoints",
        ):
            checkpoint256 = checkpoint_dir / "vit256_small_dino.pth"
            checkpoint4k = checkpoint_dir / "vit4k_xs_dino.pth"
            if checkpoint256.is_file() and checkpoint4k.is_file():
                return source_dir, checkpoint256, checkpoint4k
    raise FileNotFoundError(
        "The configured HIPT directory is incomplete. Expected the official "
        "HIPT_4K source files and Checkpoints/vit256_small_dino.pth plus "
        "Checkpoints/vit4k_xs_dino.pth under:\n"
        f"{directory}"
    )


def extract_histology_features(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    config: HistologyFeatureConfig | None = None,
) -> HistologyFeatureResult:
    """Extract shifted-tile HIPT features from one high-resolution image."""
    config = config or HistologyFeatureConfig()
    output_dir = Path(output_dir).expanduser().resolve()
    prepared, preparation = prepare_histology_image(
        image_path,
        output_dir,
        tile_multiple=config.tile_multiple,
        jpeg_quality=config.jpeg_quality,
        target_microns_per_pixel=config.target_microns_per_pixel,
        source_microns_per_pixel=config.source_microns_per_pixel,
    )
    hipt_dir, checkpoint256, checkpoint4k = _resolve_hipt_assets(config)
    extractor_python = Path(
        config.extractor_python
        or os.environ.get("SPALIGNDE_HIPT_PYTHON")
        or sys.executable
    ).expanduser().resolve()
    if not extractor_python.is_file():
        raise FileNotFoundError(
            f"HIPT Python executable not found: {extractor_python}"
        )
    device = config.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    command = [
        str(extractor_python),
        str(Path(__file__).with_name("_hipt_feature_extractor.py")),
        "--image",
        str(prepared),
        "--output",
        str(output_dir / "embeddings-hist-vit.pickle"),
        "--hipt-dir",
        str(hipt_dir),
        "--model256",
        str(checkpoint256),
        "--model4k",
        str(checkpoint4k),
        "--device",
        str(device),
    ]
    if not config.shifted_tiles:
        command.append("--no-shift")
    subprocess_environment = os.environ.copy()
    subprocess_environment.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    subprocess.run(
        command,
        cwd=output_dir,
        env=subprocess_environment,
        check=True,
    )
    feature_path = output_dir / "embeddings-hist-vit.pickle"
    if not feature_path.is_file():
        raise RuntimeError("HIPT extraction did not produce the feature pickle")
    feature_grid_shape = [
        int(value) for value in preparation["feature_grid_shape_hw"]
    ]
    shift_offsets = [0, 64, 128, 192] if config.shifted_tiles else [0]
    manifest = {
        **preparation,
        "feature_backend": "HIPT ViT-256 + ViT-4K",
        "feature_implementation": "spAlignDE dense shifted-view extractor",
        "model_sources": {
            "hipt_code": "https://github.com/mahmoodlab/HIPT",
        },
        "model_checkpoints": {
            "vit256_small_dino": {
                "file": checkpoint256.name,
                "bytes": int(checkpoint256.stat().st_size),
                "sha256": _file_sha256(checkpoint256),
                "source": (
                    "https://github.com/mahmoodlab/HIPT/blob/master/"
                    "HIPT_4K/Checkpoints/vit256_small_dino.pth"
                ),
            },
            "vit4k_xs_dino": {
                "file": checkpoint4k.name,
                "bytes": int(checkpoint4k.stat().st_size),
                "sha256": _file_sha256(checkpoint4k),
                "source": (
                    "https://github.com/mahmoodlab/HIPT/blob/master/"
                    "HIPT_4K/Checkpoints/vit4k_xs_dino.pth"
                ),
            },
        },
        "shifted_tiles": bool(config.shifted_tiles),
        "extraction_geometry": {
            "global_context_tile_px": 4096,
            "local_patch_px": 256,
            "feature_stride_px": 16,
            "shift_margin_px": 256 if config.shifted_tiles else 0,
            "shift_stride_px": 64 if config.shifted_tiles else 0,
            "shift_offsets_per_axis_px": shift_offsets,
            "shifted_views": int(len(shift_offsets) ** 2),
            "cls_smoothing_window_feature_pixels": 16,
            "sub_smoothing_window_feature_pixels": 4,
            "smoothing_kernel": "uniform",
        },
        "feature_schema": {
            "cls": {
                "description": "ViT-4K context features",
                "channels": 192,
                "channel_shape_hw": feature_grid_shape,
                "dtype": "float32",
            },
            "sub": {
                "description": "ViT-256 subpatch features",
                "channels": 384,
                "channel_shape_hw": feature_grid_shape,
                "dtype": "float32",
            },
            "rgb": {
                "description": "16-fold mean-downsampled RGB",
                "channels": 3,
                "channel_shape_hw": feature_grid_shape,
                "dtype": "float32",
            },
        },
        "random_seed": 0,
        "device": str(device),
        "extractor_python": extractor_python.name,
        "feature_file": feature_path.name,
        "feature_bytes": int(feature_path.stat().st_size),
        "feature_sha256": _file_sha256(feature_path),
    }
    _write_json(output_dir / "histology_feature_manifest.json", manifest)
    return HistologyFeatureResult(
        source_image_path=Path(image_path).expanduser().resolve(),
        prepared_image_path=prepared,
        feature_path=feature_path,
        image_size_wh=tuple(preparation["padded_size_wh"]),
        feature_grid_shape_hw=tuple(preparation["feature_grid_shape_hw"]),
        output_dir=output_dir,
        manifest=manifest,
    )


def load_histology_features(
    image_path: str | Path,
    feature_path: str | Path,
) -> HistologyFeatureResult:
    """Construct the feature-stage result from explicit image and feature files."""
    image_path = Path(image_path).expanduser().resolve()
    feature_path = Path(feature_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    with Image.open(image_path) as image:
        size_wh = tuple(int(value) for value in image.size)
    grid = (int(size_wh[1] // 16), int(size_wh[0] // 16))
    manifest = {
        "source_image": image_path.name,
        "prepared_image": image_path.name,
        "feature_file": feature_path.name,
        "padded_size_wh": list(size_wh),
        "feature_grid_shape_hw": list(grid),
        "source": "explicit files",
    }
    return HistologyFeatureResult(
        source_image_path=image_path,
        prepared_image_path=image_path,
        feature_path=feature_path,
        image_size_wh=size_wh,
        feature_grid_shape_hw=grid,
        output_dir=feature_path.parent,
        manifest=manifest,
    )


def _n_regions(labels: np.ndarray) -> int:
    foreground = np.asarray(labels)[np.asarray(labels) >= 0]
    return int(len(np.unique(foreground)))


def cluster_histology_features(
    features: HistologyFeatureResult | str | Path,
    output_dir: str | Path,
    *,
    image_path: str | Path | None = None,
    config: HistologyClusteringConfig | None = None,
) -> HistologyClusteringResult:
    """Cluster HIPT features with the deterministic paper profile."""
    from . import _histology_clustering_core as core

    config = config or HistologyClusteringConfig()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(features, HistologyFeatureResult):
        feature_path = features.feature_path
        image = features.prepared_image_path
    else:
        feature_path = Path(features).expanduser().resolve()
        image = (
            Path(image_path).expanduser().resolve()
            if image_path
            else feature_path.parent / "he.jpg"
        )
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if not image.is_file():
        raise FileNotFoundError(image)
    core.POST_SYMMETRY_MERGE = True
    core.POST_SYMMETRY_AXIS = config.symmetry_axis
    core.POST_MAX_MERGES = int(config.symmetry_max_merges)
    core.POST_MIN_SCORE_GAIN = float(config.symmetry_min_score_gain)
    core.POST_MIN_REFLECTED_DICE = float(config.symmetry_min_reflected_dice)
    core.POST_MAX_REFLECTED_CENTROID_DIST = float(
        config.symmetry_max_centroid_distance
    )
    core.POST_MIN_FEATURE_COSINE = float(config.symmetry_min_feature_cosine)
    stage = core.run_he_kmeans_crisp(
        pickle_path=str(feature_path),
        out_dir=str(output_dir),
        k_bg=config.background_clusters,
        k_slide=config.image_clusters,
        n_iters=config.kmeans_iterations,
        init_samples=config.initialization_samples,
        assign_chunk=config.assignment_chunk,
        border_trim_px=config.border_trim_pixels,
        rgb_weight=config.rgb_weight,
        xy_weight=config.coordinate_weight,
        random_state=config.random_state,
        max_fit=120_000,
        fast_mode=False,
        cpu_threads=config.cpu_threads,
        keep_largest_tissue_cc=True,
        min_tissue_cc_area=0,
        min_slide_fraction=0.01,
        stage1_use_rgb_only=True,
        debug_mask=True,
        reflection_axis=config.symmetry_axis,
    )
    labels_raw = stage["labels_full"].copy()
    labels_filled, holes = core.fill_internal_label_holes(labels_raw, bg=-1)
    labels_slide = labels_raw[stage["slide_mask"]].astype(np.int32)
    merged = core.merge_he_clusters(
        labels_full=labels_filled,
        slide_x_raw=stage["Xp"],
        labels_slide=labels_slide,
        k_slide=config.image_clusters,
        k_merge=config.merged_clusters,
        do_reflection_merge=True,
        reflection_axis=config.symmetry_axis,
    )
    labels_merged, symmetry_history = core.post_merge_by_symmetry_score(
        merged["labels_merged"],
        stage["Xp"],
        axis=config.symmetry_axis,
        bg=-1,
        feature_mask=(labels_raw >= 0),
    )
    labels_clean = core.cleanup_label_islands(
        labels_merged,
        bg=-1,
        min_size=config.cleanup_min_size,
    )
    tissue_mask = labels_clean >= 0
    for filename, array in {
        "slide_mask.npy": tissue_mask,
        "labels_full_raw.npy": labels_raw,
        "labels_full.npy": labels_filled,
        "labels_merged.npy": labels_merged,
        "labels_merged_clean.npy": labels_clean,
    }.items():
        np.save(output_dir / filename, array)
    summary = {
        "profile": "paper_he_24_structure",
        "input_image": image.name,
        "input_feature": feature_path.name,
        "feature_grid_shape_hw": list(labels_clean.shape),
        "tissue_pixels": int(tissue_mask.sum()),
        "internal_holes_filled": int(holes.sum()),
        "regions": {
            "labels_full_raw": _n_regions(labels_raw),
            "labels_full": _n_regions(labels_filled),
            "labels_merged": _n_regions(labels_merged),
            "labels_merged_clean": _n_regions(labels_clean),
        },
        "post_symmetry_history": symmetry_history,
        "parameters": asdict(config),
    }
    _write_json(output_dir / "cluster_summary.json", summary)
    with (output_dir / "histology_clustering_result.pickle").open("wb") as handle:
        pickle.dump(
            {
                "image_path": str(image),
                "feature_path": str(feature_path),
                "labels_raw": labels_raw,
                "labels_filled": labels_filled,
                "labels_merged": labels_merged,
                "labels_clean": labels_clean,
                "tissue_mask": tissue_mask,
                "summary": summary,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return HistologyClusteringResult(
        image_path=image,
        feature_path=feature_path,
        labels_raw=labels_raw,
        labels_filled=labels_filled,
        labels_merged=labels_merged,
        labels_clean=labels_clean,
        tissue_mask=tissue_mask,
        summary=summary,
        output_dir=output_dir,
    )


def load_histology_clustering(output_dir: str | Path) -> HistologyClusteringResult:
    """Load compact output from histology feature clustering."""
    output_dir = Path(output_dir).expanduser().resolve()
    pickle_path = output_dir / "histology_clustering_result.pickle"
    if pickle_path.is_file():
        with pickle_path.open("rb") as handle:
            payload = pickle.load(handle)
        return HistologyClusteringResult(
            image_path=Path(payload["image_path"]),
            feature_path=Path(payload["feature_path"]),
            labels_raw=np.asarray(payload["labels_raw"]),
            labels_filled=np.asarray(payload["labels_filled"]),
            labels_merged=np.asarray(payload["labels_merged"]),
            labels_clean=np.asarray(payload["labels_clean"]),
            tissue_mask=np.asarray(payload["tissue_mask"], dtype=bool),
            summary=dict(payload["summary"]),
            output_dir=output_dir,
        )
    required = (
        output_dir / "labels_full_raw.npy",
        output_dir / "labels_full.npy",
        output_dir / "labels_merged.npy",
        output_dir / "labels_merged_clean.npy",
        output_dir / "cluster_summary.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing clustering outputs:\n" + "\n".join(map(str, missing))
        )
    summary = json.loads((output_dir / "cluster_summary.json").read_text())
    image_path = output_dir.parent / summary.get("input_image", "he.jpg")
    feature_path = output_dir.parent / summary.get(
        "input_feature", "embeddings-hist-vit.pickle"
    )
    labels_clean = np.load(output_dir / "labels_merged_clean.npy")
    return HistologyClusteringResult(
        image_path=image_path,
        feature_path=feature_path,
        labels_raw=np.load(output_dir / "labels_full_raw.npy"),
        labels_filled=np.load(output_dir / "labels_full.npy"),
        labels_merged=np.load(output_dir / "labels_merged.npy"),
        labels_clean=labels_clean,
        tissue_mask=labels_clean >= 0,
        summary=summary,
        output_dir=output_dir,
    )


def _label_display(labels: np.ndarray) -> np.ndarray:
    output = np.asarray(labels, dtype=float).copy()
    output[output < 0] = np.nan
    return output


def _categorical_label_rgba(labels: np.ndarray) -> np.ndarray:
    """Render integer labels with a stable palette and white background."""
    colors: list[Any] = []
    for name in ("tab20", "tab20b", "tab20c", "Set3", "Paired"):
        color_map = plt.get_cmap(name)
        colors.extend(
            getattr(
                color_map,
                "colors",
                color_map(np.linspace(0, 1, 20, endpoint=True)),
            )
        )
    rgba = np.ones(np.asarray(labels).shape + (4,), dtype=np.float32)
    for label in np.unique(np.asarray(labels)[np.asarray(labels) >= 0]):
        rgba[np.asarray(labels) == label] = to_rgba(colors[int(label) % len(colors)])
    return rgba


def plot_histology_feature_clusters(
    result: HistologyClusteringResult,
    *,
    figsize: tuple[float, float] = (13.0, 7.5),
) -> tuple[Any, np.ndarray]:
    """Show the image and three structure-construction stages."""
    with Image.open(result.image_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (result.shape[1], result.shape[0]),
                Image.Resampling.BILINEAR,
            )
        )
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    panels = (
        (axes[0, 0], rgb, "High-resolution H&E image"),
        (
            axes[0, 1],
            _categorical_label_rgba(result.labels_raw),
            "Image-feature clusters",
        ),
        (
            axes[1, 0],
            _categorical_label_rgba(result.labels_merged),
            "Symmetry-aware merging",
        ),
        (
            axes[1, 1],
            _categorical_label_rgba(result.labels_clean),
            f"Cleaned structures (n={result.n_structures})",
        ),
    )
    for axis, image_data, title in panels:
        axis.imshow(
            image_data,
            origin="upper",
            interpolation="nearest",
            cmap=None if image_data.ndim == 3 else "tab20",
        )
        axis.set_title(title)
        axis.axis("off")
    return fig, axes


def build_st_histology_structures(
    adata: ad.AnnData,
    *,
    config: STHistologyStructureConfig | None = None,
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    layer: str | None = None,
    copy: bool = True,
) -> tuple[ad.AnnData, tuple[str, ...]]:
    """Add expression-based ST hierarchy levels used for image matching."""
    from .atlas import STAtlasAlignmentConfig, _build_hierarchy_tables

    config = config or STHistologyStructureConfig()
    atlas_compatible = STAtlasAlignmentConfig(
        n_levels=config.n_levels,
        variance_fraction=config.variance_fraction,
        min_genes=config.min_genes,
        drop_blank_genes=config.drop_blank_genes,
    )
    table, hierarchy_columns, _, _, _ = _build_hierarchy_tables(
        adata,
        config=atlas_compatible,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        layer=layer,
    )
    source = adata.to_memory() if adata.isbacked else adata
    output = source.copy() if copy else source
    for column in hierarchy_columns:
        output.obs[column] = table[column].copy()
    output.uns.pop("spalignde", None)
    output.uns.setdefault("spAlignDE", {})["st_histology_hierarchy"] = {
        "cluster_key": cluster_key,
        "hierarchy_columns": list(hierarchy_columns),
        **_json_ready(asdict(config)),
    }
    return output, hierarchy_columns


def plot_st_histology_structures(
    adata: ad.AnnData,
    hierarchy_columns: tuple[str, ...] | list[str],
    *,
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    selected_key: str | None = None,
    max_points: int = 80_000,
    random_state: int = 0,
    point_size: float = 0.30,
    alpha: float = 0.75,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray]:
    """Compare the original ST partition with its hierarchy levels."""
    validate_single_sample_anndata(
        adata,
        spatial_key=spatial_key,
        cluster_key=cluster_key,
        require_cluster=True,
    )
    columns = [cluster_key, *hierarchy_columns]
    missing = [column for column in columns if column not in adata.obs]
    if missing:
        raise ValueError(f"Missing ST structure columns: {missing}")
    selected_key = selected_key or (
        hierarchy_columns[0] if hierarchy_columns else cluster_key
    )
    if selected_key not in columns:
        raise ValueError("selected_key must be one of the displayed columns")
    coordinates = spatial_coordinates(adata, spatial_key=spatial_key)
    indices = np.arange(adata.n_obs)
    if len(indices) > max_points:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    n_columns = min(3, len(columns))
    n_rows = int(np.ceil(len(columns) / n_columns))
    figsize = figsize or (4.2 * n_columns, 4.0 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=figsize,
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, column in zip(flat_axes, columns, strict=False):
        labels = adata.obs[column].astype(str).to_numpy()
        unique = sorted(pd.unique(labels), key=str)
        palette = {
            label: plt.get_cmap("tab20")(index % 20)
            for index, label in enumerate(unique)
        }
        axis.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            c=np.asarray([palette[label] for label in labels[indices]]),
            s=point_size,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )
        suffix = " (selected)" if column == selected_key else ""
        axis.set_title(f"{column}{suffix}")
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.axis("off")
    for axis in flat_axes[len(columns):]:
        axis.axis("off")
    return fig, axes


def _manual_matrix(config: ManualPrealignmentConfig) -> np.ndarray:
    theta = np.deg2rad(float(config.theta_deg))
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    matrix = np.eye(3)
    matrix[:2, :2] = float(config.scale) * rotation
    matrix[:2, 2] = [config.translation_x, config.translation_y]
    return matrix


def prealign_st_to_histology(
    adata: ad.AnnData,
    histology: HistologyClusteringResult,
    *,
    config: HistologyPrealignmentConfig | None = None,
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    copy: bool = True,
) -> HistologyPrealignmentResult:
    """Initialize ST by whole-mask overlap or a manual similarity transform."""
    config = config or HistologyPrealignmentConfig()
    validate_single_sample_anndata(
        adata,
        spatial_key=spatial_key,
        cluster_key=cluster_key,
        require_cluster=True,
    )
    source = adata.to_memory() if adata.isbacked else adata
    output = source.copy() if copy else source
    coordinates = spatial_coordinates(output, spatial_key=spatial_key)
    method = config.method.strip().lower().replace("-", "_")
    if method in {"manual", "manual_similarity"}:
        manual = config.manual
        transformed = apply_similarity_transform(coordinates, manual)
        params = {
            "method": "manual_similarity",
            "scale": float(manual.scale),
            "theta_deg": float(manual.theta_deg),
            "translation_x": float(manual.translation_x),
            "translation_y": float(manual.translation_y),
            "matrix": _manual_matrix(manual).tolist(),
            "iou": None,
        }
    elif method in {"mask", "mask_overlap", "automatic"}:
        from . import _atlas_core as atlas_core

        rng = np.random.default_rng(config.random_state)
        indices = np.arange(len(coordinates))
        if len(indices) > config.search_max_points:
            indices = np.sort(
                rng.choice(indices, size=config.search_max_points, replace=False)
            )
        search = pd.DataFrame(
            {"x": coordinates[indices, 0], "y": coordinates[indices, 1]}
        )
        source_span = float(np.max(np.ptp(search[["x", "y"]].to_numpy(), axis=0)))
        target_span = float(max(histology.shape))
        had_resolution = hasattr(atlas_core, "resolution")
        previous_resolution = getattr(atlas_core, "resolution", None)
        try:
            atlas_core.resolution = max(
                source_span / max(target_span, 1.0),
                1.0,
            )
            _, automatic = atlas_core.align_omics_no_flip_max_iou(
                search,
                histology.tissue_mask.astype(np.uint8),
                np.arange(histology.shape[1], dtype=float),
                np.arange(histology.shape[0], dtype=float),
                close_ksize=config.close_kernel,
                angle_step_deg=config.angle_step_degrees,
                scale_tweak=config.scale_tweak,
                scale_steps=config.scale_steps,
            )
        finally:
            if had_resolution:
                atlas_core.resolution = previous_resolution
            else:
                del atlas_core.resolution
        normalized_angle = (
            (float(automatic["best_angle_deg"]) + 180.0) % 360.0
        ) - 180.0
        manual = ManualPrealignmentConfig(
            scale=float(automatic["best_scale"]),
            theta_deg=normalized_angle,
            translation_x=float(automatic["tx"]),
            translation_y=float(automatic["ty"]),
        )
        transformed = apply_similarity_transform(coordinates, manual)
        params = {
            "method": "whole_tissue_mask_overlap",
            "scale": manual.scale,
            "theta_deg": manual.theta_deg,
            "translation_x": manual.translation_x,
            "translation_y": manual.translation_y,
            "matrix": _manual_matrix(manual).tolist(),
            "iou": float(automatic["iou"]),
            "search_observations": int(len(indices)),
        }
        if params["iou"] < config.minimum_recommended_iou:
            warnings.warn(
                "Mask-overlap IoU is low. Use "
                "interactive_histology_prealignment before S-LDDMM.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        raise ValueError("method must be mask_overlap or manual")
    output.obs["x_prealigned"] = transformed[:, 0]
    output.obs["y_prealigned"] = transformed[:, 1]
    output.obs["x_aligned"] = transformed[:, 0]
    output.obs["y_aligned"] = transformed[:, 1]
    output.uns.pop("spalignde", None)
    metadata = output.uns.setdefault("spAlignDE", {})
    metadata.setdefault("st_to_histology", {})["prealignment"] = _anndata_ready({
        **params,
        "cluster_key": cluster_key,
        "spatial_key": spatial_key,
        "histology_grid_shape_hw": list(histology.shape),
    })
    return HistologyPrealignmentResult(output, histology, params)


def plot_histology_prealignment_preview(
    adata: ad.AnnData,
    histology: HistologyClusteringResult,
    *,
    config: ManualPrealignmentConfig,
    spatial_key: str = "spatial",
    max_points: int = 50_000,
    random_state: int = 0,
    point_size: float = 0.25,
    alpha: float = 0.18,
    transformed_title: str = "Pre-aligned to tissue mask",
    figsize: tuple[float, float] = (11.0, 5.5),
) -> tuple[Any, np.ndarray]:
    """Preview a manual transform against the image tissue mask."""
    coordinates = spatial_coordinates(adata, spatial_key=spatial_key)
    transformed = apply_similarity_transform(coordinates, config)
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(coordinates))
    if len(indices) > max_points:
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    axes[0].scatter(
        coordinates[indices, 0],
        coordinates[indices, 1],
        s=point_size,
        color="#C23B32",
        alpha=max(alpha, 0.30),
        edgecolors="none",
        rasterized=True,
    )
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].set_title("Original ST coordinates")
    axes[0].axis("off")

    axes[1].imshow(
        histology.tissue_mask,
        cmap="Greys",
        origin="upper",
        interpolation="nearest",
        alpha=0.55,
    )
    axes[1].scatter(
        transformed[indices, 0],
        transformed[indices, 1],
        s=point_size,
        color="#C23B32",
        alpha=alpha,
        edgecolors="none",
        rasterized=True,
    )
    axes[1].set_xlim(0, histology.shape[1] - 1)
    axes[1].set_ylim(histology.shape[0] - 1, 0)
    axes[1].set_aspect("equal")
    axes[1].set_title(transformed_title)
    axes[1].axis("off")
    return fig, axes


@dataclass
class HistologyPrealignmentUI:
    """Interactive controller for manual ST-to-histology initialization."""

    adata: ad.AnnData
    histology: HistologyClusteringResult
    controls: dict[str, Any]
    widget: Any
    cluster_key: str = "cluster"
    spatial_key: str = "spatial"

    @property
    def selected_config(self) -> ManualPrealignmentConfig:
        return ManualPrealignmentConfig(
            scale=float(self.controls["scale"].value),
            theta_deg=float(self.controls["theta_deg"].value),
            translation_x=float(self.controls["translation_x"].value),
            translation_y=float(self.controls["translation_y"].value),
        )

    def display(self) -> "HistologyPrealignmentUI":
        from IPython.display import display

        display(self.widget)
        return self

    def preview(self, **kwargs: Any) -> tuple[Any, np.ndarray]:
        return plot_histology_prealignment_preview(
            self.adata,
            self.histology,
            config=self.selected_config,
            spatial_key=self.spatial_key,
            transformed_title="Manual pre-alignment",
            **kwargs,
        )

    def apply(self, *, copy: bool = True) -> HistologyPrealignmentResult:
        return prealign_st_to_histology(
            self.adata,
            self.histology,
            config=HistologyPrealignmentConfig(
                method="manual",
                manual=self.selected_config,
            ),
            cluster_key=self.cluster_key,
            spatial_key=self.spatial_key,
            copy=copy,
        )


def interactive_histology_prealignment(
    adata: ad.AnnData,
    histology: HistologyClusteringResult,
    *,
    initial_config: ManualPrealignmentConfig | None = None,
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    display_ui: bool = True,
) -> HistologyPrealignmentUI:
    """Open sliders when whole-mask overlap is not anatomically valid."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError("Install spAlignDE[tutorial] for the interactive UI") from exc
    initial = initial_config or ManualPrealignmentConfig()
    span = float(max(histology.shape))
    controls = {
        "scale": widgets.FloatSlider(
            value=initial.scale,
            min=max(0.0001, initial.scale * 0.5),
            max=max(initial.scale * 1.5, 0.001),
            step=max(initial.scale / 500.0, 0.0001),
            description="scale",
            continuous_update=False,
        ),
        "theta_deg": widgets.FloatSlider(
            value=initial.theta_deg,
            min=initial.theta_deg - 180,
            max=initial.theta_deg + 180,
            step=0.5,
            description="theta",
            continuous_update=False,
        ),
        "translation_x": widgets.FloatSlider(
            value=initial.translation_x,
            min=initial.translation_x - span,
            max=initial.translation_x + span,
            step=max(span / 200, 0.1),
            description="tx",
            continuous_update=False,
        ),
        "translation_y": widgets.FloatSlider(
            value=initial.translation_y,
            min=initial.translation_y - span,
            max=initial.translation_y + span,
            step=max(span / 200, 0.1),
            description="ty",
            continuous_update=False,
        ),
    }

    def update_preview(scale, theta_deg, translation_x, translation_y):
        fig, _ = plot_histology_prealignment_preview(
            adata,
            histology,
            config=ManualPrealignmentConfig(
                scale=scale,
                theta_deg=theta_deg,
                translation_x=translation_x,
                translation_y=translation_y,
            ),
            spatial_key=spatial_key,
            transformed_title="Manual pre-alignment",
        )
        display(fig)
        plt.close(fig)

    output = widgets.interactive_output(update_preview, controls)
    widget = widgets.VBox([widgets.VBox(list(controls.values())), output])
    controller = HistologyPrealignmentUI(
        adata, histology, controls, widget, cluster_key, spatial_key
    )
    if display_ui:
        controller.display()
    return controller


def _overall_st_mask(coordinates: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from skimage import morphology

    height, width = shape
    x = np.rint(coordinates[:, 0]).astype(int)
    y = np.rint(coordinates[:, 1]).astype(int)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    accumulator = np.zeros(shape, dtype=np.float32)
    np.add.at(accumulator, (y[valid], x[valid]), 1.0)
    accumulator = ndimage.gaussian_filter(accumulator, sigma=3.0)
    if accumulator.max() > 0:
        accumulator /= accumulator.max()
    mask = accumulator >= 0.015
    mask = morphology.binary_closing(mask, morphology.disk(8))
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=500)
    return mask.astype(np.uint8)


def _histology_pairing_weights(
    config: STHistologyAlignmentConfig,
) -> dict[str, float]:
    """Return the normalized H&E--ST composite-score weights."""
    weights = {
        "sdf_corr": float(config.pairing_weight_sdf),
        "chamfer_sim": float(config.pairing_weight_chamfer),
        "dice": float(config.pairing_weight_dice),
        "area_sim": float(config.pairing_weight_area),
        "thick_sim": float(config.pairing_weight_thickness),
    }
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Histology pairing weights must be finite and non-negative")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-8):
        raise ValueError("Histology pairing weights must sum to 1.0")
    return weights


def align_st_to_histology(
    prealigned: HistologyPrealignmentResult | ad.AnnData,
    histology: HistologyClusteringResult | None = None,
    *,
    config: STHistologyAlignmentConfig | None = None,
    cluster_key: str = "cluster",
    structure_key: str | None = None,
    spatial_key: str = "spatial",
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> STHistologyAlignmentResult:
    """Pair image/ST masks and run the paper single-stage S-LDDMM."""
    import torch
    from . import _histology_alignment_core as core
    from . import _slddmm_core as lddmm

    config = config or STHistologyAlignmentConfig()
    if isinstance(prealigned, HistologyPrealignmentResult):
        adata = prealigned.adata
        histology = prealigned.histology
        prealignment_parameters = dict(prealigned.params)
    else:
        adata = prealigned
        if histology is None:
            raise ValueError("histology is required when prealigned is AnnData")
        prealignment_parameters = dict(
            adata.uns.get("spAlignDE", {})
            .get("st_to_histology", {})
            .get("prealignment", {})
        )
    structure_key = structure_key or cluster_key
    validate_single_sample_anndata(
        adata,
        spatial_key=spatial_key,
        cluster_key=structure_key,
        require_cluster=True,
    )
    missing = {"x_prealigned", "y_prealigned"}.difference(adata.obs.columns)
    if missing:
        raise ValueError("Run prealign_st_to_histology first")
    output = adata.copy()
    frame = output.obs.copy()
    raw = spatial_coordinates(output, spatial_key=spatial_key)
    frame["x"] = raw[:, 0]
    frame["y"] = raw[:, 1]
    frame[structure_key] = frame[structure_key].astype(str)
    shape = histology.shape
    grid_x = np.arange(shape[1], dtype=float)
    grid_y = np.arange(shape[0], dtype=float)
    filtered, removed, filter_stats = core.filter_cluster_for_mask(
        frame,
        label_col=structure_key,
        x_col="x_prealigned",
        y_col="y_prealigned",
        base_k=5,
        detail_area_quantile=0.15,
        area_mode="bbox",
        detail_mad_k=3.0,
        normal_mad_k=2.2,
        apply_grid_thin=False,
    )
    mask_result = core.build_cluster_masks(
        df_smooth=filtered,
        sl=histology.tissue_mask.astype(np.uint8),
        xJ=grid_x,
        yJ=grid_y,
        x_col="x_prealigned",
        y_col="y_prealigned",
        label_col=structure_key,
        params=core.DEFAULT_MASK_PARAMS,
        params_thin=core.MASK_PARAMS_THIN,
        shape_type_col="shape_type",
        thin_values=("detail",),
        thin_rule="mode",
    )
    st_masks = mask_result["st_masks"]
    he_masks, _, he_mask_table, _ = core.build_he_masks_from_labels(
        histology.labels_clean,
        bg_val=-1,
        min_area_raw=100,
        close_r=4,
        open_r=1,
        smooth_sigma=0.8,
        smooth_thr=0.5,
        min_area_clean=100,
        keep_largest=False,
        island_mode="fraction",
        frac_keep=0.3,
        top_k=4,
    )
    pairing_weights = _histology_pairing_weights(config)
    pair_scores = core.all_pair_scores(
        st_masks,
        he_masks,
        min_intersection=config.minimum_intersection,
        weights=pairing_weights,
    )
    if pair_scores.empty:
        raise RuntimeError(
            "No overlapping structures; use interactive_histology_prealignment"
        )
    accepted = pair_scores[
        (pair_scores["align_score"] >= config.pair_score_threshold)
        & (pair_scores["asd"] <= config.pair_asd_threshold)
    ]
    matched = core.greedy_nonoverlap_selection(
        accepted, max_pairs=config.maximum_pairs
    )
    if matched.empty:
        raise RuntimeError("No structure pair passed the score and ASD gates")
    source_pairs, target_pairs, pair_meta = core.build_pair_onehot_from_masks(
        matched, he_masks, st_masks, add_other=False
    )
    pre_xy = frame[["x_prealigned", "y_prealigned"]].to_numpy(float)
    source_onehot = np.concatenate(
        [source_pairs, _overall_st_mask(pre_xy, shape)[None]], axis=0
    )
    target_onehot = np.concatenate(
        [target_pairs, histology.tissue_mask.astype(np.uint8)[None]], axis=0
    )
    source_binary, target_binary = core.preprocess_onehot_asymmetric(
        source_onehot,
        target_onehot,
        st_cfg=dict(sigma_pre=1.4, thr=0.5, close_r=2, open_r=1, min_area=50),
        he_cfg=dict(sigma_pre=0.4, thr=0.5, close_r=1, open_r=1, min_area=80),
    )
    source_sdt, target_sdt = core._build_equalized_sdt(
        source_binary,
        target_binary,
        clip_dist=60.0,
        sigma_sdt=0.9,
        band=4.0,
        tau=2.0,
    )
    weights = core._area_balance_weights(
        source_binary, target_binary, power=0.8, w_min=0.5, w_max=2.5
    )
    weights[-1] *= config.global_shape_weight
    source_image = (source_sdt * weights[:, None, None]).astype(np.float32)
    target_image = (target_sdt * weights[:, None, None]).astype(np.float32)
    source_grid_y = grid_y.copy()
    source_grid_x = grid_x.copy()
    if config.zoom_scale < 1.0:
        _, height, width = source_image.shape
        height_new = max(16, int(round(height * config.zoom_scale)))
        width_new = max(16, int(round(width * config.zoom_scale)))
        factors = (1, height_new / height, width_new / width)
        source_image = zoom(source_image, factors, order=1).astype(np.float32)
        target_image = zoom(target_image, factors, order=1).astype(np.float32)
        source_grid_y = np.linspace(0, shape[0] - 1, height_new)
        source_grid_x = np.linspace(0, shape[1] - 1, width_new)
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    lddmm_output = lddmm.LDDMM_shooting(
        [source_grid_y, source_grid_x],
        source_image,
        [source_grid_y, source_grid_x],
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
            "diffeo_start": 0,
            "lrL": config.affine_linear_lr,
            "lrT": config.affine_translation_lr,
            "lrM": config.momentum_lr,
            "affine_slowdown": 10.0,
            "lrM_decay": config.momentum_lr_decay,
            "lrM_min": config.minimum_momentum_lr,
            "restore_best": config.restore_best_checkpoint,
        },
        em_cfg={"update_every": 5, "start_iter": 50},
        intensity_cfg={
            "sigmaM": 1.0,
            "sigmaB": 2.0,
            "sigmaA": 5.0,
            "sigmaR": 5e5,
        },
        device=device,
        dtype=dtype,
        verbose=verbose,
        print_every=100,
    )
    mapped = lddmm.map_points_source_to_target(
        lddmm_output["xv"],
        lddmm_output["v"],
        lddmm_output["A"],
        np.stack([pre_xy[:, 1], pre_xy[:, 0]], axis=1),
    )
    if torch.is_tensor(mapped):
        mapped = mapped.detach().cpu().numpy()
    output.obs["x_aligned"] = mapped[:, 1].astype(float)
    output.obs["y_aligned"] = mapped[:, 0].astype(float)

    # Re-rasterize the accepted ST structures at their final coordinates so
    # users can inspect the exact pairwise evidence before and after S-LDDMM.
    # The same filtering decisions and mask-cleanup settings are reused; only
    # the ST coordinates change.
    final_filtered = filtered.copy()
    final_filtered["x_aligned"] = output.obs.loc[
        final_filtered.index, "x_aligned"
    ].to_numpy(float)
    final_filtered["y_aligned"] = output.obs.loc[
        final_filtered.index, "y_aligned"
    ].to_numpy(float)
    final_mask_result = core.build_cluster_masks(
        df_smooth=final_filtered,
        sl=histology.tissue_mask.astype(np.uint8),
        xJ=grid_x,
        yJ=grid_y,
        x_col="x_aligned",
        y_col="y_aligned",
        label_col=structure_key,
        params=core.DEFAULT_MASK_PARAMS,
        params_thin=core.MASK_PARAMS_THIN,
        shape_type_col="shape_type",
        thin_values=("detail",),
        thin_rule="mode",
        verbose=False,
    )
    final_source_pairs, final_target_pairs, _ = core.build_pair_onehot_from_masks(
        matched,
        he_masks,
        final_mask_result["st_masks"],
        add_other=False,
    )
    final_source_binary, final_target_binary = core.preprocess_onehot_asymmetric(
        final_source_pairs,
        final_target_pairs,
        st_cfg=dict(sigma_pre=1.4, thr=0.5, close_r=2, open_r=1, min_area=50),
        he_cfg=dict(sigma_pre=0.4, thr=0.5, close_r=1, open_r=1, min_area=80),
    )
    pair_metric_rows = []
    for pair_order, (_, pair) in enumerate(matched.iterrows()):
        for stage, source_mask, target_mask in (
            ("before", source_binary[pair_order], target_binary[pair_order]),
            (
                "after",
                final_source_binary[pair_order],
                final_target_binary[pair_order],
            ),
        ):
            metrics = core.compute_all_metrics(source_mask, target_mask)
            pair_metric_rows.append(
                {
                    "pair_order": pair_order,
                    "st": pair["st"],
                    "he": pair["he"],
                    "stage": stage,
                    **metrics,
                }
            )
    pair_overlap_metrics = pd.DataFrame(pair_metric_rows)

    metadata = output.uns.setdefault("spAlignDE", {})
    metadata.setdefault("st_to_histology", {})["alignment"] = _anndata_ready({
        "cluster_key": cluster_key,
        "structure_key": structure_key,
        "spatial_key": spatial_key,
        "histology_grid_shape_hw": list(shape),
        "histology_pixels_per_grid_unit": 16,
        "matched_pairs": int(len(matched)),
        "parameters": asdict(config),
    })
    destination = None
    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output.write_h5ad(destination / "st_to_histology_aligned.h5ad")
        matched.to_csv(destination / "matched_structure_pairs.csv", index=False)
        pair_overlap_metrics.to_csv(
            destination / "matched_structure_pair_overlap_metrics.csv",
            index=False,
        )
        filter_stats.to_csv(destination / "st_mask_filter_summary.csv", index=False)
        _write_json(
            destination / "alignment_manifest.json",
            {
                "observations": int(output.n_obs),
                "histology_structures": int(histology.n_structures),
                "matched_pairs": int(len(matched)),
                "prealignment": prealignment_parameters,
                "alignment": metadata["st_to_histology"]["alignment"],
            },
        )
    return STHistologyAlignmentResult(
        adata=output,
        histology=histology,
        matched_pairs=matched,
        prealignment_parameters=prealignment_parameters,
        output_dir=destination,
        context={
            "st_masks": st_masks,
            "histology_masks": he_masks,
            "histology_mask_table": he_mask_table,
            "pair_scores": pair_scores,
            "removed_st_observations": removed,
            "source_binary": source_binary,
            "target_binary": target_binary,
            "final_source_binary": final_source_binary,
            "final_target_binary": final_target_binary,
            "pair_overlap_metrics": pair_overlap_metrics,
            "channel_weights": weights,
            "pair_metadata": pair_meta,
            "lddmm_output": lddmm_output,
        },
    )


def plot_st_histology_pair_overlap(
    result: STHistologyAlignmentResult,
    *,
    stage: str = "after",
    padding: int = 18,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    """Plot each accepted ST--H&E mask pair before or after S-LDDMM.

    The returned long-form table reports Dice, IoU, ASD, and the other mask
    metrics at both stages. The final-stage plot annotates the before-to-after
    changes for direct pair-level quality control.
    """
    from matplotlib.patches import Patch
    from . import _histology_alignment_core as core

    if stage not in {"before", "after"}:
        raise ValueError("stage must be 'before' or 'after'")
    if result.context is None:
        raise ValueError("Pair-overlap plotting requires the in-memory result context")

    context = result.context
    if stage == "before":
        source_masks = np.asarray(context["source_binary"][:-1]).astype(bool)
        target_masks = np.asarray(context["target_binary"][:-1]).astype(bool)
    else:
        try:
            source_masks = np.asarray(context["final_source_binary"]).astype(bool)
            target_masks = np.asarray(context["final_target_binary"]).astype(bool)
        except KeyError as error:
            raise ValueError(
                "This result predates final pair-mask capture; rerun "
                "align_st_to_histology before plotting stage='after'"
            ) from error

    pair_count = len(result.matched_pairs)
    if pair_count == 0:
        raise ValueError("No matched ST--H&E pairs are available")
    if source_masks.shape[0] != pair_count or target_masks.shape[0] != pair_count:
        raise ValueError("Stored pair masks do not match matched_pairs")

    metrics_table = context.get("pair_overlap_metrics")
    if metrics_table is None:
        rows = []
        before_sources = np.asarray(context["source_binary"][:-1]).astype(bool)
        before_targets = np.asarray(context["target_binary"][:-1]).astype(bool)
        after_sources = np.asarray(context["final_source_binary"]).astype(bool)
        after_targets = np.asarray(context["final_target_binary"]).astype(bool)
        for pair_order, (_, pair) in enumerate(result.matched_pairs.iterrows()):
            for current_stage, source_mask, target_mask in (
                ("before", before_sources[pair_order], before_targets[pair_order]),
                ("after", after_sources[pair_order], after_targets[pair_order]),
            ):
                rows.append(
                    {
                        "pair_order": pair_order,
                        "st": pair["st"],
                        "he": pair["he"],
                        "stage": current_stage,
                        **core.compute_all_metrics(source_mask, target_mask),
                    }
                )
        metrics_table = pd.DataFrame(rows)
    else:
        metrics_table = pd.DataFrame(metrics_table).copy()

    source_color = (0.00, 0.45, 0.70, 1.00)
    target_color = (0.90, 0.60, 0.00, 1.00)
    overlap_color = (0.80, 0.47, 0.65, 1.00)
    if figsize is None:
        figsize = (11.0, 3.6 * pair_count + 0.45)
    fig, axes = plt.subplots(
        pair_count,
        3,
        figsize=figsize,
        constrained_layout=False,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.96,
        bottom=0.10,
        hspace=0.36,
        wspace=0.10,
    )

    for pair_order, ((_, pair), source_mask, target_mask) in enumerate(
        zip(
            result.matched_pairs.iterrows(),
            source_masks,
            target_masks,
            strict=True,
        )
    ):
        union = source_mask | target_mask
        yy, xx = union.nonzero()
        if yy.size:
            y0 = max(0, int(yy.min()) - padding)
            y1 = min(union.shape[0], int(yy.max()) + padding + 1)
            x0 = max(0, int(xx.min()) - padding)
            x1 = min(union.shape[1], int(xx.max()) + padding + 1)
        else:
            y0, y1, x0, x1 = 0, union.shape[0], 0, union.shape[1]

        panels = []
        for mask, color in (
            (source_mask, source_color),
            (target_mask, target_color),
        ):
            image = np.ones((*mask.shape, 4), dtype=float)
            image[mask] = color
            panels.append(image)
        overlap_image = panels[1].copy()
        overlap_image[source_mask] = source_color
        overlap_image[source_mask & target_mask] = overlap_color
        panels.append(overlap_image)

        if stage == "after":
            source_title = "ST structure (after S-LDDMM)"
            overlap_title = "After S-LDDMM overlap"
        else:
            source_title = "ST structure (pre-aligned)"
            overlap_title = "Pre-aligned overlap"
        titles = (
            source_title,
            "H&E structure",
            f"{overlap_title}\nST {pair['st']} ↔ H&E {pair['he']}",
        )
        for axis, image, title in zip(axes[pair_order], panels, titles, strict=True):
            axis.imshow(
                image[y0:y1, x0:x1],
                origin="lower",
                interpolation="nearest",
                rasterized=True,
            )
            axis.set_title(title)
            axis.axis("off")

        selected = metrics_table[
            (metrics_table["pair_order"] == pair_order)
            & (metrics_table["stage"] == stage)
        ].iloc[0]
        if stage == "after":
            before = metrics_table[
                (metrics_table["pair_order"] == pair_order)
                & (metrics_table["stage"] == "before")
            ].iloc[0]
            annotation = (
                f"Dice {before['dice']:.3f}→{selected['dice']:.3f}; "
                f"IoU {before['iou']:.3f}→{selected['iou']:.3f}; "
                f"ASD {before['asd']:.1f}→{selected['asd']:.1f}"
            )
        else:
            annotation = (
                f"Dice={selected['dice']:.3f}; IoU={selected['iou']:.3f}; "
                f"ASD={selected['asd']:.1f}"
            )
        axes[pair_order, 2].text(
            0.5,
            -0.04,
            annotation,
            transform=axes[pair_order, 2].transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )

    fig.legend(
        handles=(
            Patch(facecolor=source_color, label="ST mask"),
            Patch(facecolor=target_color, label="H&E mask"),
            Patch(facecolor=overlap_color, label="overlap"),
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    return fig, axes, metrics_table


def plot_st_histology_alignment(
    result: STHistologyAlignmentResult,
    *,
    cluster_key: str = "cluster",
    max_points: int = 80_000,
    random_state: int = 0,
    point_size: float = 0.30,
    alpha: float = 0.22,
    color_by_cluster: bool = False,
    point_color: str = "#0066CC",
    figsize: tuple[float, float] = (12.0, 5.8),
) -> tuple[Any, np.ndarray]:
    """Compare global initialization with final S-LDDMM on the image."""
    adata = result.adata
    rng = np.random.default_rng(random_state)
    indices = np.arange(adata.n_obs)
    if len(indices) > max_points:
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    unique = sorted(pd.unique(labels), key=str)
    color_map = {
        label: plt.get_cmap("tab20")(index % 20)
        for index, label in enumerate(unique)
    }
    colors = np.asarray([color_map[label] for label in labels])
    with Image.open(result.histology.image_path) as image:
        rgb = np.asarray(
            image.convert("RGB").resize(
                (result.histology.shape[1], result.histology.shape[0]),
                Image.Resampling.BILINEAR,
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for axis, x_column, y_column, title in (
        (axes[0], "x_prealigned", "y_prealigned", "Global pre-alignment"),
        (axes[1], "x_aligned", "y_aligned", "After structure-guided S-LDDMM"),
    ):
        axis.imshow(rgb, origin="upper", interpolation="nearest")
        axis.scatter(
            adata.obs[x_column].to_numpy()[indices],
            adata.obs[y_column].to_numpy()[indices],
            c=colors[indices] if color_by_cluster else point_color,
            s=point_size,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
        )
        axis.set_xlim(0, result.histology.shape[1] - 1)
        axis.set_ylim(result.histology.shape[0] - 1, 0)
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.axis("off")
    return fig, axes


__all__ = [
    "HistologyClusteringConfig",
    "HistologyClusteringResult",
    "HistologyFeatureConfig",
    "HistologyFeatureResult",
    "HistologyPrealignmentConfig",
    "HistologyPrealignmentResult",
    "HistologyPrealignmentUI",
    "STHistologyStructureConfig",
    "STHistologyAlignmentConfig",
    "STHistologyAlignmentResult",
    "align_st_to_histology",
    "build_st_histology_structures",
    "cluster_histology_features",
    "extract_histology_features",
    "interactive_histology_prealignment",
    "load_histology_clustering",
    "load_histology_features",
    "plot_histology_feature_clusters",
    "plot_histology_prealignment_preview",
    "plot_st_histology_pair_overlap",
    "plot_st_histology_alignment",
    "plot_st_histology_structures",
    "prealign_st_to_histology",
    "prepare_histology_image",
]
