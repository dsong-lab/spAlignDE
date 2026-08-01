"""spAlignDE public Python interface."""

__version__ = "0.1.0"

from .alignment import (
    ATACSTAlignmentConfig,
    ATACSTAlignmentResult,
    ATACSTPrealignmentConfig,
    ATACSTPrealignmentResult,
    AllenCCFReference,
    CrossSampleAlignmentResult,
    CrossSampleFields,
    ManualPrealignmentConfig,
    ManualPrealignmentUI,
    PrealignmentConfig,
    PrealignmentResult,
    RasterizationConfig,
    SLDDMMConfig,
    STAtlasAlignmentConfig,
    STAtlasAlignmentResult,
    UIAtlasAlignmentConfig,
    UIAtlasPairing,
    HistologyClusteringConfig,
    HistologyClusteringResult,
    HistologyFeatureConfig,
    HistologyFeatureResult,
    HistologyPrealignmentConfig,
    HistologyPrealignmentResult,
    HistologyPrealignmentUI,
    STHistologyStructureConfig,
    STHistologyAlignmentConfig,
    STHistologyAlignmentResult,
    align_st_to_histology,
    align_atac_to_st,
    build_st_histology_structures,
    align_st_to_allen_atlas,
    align_st_to_allen_atlas_from_ui_pairs,
    align_cross_sample,
    apply_similarity_transform,
    build_st_cluster_hierarchy,
    interactive_manual_prealignment,
    interactive_histology_prealignment,
    load_allen_ccf_reference,
    load_atlas_label_color_map,
    load_atlas_structure_color_map,
    load_st_atlas_alignment,
    load_ui_atlas_pairing,
    load_histology_clustering,
    load_histology_features,
    plot_atlas_label_transfer,
    plot_alignment_result,
    plot_cluster_alignment_result,
    plot_manual_prealignment_preview,
    plot_prealignment_result,
    plot_rasterized_fields,
    plot_st_atlas_alignment,
    plot_histology_feature_clusters,
    plot_histology_prealignment_preview,
    plot_st_histology_alignment,
    plot_st_histology_structures,
    plot_atac_st_alignment,
    plot_atac_st_matched_structures,
    plot_atac_st_prealignment,
    prealign_cross_sample,
    prealign_cross_sample_manual,
    prealign_st_to_histology,
    prealign_atac_to_st,
    prepare_histology_image,
    cluster_histology_features,
    extract_histology_features,
    rasterize_cross_sample,
    run_slddmm_alignment,
)
from .io import (
    REQUIRED_OUTPUT_COLUMNS,
    load_cross_sample_data,
    load_single_sample_data,
    read_cross_sample_csv,
    read_single_sample_csv,
    validate_cross_sample_anndata,
    validate_single_sample_anndata,
)
from .inference import (
    LocalDEResult,
    PreparedInference,
    TrajectoryResult,
    acat_pvalue,
    cluster_trajectories,
    fit_local_de,
    plot_local_result,
    prepare_inference,
)
from .datasets import VisiumInferenceInput, build_visium_inference_table
from . import uncertainty


def cluster_joint(*args, **kwargs):
    """Lazily import the optional BANKSY/Harmony clustering workflow."""
    from .clustering import cluster_joint as implementation

    return implementation(*args, **kwargs)


def plot_joint_cluster_refinement(*args, **kwargs):
    """Plot raw and boundary-refined joint clusters with shared colors."""
    from .clustering import plot_joint_cluster_refinement as implementation

    return implementation(*args, **kwargs)


def cluster_single(*args, **kwargs):
    """Lazily import the optional single-sample BANKSY workflow."""
    from .clustering import cluster_single as implementation

    return implementation(*args, **kwargs)


def plot_single_cluster_refinement(*args, **kwargs):
    """Plot raw and boundary-refined single-sample clusters."""
    from .clustering import plot_single_cluster_refinement as implementation

    return implementation(*args, **kwargs)


def __getattr__(name):
    if name == "JointClusteringConfig":
        from .clustering import JointClusteringConfig

        return JointClusteringConfig
    if name == "SingleClusteringConfig":
        from .clustering import SingleClusteringConfig

        return SingleClusteringConfig
    raise AttributeError(name)


__all__ = [
    "__version__",
    "ATACSTAlignmentConfig",
    "ATACSTAlignmentResult",
    "ATACSTPrealignmentConfig",
    "ATACSTPrealignmentResult",
    "AllenCCFReference",
    "CrossSampleAlignmentResult",
    "CrossSampleFields",
    "JointClusteringConfig",
    "SingleClusteringConfig",
    "ManualPrealignmentConfig",
    "ManualPrealignmentUI",
    "LocalDEResult",
    "PrealignmentConfig",
    "PrealignmentResult",
    "PreparedInference",
    "REQUIRED_OUTPUT_COLUMNS",
    "RasterizationConfig",
    "SLDDMMConfig",
    "STAtlasAlignmentConfig",
    "STAtlasAlignmentResult",
    "UIAtlasAlignmentConfig",
    "UIAtlasPairing",
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
    "TrajectoryResult",
    "VisiumInferenceInput",
    "uncertainty",
    "acat_pvalue",
    "align_st_to_histology",
    "align_atac_to_st",
    "build_st_histology_structures",
    "align_st_to_allen_atlas",
    "align_st_to_allen_atlas_from_ui_pairs",
    "align_cross_sample",
    "apply_similarity_transform",
    "build_st_cluster_hierarchy",
    "build_visium_inference_table",
    "interactive_manual_prealignment",
    "interactive_histology_prealignment",
    "load_allen_ccf_reference",
    "load_atlas_label_color_map",
    "load_atlas_structure_color_map",
    "load_st_atlas_alignment",
    "load_ui_atlas_pairing",
    "load_histology_clustering",
    "load_histology_features",
    "cluster_joint",
    "cluster_single",
    "cluster_trajectories",
    "fit_local_de",
    "load_cross_sample_data",
    "load_single_sample_data",
    "plot_alignment_result",
    "plot_atlas_label_transfer",
    "plot_cluster_alignment_result",
    "plot_manual_prealignment_preview",
    "plot_local_result",
    "plot_joint_cluster_refinement",
    "plot_single_cluster_refinement",
    "plot_prealignment_result",
    "plot_rasterized_fields",
    "plot_st_atlas_alignment",
    "plot_histology_feature_clusters",
    "plot_histology_prealignment_preview",
    "plot_st_histology_alignment",
    "plot_st_histology_structures",
    "plot_atac_st_alignment",
    "plot_atac_st_matched_structures",
    "plot_atac_st_prealignment",
    "prealign_cross_sample",
    "prealign_cross_sample_manual",
    "prealign_st_to_histology",
    "prealign_atac_to_st",
    "prepare_inference",
    "prepare_histology_image",
    "cluster_histology_features",
    "extract_histology_features",
    "rasterize_cross_sample",
    "read_cross_sample_csv",
    "read_single_sample_csv",
    "run_slddmm_alignment",
    "validate_cross_sample_anndata",
    "validate_single_sample_anndata",
]
