"""Mismatch-aware post-alignment spatial differential expression."""

from ._types import LocalDEResult, PreparedInference, TrajectoryResult
from .plotting import plot_local_result
from .prepare import prepare_inference
from .summaries import acat_pvalue, cluster_trajectories, gene_level_acat_pvalue
from .testing import fit_local_de

__all__ = [
    "LocalDEResult",
    "PreparedInference",
    "TrajectoryResult",
    "acat_pvalue",
    "cluster_trajectories",
    "gene_level_acat_pvalue",
    "fit_local_de",
    "plot_local_result",
    "prepare_inference",
]
