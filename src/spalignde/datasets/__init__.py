"""Dataset handoff helpers used by the spAlignDE tutorials."""

from .aging_brain import (
    AGING_BRAIN_FIGURE5A_QUERIES,
    AGING_BRAIN_FIGURE5A_REFERENCE,
    AGING_BRAIN_FIGURE5A_SAMPLES,
    aging_brain_figure5a_genes,
    aging_brain_figure5a_metadata,
    load_aging_brain_figure5a,
)
from .examples import make_cross_sample_example
from .kidney import (
    KIDNEY_SAMPLES,
    kidney_alignment_metadata,
    load_kidney_aligned_coordinates,
)
from .visium import (
    VisiumInferenceInput,
    build_visium_coordinate_table,
    build_visium_inference_table,
    canonical_visium_barcodes,
    summarize_raw_genes,
)

__all__ = [
    "AGING_BRAIN_FIGURE5A_QUERIES",
    "AGING_BRAIN_FIGURE5A_REFERENCE",
    "AGING_BRAIN_FIGURE5A_SAMPLES",
    "KIDNEY_SAMPLES",
    "VisiumInferenceInput",
    "aging_brain_figure5a_genes",
    "aging_brain_figure5a_metadata",
    "build_visium_coordinate_table",
    "build_visium_inference_table",
    "canonical_visium_barcodes",
    "kidney_alignment_metadata",
    "load_aging_brain_figure5a",
    "load_kidney_aligned_coordinates",
    "make_cross_sample_example",
    "summarize_raw_genes",
]
