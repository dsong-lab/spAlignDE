"""Dataset handoff helpers used by the spAlignDE tutorials."""

from .aging_brain import (
    AGING_BRAIN_QUERIES,
    AGING_BRAIN_REFERENCE,
    AGING_BRAIN_SAMPLES,
    aging_brain_genes,
    aging_brain_metadata,
    load_aging_brain,
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
    "AGING_BRAIN_QUERIES",
    "AGING_BRAIN_REFERENCE",
    "AGING_BRAIN_SAMPLES",
    "KIDNEY_SAMPLES",
    "VisiumInferenceInput",
    "aging_brain_genes",
    "aging_brain_metadata",
    "build_visium_coordinate_table",
    "build_visium_inference_table",
    "canonical_visium_barcodes",
    "kidney_alignment_metadata",
    "load_aging_brain",
    "load_kidney_aligned_coordinates",
    "make_cross_sample_example",
    "summarize_raw_genes",
]
