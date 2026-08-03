"""Dataset handoff helpers used by the spAlignDE tutorials."""

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
)

__all__ = [
    "KIDNEY_SAMPLES",
    "VisiumInferenceInput",
    "build_visium_coordinate_table",
    "build_visium_inference_table",
    "canonical_visium_barcodes",
    "kidney_alignment_metadata",
    "load_kidney_aligned_coordinates",
    "make_cross_sample_example",
]
