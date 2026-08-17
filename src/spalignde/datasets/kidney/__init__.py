"""Precomputed mouse-kidney alignment coordinates used by the tutorial."""

from __future__ import annotations

import json
from importlib.resources import files

import pandas as pd


KIDNEY_SAMPLES = ("NL3", "IL3")


def load_kidney_aligned_coordinates(sample_id: str) -> pd.DataFrame:
    """Load the packaged fixed-seed manual-alignment coordinates for one sample."""

    sample_id = str(sample_id).upper()
    if sample_id not in KIDNEY_SAMPLES:
        raise ValueError(
            f"sample_id must be one of {KIDNEY_SAMPLES}, got {sample_id!r}."
        )
    resource = files(__package__).joinpath(
        f"aligned_coords_{sample_id}.csv.gz"
    )
    with resource.open("rb") as stream:
        return pd.read_csv(stream, compression="gzip")


def kidney_alignment_metadata() -> dict:
    """Return provenance and schema metadata for the packaged coordinates."""

    resource = files(__package__).joinpath("metadata.json")
    with resource.open("r", encoding="utf-8") as stream:
        return json.load(stream)


__all__ = [
    "KIDNEY_SAMPLES",
    "kidney_alignment_metadata",
    "load_kidney_aligned_coordinates",
]
