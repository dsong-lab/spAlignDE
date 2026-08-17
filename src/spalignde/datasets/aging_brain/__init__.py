"""Packaged aligned aging mouse-brain data for post-alignment inference."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files

import pandas as pd
from scipy import sparse


AGING_BRAIN_REFERENCE = "age_4.3"
AGING_BRAIN_QUERIES = (
    "age_6.6",
    "age_15.8",
    "age_30.9",
    "age_34.5",
)
AGING_BRAIN_SAMPLES = (
    AGING_BRAIN_REFERENCE,
    *AGING_BRAIN_QUERIES,
)


def aging_brain_metadata() -> dict:
    """Return source, schema, and manuscript settings for the dataset."""

    resource = files(__package__).joinpath("metadata.json")
    with resource.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def aging_brain_genes() -> tuple[str, ...]:
    """Return the ordered 300-gene candidate panel stored with the dataset."""

    resource = files(__package__).joinpath("genes.json")
    with resource.open("r", encoding="utf-8") as stream:
        return tuple(json.load(stream))


def _normalize_selection(
    values: Sequence[str] | None,
    *,
    available: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    selected = available if values is None else tuple(str(value) for value in values)
    if not selected:
        raise ValueError(f"{label} must contain at least one value.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{label} contains duplicate values.")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}.")
    return selected


def load_aging_brain(
    *,
    samples: Sequence[str] | None = None,
    genes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load the compact five-section MERFISH dataset as a long table.

    The default loads all five sections and all 300 genes. Keeping the full
    gene panel is required for the manuscript mismatch-risk construction;
    ``genes`` is primarily useful for inspecting the packaged data.
    """

    available_genes = aging_brain_genes()
    selected_samples = _normalize_selection(
        samples,
        available=AGING_BRAIN_SAMPLES,
        label="samples",
    )
    selected_genes = _normalize_selection(
        genes,
        available=available_genes,
        label="genes",
    )
    gene_indices = [available_genes.index(gene) for gene in selected_genes]
    package_files = files(__package__)
    frames: list[pd.DataFrame] = []

    for sample_id in selected_samples:
        stem = sample_id.replace(".", "_")
        observation_resource = package_files.joinpath(
            f"{stem}_observations.csv.gz"
        )
        count_resource = package_files.joinpath(f"{stem}_counts.npz")
        with observation_resource.open("rb") as stream:
            observations = pd.read_csv(stream, compression="gzip")
        with count_resource.open("rb") as stream:
            counts = sparse.load_npz(stream).tocsr()[:, gene_indices]
        if counts.shape[0] != len(observations):
            raise RuntimeError(
                f"Packaged observations and counts disagree for {sample_id}."
            )
        expression = pd.DataFrame(
            counts.toarray(),
            columns=selected_genes,
            index=observations.index,
        )
        frames.append(pd.concat([observations, expression], axis=1))

    return pd.concat(frames, ignore_index=True)


# Explicit Figure 5A names used by the current inference documentation.  The
# generic 0.1 names above remain public aliases for backward compatibility.
AGING_BRAIN_FIGURE5A_REFERENCE = AGING_BRAIN_REFERENCE
AGING_BRAIN_FIGURE5A_QUERIES = AGING_BRAIN_QUERIES
AGING_BRAIN_FIGURE5A_SAMPLES = AGING_BRAIN_SAMPLES
aging_brain_figure5a_genes = aging_brain_genes
aging_brain_figure5a_metadata = aging_brain_metadata
load_aging_brain_figure5a = load_aging_brain


__all__ = [
    "AGING_BRAIN_FIGURE5A_QUERIES",
    "AGING_BRAIN_FIGURE5A_REFERENCE",
    "AGING_BRAIN_FIGURE5A_SAMPLES",
    "AGING_BRAIN_QUERIES",
    "AGING_BRAIN_REFERENCE",
    "AGING_BRAIN_SAMPLES",
    "aging_brain_figure5a_genes",
    "aging_brain_figure5a_metadata",
    "aging_brain_genes",
    "aging_brain_metadata",
    "load_aging_brain_figure5a",
    "load_aging_brain",
]
