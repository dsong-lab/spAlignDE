"""Configuration checks for alignment-mismatch risk estimation."""

from __future__ import annotations


def validate_density_energy_share(value: float) -> float:
    """Validate the density-channel feature-energy share used by the risk kernel.

    The value controls the target share of standardized feature-vector energy
    assigned to the density channel. It is not a direct linear weight on the
    final mismatch-risk map or its variance multiplier. The notebook
    implementation requires a value strictly between zero and one.
    """

    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("density_energy_share must be strictly between 0 and 1.")
    return value
