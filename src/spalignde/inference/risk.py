"""Configuration checks for alignment-mismatch risk estimation."""

from __future__ import annotations


def validate_density_energy_share(value: float) -> float:
    """Validate the density-channel share used by the current risk kernel.

    The notebook implementation requires a value strictly between zero and
    one. A later refactor may support zero as an explicit density-channel-off
    setting, but that is not part of the validated implementation yet.
    """

    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("density_energy_share must be strictly between 0 and 1.")
    return value
