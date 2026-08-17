"""Typed public result containers for mismatch-aware inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class PreparedInference:
    """Fixed shared-grid quantities reused across genes."""

    shared: dict[str, Any]
    data: pd.DataFrame
    genes: tuple[str, ...]
    reference: str
    library_size: float | None
    density_energy_share: float
    alignment_uncertainty_key: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LocalDEResult:
    """Local differential-expression result for one or more genes."""

    fits: dict[str, dict[str, Any]]
    prepared: PreparedInference
    alpha: float
    contrast: str
    mismatch_aware: bool
    technical_adjustment: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, gene: str) -> dict[str, Any]:
        return self.fits[gene]


@dataclass(slots=True)
class TrajectoryResult:
    """Trajectory-clustering output for one fitted gene."""

    result: dict[str, Any]
    gene: str
    n_clusters: int | str
    metadata: dict[str, Any] = field(default_factory=dict)

