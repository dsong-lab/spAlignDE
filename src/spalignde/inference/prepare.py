"""Prepare shared-grid objects for mismatch-aware local inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import _legacy_core
from ._types import PreparedInference
from .risk import validate_density_energy_share


_REQUIRED_COLUMNS = ("sample_id", "x", "y", "batch")
_MISSING_CELL_TYPE = "__spalignde_celltype_unavailable__"
_METADATA_COLUMNS = {
    "barcode",
    "cell_id",
    "condition",
    "spot_id",
    "sample_id",
    "x",
    "y",
    "x_aligned",
    "y_aligned",
    "batch",
    "celltype",
}


def _rename_input_columns(
    data: pd.DataFrame,
    *,
    sample_key: str,
    coordinate_key: tuple[str, str],
    aligned_coordinate_key: tuple[str, str] | None,
    batch_key: str,
    cell_type_key: str | None,
) -> pd.DataFrame:
    rename = {
        sample_key: "sample_id",
        coordinate_key[0]: "x",
        coordinate_key[1]: "y",
        batch_key: "batch",
    }
    if cell_type_key is not None and cell_type_key in data.columns:
        rename[cell_type_key] = "celltype"
    if aligned_coordinate_key is not None:
        rename.update({
            aligned_coordinate_key[0]: "x_aligned",
            aligned_coordinate_key[1]: "y_aligned",
        })
    if len(set(rename.values())) != len(rename):
        raise ValueError("Column mapping assigns more than one input column to the same role.")
    return data.rename(columns=rename).copy()


def _infer_expression_columns(frame: pd.DataFrame) -> list[str]:
    """Return non-negative numeric columns eligible for risk estimation."""

    genes: list[str] = []
    for column in frame.columns:
        if column in _METADATA_COLUMNS:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and finite.min() >= 0 and np.mean(finite > 0) >= 0.001:
            genes.append(str(column))
    return genes


def prepare_inference(
    data: pd.DataFrame,
    *,
    reference: str,
    genes: Sequence[str],
    risk_genes: Sequence[str] | None = None,
    sample_key: str = "sample_id",
    coordinate_key: tuple[str, str] = ("x", "y"),
    aligned_coordinate_key: tuple[str, str] | None = ("x_aligned", "y_aligned"),
    batch_key: str = "batch",
    cell_type_key: str | None = "celltype",
    control_samples: Sequence[str] | None = None,
    alignment_uncertainty_key: str | None = None,
    density_energy_share: float = 0.25,
    library_size: float | None = None,
    grid_n: int | None = None,
    n_jobs: int = 1,
    random_state: int | None = None,
) -> PreparedInference:
    """Build the fixed shared-grid analysis object.

    Parameters
    ----------
    data
        Long spot/cell table. Expression columns must be non-negative numeric
        values. `coordinate_key` contains original coordinates; when
        `aligned_coordinate_key` is supplied, those coordinates define the
        shared local-inference space.
    reference
        Identifier of the reference sample.
    genes
        Genes intended for downstream fitting. They are retained in the public
        prepared object but do not limit mismatch-risk estimation.
    risk_genes
        Candidate expression columns for stable-gene mismatch-risk estimation.
        If omitted, all eligible non-negative numeric expression columns in
        `data` are candidates. The kernel screens this pool and retains up to
        300 stable/informative markers; the input pool does not need to contain
        300 genes. Supplying only the genes being tested is usually
        inappropriate because risk estimation requires a broad candidate set.
    library_size
        `None` leaves input expression unchanged. A positive number rescales
        each spot/cell to this library size before local testing.
    alignment_uncertainty_key
        Optional spot-level alignment-uncertainty column. When omitted,
        mismatch risk is estimated from stable-gene profiles and local density.
    density_energy_share
        Target share of the standardized mismatch-feature-vector energy assigned
        to the density channel before the comparison-level risk map is formed.
        It is not a direct linear weight on the final risk or variance factor.
        The current validated kernel accepts values in `(0, 1)`.
    grid_n
        Explicit Cartesian grid resolution. By default, the R-driven candidate
        is retained when its estimated number of tissue-valid locations lies
        between the median per-sample observation count and twice that count;
        otherwise resolution is adjusted toward the corresponding bound after
        accounting for tissue-mask occupancy. An explicit value takes priority
        over this automatic location-count rule.
    n_jobs
        Worker count for parallel preparation stages. Seeded auto-geometry
        subsampling and parameter estimation are run serially so their result
        does not depend on thread scheduling.
    random_state
        Seed used by stochastic preparation steps, including auto-geometry
        subsampling when a sample exceeds the internal size threshold.
    cell_type_key
        Cell-type annotation column. The column may be absent when downstream
        fitting uses `cell_type_adjustment=False`. The optional composition
        adjustment compares kernel-smoothed local cell-type proportions, so it
        requires a complete, non-missing annotation column.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame in the initial release.")
    if not genes:
        raise ValueError("genes must contain at least one expression column.")
    if library_size is not None and float(library_size) <= 0:
        raise ValueError("library_size must be None or a positive number.")
    if grid_n is not None:
        if isinstance(grid_n, bool) or int(grid_n) != grid_n or int(grid_n) < 2:
            raise ValueError("grid_n must be None or an integer of at least 2.")
        grid_n = int(grid_n)
    density_energy_share = validate_density_energy_share(density_energy_share)
    n_jobs = max(int(n_jobs), 1)

    frame = _rename_input_columns(
        data,
        sample_key=sample_key,
        coordinate_key=coordinate_key,
        aligned_coordinate_key=aligned_coordinate_key,
        batch_key=batch_key,
        cell_type_key=cell_type_key,
    )
    missing = set(_REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns after mapping: {sorted(missing)}")
    missing_genes = set(genes).difference(frame.columns)
    if missing_genes:
        raise ValueError(f"Expression columns not found: {sorted(missing_genes)}")
    if alignment_uncertainty_key is not None and alignment_uncertainty_key not in frame.columns:
        raise ValueError(f"alignment_uncertainty_key={alignment_uncertainty_key!r} is not a column in data.")

    cell_type_available = False
    n_missing_cell_types = len(frame)
    if "celltype" in frame.columns:
        cell_type_values = frame["celltype"].astype("string")
        valid_cell_types = (
            cell_type_values.notna()
            & cell_type_values.str.strip().ne("")
            & ~cell_type_values.str.lower().isin({"nan", "none"})
        )
        n_missing_cell_types = int((~valid_cell_types).sum())
        cell_type_available = bool(len(frame) and valid_cell_types.all())
        frame["celltype"] = cell_type_values.where(
            valid_cell_types,
            _MISSING_CELL_TYPE,
        ).astype(str)
    else:
        frame["celltype"] = _MISSING_CELL_TYPE

    inferred_risk_genes = _infer_expression_columns(frame) if risk_genes is None else list(risk_genes)
    inferred_risk_genes = list(dict.fromkeys([*inferred_risk_genes, *genes]))
    missing_risk_genes = set(inferred_risk_genes).difference(frame.columns)
    if missing_risk_genes:
        raise ValueError(f"risk_genes not found in data: {sorted(missing_risk_genes)}")
    if len(inferred_risk_genes) < 10:
        raise ValueError(
            "Mismatch-risk estimation requires at least 10 expression genes. "
            "Provide a broader risk_genes set or an input table with more genes."
        )

    shared = _legacy_core.batch_prepare_once_multi(
        frame,
        ref_sample_id=str(reference),
        control_ids=None if control_samples is None else [str(x) for x in control_samples],
        user_var_col=alignment_uncertainty_key or "__spalignde_no_user_risk__",
        s=density_energy_share,
        core=n_jobs,
        seed=random_state,
        use_libsize_norm=library_size is not None,
        target_total=1.0 if library_size is None else float(library_size),
        gene_cols_hint=inferred_risk_genes,
        grid_n=grid_n,
    )
    return PreparedInference(
        shared=shared,
        data=frame,
        genes=tuple(genes),
        reference=str(reference),
        library_size=None if library_size is None else float(library_size),
        density_energy_share=density_energy_share,
        alignment_uncertainty_key=alignment_uncertainty_key,
        metadata={
            "n_jobs": n_jobs,
            "auto_geometry_n_jobs": 1,
            "random_state": random_state,
            "risk_genes": tuple(inferred_risk_genes),
            "cell_type_available": cell_type_available,
            "n_missing_cell_types": n_missing_cell_types,
            "cell_type_adjustment_method": (
                shared.get("celltype_adjustment_info", {}).get("method")
                if cell_type_available
                else None
            ),
            "shared_grid_spacing": float(shared["grid_spacing"]),
            "risk_map_radius": float(shared["R_map"]),
            "risk_map_grid_multiplier": float(
                shared["PARAMS"]["uncert"]["R_map_grid_multiplier"]
            ),
            "grid_n": int(shared["PARAMS"]["grid_n"]),
            "grid_n_source": shared["auto_geometry"]["diagnostics"]["selection"],
            "n_typ": float(shared["auto_geometry"]["diagnostics"]["n_typ"]),
            "target_grid_locations": tuple(
                shared["auto_geometry"]["diagnostics"]["target_grid_locations"]
            ),
            "shared_grid_locations": int(len(shared["grid_eval"])),
        },
    )
