"""Run mismatch-aware local spatial differential-expression tests."""

from __future__ import annotations

from collections.abc import Sequence

from . import _legacy_core
from ._calibration import MISMATCH_CALIBRATION_MODE
from ._types import LocalDEResult, PreparedInference


def fit_local_de(
    prepared: PreparedInference,
    *,
    genes: Sequence[str] | None = None,
    contrast: str = "vs_reference",
    alpha: float = 0.05,
    mismatch_aware: bool = True,
    technical_adjustment: bool = True,
    cell_type_adjustment: bool = True,
    global_offset: bool = True,
    region_cleanup: bool = False,
    n_jobs: int = 1,
    random_state: int | None = None,
    verbose: bool = True,
) -> LocalDEResult:
    """Fit local tests for one or more genes.

    `contrast="vs_reference"` compares every non-reference sample to the
    reference. `contrast="sequential"` compares each ordered sample with its
    predecessor. The BH adjustment is applied separately within each gene and
    sample contrast across valid grid locations.

    When `cell_type_adjustment=True`, the fitted kernel uses its
    `sampling_gap_adjust` path. It builds local shared cell-type coverage from
    smoothed cell-type proportions and type-specific effective support, then
    multiplies the precision weight by the resulting support precision. Because
    the weight is inverse variance, this is equivalent to variance inflation by
    the reciprocal support precision at low-support locations.

    When `mismatch_aware=True`, each gene is first fitted without alignment-risk
    inflation. Its local statistics are binned by normalized local risk; each
    bin is median-centered, and its MAD is standardized by the Student-t null
    MAD. Nonnegative excess variance is constrained to be nondecreasing with
    risk, and a weighted quadratic calibration through the origin is boundedly
    rescaled at the risk bin nearest the 80th percentile to estimate the
    gene-specific local coefficient. The origin constraint fixes the
    gene-specific global mismatch coefficient at zero. Cell-type support is a
    separate precision adjustment. With multiple query samples, the first
    available contrast is the calibration anchor and its gene-specific
    coefficient is reused across the remaining contrasts.

    When `region_cleanup=True`, isolated calls and unsupported fragments are
    removed from the reported significant-region mask after grid-level testing.
    The local statistics and q-values are unchanged.
    """

    if not isinstance(prepared, PreparedInference):
        raise TypeError("prepared must be returned by prepare_inference.")
    if contrast not in {"vs_reference", "sequential"}:
        raise ValueError("contrast must be 'vs_reference' or 'sequential'.")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    if cell_type_adjustment and not bool(prepared.metadata.get("cell_type_available", True)):
        raise ValueError(
            "cell_type_adjustment=True requires a complete cell-type annotation "
            "column in prepare_inference(). Set cell_type_adjustment=False when "
            "reliable cell-type annotations are unavailable."
        )
    gene_list = list(prepared.genes if genes is None else genes)
    unknown = set(gene_list).difference(prepared.genes)
    if unknown:
        raise ValueError(f"Genes were not included in prepare_inference: {sorted(unknown)}")

    fits = _legacy_core.batch_run_one_gene_and_save_multi_conditional(
        prepared.shared,
        gene_list,
        alpha=float(alpha),
        include_Psi=bool(technical_adjustment),
        sampling_gap_adjust=bool(cell_type_adjustment),
        include_intercept=bool(global_offset),
        time_contrast="vs_ref" if contrast == "vs_reference" else "sequential",
        risk_in_Wv=bool(mismatch_aware),
        traj_pretest=False,
        screen_consensus=False,
        drawmask_cleanup=bool(region_cleanup),
        core=max(int(n_jobs), 1),
        seed=random_state,
        show_progress=bool(verbose),
    )
    # The legacy kernel returns a single fitted-gene object when one gene is
    # requested, and a gene-to-fit mapping otherwise. Normalize both forms.
    if not isinstance(fits, dict) or "gene" in fits:
        fits = {gene_list[0]: fits}

    calibration_modes: dict[str, str] = {}
    lambda_local_by_gene: dict[str, float] = {}
    lambda_global_by_gene: dict[str, float] = {}
    for gene, fit in fits.items():
        terrain = fit.get("terrain_data", {}) if isinstance(fit, dict) else {}
        risk_calibration = (
            terrain.get("risk_calibration") if isinstance(terrain, dict) else None
        )
        if not mismatch_aware:
            calibration_modes[gene] = "none"
            lambda_local_by_gene[gene] = 0.0
            lambda_global_by_gene[gene] = 0.0
            continue
        if not isinstance(risk_calibration, dict):
            calibration_modes[gene] = "unavailable"
            lambda_local_by_gene[gene] = 0.0
            lambda_global_by_gene[gene] = 0.0
            continue
        detail = risk_calibration.get("calibration")
        if not isinstance(detail, dict):
            detail = risk_calibration.get("empnull")
        diag = detail.get("diag", {}) if isinstance(detail, dict) else {}
        calibration_modes[gene] = str(diag.get("mode", "unavailable"))
        lambda_local_by_gene[gene] = float(
            risk_calibration.get("lambda_local_hat", 0.0)
        )
        lambda_global_by_gene[gene] = float(
            risk_calibration.get("lambda_global_hat", 0.0)
        )

    return LocalDEResult(
        fits=fits,
        prepared=prepared,
        alpha=float(alpha),
        contrast=contrast,
        mismatch_aware=bool(mismatch_aware),
        technical_adjustment=bool(technical_adjustment),
        metadata={
            "cell_type_adjustment": bool(cell_type_adjustment),
            "global_offset": bool(global_offset),
            "region_cleanup": bool(region_cleanup),
            "n_jobs": max(int(n_jobs), 1),
            "mismatch_calibration": (
                MISMATCH_CALIBRATION_MODE if mismatch_aware else "none"
            ),
            "mismatch_calibration_by_gene": calibration_modes,
            "mismatch_lambda_local_by_gene": lambda_local_by_gene,
            "mismatch_lambda_global_by_gene": lambda_global_by_gene,
        },
    )
