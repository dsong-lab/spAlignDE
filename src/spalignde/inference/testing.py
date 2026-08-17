"""Run mismatch-aware local spatial differential-expression tests."""

from __future__ import annotations

from collections.abc import Sequence

from . import _legacy_core
from ._calibration import (
    MISMATCH_CALIBRATION_MODE,
    MULTI_CONTRAST_CALIBRATION_MODE,
)
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

    When `cell_type_adjustment=True`, kernel-smoothed local cell-type
    proportions are compared with the normalized Jensen--Shannon distance.
    The resulting distance lies in [0, 1] and enters the variance through
    `exp(distance)`, so identical compositions receive factor 1 and the maximum
    possible adjustment is `e`.

    `technical_adjustment=True` includes local covariates that capture
    confounding variation such as effects of local library size and detection
    rate. `global_offset` separately controls the gene-specific condition
    offset in the mean model.

    When `mismatch_aware=True`, each gene is first fitted without alignment-risk
    inflation. Its local statistics are binned by normalized local risk; each
    bin is median-centered, and its MAD is standardized by the Student-t null
    MAD. Nonnegative excess variance is constrained to be nondecreasing with
    risk, and a weighted quadratic calibration through the origin is boundedly
    rescaled at the risk bin nearest the 80th percentile to estimate a
    provisional gene-by-contrast coefficient. With multiple contrasts, valid
    provisional coefficients are combined by an equal-weight Huber center to
    obtain one robust gene-specific coefficient shared across contrasts. This
    avoids dependence on a single chosen contrast while preventing each
    contrast from calibrating its own test in isolation. A successful
    single-contrast fit is unchanged because its Huber center is the original
    coefficient. Validity requires a successful within-contrast fit, adequate
    usable locations and at least four distinct risk bins with positive-risk
    support, finite calibration quantities, a finite nonnegative capped local
    coefficient, and a zero global coefficient. A successful zero coefficient
    remains valid; failed contrasts are excluded. The mismatch variance factor
    equals one at zero normalized local risk. Cell-type composition is a
    separate variance adjustment.

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
        celltype_adjustment=bool(cell_type_adjustment),
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
    for gene, fit in fits.items():
        terrain = fit.get("terrain_data", {}) if isinstance(fit, dict) else {}
        risk_calibration = (
            terrain.get("risk_calibration") if isinstance(terrain, dict) else None
        )
        if not mismatch_aware:
            calibration_modes[gene] = "none"
            lambda_local_by_gene[gene] = 0.0
            continue
        if not isinstance(risk_calibration, dict):
            calibration_modes[gene] = "unavailable"
            lambda_local_by_gene[gene] = 0.0
            continue
        calibration_modes[gene] = str(
            risk_calibration.get("method", "unavailable")
        )
        lambda_local_by_gene[gene] = float(
            risk_calibration.get("lambda_local_hat", 0.0)
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
            "cell_type_adjustment_method": (
                prepared.shared.get("celltype_adjustment_info", {}).get("method")
                if cell_type_adjustment
                else "none"
            ),
            "global_offset": bool(global_offset),
            "region_cleanup": bool(region_cleanup),
            "n_jobs": max(int(n_jobs), 1),
            "mismatch_calibration": (
                MULTI_CONTRAST_CALIBRATION_MODE if mismatch_aware else "none"
            ),
            "within_contrast_mismatch_calibration": (
                MISMATCH_CALIBRATION_MODE if mismatch_aware else "none"
            ),
            "mismatch_calibration_by_gene": calibration_modes,
            "mismatch_lambda_local_by_gene": lambda_local_by_gene,
        },
    )
