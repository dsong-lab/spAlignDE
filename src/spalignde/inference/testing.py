"""Run mismatch-aware local spatial differential-expression tests."""

from __future__ import annotations

from collections.abc import Sequence

from . import _legacy_core
from ._types import LocalDEResult, PreparedInference


def fit_local_de(
    prepared: PreparedInference,
    *,
    genes: Sequence[str] | None = None,
    contrast: str = "vs_reference",
    alpha: float = 0.05,
    mismatch_aware: bool = True,
    technical_adjustment: bool = True,
    cell_type_adjustment: bool = False,
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

    Cell-type adjustment is disabled by default because many public spatial
    inputs do not contain complete cell-type annotations. When
    `cell_type_adjustment=True`, the fitted kernel uses its
    `sampling_gap_adjust` path. It builds local shared cell-type coverage from
    smoothed cell-type proportions and type-specific effective support, then
    multiplies the precision weight by the resulting support precision. Because
    the weight is inverse variance, this is equivalent to variance inflation by
    the reciprocal support precision at low-support locations.

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
        },
    )
