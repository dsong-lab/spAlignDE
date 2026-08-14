Post-Alignment Inference
========================

These workflows test location-resolved expression differences after samples
have been transformed into a common coordinate frame. Both examples begin with
coordinates produced by spAlignDE and then construct mismatch risk, fit local
differential-expression tests and display the resulting spatial statistics.

Injured Mouse Kidney
--------------------

The kidney workflow continues from the validated fixed-seed Visium
cross-sample alignment (seed ``1000``). It joins the packaged coordinates to
the public NL3 and IL3 10x counts by terminal barcode, then fits local tests for ``Cbr1``, ``Cd44``
and ``Myo5a``. The raw count matrices and tissue-position tables are available
from `Zenodo record 17676992 <https://zenodo.org/records/17676992>`_. Users can
replace the packaged coordinates with their own spAlignDE output through
``SPALIGNDE_ALIGNMENT_DIR``.

.. toctree::
   :maxdepth: 1

   post_alignment_inference_nb

The saved notebook is fully executed. For every representative gene it shows
NL3 expression, IL3 expression, the zero-centered Mismatch-aware local
statistic, direct ``q < 0.05`` contours and the gene-level ACAT P value. The
complete input contract and statistical model are documented in
:doc:`../tutorials/post_alignment_inference`.

Aging Mouse Brain
-----------------

The aging-brain workflow compares the 6.6-, 15.8-, 30.9- and 34.5-month
MERFISH sections with a 4.3-month reference and fits a local test for ``Gamt``.
The original 300-gene MERFISH data were published with `Spatial transcriptomic
clocks reveal cell proximity effects in brain ageing
<https://doi.org/10.1038/s41586-024-08334-8>`_ and are available from `Zenodo
record 13883177 <https://doi.org/10.5281/zenodo.13883177>`_.

The package contains a compact five-section subset with public raw counts,
original coordinates and cell-type labels, plus coordinates from the validated
fixed-seed cross-sample workflow (seed ``1000``). The notebook therefore
starts from the aligned ``x_aligned`` and ``y_aligned`` fields and does not
rerun alignment. No external download is required to execute the packaged
example.

.. toctree::
   :maxdepth: 1

   post_alignment_inference_aging_brain_nb

The saved notebook uses the automatic shared grid, a broad 300-gene
mismatch-risk candidate panel, local technical covariates, a gene-specific
global offset, mismatch-aware variance adjustment and connected-region
cleanup. Cell-type adjustment is disabled, and grid-level significance is
reported at ``q <= 0.05``.

For either workflow, tune inference only after geometric QC passes. Risk genes
should be broad and selected independently of tested genes; cell-type
adjustment requires complete validated labels in every sample. See
:doc:`../tutorials/parameter_tuning` for filtering, density-risk and region-
cleanup guidance.

The public kidney and aging-brain notebooks use workflow seed 1 and
``n_jobs=1`` for both inference preparation and fitting. Independent clean
runs reproduced all sanitized saved outputs exactly: 6,187 shared-grid
locations for kidney and 76,124 for aging brain. Parallel workers can be used
for exploratory acceleration, but their diagnostic messages may be emitted in
a different order even when the fitted scientific results are unchanged.
