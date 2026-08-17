Post-Alignment Inference
========================

These workflows test location-resolved expression differences after samples
have been transformed into a common coordinate frame. Both examples begin with
coordinates produced by spAlignDE and then construct mismatch risk, fit local
differential-expression tests and display the resulting spatial statistics.

Alignment-to-inference handoff
------------------------------

The alignment workflows save ``x_aligned`` and ``y_aligned`` as the explicit
handoff to inference; the inference notebooks consume these coordinates
without estimating another transformation. For kidney, the recorded source is
the fixed-seed manual-prealignment IL3-to-NL3 tutorial output (resolution
``0.2``, 5,000 iterations, seed ``1000``). The selected similarity transform
uses scale ``1``, rotation ``0`` degrees and translation
``(-36.20040965, -153.38356513)`` in the scaled alignment coordinates. IL3
contributes 2,965 transformed query spots, while the 3,215 NL3 reference
coordinates remain unchanged. The repository packages a compact,
hash-tracked copy of these coordinates; inference uses seed ``1`` and
``n_jobs=1``.

The aging-brain notebook is explicitly a five-section website example drawn
from the formal 19-query fixed-seed archive (resolution ``0.8``, 800
iterations). It uses the saved coordinates for the 6.6-, 15.8-, 30.9- and
34.5-month queries plus the unchanged 4.3-month reference; it does not rerun
alignment. See
:doc:`../tutorials/post_alignment_inference` for the coordinate contract,
scaling rule and complete kidney handoff.

Injured Mouse Kidney
--------------------

The kidney workflow continues from the fixed-seed Visium cross-sample
alignment with the selected manual initialization. It joins the packaged coordinates to
the public NL3 and IL3 10x counts by terminal barcode, then fits local tests for ``Cbr1``, ``Cd44``
and ``Myo5a``. The raw count matrices and tissue-position tables are available
from `Zenodo record 17676992 <https://zenodo.org/records/17676992>`_. Users can
replace the packaged coordinates with their own spAlignDE output through
``SPALIGNDE_KIDNEY_ALIGNED_H5AD`` or ``SPALIGNDE_ALIGNMENT_DIR``.
The upstream clustering/alignment input and region annotations are documented
separately under `STcompare record 20647680
<https://zenodo.org/records/20647680>`_; aligned coordinates and source 10x
tables are joined by terminal barcode rather than row order.

.. toctree::
   :maxdepth: 1

   post_alignment_inference_nb

The saved notebook is fully executed. For every representative gene it shows
NL3 expression, IL3 expression, the zero-centered mismatch-aware local
statistic, direct ``q < 0.05`` contours and the gene-level ACAT P value. The
complete input contract and statistical model are documented in
:doc:`../tutorials/post_alignment_inference`.

Aging Mouse Brain
-----------------

The aging-brain workflow compares the 6.6-, 15.8-, 30.9- and 34.5-month
MERFISH sections with a 4.3-month alignment reference and fits local tests for
``Gamt`` and ``Vip``.
The original 300-gene MERFISH data were published with `Spatial transcriptomic
clocks reveal cell proximity effects in brain ageing
<https://doi.org/10.1038/s41586-024-08334-8>`_ and are available from `Zenodo
record 13883177 <https://doi.org/10.5281/zenodo.13883177>`_.

The current PyPI source contains public raw counts and annotations for the
compact Figure 5A subset. The repository coordinate handoff is generated from
the formal 19-query archive and supplies the exact ``x_aligned`` and
``y_aligned`` values used by the notebook. No external download or alignment
rerun is required for the packaged example.

.. toctree::
   :maxdepth: 1

   post_alignment_inference_aging_brain_nb

The saved notebook uses the automatic shared grid, a broad 300-gene
mismatch-risk candidate panel, local technical covariates, a gene-specific
global offset, mismatch-aware variance adjustment and connected-region
cleanup. Cell-type adjustment is disabled, and grid-level significance is
reported at ``q <= 0.05``. The reported multi-age P value comes from
``gene_level_age_trend_acat``; the older any-spatial-change ACAT is displayed
only as a diagnostic. Automatic trajectory K is selected by the public
``cluster_trajectories`` function, and the notebook displays its
``fine_to_coarse_scan`` metadata.

For either workflow, tune inference only after geometric QC passes. Risk genes
should be broad and selected independently of tested genes; cell-type
adjustment requires complete validated labels in every sample. See
:doc:`../tutorials/parameter_tuning` for filtering, density-risk and region-
cleanup guidance.

The public kidney and aging-brain notebooks use workflow seed 1 and
``n_jobs=1`` for both inference preparation and fitting. The saved executions
contain 6,187 shared-grid locations for kidney and 75,868 for the compact
aging-brain example. Parallel workers can be used for exploratory acceleration,
but their diagnostic messages may be emitted in a different order even when
the fitted scientific results are unchanged.
