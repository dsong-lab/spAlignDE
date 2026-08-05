Post-Alignment Inference - Injured Mouse Kidney
================================================

This section continues from the Visium kidney cross-sample workflow. The
notebook joins the packaged manuscript ``aligned_317`` coordinates to raw NL3
and IL3 10x counts by terminal barcode, estimates post-alignment mismatch risk,
and fits local tests for ``Cbr1``, ``Cd44`` and ``Myo5a``. Users can replace the
packaged coordinates with their own alignment output through
``SPALIGNDE_ALIGNMENT_DIR``.

All reusable data preparation is implemented by public package functions. The
notebook contains no local function definitions:
``build_visium_coordinate_table`` performs the position/alignment handoff and
``build_visium_inference_table`` reads raw counts, summarizes gene support,
selects the risk-gene pool and constructs the long inference table.

Input data
----------

The NL3 and IL3 count matrices and tissue-position tables are available from
`Zenodo record 17676992 <https://zenodo.org/records/17676992>`_. The packaged
coordinate files contain identifiers and aligned coordinates only; no raw
expression or precomputed local-inference result is bundled. The complete
input/output contract and statistical model are documented in
:doc:`../tutorials/post_alignment_inference`.

Source notebook
---------------

.. toctree::
   :maxdepth: 1

   post_alignment_inference_nb

The saved notebook is fully executed. For every representative gene it shows
NL3 expression, IL3 expression, the zero-centered Mismatch-aware local
statistic, and red contours directly tracing the ``q < 0.05`` grid mask. It
also reports the gene-level ACAT P value computed from retained raw local P
values. The notebook uses ``region_cleanup=False`` so the displayed regions
match the manuscript plotting logic.

Tune inference only after geometric QC passes. Risk genes should be broad and
selected independently of the tested genes; cell-type adjustment requires
complete validated labels in every sample. See
:doc:`../tutorials/parameter_tuning` for filtering, density-risk and region-
cleanup guidance.
