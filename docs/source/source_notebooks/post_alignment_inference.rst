Post-Alignment Inference - Injured Mouse Kidney
================================================

This section continues from the Visium kidney cross-sample workflow. The
notebook reads the final ``kidney_IL3_to_NL3_aligned.h5ad`` output, joins raw
NL3 and IL3 10x counts by barcode, estimates post-alignment mismatch risk and
fits Mismatch-aware local differential-expression tests for ``Cbr1``,
``Cd44`` and ``Slc5a2``.

The H5AD/raw-count handoff is implemented by the public
``spAlignDE.build_visium_inference_table`` function; the notebook contains no
dataset-specific reader, barcode-matching or risk-gene-filtering function.

Input data
----------

The NL3 and IL3 count matrices are available from the `STcompare Zenodo record
<https://zenodo.org/records/19486091>`_. Run the kidney clustering and
alignment notebooks first; the alignment notebook produces the standardized
H5AD consumed here. The complete input/output contract and statistical model
are documented in :doc:`../tutorials/post_alignment_inference`.

Source notebook
---------------

.. toctree::
   :maxdepth: 1

   post_alignment_inference_nb

The saved notebook is fully executed. It shows the original and aligned
geometry, the mismatch-risk map, grid-level result summary, and the expression
and local-statistic panels for all three representative genes.

Tune inference only after geometric QC passes. Risk genes should be broad and
selected independently of the tested genes; cell-type adjustment requires
complete validated labels in every sample. See
:doc:`../tutorials/parameter_tuning` for filtering, density-risk and region-
cleanup guidance.
