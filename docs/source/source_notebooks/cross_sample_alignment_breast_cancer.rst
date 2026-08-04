Cross-Sample Alignment - Two Breast Cancer Sections from Xenium
===============================================================

This workflow aligns the ``Rep2`` query section to the fixed ``Rep1``
reference section. It starts from 286,532 cells in the original Xenium inputs
and uses all 279,625 cells that pass quality control.

Source notebooks
----------------

.. toctree::
   :maxdepth: 1

   cross_sample_alignment_breast_cancer_clustering_nb
   cross_sample_alignment_breast_cancer_alignment_nb

Input audit and preprocessing
-----------------------------

The Rep1 and Rep2 cell-feature matrices and cells tables are from the
`10x Genomics Xenium human-breast dataset
<https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast>`_.
The clustering notebook verifies that each cells table and H5 matrix contain
identical cell IDs in the same order and that per-cell gene sums equal the
reported transcript counts.

Only the 313 features annotated as ``Gene Expression`` are retained. The
workflow excludes 159 blank codewords, 41 negative-control codewords and 28
negative-control probes. Cells require at least 20 gene counts, at least 10
detected genes and at most 5% control counts. This retains 161,995 of 167,780
Rep1 cells and 117,630 of 118,752 Rep2 cells.

Counts are normalized to 10,000 per cell and ``log1p`` transformed while raw
gene counts remain in ``layers["counts"]``. No highly variable gene selection
is applied because Xenium measures a targeted 313-gene panel.

Joint clustering
----------------

BANKSY features are computed independently within each sample with 30 spatial
neighbors, ``scaled_gaussian`` decay and ``max_m=1``. The workflow compares
``lambda`` values 0, 0.2, 0.5, 0.8 and 1. Each joint representation uses 30
PCs, Harmony with ``sample_id`` and ``theta=4``, a 50-neighbor graph, and
Leiden resolution 0.3.

Harmony improves same-batch 30-nearest-neighbor mixing and iLISI at every
lambda. ``lambda=0`` provides the strongest batch mixing but is an
expression-only control. ``lambda=0.2`` is selected because it preserves a
modest spatial contribution while maintaining good mixing; larger spatial
weights leave progressively stronger sample structure. The selected result
contains 10 shared Leiden clusters and does not introduce an HVG subset.

Rep2-to-Rep1 alignment
----------------------

The selected orientation-preserving manual initialization uses scale 1,
rotation 2 degrees, x translation -250 micrometers and y translation 1750
micrometers. It raises nearest-neighbor cluster agreement to 0.495. An optional
interactive control is available by setting ``SPALIGNDE_ENABLE_MANUAL_UI=1``.

Rasterization uses the 10 shared cluster-composition channels on a 30
micrometer grid. The density channel receives zero optimization weight because
the serial sections differ in coverage and sampling density. S-LDDMM then uses
``kernel_scale=300``, ``time_steps=3``, ``velocity_grid_spacing=100``,
``momentum_lr=4000`` and 500 iterations.

EM intensity updates change the objective scale during this run, so
``restore_best_checkpoint=False`` preserves the requested final-iteration
transformation instead of restoring an incomparable pre-EM checkpoint. Final
nearest-neighbor cluster agreement is 0.507. This cell-level agreement is an
alignment diagnostic, not a replicate-level inferential statistic.

Output contract
---------------

The clustered H5AD contains normalized expression, raw counts, BANKSY/Harmony
embeddings, UMAP coordinates and the selected 10-cluster labels. The aligned
H5AD additionally stores ``x_prealigned``, ``y_prealigned``, ``x_aligned`` and
``y_aligned`` while preserving observation order, expression values,
``obsm["spatial"]`` and all Rep1 coordinates.

Manual-transform and S-LDDMM distances use Xenium coordinate units. Inspect
the manual initialization, raster fields, energy history and final local
structures before transferring these values to another dataset; see
:doc:`../tutorials/parameter_tuning` for the tuning order.
