Cross-Sample Alignment - Two Breast Cancer Sections from Xenium
===============================================================

This single-cell Xenium workflow aligns the ``Rep2`` query section to the
``Rep1`` reference section. It starts from the original combined dataset
containing 286,532 cells and 541 measured genes.

Input data and quality control
------------------------------

In Situ Sample 1 Replicates 1 and 2 are publicly available from the
`10x Genomics Xenium human-breast dataset
<https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast>`_,
as listed in the manuscript Data Availability section.

The notebooks accept a combined AnnData file; paired CSV input must follow the
:ref:`cross-sample-csv-contract`. In the recorded run, quality control is
performed independently within each replicate using transcript-count and
cell/nucleus-area thresholds. It retains 147,333 ``Rep1`` cells and 104,365
``Rep2`` cells.

Source notebooks
----------------

.. toctree::
   :maxdepth: 1

   cross_sample_alignment_breast_cancer_clustering_nb
   cross_sample_alignment_breast_cancer_alignment_nb

The clustering notebook recomputes the workflow from the original data,
selects 300 batch-aware highly variable genes, and displays raw and
boundary-refined clusters for both sections in the same 2 × 2 layout used by
the MERFISH notebook. Four refined shared clusters are used for alignment.

Automatic centroid pre-alignment is not used. The alignment notebook retains
the selected manual similarity coordinates and S-LDDMM settings from the
original breast-cancer analysis. It displays the manual
pre-alignment, rasterized cluster and density fields, and before/after overlays
both without and with cluster colors. Nearest-neighbor cluster agreement
increases from 0.625 to 0.725 in the executed result.

The final H5AD adds ``x_prealigned``, ``y_prealigned``, ``x_aligned`` and
``y_aligned`` while preserving ``obsm["spatial"]``. Manual-transform and
S-LDDMM distances use Xenium spatial units; see
:doc:`../tutorials/parameter_tuning` before transferring the example values.
