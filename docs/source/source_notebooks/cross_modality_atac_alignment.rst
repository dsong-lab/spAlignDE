Spatial ATAC-to-ST Notebooks
============================

These two executable notebooks implement the complete P22 spatial ATAC to
MERFISH S3R1 workflow. Run them in order. The first notebook produces the
clustered ATAC AnnData consumed by the second; the second performs a fresh
global pre-alignment, automatic structure pairing, and S-LDDMM optimization.

For a new dataset, validate the partial field of view before changing pair or
deformation settings. The accepted/rejected masks and cropped overlay are the
first diagnostics; ``a`` and ``grid_step`` are interpreted in the cropped
raster-canvas units. See :doc:`../tutorials/cross_modality_atac_alignment` for
the data and output fields and :doc:`../tutorials/parameter_tuning` for failure-
oriented tuning.

Both notebooks use seed ``1234``. The fixed result contains 17 ATAC
structures and eight accepted ATAC-to-ST pairs. In two fresh processes, the
pair table was identical and the aligned coordinates agreed within
``1e-12``; this numerical tolerance is used for float64 GPU output rather
than requiring byte-identical H5AD files.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   cross_modality/atac_st_single_clustering_nb
   cross_modality/atac_st_alignment_nb
