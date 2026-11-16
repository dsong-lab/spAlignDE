Spatial ATAC-to-ST Notebooks
============================

Prepare the ST reference and ATAC query independently, then align them:

1. Download the **MERFISH S3R1** expression matrix and cell metadata, and
   follow :doc:`Single Clustering <../tutorials/clustering_single>` using
   S3R1 in place of the example's S2R1 input. Save the result as
   ``data/cross_modality/atac/merfish_s3r1_clustered.h5ad``. See
   :ref:`atac-st-prepare-s3r1` for the input and save-path example.
2. Run the ATAC single-clustering notebook below to produce
   ``tutorials/cross_modality/atac/output/p22_atac_single_clustered.h5ad``.
3. Run the alignment notebook with both clustered files. It performs global
   pre-alignment, automatic structure pairing, and S-LDDMM optimization.

The ST input should retain the full S3R1 sample, original spatial coordinates,
and ``obs["cluster"]``. The alignment notebook selects the matching half brain.

For a new dataset, validate the partial field of view before changing pair or
deformation settings. The accepted/rejected masks and cropped overlay are the
first diagnostics; ``a`` and ``grid_step`` are interpreted in the cropped
raster-canvas units. See :doc:`../tutorials/cross_modality_atac_alignment` for
the data and output fields and :doc:`../tutorials/parameter_tuning` for failure-
oriented tuning.

The example produces 17 ATAC structures, eight accepted ATAC-to-ST pairs,
and an AnnData with the ATAC coordinates aligned to the ST reference.
See :doc:`../tutorials/reproducibility` for comparisons across runs.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   cross_modality/atac_st_single_clustering_nb
   cross_modality/atac_st_alignment_nb
