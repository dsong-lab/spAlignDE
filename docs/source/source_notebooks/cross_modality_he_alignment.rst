ST-to-Histology Image Workflow
==============================

First download the **Replicate 1 Xenium output bundle** from the
`10x Fresh Frozen Mouse Brain dataset
<https://www.10xgenomics.com/datasets/fresh-frozen-mouse-brain-replicates-1-standard>`_.
Use its expression matrix and cell coordinates as the input to the
:doc:`single-clustering workflow <../tutorials/clustering_single>`.
When adapting the :doc:`example notebook <clustering/clustering_single_nb>`,
replace its MERFISH S2R1 input paths with your Xenium Rep1 inputs and save the
clustered AnnData as ``xenium_rep1_single_clustered.h5ad`` under
``data/cross_modality/histology/mouse_brain/``. It should contain
``obsm["spatial"]``, ``obs["cluster_raw"]`` and ``obs["cluster"]``.

Once the Xenium clustering result is ready, run these three notebooks in
order to prepare the H&E reference image and align Xenium Rep1 to it:

.. toctree::
   :maxdepth: 1
   :titlesonly:

   cross_modality/st_he_feature_extraction_nb
   cross_modality/st_he_feature_clustering_nb
   cross_modality/st_he_alignment_nb

The histology-side dataset input is one high-resolution image. Feature files,
label rasters, and aligned coordinates shown in later notebooks are outputs of
the preceding notebook, not additional raw-data requirements.

The image pixel scale must be correct before feature extraction. Inspect the
prepared image and feature field before tuning image clusters, then inspect
whole-mask pre-alignment and accepted structure pairs before changing
S-LDDMM. See :doc:`../tutorials/st_histology_image_processing` for the exact
handoff files and :doc:`../tutorials/parameter_tuning` for parameter failure
modes.

Image clustering produces 21 final structures, and alignment accepts two
ST/H&E structure pairs. The final AnnData contains aligned Xenium coordinates
in the H&E frame. See :doc:`../tutorials/reproducibility` for comparisons
across runs.
