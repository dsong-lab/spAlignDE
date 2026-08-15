ST-to-Histology Image Workflow
==============================

These three notebooks implement the complete image-only reference
workflow. Run them in order. Prepare the query AnnData first with the canonical
:doc:`single-clustering notebook <clustering/clustering_single_nb>`.

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

The three notebooks use seed ``0``. Image clustering first targets 26 merged
regions; reflection-aware and post-symmetry cleanup leaves 21 final image
structures. Two ST/H&E structure pairs are accepted. Manuscript-grade S-LDDMM
is run in float64 and checked with a numerical coordinate tolerance.
