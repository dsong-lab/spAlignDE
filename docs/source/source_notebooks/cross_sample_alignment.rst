Cross-Sample Alignment
======================

The source notebooks are organized by dataset. Each workflow uses the same
AnnData-native spAlignDE interface and reports the query and reference samples,
pre-alignment, rasterized fields, and final S-LDDMM result.

The MERFISH joint-clustering notebook has one canonical location under
:doc:`Clustering <clustering>` and is linked from the mouse-brain workflow.
It is intentionally not repeated in this section's navigation.

.. toctree::
   :maxdepth: 2

   cross_sample_alignment_mouse_brain
   cross_sample_alignment_mouse_kidney
   cross_sample_alignment_breast_cancer
   cross_sample_uncertainty_qualification

See :doc:`../tutorials/cross_sample_alignment` for installation, the canonical
CSV and AnnData input contracts, output fields, and underlying equations.
The :doc:`../tutorials/parameter_tuning` page explains how to adapt clustering,
rasterization and S-LDDMM settings to a new coordinate system.
All cross-sample clustering notebooks use seed ``1000`` and explicitly pin
the partition backend/iteration policy. See
:doc:`../tutorials/reproducibility` for the exact-label and scale-aware CUDA
contracts, including the validated aging-brain series.
