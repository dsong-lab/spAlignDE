ST-to-Allen-CCF Workflow — MERFISH Adult Mouse Brain
====================================================

This landing page links the automatic and UI-curated Atlas workflows. Single
clustering remains in the **Clustering** section; it is linked here
rather than duplicated in the Cross-Modality navigation.

- :doc:`Single clustering for cross-modality alignment
  <clustering/clustering_single_nb>`
- :doc:`MERFISH S2R1 to Allen CCF slice 675
  <cross_modal_atlas_alignment_nb>`
- :doc:`UI-curated ST-to-Allen-CCF alignment — MERFISH S2R1
  <cross_modality/ui_paired_atlas_alignment_nb>`

Use :doc:`Interactive region pairing and refinement
<cross_modality/interactive_region_pairing_nb>` to create the pairing CSV,
then pass that export together with the clustered H5AD from the same run to
the UI-curated alignment notebook. The pairing CSV must refer to the cluster
IDs in that H5AD.

Use :doc:`../tutorials/cross_modality_atlas_alignment` for the Allen input and
output contracts. The :doc:`../tutorials/parameter_tuning` guide explains
hierarchy depth, whole-mask initialization, global area/thickness weighting,
pair gates and the separate UI-curated deformation controls.

The automatic example produces 18 final structure pairs and an AnnData
with coordinates aligned to the Allen slice. See
:doc:`../tutorials/reproducibility` for seeds and comparisons across runs.

.. toctree::
   :hidden:
   :maxdepth: 1

   cross_modal_atlas_alignment_nb
   cross_modality/ui_paired_atlas_alignment_nb
