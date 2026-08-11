ST-to-Allen-CCF Workflow — MERFISH Adult Mouse Brain
====================================================

This landing page links the automatic and UI-curated Atlas workflows. Single
clustering remains in the canonical **Clustering** section; it is linked here
rather than duplicated in the Cross-Modality navigation.

- :doc:`Single clustering for cross-modality alignment
  <clustering/clustering_single_nb>`
- :doc:`MERFISH S2R1 to Allen CCF slice 675
  <cross_modal_atlas_alignment_nb>`
- :doc:`UI-curated ST-to-Allen-CCF alignment — MERFISH S2R1
  <cross_modality/ui_paired_atlas_alignment_nb>`

Use :doc:`Interactive region pairing and refinement
<cross_modality/interactive_region_pairing_nb>` to create the pairing CSV,
then pass that unedited export to the UI-curated alignment notebook.

Use :doc:`../tutorials/cross_modality_atlas_alignment` for the Allen input and
output contracts. The :doc:`../tutorials/parameter_tuning` guide explains
hierarchy depth, whole-mask initialization, global area/thickness weighting,
pair gates and the separate UI-curated deformation controls.

The automatic workflow uses seed ``1234``, three ST levels (7/16/25
structures), scheduled-stage pair counts 3/7/15 and a final post-continuation
set of 17. Allen hierarchy depths 2–10 remain searchable at every stage.

.. toctree::
   :hidden:
   :maxdepth: 1

   cross_modal_atlas_alignment_nb
   cross_modality/ui_paired_atlas_alignment_nb
