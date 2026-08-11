Cross-Modality Alignment Notebooks
==================================

This section collects executable notebooks for cross-modality workflows. The
package-integrated examples map MERFISH S2R1 to an Allen CCF slice, Xenium
mouse brain to an H&E image, and P22 spatial ATAC to MERFISH S3R1. Their
clustering notebooks are linked from each workflow landing page, followed by
the corresponding alignment notebooks. A separate interactive notebook shows
how to visually review, group, pair, and refine regions across these data
types; the Atlas landing page also includes a complete notebook that consumes
its exported pairing CSV directly.

Use :doc:`../tutorials/parameter_tuning` when changing hierarchy depth, image
resolution, mask construction, pairing gates or deformation settings. The
distance-valued defaults belong to each example's own coordinate frame.
The workflow-specific seeds and continuous-coordinate tolerances are listed in
:doc:`../tutorials/reproducibility`.

.. toctree::
   :maxdepth: 2
   :titlesonly:

   cross_modality_atlas_alignment
   cross_modality_he_alignment
   cross_modality_atac_alignment
   cross_modality/interactive_region_pairing_nb
