Cross-Modality Alignment
========================

Cross-modality alignment registers a spatial assay to a reference that does
not necessarily share measured molecular features. Each workflow converts
modality-specific structures into comparable spatial fields before S-LDDMM.

The ST-to-Allen-CCF, ST-to-H&E, and spatial-ATAC-to-ST workflows are
package-integrated reference implementations. They follow the same
documentation framework as cross-sample alignment: explicit input contracts,
package functions, ordered executable notebooks, standard coordinate outputs,
visual QC and reproducible provenance.

Cross-modality workflows add modality-specific choices for hierarchy depth,
mask construction, geometric pairing, image resolution and partial
field-of-view selection. See :doc:`Parameter Tuning Guide <parameter_tuning>`
for the recommended tuning order and the relationship between these settings
and S-LDDMM parameters.

.. toctree::
   :maxdepth: 1

   clustering_single
   cross_modality_atlas_alignment
   st_histology_image_processing
   cross_modality_atac_alignment
