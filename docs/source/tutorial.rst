.. _tutorial:

Tutorials
=========

Choose your analysis below. **Guides** explain the method and data requirements;
**notebooks** provide runnable examples with saved outputs. Before running a
notebook, follow the :doc:`installation` instructions and select the
**Python (spAlignDE-notebooks)** kernel.

.. list-table::
   :class: workflow-table
   :header-rows: 1
   :widths: 28 32 40

   * - Task
     - Read the guide
     - Run the notebooks
   * - Align spatial transcriptomics samples
     - :doc:`Cross-sample alignment <tutorials/cross_sample_alignment>`
     - :doc:`Mouse brain <source_notebooks/cross_sample_alignment_mouse_brain>`,
       :doc:`kidney <source_notebooks/cross_sample_alignment_mouse_kidney>` or
       :doc:`breast cancer <source_notebooks/cross_sample_alignment_breast_cancer>`
   * - Align to the Allen brain atlas
     - :doc:`Atlas alignment <tutorials/cross_modality_atlas_alignment>`
     - :doc:`Automatic and interactive pairing <source_notebooks/cross_modality_atlas_alignment>`
   * - Align to a histology image
     - :doc:`Histology alignment <tutorials/st_histology_image_processing>`
     - :doc:`Mouse brain H&E <source_notebooks/cross_modality_he_alignment>`
   * - Align spatial ATAC-seq to spatial transcriptomics
     - :doc:`ATAC alignment <tutorials/cross_modality_atac_alignment>`
     - :doc:`Mouse brain ATAC <source_notebooks/cross_modality_atac_alignment>`
   * - Assess alignment stability
     - :doc:`Subsampling and variability <source_notebooks/cross_sample_uncertainty_qualification>`
     - :doc:`Stability notebook <source_notebooks/cross_sample_uncertainty_report>`
   * - Test local differential expression after alignment
     - :doc:`Local inference <tutorials/post_alignment_inference>`
     - :doc:`Kidney and aging brain <source_notebooks/post_alignment_inference>`

Each notebook page lists the run order and required downloads. Alignment
produces the coordinates used for local differential-expression analysis.
The aging-brain inference example covers five sections; the manuscript's full
20-section analysis and Nissl analysis are not provided as complete public
notebooks.

When using your own data, start with the
:doc:`parameter guide <tutorials/parameter_tuning>`. To repeat a published
example, follow the input, seed and comparison guidance in
:doc:`tutorials/reproducibility`.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/cross_sample_alignment
   tutorials/cross_modality_alignment
   tutorials/post_alignment_inference
   tutorials/parameter_tuning
   tutorials/reproducibility
