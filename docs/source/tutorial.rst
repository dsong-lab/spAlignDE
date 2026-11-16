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
     - :doc:`Local inference <tutorials/post_alignment_inference>`;
       :doc:`use your own aligned data <tutorials/own_aligned_data>`
     - :doc:`Kidney and aging brain <source_notebooks/post_alignment_inference>`

Each notebook page lists the run order and required downloads. Alignment
produces the coordinates used for local differential-expression analysis.
The aging-brain inference example covers five sections; the manuscript's full
20-section analysis and Nissl analysis are not included as complete executable
tutorials on this site.

For your own data:

1. Choose a workflow from the table above.
2. Check its input format: :ref:`paired CSV <cross-sample-csv-format>` or
   :ref:`AnnData <cross-sample-anndata-format>` for cross-sample alignment,
   the corresponding guide for cross-modality inputs, or
   :doc:`aligned data for inference <tutorials/own_aligned_data>`.
3. Run the :ref:`Quick Start <quick-start>` to try alignment, then follow the
   matching workflow's example with your data.
4. Use the :doc:`parameter guide <tutorials/parameter_tuning>` to adjust settings.

For reproducing a reported example, see :doc:`tutorials/reproducibility`.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials/cross_sample_alignment
   tutorials/cross_modality_alignment
   tutorials/post_alignment_inference
   tutorials/own_aligned_data
   tutorials/parameter_tuning
   tutorials/reproducibility
