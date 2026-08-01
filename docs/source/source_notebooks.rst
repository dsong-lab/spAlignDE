Source Notebooks
================

This section collects executable source notebooks for spAlignDE workflows. Use
these notebooks as runnable examples for clustering, prealignment, LDDMM
alignment, quality control, cross-modality registration, and post-alignment
local inference.

Run all notebooks with the kernel created from the :doc:`validated notebook
environment <installation>`. The downloadable environment file pins the
Python, PyTorch/CUDA, clustering, image-processing and Jupyter dependencies
used by these workflows.

When adapting an example rather than reproducing it unchanged, read the
:doc:`Parameter Tuning Guide <tutorials/parameter_tuning>` first. It explains
coordinate-dependent settings, legacy S-LDDMM symbols, parameter coupling and
the required quality-control sequence.

.. toctree::
   :maxdepth: 1
   :caption: Executable Notebooks

   source_notebooks/quickstart_nb
   source_notebooks/clustering
   source_notebooks/cross_sample_alignment
   source_notebooks/cross_modality_alignment
   source_notebooks/post_alignment_inference

Workflow Categories
-------------------

- :doc:`CPU Quickstart <source_notebooks/quickstart_nb>`
- :doc:`Clustering <source_notebooks/clustering>`
- :doc:`Cross-Sample Alignment <source_notebooks/cross_sample_alignment>`
- :doc:`Cross-Modality Alignment <source_notebooks/cross_modality_alignment>`
- :doc:`Post-Alignment Inference <source_notebooks/post_alignment_inference>`
