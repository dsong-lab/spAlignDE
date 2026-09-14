.. _welcome-to-spalignde-s-documentation:

spAlignDE
=========

**spAlignDE** uses tissue structure to align spatial omics datasets across
samples and modalities. For aligned spatial transcriptomics samples, it tests
local differences in gene expression while accounting for residual alignment
mismatch.

Use it to align tissue sections, register data with histology images or
anatomical atlases, align spatial ATAC-seq to spatial transcriptomics, and
compare gene expression at corresponding locations.

Getting started
---------------

1. :doc:`Install spAlignDE <installation>` and set up the notebook environment.
2. :doc:`Choose a workflow <tutorial>` for your data and analysis goal.
3. Run its notebooks in the listed order, then use the
   :doc:`parameter guide <tutorials/parameter_tuning>` when adapting the example.

For the method and supported applications, read the :doc:`overview`.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   overview
   installation
   tutorial

Reference
---------

* :doc:`source_notebooks`: runnable examples with saved outputs.
* :doc:`api`: Python functions, parameters and output fields.
* :doc:`tutorials/reproducibility`: random seeds and comparisons across runs.
* :doc:`license`: MIT License.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   source_notebooks
   api
   license
