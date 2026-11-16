.. _welcome-to-spalignde-s-documentation:

spAlignDE
=========

**spAlignDE** uses tissue structure to align spatial omics datasets across
samples and modalities. For aligned spatial transcriptomics samples, it tests
local differences in gene expression while accounting for residual alignment
mismatch.

`GitHub <https://github.com/dsong-lab/spAlignDE>`_ ·
`Paper <https://doi.org/10.64898/2026.09.05.749632>`_ ·
:doc:`Overview <overview>`

Use it to align tissue sections, register data with histology images or
anatomical atlases, align spatial ATAC-seq to spatial transcriptomics, and
compare gene expression at corresponding locations.

.. _quick-start:

Quick Start
-----------

Follow :doc:`installation`, then run this small CPU example. It creates two
synthetic samples, aligns them with two optimization iterations and saves
``cross_sample_example_aligned.h5ad``. No data download is needed.

.. code-block:: python

   import spAlignDE

   spAlignDE.set_random_seed(0)
   adata = spAlignDE.make_cross_sample_example(n_per_cluster=10, random_state=0)

   result = spAlignDE.align_cross_sample(
       adata,
       query_sample="query",
       reference_sample="reference",
       rasterization_config=spAlignDE.RasterizationConfig(grid_spacing=0.3),
       slddmm_config=spAlignDE.SLDDMMConfig(
           iterations=2,
           kernel_scale=1.0,
           velocity_grid_spacing=0.5,
           momentum_lr=1.0,
           minimum_momentum_lr=1.0,
           sigma_regularization=100.0,
       ),
       device="cpu",
       verbose=False,
       return_result=True,
   )
   result.adata.write_h5ad("cross_sample_example_aligned.h5ad")
   print(result.metrics)

The saved AnnData retains the original coordinates and adds ``x_aligned`` and
``y_aligned`` to ``obs``. For real data, choose a :doc:`workflow <tutorial>`,
check its input requirements and run the listed notebooks in order.
Use the :doc:`parameter guide <tutorials/parameter_tuning>` to adapt settings.

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
