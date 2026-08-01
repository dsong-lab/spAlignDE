CPU Quickstart
==============

This quickstart verifies the complete cross-sample alignment handoff without
external data, a GPU or a preceding clustering run. It uses a deterministic,
synthetic AnnData object containing two samples and three known shared
structures. The example is for software validation only, not biological
inference.

After creating the :doc:`validated environment <../installation>`, run:

.. code-block:: bash

   cd /path/to/spAlignDE
   jupyter lab source_notebooks/quickstart_nb.ipynb

The notebook performs these executable checks:

1. creates the small input with ``spAlignDE.make_cross_sample_example``;
2. validates the public AnnData contract;
3. runs automatic pre-alignment, rasterization and two CPU S-LDDMM iterations;
4. verifies preservation of expression, observation order and fixed-reference
   coordinates; and
5. verifies ``x_prealigned``, ``y_prealigned``, ``x_aligned`` and
   ``y_aligned`` in the returned AnnData.

The short optimization is deliberately a smoke-test configuration. For real
data, continue with :doc:`Cross-Sample Alignment <cross_sample_alignment>` and
select parameters using :doc:`Parameter Tuning <parameter_tuning>`.

The same notebook is executed from a clean checkout in continuous integration,
so import failures and runtime API breakage fail the build.

Notebook
--------

:doc:`CPU quickstart — synthetic cross-sample alignment
<../source_notebooks/quickstart_nb>`
