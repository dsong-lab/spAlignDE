Cross-Sample Alignment - Two Mouse Kidney Sections from Visium
==============================================================

This spot-resolution workflow aligns the ``IL3`` query section to the ``NL3``
reference section. After spatial filtering, the combined input contains 6,180
spots: 2,965 from ``IL3`` and 3,215 from ``NL3``.

Input data
----------

Download the raw Visium inputs from `Zenodo record 17676992
<https://zenodo.org/records/17676992>`_:

- ``IL3_filtered_feature_bc_matrix.h5`` and ``NL3_filtered_feature_bc_matrix.h5``;
- ``IL3_tissue_positions.csv`` and ``NL3_tissue_positions.csv``.

Place them in ``data/cross_sample/kidney/`` or set
``SPALIGNDE_KIDNEY_DATA_DIR`` to their directory. The clustering notebook
creates ``kidney_combined.h5ad`` by:

1. matching expression and coordinate rows by barcode, after removing the
   coordinate tables' ``IR3_`` or ``CTRL3_`` prefixes;
2. retaining the barcodes present in the coordinate tables, in expression-matrix
   order, with ``x=array_col`` and ``y=array_row``; and
3. merging IL3 and NL3 into one AnnData with raw counts, ``sample_id`` and
   ``obsm["spatial"]``.

This yields the 6,180 spots used by the example. The spatial selection is
barcode membership in the supplied coordinate tables, with no additional
distance or tissue-boundary cutoff. Region annotations from the
`STcompare record <https://zenodo.org/records/20647680>`_ are optional comparison
data, not inputs to clustering or alignment.

An existing combined H5AD or :ref:`paired per-sample CSV directory
<cross-sample-csv-format>` can be supplied through ``SPALIGNDE_KIDNEY_INPUT``.
The combined input contains one row per retained spot. This same 6,180-spot
pair continues through clustering, alignment, and kidney inference.

Source notebooks
----------------

.. toctree::
   :maxdepth: 1

   cross_sample_alignment_mouse_kidney_clustering_nb
   cross_sample_alignment_mouse_kidney_alignment_nb

Save one final alignment result:
``tutorials/cross_sample/kidney/output/kidney_IL3_to_NL3_aligned.h5ad``.
It contains both IL3 and NL3 with their ``x_aligned`` and ``y_aligned``
coordinates. The :doc:`kidney inference example <post_alignment_inference_nb>`
uses the same saved alignment shown in this notebook. The package supplies
its coordinates as one table per section, with ``x_aligned`` and
``y_aligned`` multiplied by 50 for the inference coordinate scale.

The clustering notebook follows the same presentation as the MERFISH example:
raw joint clusters are shown in the top row and boundary-refined clusters in
the bottom row, with a shared palette across ``IL3`` and ``NL3``. The executed
workflow identifies four refined shared clusters.
With seed ``1000``, both fresh-process runs produced the same four raw,
refined and final cluster labels.

Because only four shared structures are available for this spot-level pair,
the automatic similarity fit is unstable. The validated workflow therefore
uses an explicit manual transform: unit scale, zero rotation and a fixed
center-to-center translation. It then shows:

1. the fixed manual pre-alignment;
2. rasterized shared-cluster and density fields;
3. overlays before and after S-LDDMM without cluster colors; and
4. overlays before and after S-LDDMM with cluster colors.

Alignment parameters
--------------------

The requested kidney S-LDDMM settings are used exactly:

.. code-block:: python

   config = spAlignDE.SLDDMMConfig(
       kernel_scale=500,          # a
       time_steps=5,              # nt
       velocity_grid_spacing=250, # grid_step
       momentum_lr=50,            # lrM
       minimum_momentum_lr=50,
       iterations=5000,           # niter
       restore_best_checkpoint=False,
   )

Coordinates are multiplied by 50 internally, rasterized with grid spacing 30,
and returned in the original coordinate scale. The recorded manual values in
the internally scaled coordinate system are ``scale=1.0``, ``theta=0.0``,
``tx=-36.20040965`` and ``ty=-153.38356513``. The recorded alignment used by
inference improves nearest-neighbor shared-cluster agreement from ``0.6587``
to ``0.7417``.

Independent fresh-process validation uses the same fixed cluster labels,
observation order, manual transform and S-LDDMM configuration. Continuous
float32 CUDA coordinates are compared using a ``0.01``-unit tolerance in the
original coordinate system because GPU computation can introduce small
numerical differences.

Here ``a`` controls deformation smoothness and ``grid_step`` controls velocity
field resolution. Increasing either makes the warp more global/coarse;
decreasing either permits more local motion and increases memory or
overfitting risk. ``nt`` changes flow-integration accuracy, while ``niter`` and
``lrM`` control optimization duration and step size. See
:doc:`../tutorials/parameter_tuning` before transferring these values to
unscaled coordinates or a different tissue extent.
