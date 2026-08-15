Cross-Sample Alignment - Two Mouse Kidney Sections from Visium
==============================================================

This spot-resolution workflow aligns the ``IL3`` query section to the ``NL3``
reference section. After spatial filtering, the combined input contains 6,180
spots: 2,965 from ``IL3`` and 3,215 from ``NL3``.

Input data
----------

The IL3 and NL3 Visium matrices, spatial coordinates and region annotations
are publicly available from the `STcompare Zenodo record
<https://zenodo.org/records/20647680>`_, as listed in the manuscript Data
Availability section.

The input may be a combined AnnData file or paired per-sample CSV files
following the :ref:`cross-sample-csv-contract`. The recorded run starts from
``kidney_combined.h5ad`` and filters observations using
``adata.obs["is_spatial_filtered"]`` before validation. Public paths in both
notebooks are configured through environment variables.

Source notebooks
----------------

.. toctree::
   :maxdepth: 1

   cross_sample_alignment_mouse_kidney_clustering_nb
   cross_sample_alignment_mouse_kidney_alignment_nb

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
``tx=-36.20040965`` and ``ty=-153.38356513``. The executed alignment improves
nearest-neighbor shared-cluster agreement from ``0.6637`` to ``0.7366``.

Independent fresh-process validation uses the same fixed cluster labels,
observation order, manual transform and S-LDDMM configuration. Continuous
float32 CUDA coordinates are compared with a predeclared ``0.01``-unit
tolerance in the original coordinate system rather than requiring bitwise
identity.

Here ``a`` controls deformation smoothness and ``grid_step`` controls velocity
field resolution. Increasing either makes the warp more global/coarse;
decreasing either permits more local motion and increases memory or
overfitting risk. ``nt`` changes flow-integration accuracy, while ``niter`` and
``lrM`` control optimization duration and step size. See
:doc:`../tutorials/parameter_tuning` before transferring these values to
unscaled coordinates or a different tissue extent.
