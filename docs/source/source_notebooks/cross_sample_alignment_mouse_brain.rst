Cross-Sample Alignment - Two Adult Mouse Brain Sections from MERFISH
====================================================================

This single-cell workflow aligns the 85,958-cell ``S2R3`` query section to the
84,172-cell ``S2R2`` reference section from the Vizgen MERFISH Mouse Brain
Receptor Map.

First run :doc:`the mouse-brain joint-clustering notebook
<clustering/clustering_joint_nb>`, which is maintained under the
:doc:`Clustering <clustering>` section. It accepts either a combined AnnData
file or paired CSV files following the :ref:`cross-sample-csv-contract`.

Then run the alignment notebook below. It displays the automatic or
interactive manual pre-alignment, the rasterized cluster and density fields,
and the final overlays with and without cluster colors.

.. toctree::
   :maxdepth: 1

   cross_sample_alignment_nb

The output AnnData retains the original coordinates and adds
``x_prealigned``, ``y_prealigned``, ``x_aligned`` and ``y_aligned`` to
``adata.obs``. See :doc:`../tutorials/cross_sample_alignment` for the complete
data contract and workflow explanation.

Use :doc:`../tutorials/parameter_tuning` to adapt clustering, raster spacing,
``a``, ``grid_step``, integration steps and optimizer settings; first correct
an implausible pre-alignment rather than asking S-LDDMM to repair it.
