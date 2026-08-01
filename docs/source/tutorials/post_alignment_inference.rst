Post-Alignment Inference - Injured Mouse Kidney
================================================

spAlignDE continues from a completed cross-sample alignment to test
location-resolved expression differences in the common coordinate frame. The
public example compares an injured mouse-kidney Visium section (``IL3``, query)
with a normal section (``NL3``, reference). It uses the exact aligned AnnData
written by the kidney alignment tutorial rather than a separate coordinate
cache.

The workflow has four stages:

1. validate the standardized alignment output and extract its final
   ``x_aligned`` and ``y_aligned`` coordinates;
2. join raw 10x Visium counts to aligned spots by terminal barcode;
3. construct a shared testing grid and estimate stable-gene/density mismatch
   risk; and
4. fit Mismatch-aware local differential-expression tests and report one
   statistic, P value, q-value and significance call per valid grid location.

Installation
------------

Install the integrated package and tutorial dependencies from the repository
root:

.. code-block:: bash

   cd /path/to/spAlignDE
   python -m pip install -e ".[tutorial]"

``prepare_inference``, ``fit_local_de`` and ``plot_local_result`` are part of
the main ``spAlignDE`` package; no second inference package or private module
path is required.

Example data and execution order
--------------------------------

The NL3 and IL3 Visium data are publicly available from the `STcompare Zenodo
record <https://zenodo.org/records/19486091>`_, the source listed in the
manuscript Data Availability section. The recorded pair contains 6,180 spots:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Sample
     - Role
     - Spots
     - Description
   * - ``NL3``
     - Reference
     - 3,215
     - Normal kidney section defining the fixed coordinate frame.
   * - ``IL3``
     - Query
     - 2,965
     - Ischemia--reperfusion-injured section transformed into the NL3 frame.

Run the notebooks in this order:

1. :doc:`Kidney joint clustering
   <../source_notebooks/cross_sample_alignment_mouse_kidney_clustering_nb>`
2. :doc:`Kidney pre-alignment, rasterization and S-LDDMM
   <../source_notebooks/cross_sample_alignment_mouse_kidney_alignment_nb>`
3. :doc:`Kidney post-alignment local inference
   <../source_notebooks/post_alignment_inference_nb>`

The second notebook writes ``kidney_IL3_to_NL3_aligned.h5ad``. Configure the
inference notebook with:

.. code-block:: bash

   export SPALIGNDE_KIDNEY_ALIGNED_H5AD=/path/to/kidney_IL3_to_NL3_aligned.h5ad
   export SPALIGNDE_KIDNEY_RAW_DIR=/path/to/kidney/raw
   export SPALIGNDE_POST_INFERENCE_OUTPUT_DIR=/path/to/kidney/post_alignment_output

Input contract
--------------

Aligned AnnData
~~~~~~~~~~~~~~~

The alignment handoff must contain one observation per retained spot and these
finite columns in ``adata.obs``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Meaning
   * - ``sample_id``
     - Query/reference identity for every spot.
   * - ``x``, ``y``
     - Original spot coordinates retained for provenance and visualization.
   * - ``x_aligned``, ``y_aligned``
     - Final coordinates defining the shared local-inference space.
   * - ``obs_names``
     - Unique identifiers containing a terminal 10x barcode.

The kidney H5AD also contains clustering and pre-alignment fields, but they are
not required by local inference. The final coordinates, not
``adata.obsm["spatial"]``, define the comparison neighborhoods.

Raw expression
~~~~~~~~~~~~~~

The raw directory requires:

- ``NL3_filtered_feature_bc_matrix.h5``;
- ``IL3_filtered_feature_bc_matrix.h5``.

The H5AD expression matrix is deliberately not treated as raw counts because
alignment workflows may retain normalized or feature-transformed values in
``adata.X``. The notebook reads the public 10x matrices and performs a
validated one-to-one barcode join. Row-order matching is never used.

The complete handoff is a package function rather than notebook-local loading
code:

.. code-block:: python

   visium_input = spAlignDE.build_visium_inference_table(
       "/path/to/kidney_IL3_to_NL3_aligned.h5ad",
       {
           "NL3": "/path/to/NL3_filtered_feature_bc_matrix.h5",
           "IL3": "/path/to/IL3_filtered_feature_bc_matrix.h5",
       },
       genes=["Cbr1", "Cd44", "Slc5a2"],
       min_detected_spots=10,
       min_total_counts=10,
       batch="kidney_pair",
   )

   inference_data = visium_input.data
   risk_genes = list(visium_input.risk_genes)

``VisiumInferenceInput`` also exposes the validated coordinate table, tested
genes, per-sample spot counts and number of genes shared by the raw matrices.

Shared grid and local target
----------------------------

Let :math:`\boldsymbol{\xi}_i` denote a retained location of the shared grid.
Nearby aligned spots from each sample form Gaussian kernel-weighted
neighborhoods. For gene :math:`g`, the local contrast is

.. math::

   \Delta_{ig}=\mu_{\mathrm{IL3},ig}-\mu_{\mathrm{NL3},ig}.

spAlignDE does not test the unadjusted null :math:`\Delta_{ig}=0`. It models a
comparison-wide gene-specific baseline and an optional grid-varying technical
profile:

.. math::

   \theta_{ig}
   =\Delta_{ig}-\left(\gamma_g+
   \mathbf z_i^{\mathsf T}\boldsymbol\beta_g\right),
   \qquad H_{0,ig}:\theta_{ig}=0.

Here, :math:`\mathbf z_i` is constructed from local library size and detection
rate. In the recorded run, spot counts are normalized to a target library size
of 10,000 and technical adjustment is enabled.

Mismatch-aware uncertainty
--------------------------

A broad stable-gene candidate pool is formed from genes detected in at least
10 spots with at least 10 total counts across the pair. Differences in local
expression magnitude, overdispersion and excess sparsity are combined with
sampling-density discordance to create a contrast-specific mismatch-risk map.
The density channel contributes 0.25 of the risk energy in this tutorial.

The risk score changes uncertainty, not the fitted local contrast. With
cell-type adjustment available, the final variance is

.. math::

   V_{ig}=V_{ig}^{\mathrm{base}}
   \phi_i^{\mathrm{align}}\phi_i^{\mathrm{cell}},
   \qquad
   t_{ig}=\frac{\theta_{ig}}{\sqrt{V_{ig}}}.

Both inflation factors are at least one. Reliable cell-type or deconvolution
labels are not included with the public kidney inputs, so the recorded run uses
``cell_type_adjustment=False`` and :math:`\phi_i^{\mathrm{cell}}=1`.

Public API
----------

First construct the standardized long table with
``build_visium_inference_table`` as shown above. It accepts an aligned H5AD or
in-memory AnnData and a mapping of sample IDs to 10x HDF5 paths or raw-count
AnnData objects. It validates the complete handoff and selects the broad risk
gene pool without reading expression from the aligned H5AD.

Then prepare the reusable geometry and risk object once:

.. code-block:: python

   prepared = spAlignDE.prepare_inference(
       inference_data,
       reference="NL3",
       genes=["Cbr1", "Cd44", "Slc5a2"],
       risk_genes=risk_genes,
       aligned_coordinate_key=("x_aligned", "y_aligned"),
       cell_type_key=None,
       density_energy_share=0.25,
       library_size=10_000,
       random_state=1,
   )

Fit the Mismatch-aware local tests:

.. code-block:: python

   result = spAlignDE.fit_local_de(
       prepared,
       mismatch_aware=True,
       technical_adjustment=True,
       cell_type_adjustment=False,
       global_offset=False,
       region_cleanup=True,
       random_state=1,
   )

``mismatch_aware=False`` runs a Naive comparison on the same prepared grid.
For ordered or multi-query designs, ``contrast="sequential"`` and
``contrast="vs_reference"`` provide the two supported contrast schemes.

Parameter adaptation
--------------------

Choose ``risk_genes`` independently from the genes being tested and retain a
broad set with adequate detection. ``min_detected_spots`` and
``min_total_counts`` remove extremely sparse candidates before risk
estimation. ``density_energy_share`` must lie between zero and one; increase it
only when local sampling-density mismatch is a meaningful reliability signal.
Enable cell-type adjustment only with complete, validated annotations in every
sample. Region cleanup changes the topology of reported significant regions,
so save and report this choice. See :doc:`Parameter Tuning Guide
<parameter_tuning>` for the alignment-to-inference checklist.

Recorded result
---------------

The fully executed notebook constructs 52,877 shared-grid locations and uses
16,446 risk genes. It reports:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Gene
     - Significant grid locations
     - Minimum q-value
     - Median absolute statistic
   * - ``Cbr1``
     - 31,743
     - :math:`1.05\times10^{-41}`
     - 2.755
   * - ``Cd44``
     - 14,608
     - :math:`9.88\times10^{-6}`
     - 1.812
   * - ``Slc5a2``
     - 19,502
     - :math:`3.70\times10^{-8}`
     - 1.429

The notebook displays the alignment handoff, mismatch-risk map and one
reference-expression, query-expression and local-statistic panel for each
gene.

Output contract
---------------

The in-memory outputs are a ``PreparedInference`` object and a
``LocalDEResult``. The notebook also writes:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Output
     - Content
   * - ``<gene>_IL3_vs_NL3_local_de.csv.gz``
     - Grid ``x``/``y``, statistic, two-sided P value, BH q-value and final
       significance flag.
   * - ``kidney_local_de_summary.csv``
     - One-row-per-gene summary of significant locations and statistic/q-value
       diagnostics.
   * - ``result.fits[gene]["terrain_data"]``
     - In-memory contrast-specific local maps and region-cleanup metadata.

Validation against the previous kidney workflow
-----------------------------------------------

The new H5AD handoff was compared with the coordinate set that produced the
backup notebook. Sample counts, raw counts, filtering and model settings were
held fixed. The shared-grid size changed by 0.44% (52,877 versus 53,110).
Nearest-grid statistic-map correlations were 0.918 for ``Cbr1``, 0.980 for
``Cd44`` and 0.834 for ``Slc5a2``. Significant-region Dice overlaps were
0.976, 0.959 and 0.930, respectively, and statistic signs agreed at 98--99% of
matched grid locations. The updated handoff therefore preserves the previous
spatial conclusions while using the current website-generated alignment.

Interpretation and limitations
------------------------------

This example is a matched-section analysis, not replicate-level population
inference. Mismatch risk is a diagnostic of local comparability and can be
conservative when genuinely changing genes enter the stable-gene pool. Users
should inspect alignment quality, risk maps and local support together, and
should add cell-type adjustment only when complete, validated cell-type or
spot-deconvolution annotations are available.
