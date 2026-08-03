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
3. construct a sample-size- and tissue-mask-aware shared testing grid and
   estimate stable-gene/density mismatch risk; and
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
       genes=["Cbr1", "Cd44", "Myo5a"],
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

The default R-driven Cartesian resolution is retained when the *actual*
number of tissue-valid grid locations lies between the median per-sample spot
count, :math:`N_{typ}`, and :math:`2N_{typ}`. Otherwise spAlignDE moves the
resolution toward the nearest boundary using the same tissue mask that defines
the final grid. Passing an integer ``grid_n`` explicitly overrides this rule;
``grid_n`` is the number of Cartesian points per axis, not the number of
retained tissue locations.

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
The density channel contributes 0.75 of the risk energy in this recorded kidney
analysis. This dataset-specific value reflects the importance of local
sampling-density mismatch in the aligned sections and is not a universal
default.

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
       genes=["Cbr1", "Cd44", "Myo5a"],
       risk_genes=risk_genes,
       aligned_coordinate_key=("x_aligned", "y_aligned"),
       cell_type_key=None,
       density_energy_share=0.75,
       library_size=10_000,
       grid_n=None,
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
       region_cleanup=False,
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
The recorded kidney analysis uses ``0.75`` because sampling-density mismatch is
important for these aligned sections; this is a dataset-specific setting, not a
universal default. ``region_cleanup=False`` reports the direct FDR masks without
topological post-processing.
Leave ``grid_n=None`` for the sample-size-aware automatic rule. Increasing an
explicit ``grid_n`` improves spatial resolution but increases runtime and
memory; decreasing it coarsens small structures. Always report the selected
resolution, its source, and the final tissue-valid location count.
Enable cell-type adjustment only with complete, validated annotations in every
sample. Region cleanup changes the topology of reported significant regions,
so save and report this choice. See :doc:`Parameter Tuning Guide
<parameter_tuning>` for the alignment-to-inference checklist.

Recorded result
---------------

The fully executed notebook uses 16,446 risk genes. The automatic rule selects
an 86-by-86 Cartesian resolution and retains 6,187 tissue-valid grid locations
(``N_typ = 3,090``; target interval 3,090--6,180). It reports:

.. list-table::
   :header-rows: 1
   :widths: 17 23 22 19 19

   * - Gene
     - Gene-level ACAT P value
     - Significant grid locations
     - Minimum q-value
     - Median absolute statistic
   * - ``Cbr1``
     - :math:`1.48\times10^{-14}`
     - 3,482
     - :math:`1.21\times10^{-31}`
     - 3.222
   * - ``Cd44``
     - :math:`2.48\times10^{-6}`
     - 1,629
     - :math:`7.68\times10^{-6}`
     - 1.802
   * - ``Myo5a``
     - :math:`5.28\times10^{-14}`
     - 2,996
     - :math:`7.08\times10^{-21}`
     - 2.408

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
     - One-row-per-gene summary of the gene-level ACAT omnibus P value,
       significant locations and statistic/q-value diagnostics.
   * - ``result.fits[gene]["terrain_data"]``
     - In-memory contrast-specific local maps and region-cleanup metadata.

Gene-level summary
------------------

``spAlignDE.gene_level_acat_pvalue(result, gene)`` combines dependent local P
values across valid grid locations within each contrast. For multi-query
results, contrast-level ACAT P values are combined again with weights
proportional to the number of valid local tests. Compact results from earlier
package versions are supported by reconstructing two-sided local P values from
the stored statistic and degrees of freedom. The returned value is an omnibus
P value for a spatial change somewhere on the tested grid; it is neither a
local P value nor a genome-wide FDR-adjusted gene-discovery value.

Interpretation and limitations
------------------------------

This example is a matched-section analysis, not replicate-level population
inference. Mismatch risk is a diagnostic of local comparability and can be
conservative when genuinely changing genes enter the stable-gene pool. Users
should inspect alignment quality, risk maps and local support together, and
should add cell-type adjustment only when complete, validated cell-type or
spot-deconvolution annotations are available.
