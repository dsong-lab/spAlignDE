Post-Alignment Inference - Injured Mouse Kidney
================================================

spAlignDE tests location-resolved expression differences after samples have
been transformed into a common coordinate frame. The real-data example uses a
normal mouse-kidney Visium section (``NL3``) as reference and an injured
section (``IL3``) as query. It reports local results for ``Cbr1``, ``Cd44`` and
``Myo5a`` together with one gene-level ACAT omnibus P value per gene.

The fully executed notebook uses the packaged manuscript ``aligned_317``
coordinates. These files contain spot identifiers and aligned coordinates
only. Raw expression and tissue-position tables remain external public inputs,
and users can substitute coordinates from their own spAlignDE run.

Installation and data
---------------------

Install the integrated package and tutorial dependencies from the repository
root:

.. code-block:: bash

   python -m pip install -e ".[tutorial]"

Download these files from `Zenodo record 17676992
<https://zenodo.org/records/17676992>`_:

- ``NL3_filtered_feature_bc_matrix.h5``;
- ``IL3_filtered_feature_bc_matrix.h5``;
- ``NL3_tissue_positions.csv``; and
- ``IL3_tissue_positions.csv``.

Configure their directory without editing the notebook:

.. code-block:: bash

   export SPALIGNDE_KIDNEY_DATA_DIR=/path/to/raw_kidney_files
   export SPALIGNDE_TUTORIAL_WORK_DIR=/path/to/tutorial_work

To replace the packaged alignment, provide standardized coordinate files named
``aligned_coords_NL3.csv`` and ``aligned_coords_IL3.csv``:

.. code-block:: bash

   export SPALIGNDE_ALIGNMENT_DIR=/path/to/custom_alignment_output

NL3 contributes 3,215 spots and IL3 contributes 2,965 spots. The notebook
canonicalizes terminal 10x barcodes, rejects missing or duplicated identifiers,
and performs a one-to-one join between raw expression, tissue positions and
aligned coordinates. Row order is never used as identity.

Stable-gene candidates and density risk
---------------------------------------

The three tested genes are kept separate from the broad mismatch-risk candidate
pool. Candidates must be detected in at least 10 spots and have at least 10
total counts across the pair. The executed analysis retains 16,446 candidates
from 32,285 shared genes and normalizes each spot to a target library size of
10,000.

``prepare_inference`` internally screens the candidate pool for stable and
informative genes, then combines their local distributional disagreement with
sampling-density discordance. The kidney run uses
``density_energy_share=0.75``: density receives 75% of the standardized
mismatch-feature energy before the combined risk map is formed. This is not a
direct weight on the final risk map or variance multiplier, and it is a
dataset-specific choice rather than a universal default.

Reliable biological cell-type annotations are not included with the public
inputs, so the recorded run uses ``cell_type_adjustment=False``. When validated
annotations are available in every sample, cell-type support is a separate
precision adjustment; it is not part of the gene-specific alignment-risk
calibration.

Shared-grid rule
----------------

Let :math:`N_{typ}` be the median observation count per sample. With
``grid_n=None``, spAlignDE begins from the R-driven Cartesian resolution and
counts the locations retained by the same tissue-occupancy mask used for the
final grid. The candidate resolution is retained when the valid count lies
between :math:`N_{typ}` and :math:`2N_{typ}`; otherwise the resolution is moved
toward the nearest boundary.

An explicit integer ``grid_n`` has priority over this rule. It is the number of
Cartesian points per axis, not the number of tissue-valid locations. The
selected value, its ``automatic`` or ``manual`` source, the target interval and
the final valid count are recorded in ``prepared.metadata``.

Gene-specific local-only calibration
------------------------------------

For each gene, Mismatch-aware fitting first obtains local statistics without
mismatch inflation. Let :math:`r_i` denote normalized local risk. The
calibration then:

1. groups first-pass statistics by :math:`r_i`;
2. subtracts the median within each risk bin;
3. divides each raw MAD by the Student-t null MAD at the corresponding degrees
   of freedom;
4. truncates excess variance below at zero and constrains it to be
   nondecreasing with risk by isotonic regression;
5. fits a quadratic :math:`B r^2` through the origin; and
6. applies a bounded anchor adjustment at the retained risk bin nearest the
   80th percentile.

The resulting mismatch factor for gene :math:`g` and location :math:`i` is

.. math::

   \phi_{ig}^{\mathrm{align}}
   =1+\lambda_{\mathrm{local},g}r_i^2,
   \qquad \lambda_{\mathrm{global},g}=0.

The comparison-level global risk score remains in metadata as provenance but
does not create a spatially uniform variance penalty. Consequently, a location
with zero local risk receives no mismatch inflation. This avoids the former
failure mode in which a small number of genes were over-shrunk everywhere by a
large gene-specific global factor.

Public workflow
---------------

The package performs the complete reusable handoff. After concatenating the
standardized outputs of ``build_visium_coordinate_table`` into
``coordinate_data``, the executed analysis runs:

.. code-block:: python

   visium_input = spAlignDE.build_visium_inference_table(
       coordinate_data,
       {
           "NL3": "/path/to/NL3_filtered_feature_bc_matrix.h5",
           "IL3": "/path/to/IL3_filtered_feature_bc_matrix.h5",
       },
       genes=["Cbr1", "Cd44", "Myo5a"],
       min_detected_spots=10,
       min_total_counts=10,
       batch="kidney_pair",
   )

   prepared = spAlignDE.prepare_inference(
       visium_input.data,
       reference="NL3",
       genes=visium_input.genes,
       risk_genes=visium_input.risk_genes,
       aligned_coordinate_key=("x_aligned", "y_aligned"),
       cell_type_key=None,
       density_energy_share=0.75,
       library_size=10_000,
       grid_n=None,
       random_state=1,
   )

   result = spAlignDE.fit_local_de(
       prepared,
       genes=visium_input.genes,
       contrast="vs_reference",
       mismatch_aware=True,
       technical_adjustment=True,
       cell_type_adjustment=False,
       global_offset=False,
       region_cleanup=False,
       random_state=1,
   )

``summarize_raw_genes`` is also public for workflows that need the per-gene
``detected_spots`` and ``total_counts`` table directly. The kidney notebook
does not reimplement that summary or any dataset-processing helper locally.

``mismatch_aware=False`` runs the Naive comparison on the same prepared grid.
For multiple queries, the first available contrast calibrates each gene's
local coefficient, which is then reused across its remaining contrasts.
Requested and actual calibration modes plus local/global coefficients are
stored in ``result.metadata``.

Grid-level regions and gene-level ACAT
--------------------------------------

BH adjustment is performed separately for every gene and contrast across valid
grid locations. The tutorial deliberately sets ``region_cleanup=False``;
therefore red contours directly trace connected components of the raw
``q < 0.05`` grid mask, matching the manuscript plotting logic. Enabling cleanup
is an optional reporting choice that can remove small or unsupported fragments
without changing local statistics, P values or q-values.

Compact fitted results now retain ``p_by_time``.
``spAlignDE.gene_level_acat_pvalue(result, gene)`` strictly combines those raw
local P values; q-values are never substituted. The return value is an omnibus
P value for spatial change somewhere on the tested grid. It is neither a local
P value nor a genome-wide FDR-adjusted gene discovery value.

Recorded result
---------------

The executed notebook retains 6,169 tissue-valid shared-grid locations and
reports:

.. list-table::
   :header-rows: 1
   :widths: 17 23 22 19 19

   * - Gene
     - Gene-level ACAT P value
     - Significant grid locations
     - Minimum q-value
     - Median absolute statistic
   * - ``Cbr1``
     - :math:`1.482148\times10^{-14}`
     - 2,989
     - :math:`5.184571\times10^{-31}`
     - 2.456205
   * - ``Cd44``
     - :math:`6.146148\times10^{-6}`
     - 1,449
     - :math:`1.764949\times10^{-5}`
     - 1.762186
   * - ``Myo5a``
     - :math:`8.038015\times10^{-14}`
     - 2,438
     - :math:`1.397810\times10^{-19}`
     - 2.010681

For each gene, the saved notebook displays NL3 expression, IL3 expression, the
zero-centered Mismatch-aware local statistic, the red ``q < 0.05`` contours,
and the gene-level ACAT P value. Figures are preserved as genuine execution
outputs inside the notebook; no separate precomputed image file is required.

Interpretation and limitations
------------------------------

This is a matched-section IL3-versus-NL3 analysis, not replicate-level
population inference. Mismatch risk describes local comparability and does not
replace geometric quality control. Inspect the aligned geometry, shared-grid
occupancy, stable-gene screen, density risk and local support before assigning
biological meaning to significant regions.
