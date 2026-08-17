Post-Alignment Inference
========================

spAlignDE tests location-resolved expression differences after samples have
been transformed into a common coordinate frame. The public tutorials cover
two aligned-tissue examples: injured mouse kidney and aging mouse brain.

Alignment-to-inference handoff
------------------------------

Alignment and post-alignment inference are consecutive but modular stages.
The alignment notebooks write ``x_aligned`` and ``y_aligned`` for every
retained observation; the inference notebooks consume those coordinates
directly, join them to the original count data by stable observation identity,
and do not estimate another spatial transformation. Thus the fixed-seed
alignment output is the explicit handoff artifact between the two stages.

For kidney, the recorded inference example starts from the formal fixed-seed
reproducibility audit:

.. code-block:: text

   reproducibility_audit_0810/.../kidney/run_1/
       -> alignments/IL3_to_NL3/query_coordinates.csv.gz
       +  cluster_labels.csv.gz (unchanged NL3 reference)
       -> packaged, hash-tracked coordinate tables
       -> post_alignment_inference_nb.ipynb

The formal run uses clustering/alignment seed ``1000``, Leiden resolution
``0.2``, 5,000 S-LDDMM iterations and
``restore_best_checkpoint=False``. The inference stage uses seed ``1`` and
``n_jobs=1``. The compact package copies IL3 ``x_aligned`` and ``y_aligned``
from ``run_1`` and the unchanged NL3 ``x`` and ``y`` from the same run's
cluster table, then restores the recorded 50-fold inference coordinate scale.
The notebook joins these coordinates one-to-one to public Visium counts by
terminal 10x barcode. ``run_2`` is the reproducibility repeat and is not used
for the reported inference result.

Custom alignment H5AD or standardized coordinate CSV inputs remain supported
through ``SPALIGNDE_KIDNEY_ALIGNED_H5AD`` and
``SPALIGNDE_ALIGNMENT_DIR``. They reproduce the website numbers only when
their evaluated spot identities and coordinates are identical to formal
``run_1``.

For aging brain, the formal archive contains 19 fixed-seed query-age
alignments to the unchanged 4.3-month reference, using resolution ``0.8`` and
800 iterations. The executable website example uses four of those query
outputs—6.6, 15.8, 30.9 and 34.5 months—plus the reference. The package stores
the exact selected ``x_aligned`` and ``y_aligned`` values and records the
source hashes. This five-section subset demonstrates the same
alignment-output-to-inference contract; the remaining formal query files are
not needed for the Figure 5A example.

Injured Mouse Kidney
--------------------

The first example uses a normal mouse-kidney Visium section (``NL3``) as
reference and an injured section (``IL3``) as query. It reports local results
for ``Cbr1``, ``Cd44`` and ``Myo5a`` together with one gene-level ACAT omnibus
P value per gene.

The fully executed notebook uses coordinates from formal fixed-seed kidney
``run_1``. The package records the source coordinate and cluster-label hashes.
Raw expression and tissue-position tables remain external public inputs, and
users can substitute coordinates from their own spAlignDE run.

Installation and data
~~~~~~~~~~~~~~~~~~~~~

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

The upstream alignment notebook uses the same NL3/IL3 sections plus region
annotations from the `STcompare record 20647680
<https://zenodo.org/records/20647680>`_. The inference notebook deliberately
reloads the source 10x files above and joins them to aligned coordinates by
terminal barcode, so the two records have distinct and explicit roles.

Configure their directory without editing the notebook:

.. code-block:: bash

   export SPALIGNDE_KIDNEY_DATA_DIR=/path/to/raw_kidney_files
   export SPALIGNDE_TUTORIAL_WORK_DIR=/path/to/tutorial_work

To evaluate a custom alignment H5AD instead of formal ``run_1``, set:

.. code-block:: bash

   export SPALIGNDE_KIDNEY_ALIGNED_H5AD=/path/to/custom_alignment.h5ad

Alternatively, provide standardized coordinate files named
``aligned_coords_NL3.csv`` and ``aligned_coords_IL3.csv``:

.. code-block:: bash

   export SPALIGNDE_ALIGNMENT_DIR=/path/to/custom_alignment_output

NL3 contributes 3,215 spots and IL3 contributes 2,965 spots. The notebook
canonicalizes terminal 10x barcodes, rejects missing or duplicated identifiers,
and performs a one-to-one join between raw expression, tissue positions and
aligned coordinates. Row order is never used as identity.

Stable-gene candidates and density risk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~

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

Gene-specific local-risk calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each gene, mismatch-aware fitting begins with initial local statistics
obtained without mismatch inflation. Let :math:`r_i` denote normalized local
risk. The calibration then:

1. groups these initial local statistics by :math:`r_i`;
2. subtracts the median within each risk bin;
3. divides each raw MAD by the Student-t null MAD at the corresponding degrees
   of freedom;
4. sets negative excess-dispersion estimates to zero and constrains the
   remaining excess dispersion to be nondecreasing with risk by isotonic
   regression;
5. fits a quadratic relation through the origin; and
6. applies a bounded anchor adjustment at the retained risk bin nearest the
   80th percentile.

This comparison of the observed risk-stratified dispersion with the Student-t
null scale estimates a provisional nonnegative coefficient for every valid
gene--contrast fit. Validity requires a successful within-contrast fit, at
least ``max(500, 4 * min_bin_n)`` usable grid locations, at least four distinct
risk bins including positive-risk support, finite fit and rescaling quantities,
and a finite nonnegative capped local coefficient with a zero global
coefficient. Failed contrasts are excluded, whereas a successful coefficient
of zero remains valid evidence.

For multiple contrasts, the valid provisional coefficients are combined by an
equal-weight Huber center to estimate one robust, nonnegative gene-specific
coefficient :math:`\lambda_g` shared across all contrasts. With one valid
contrast, as in the kidney example, the Huber center is exactly the original
coefficient. Provisional values are diagnostics and are not applied as
contrast-specific final coefficients. The
resulting alignment-mismatch variance factor for gene :math:`g` and location
:math:`i` is

.. math::

   \phi_{ig}^{\mathrm{align}}
   =1+\lambda_g r_i^2,
   \qquad \lambda_g\geq 0.

Because the fitted relation passes through the origin, zero-risk locations
retain their base variance. The comparison-level global risk score is retained
only for diagnostics and does not impose a spatially uniform variance penalty.

Cell-type-composition adjustment is independent of this gene-specific
calibration. When enabled, spAlignDE compares kernel-smoothed local cell-type
proportion vectors using the normalized Jensen--Shannon distance
:math:`D_i=\sqrt{\operatorname{JS}(p_{\mathrm{target},i},
p_{\mathrm{reference},i})/\log 2}` and applies the additional variance factor
:math:`\exp(D_i)`. Identical local compositions therefore receive factor 1,
whereas the maximum normalized distance gives factor :math:`e`.

Public workflow
~~~~~~~~~~~~~~~

The package performs the complete reusable handoff. After concatenating the
standardized outputs of ``build_visium_coordinate_table`` into
``coordinate_data``, the executed analysis runs:

.. code-block:: python

   from spalignde import fit_local_de, prepare_inference

   prepared = prepare_inference(
       inference_data,
       reference="NL3",
       genes=["Cbr1", "Cd44", "Myo5a"],
       risk_genes=risk_genes,
       aligned_coordinate_key=("x_aligned", "y_aligned"),
       cell_type_key=None,
       density_energy_share=0.75,
       library_size=10_000,
       grid_n=None,
       n_jobs=1,
       random_state=1,
   )

   result = fit_local_de(
       prepared,
       genes=["Cbr1", "Cd44", "Myo5a"],
       contrast="vs_reference",
       mismatch_aware=True,
       technical_adjustment=True,
       cell_type_adjustment=False,
       global_offset=False,
       region_cleanup=False,
       n_jobs=1,
       random_state=1,
   )

``mismatch_aware=False`` runs the naive comparison on the same prepared grid.
For multiple queries, every contrast contributes its own provisional
calibration when it passes the public validity checks; the valid provisional
values are combined by an equal-weight Huber center. The requested mode and the
fitted gene-specific coefficient are stored in the result metadata and each
gene's ``terrain_data["risk_calibration"]`` diagnostics.

Grid-level regions and gene-level ACAT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BH adjustment is performed separately for every gene and contrast across valid
grid locations. The tutorial deliberately sets ``region_cleanup=False``;
therefore red contours directly trace connected components of the raw
``q < 0.05`` grid mask, matching the manuscript plotting logic. Enabling cleanup
is an optional reporting choice that can remove small or unsupported fragments
without changing local statistics, P values or q-values.

Compact fitted results now retain ``p_by_time``.
``spalignde.gene_level_acat_pvalue(result, gene)`` strictly combines those raw
local P values; q-values are never substituted. The return value is an omnibus
P value for spatial change somewhere on the tested grid. It is neither a local
P value nor a genome-wide FDR-adjusted gene discovery value.

Recorded result
~~~~~~~~~~~~~~~

The executed notebook retains 6,127 tissue-valid shared-grid locations and
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
     - :math:`5.606626\times10^{-15}`
     - 2,142
     - :math:`1.766371\times10^{-38}`
     - 2.497326
   * - ``Cd44``
     - :math:`8.333126\times10^{-8}`
     - 1,318
     - :math:`2.030000\times10^{-7}`
     - 1.877514
   * - ``Myo5a``
     - :math:`5.151435\times10^{-14}`
     - 2,222
     - :math:`4.889250\times10^{-22}`
     - 2.428649

For each gene, the saved notebook displays NL3 expression, IL3 expression, the
zero-centered mismatch-aware local statistic, the red ``q < 0.05`` contours,
and the gene-level ACAT P value. Figures are preserved as genuine execution
outputs inside the notebook; no separate precomputed image file is required.

Interpretation and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a matched-section IL3-versus-NL3 analysis, not replicate-level
population inference. Mismatch risk describes local comparability and does not
replace geometric quality control. Inspect the aligned geometry, shared-grid
occupancy, stable-gene screen, density risk and local support before assigning
biological meaning to significant regions.

Aging Mouse Brain
-----------------

The second example compares four aging mouse-brain MERFISH sections with a
4.3-month alignment reference in their shared spAlignDE coordinate frame. The
queries are 6.6, 15.8, 30.9 and 34.5 months. Local tests are fitted for
``Gamt`` and ``Vip``; the displayed local-statistic maps remain focused on
``Gamt``.

Data and alignment handoff
~~~~~~~~~~~~~~~~~~~~~~~~~~

The original 300-gene MERFISH data were published with `Spatial transcriptomic
clocks reveal cell proximity effects in brain ageing
<https://doi.org/10.1038/s41586-024-08334-8>`_. The processed public data are
available from `Zenodo record 13883177
<https://doi.org/10.5281/zenodo.13883177>`_.

The complete formal coordinate archive contains 19 query ages aligned to the
unchanged 4.3-month reference with resolution ``0.8``, 800 iterations and seed
``1000``. The compact Figure 5A example selects the formal 6.6-, 15.8-, 30.9-
and 34.5-month query outputs. For these five sections the package retains raw
integer counts, original coordinates, cell-type labels and the exact formal
``x_aligned`` and ``y_aligned`` coordinates. The tutorial consumes these
outputs and does not rerun alignment.

Public workflow
~~~~~~~~~~~~~~~

The complete handoff uses the dataset loader and inference API directly:

.. code-block:: python

   from spalignde import (
       cluster_trajectories,
       fit_local_de,
       gene_level_acat_pvalue,
       gene_level_age_trend_acat,
       prepare_inference,
   )
   from spalignde.datasets import (
       AGING_BRAIN_FIGURE5A_REFERENCE,
       aging_brain_figure5a_genes,
       load_aging_brain_figure5a,
   )

   genes = ["Gamt", "Vip"]
   data = load_aging_brain_figure5a()
   prepared = prepare_inference(
       data,
       reference=AGING_BRAIN_FIGURE5A_REFERENCE,
       genes=genes,
       risk_genes=aging_brain_figure5a_genes(),
       density_energy_share=0.25,
       library_size=250,
       grid_n=None,
       n_jobs=1,
       random_state=1,
   )
   result = fit_local_de(
       prepared,
       genes=genes,
       contrast="vs_reference",
       alpha=0.05,
       mismatch_aware=True,
       technical_adjustment=True,
       cell_type_adjustment=False,
       global_offset=True,
       region_cleanup=True,
       n_jobs=1,
       random_state=1,
   )

   trend_result = gene_level_age_trend_acat(
       result,
       "Gamt",
       time_values=None,
       alpha=0.05,
   )
   global_trend_p = trend_result["summary"]["gene_level_trend_acat_p"]

   trajectory = cluster_trajectories(
       result,
       "Gamt",
       n_clusters="auto",
       time_values=None,
       random_state=1,
   )

Raw counts are not log-transformed and are normalized to a library size of
250. The density channel receives 25% of standardized mismatch-feature energy,
and the broad 300-gene panel is retained for mismatch-risk screening.

Global linear age trend and trajectory auto-K
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gene_level_age_trend_acat`` is the reported multi-age global test. At every
retained grid location it regresses the unsmoothed adjusted local expression
``muA_adj_by_time`` on the query ages with an intercept, using the
mismatch-aware ``Wv_by_time`` as local precision. It applies a two-sided local
slope test and combines the resulting P values across space with ACAT. The
4.3-month reference section is not inserted as an extra regression
observation. This is a pre-trajectory test: it does not use smoothed
trajectories, cluster labels, or the selected cluster count.

``gene_level_acat_pvalue`` answers the different question of whether any
fitted contrast contains spatial signal and is retained only as a diagnostic
in this multi-age example.

``cluster_trajectories(..., n_clusters="auto")`` first evaluates dynamic
evidence with held-out complete-trajectory prediction and retains the candidate
values on the one-SE plateau of the best gain. It then scans those candidates
from the largest K toward coarser resolutions. Coarsening continues while the
fraction of grid locations in components smaller than one ``R_map`` footprint
decreases. If the next coarser candidate raises fragmentation, the current
first fine-side local minimum is retained; if fragmentation decreases across
the full plateau, its coarsest candidate is retained. A nonpositive one-SE
lower bound returns the minimum candidate. The executed notebook displays
``fine_to_coarse_scan`` and the remaining public selection diagnostics.

Recorded result
~~~~~~~~~~~~~~~

The executed notebook reports the calculated age-trend spatial ACAT P value,
the distinct any-spatial-change diagnostic, the automatically selected K, and
all per-contrast grid summaries directly from the current public result
objects. These values are not copied into this prose, which prevents a stale
hard-coded result after a package update.

Because this website notebook contains four query ages rather than all 19
query-versus-reference contrasts, its global trend and automatic K are compact-
example outputs and are not expected to equal the manuscript-scale 20-section
results. The same public calls apply to the full aligned input.

Interpretation and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each output row is one age-versus-4.3-month contrast. Red contours mark the
connected subset of grid locations passing within-contrast BH FDR at
``q <= 0.05``. The analysis compares individual aligned sections rather than
biological replicates, so its local maps should not be interpreted as
replicate-level population inference.

Fixed-seed repeat check
-----------------------

Both public workflows use ``PYTHONHASHSEED`` set before kernel startup,
workflow seed 1, deterministic library controls and ``n_jobs=1`` for
preparation and fitting. The table records the latest public-API executions;
the aging row refers only to the compact five-section example.

.. list-table::
   :header-rows: 1
   :widths: 22 22 25 31

   * - Workflow
     - Shared-grid locations
     - Saved-output SHA256
     - Repeat result
   * - Kidney NL3 versus IL3
     - 6,127
     - Recorded in notebook metadata
     - Current public-API execution
   * - Aging-brain five-section example
     - 75,868
     - Recorded in notebook metadata
     - Current public-API execution

The saved-output hash covers all sanitized code-cell outputs, including tables
and figures, but excludes execution timing metadata. With ``n_jobs>1`` the
scientific arrays remain deterministic in this workflow, while diagnostic log
messages from parallel workers may arrive in a different order and therefore
change a whole-notebook byte/hash comparison.
