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

For kidney, the recorded inference example starts from the public fixed-seed
manual-alignment workflow:

.. code-block:: text

   cross_sample_alignment_mouse_kidney_alignment_nb.ipynb
       -> fixed-seed manual-alignment coordinates
       -> packaged aligned_coords_NL3.csv.gz and aligned_coords_IL3.csv.gz
       -> barcode-validated coordinate tables
       -> post_alignment_inference_nb.ipynb

Upstream clustering and alignment use seed ``1000``, Leiden resolution ``0.2``
and a selected manual similarity pre-alignment with scale ``1``, rotation ``0``
degrees and translation ``(-36.20040965, -153.38356513)`` in the scaled
alignment coordinates. The workflow then runs 5,000 S-LDDMM iterations with
``restore_best_checkpoint=False``. The inference stage uses seed ``1`` and
requests ``n_jobs=1``. The notebook loads the packaged fixed-seed coordinate
copy through the public dataset API and joins it one-to-one to public Visium
counts by terminal 10x barcode.

Standardized coordinate CSV inputs remain supported through
``SPALIGNDE_ALIGNMENT_DIR``. They reproduce the website numbers only when
their evaluated spot identities and coordinates are identical to the packaged
manual-alignment handoff.

For aging brain, the executable website example uses the current PyPI
five-section Figure 5A package: the 6.6-, 15.8-, 30.9- and 34.5-month queries
plus the 4.3-month reference. The package stores their raw counts, annotations,
and precomputed ``x_aligned`` and ``y_aligned`` values. This subset
demonstrates the same alignment-output-to-inference handoff without rerunning
alignment.

Injured Mouse Kidney
--------------------

The first example uses a normal mouse-kidney Visium section (``NL3``) as
reference and an injured section (``IL3``) as query. It reports local results
for ``Cbr1``, ``Cd44`` and ``Myo5a`` together with one gene-level ACAT omnibus
P value per gene.

The fully executed notebook uses the package's fixed-seed manual-alignment
coordinate tables.
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

To evaluate a custom alignment, provide standardized coordinate files named
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

The executed notebook retains 6,187 tissue-valid shared-grid locations and
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
     - :math:`1.676437\times10^{-14}`
     - 2,954
     - :math:`7.780664\times10^{-32}`
     - 2.435333
   * - ``Cd44``
     - :math:`4.099926\times10^{-6}`
     - 1,452
     - :math:`1.018210\times10^{-5}`
     - 1.747929
   * - ``Myo5a``
     - :math:`7.355228\times10^{-14}`
     - 2,268
     - :math:`6.605387\times10^{-21}`
     - 1.946836

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

The compact Figure 5A package contains the 6.6-, 15.8-, 30.9- and 34.5-month
queries plus the 4.3-month reference. It retains raw integer counts, original
coordinates, cell-type labels and the precomputed spAlignDE ``x_aligned`` and
``y_aligned`` coordinates. The tutorial treats these current PyPI files as its
complete input and does not rerun alignment.

Public workflow
~~~~~~~~~~~~~~~

The complete handoff uses the dataset loader and inference API directly:

.. code-block:: python

   from spalignde.inference import (
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

   trajectory_time_ids = list(
       result.fits[genes[0]]["terrain_data"]["time_ids"]
   )
   time_values = [
       float(str(time_id).removeprefix("age_"))
       for time_id in trajectory_time_ids
   ]

   trend_result = gene_level_age_trend_acat(
       result,
       "Gamt",
       time_values=time_values,
       alpha=0.05,
   )
   global_trend_p = trend_result["summary"]["gene_level_trend_acat_p"]

   trajectory = cluster_trajectories(
       result,
       "Gamt",
       n_clusters="auto",
       time_values=time_values,
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

``cluster_trajectories(..., n_clusters="auto")`` evaluates each candidate K by
its held-out cluster-specific trajectory gain relative to a shared time trend.
If the best gain is not positive after subtracting its one-SE uncertainty, the
smallest candidate K is selected. Otherwise the candidates within one SE of
the best dynamic gain are retained and scanned from fine to coarse. The scan
tracks the fraction of grid locations in connected components smaller than one
``R_map`` footprint and coarsens while that fraction decreases. If the next
coarser candidate fails to reduce fragmentation, the current finer-side local
minimum is retained. If fragmentation decreases throughout and supplies no
elbow, one conservative coarsening step is taken from the finest retained
candidate. The rule does not globally minimize fragmentation and does not take
the coarsest candidate in the no-elbow case. The executed notebook displays
all stable selection diagnostics, including ``fine_to_coarse_scan`` and
``no_elbow_fallback_k``.

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

Both public workflows use seed 1 with the same aligned-coordinate inputs and
documented parameters. The two seeded auto-geometry passes use one worker
internally; subsequent preparation and fitting stages use the caller-requested
``n_jobs`` value. Two independent fresh-kernel runs reproduced the saved
summaries. The aging row refers only to the compact five-section example.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Workflow
     - Shared-grid locations
     - Repeat result
   * - Kidney NL3 versus IL3
     - 6,187
     - Current public-API execution
   * - Aging-brain five-section example
     - 74,908
     - Current public-API execution

With ``n_jobs>1`` the scientific arrays remain deterministic in this workflow,
although diagnostic messages from parallel workers may appear in a different
order.
