API
===

The integrated package API covers AnnData-native cross-sample,
ST-to-Allen-CCF, ST-to-histology, and spatial-ATAC-to-ST alignment. Use the
case-preserving public import throughout Python code:

.. code-block:: python

   import spAlignDE

Configuration fields are listed with each workflow below. Their practical
effects, legacy S-LDDMM symbols and recommended tuning order are documented in
:doc:`Parameter Tuning Guide <tutorials/parameter_tuning>`.

Validation and clustering
-------------------------

``set_random_seed``
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   seed_controls = spAlignDE.set_random_seed(
       1234,
       deterministic_torch=True,
       warn_only=True,
   )

Resets Python, NumPy and Torch generators before downstream stochastic work.
The returned dictionary records the numeric seed and the generators that were
reset. Call it before the first stochastic step and use the same input order
and workflow parameters in repeated runs. Very small floating-point
differences may remain between GPU systems without changing selected clusters
or structure pairs. See :doc:`Reproducibility and fixed random seeds
<tutorials/reproducibility>`.

``load_single_sample_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # AnnData or H5AD
   adata = spAlignDE.load_single_sample_data("/path/to/sample.h5ad")

   # Paired CSV input
   adata = spAlignDE.load_single_sample_data(
       "/path/to/cell_metadata_sample.csv",
       expression_csv="/path/to/cell_by_gene_sample.csv",
   )

Normalizes one spatial sample to AnnData. ``read_single_sample_csv`` is the
explicit paired-CSV reader; ``validate_single_sample_anndata`` checks unique
observation names and a finite ``n_obs × 2`` spatial matrix.

``cluster_single``
~~~~~~~~~~~~~~~~~~

Signature::

   spAlignDE.cluster_single(adata, *, config=None, spatial_key="spatial",
                            cluster_key="cluster", copy=True,
                            banksy_output_dir=None, return_details=False)

.. code-block:: python

   clustered = spAlignDE.cluster_single(
       adata,
       config=spAlignDE.SingleClusteringConfig(
           banksy_lambda=0.8,
           resolution=1.2,
           refine_boundaries=True,
       ),
   )

Runs single-sample BANKSY and optional boundary-aware refinement while
preserving the input expression matrix and coordinates. It adds
``cluster_raw``, ``cluster_refined`` and the selected ``cluster`` labels.
``plot_single_cluster_refinement`` compares raw and refined spatial maps using
one shared label-color mapping.

``adata`` is the input AnnData; ``spatial_key`` and ``cluster_key`` select the
coordinate and output-label fields. ``copy=False`` permits mutation of the
input object. ``banksy_output_dir`` optionally preserves BANKSY diagnostics;
otherwise a temporary directory is used. With ``return_details=True``, the
return value is ``(clustered_adata, details)`` instead of AnnData alone.

.. list-table:: ``SingleClusteringConfig`` fields
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Effect
   * - ``num_neighbors``
     - ``30``
     - Spatial neighbors used by BANKSY.
   * - ``banksy_lambda``
     - ``0.8``
     - Weight of spatial-neighborhood features.
   * - ``resolution`` / ``pca_dim``
     - ``1.2`` / ``20``
     - Leiden granularity and number of PCA components.
   * - ``max_m`` / ``decay``
     - ``1`` / ``"scaled_gaussian"``
     - BANKSY neighborhood harmonics and distance decay.
   * - ``random_state``
     - ``1234``
     - Seed applied before BANKSY feature construction and randomized PCA.
   * - ``refine_boundaries``
     - ``True``
     - Enable spatial boundary refinement.
   * - ``k_interior`` / ``k_boundary``
     - ``150`` / ``8``
     - Refinement neighborhoods away from and near boundaries.
   * - ``boundary_distance_pixels``
     - ``3``
     - Boundary band width.
   * - ``protected_labels``
     - layer-1 aliases
     - Labels requiring stronger replacement support.
   * - ``min_same_fraction``
     - ``0.8``
     - Minimum local support for retaining a label.
   * - ``min_new_fraction_interior`` / ``min_new_fraction_boundary``
     - ``0.2`` / ``0.8``
     - Support required to replace interior/boundary labels.
   * - ``min_new_fraction_protected``
     - ``0.9``
     - Support required to replace a protected label.

Raises ``TypeError`` or ``ValueError`` for an invalid AnnData/configuration and
``ImportError`` when the ``clustering`` optional dependencies are absent.

``load_cross_sample_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Combined AnnData file
   adata = spAlignDE.load_cross_sample_data(
       "/path/to/combined_samples.h5ad"
   )

   # Or paired per-sample CSV files in one directory
   adata = spAlignDE.load_cross_sample_data(
       "/path/to/csv_folder"
   )

Accepts an in-memory AnnData object, a ``.h5ad`` path, or a directory of paired
``cell_metadata_<sample_id>.csv`` and
``cell_by_gene_<sample_id>.csv`` files. Every input route is normalized to the
same combined AnnData layout.

``read_cross_sample_csv``
~~~~~~~~~~~~~~~~~~~~~~~~~

Explicit CSV reader used by ``load_cross_sample_data``. Metadata files require
``cell_id``, ``x`` and ``y``; expression files require ``cell_id`` followed by
numeric gene columns. Rows are matched by ``cell_id`` and genes are aligned by
name across samples. The complete format is documented in
:ref:`cross-sample-csv-format`.

``validate_cross_sample_anndata``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validates the combined AnnData input. By default it requires
``adata.obs["sample_id"]``, ``adata.obs["cluster"]``, globally unique
observation names, and a finite ``n_obs × 2`` coordinate matrix in
``adata.obsm["spatial"]``. Set ``require_cluster=False`` before joint
clustering.

``cluster_joint``
~~~~~~~~~~~~~~~~~

Signature::

   spAlignDE.cluster_joint(adata, *, config=None, sample_key="sample_id",
                           spatial_key="spatial", cluster_key="cluster",
                           cell_id_key="cell_id", layer=None, copy=True,
                           return_details=False)

.. code-block:: python

   clustered = spAlignDE.cluster_joint(
       adata,
       config=spAlignDE.JointClusteringConfig(),
       sample_key="sample_id",
       spatial_key="spatial",
       cluster_key="cluster",
       copy=True,
   )

Calculates BANKSY features within each sample, integrates them by joint PCA and
Harmony, clusters the shared representation with Leiden, and optionally
refines boundaries within each section. It returns an AnnData object with
raw labels in ``adata.obs["cluster_raw"]``, boundary-refined labels in
``adata.obs["cluster_refined"]``, and the selected final labels in
``adata.obs["cluster"]`` while preserving the original expression matrix and
coordinates. If refinement is disabled, ``cluster`` and ``cluster_raw`` are
identical and ``cluster_refined`` is not added.

``JointClusteringConfig`` contains BANKSY neighborhood and lambda settings,
PCA dimension, SNN neighborhood size, Leiden resolution, Harmony settings,
boundary refinement, and random state. Harmony uses CPU with one PyTorch
thread by default so a fixed seed reproduces the same corrected representation.

``layer`` selects an AnnData expression layer; the default uses ``adata.X``.
``copy=False`` permits mutation of the input. With ``return_details=True``,
the return value is ``(clustered_adata, details)`` instead of AnnData alone.

.. list-table:: ``JointClusteringConfig`` fields
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Effect
   * - ``num_neighbors`` / ``banksy_lambda``
     - ``30`` / ``0.8``
     - BANKSY spatial neighborhood and its feature weight.
   * - ``pca_dim`` / ``resolution``
     - ``20`` / ``1.4``
     - Joint PCA dimension and Leiden granularity.
   * - ``snn_neighbors``
     - ``50``
     - Shared-nearest-neighbor graph size.
   * - ``harmony_theta`` / ``harmony_max_iter``
     - ``2.0`` / ``30``
     - Sample correction strength and iteration limit.
   * - ``harmony_device`` / ``harmony_threads``
     - ``"cpu"`` / ``1``
     - Deterministic default. Use an accelerator or ``None`` threads when
       speed is more important than bitwise repeatability.
   * - ``random_state``
     - ``1000``
     - Seed applied before BANKSY, joint PCA, Harmony and Leiden.
   * - ``leiden_flavor`` / ``leiden_n_iterations``
     - ``"leidenalg"`` / ``-1``
     - Pin the partition backend and iteration policy used by the fixed run.
   * - ``decay``
     - ``"scaled_gaussian"``
     - BANKSY distance-decay model.
   * - ``refine_boundaries`` / ``compute_umap``
     - ``True`` / ``False``
     - Enable refinement and optionally compute UMAP diagnostics.

Raises ``TypeError`` or ``ValueError`` for an invalid AnnData/configuration,
``ImportError`` when clustering dependencies are absent, and ``RuntimeError``
when BANKSY output cannot be mapped one-to-one to input observations.

``plot_joint_cluster_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig, axes = spAlignDE.plot_joint_cluster_refinement(
       clustered,
       samples=["S2R2", "S2R3"],
   )

Plots raw and boundary-refined joint clusters for each sample. One shared
label-to-color mapping is used across every panel, allowing cluster identities
to be compared directly between samples and refinement stages.

Cross-sample alignment
----------------------

``align_cross_sample``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   aligned = spAlignDE.align_cross_sample(
       clustered,
       query_sample="S2R3",
       reference_sample="S2R2",
       prealignment_config=spAlignDE.PrealignmentConfig(),
       manual_prealignment_config=None,
       rasterization_config=spAlignDE.RasterizationConfig(),
       slddmm_config=spAlignDE.SLDDMMConfig(),
       device="cuda:0",
       copy=True,
   )

Runs global pre-alignment, shared-field construction and S-LDDMM in one call.
The default return value is the aligned AnnData. Set ``return_result=True`` to
return a ``CrossSampleAlignmentResult`` with fields, transformation and
metrics.

Set ``manual_prealignment_config`` to a ``ManualPrealignmentConfig`` to use a
fixed manual similarity transform instead of the automatic centroid fit.
``prealignment_config`` and ``manual_prealignment_config`` are mutually
exclusive.

``prealign_cross_sample``
~~~~~~~~~~~~~~~~~~~~~~~~~

Estimates a query-to-reference similarity transformation from weighted shared
cluster centroids. The returned ``PrealignmentResult`` contains the updated
AnnData, transform parameters and centroid correspondences.

``PrealignmentConfig`` controls scaling, reflection, cluster-size weighting and
the minimum cluster size used in the fit.

``prealign_cross_sample_manual``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies an explicit scale, rotation and translation using
``ManualPrealignmentConfig``. It writes the same standardized coordinate
columns and returns the same ``PrealignmentResult`` type as the automatic
method, so both paths feed directly into ``rasterize_cross_sample`` and
``run_slddmm_alignment``.

``apply_similarity_transform(points, config)`` applies the same transform to
an arbitrary ``n × 2`` coordinate array.

``interactive_manual_prealignment``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   manual_ui = spAlignDE.interactive_manual_prealignment(
       clustered,
       query_sample="S2R3",
       reference_sample="S2R2",
       initial_config=spAlignDE.ManualPrealignmentConfig(),
   )

   # Adjust the displayed sliders, then apply their current values.
   prealignment = manual_ui.apply()

Builds the Jupyter slider panel for scale, rotation and x/y translation. It
returns a ``ManualPrealignmentUI`` controller; ``selected_config`` exposes the
current values, ``preview()`` creates a fresh static figure and ``apply()``
returns the standard ``PrealignmentResult``. Interactive controls require the
``tutorial`` optional dependencies.

``plot_manual_prealignment_preview``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots the raw and manually transformed query beside the fixed reference
without modifying AnnData. This is the static alternative for scripts,
non-Jupyter sessions and documentation builds.

``rasterize_cross_sample``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Builds one smoothed composition channel for every cluster shared by the query
and reference, plus one pooled-normalized tissue-density channel. It returns a
``CrossSampleFields`` object on a common grid.

``RasterizationConfig`` controls grid spacing, grid margin, Gaussian smoothing,
and the relative cluster and density channel weights.

``run_slddmm_alignment``
~~~~~~~~~~~~~~~~~~~~~~~~

Optimizes the shooting-LDDMM transformation on the multichannel fields and
evaluates it at the original query locations. It returns a
``CrossSampleAlignmentResult``.

``SLDDMMConfig`` contains the deformation-kernel, shooting, optimizer,
matched/unmatched EM intensity-model and numeric-precision settings. CUDA is
selected automatically when available unless ``device`` is supplied.

Cross-sample subsampling stability
----------------------------------

The reusable functions used by the MERFISH stability report are available as
``spAlignDE.uncertainty``; the notebook no longer imports a helper from a
developer workspace.

``run_or_load_alignments`` runs each prepared replicate through S-LDDMM and
saves the learned transformations. ``map_reference_points_through_transforms``
then applies every transformation to one fixed query support, avoiding a
confound between transformation variation and changing subsample membership.
``compute_repeat_point_variance`` returns pointwise coordinate and distance
variance. Plotting and report-writing helpers reproduce the documented Figure
2E analysis.

Prepared replicate inputs are ordinary ``lddmm_input_repNN.npz`` files. The
source notebook reads their directory from
``SPALIGNDE_UNCERTAINTY_INPUT_DIR`` and writes to
``SPALIGNDE_UNCERTAINTY_OUTPUT_DIR``. These functions quantify empirical
subsampling stability; they do not return posterior probabilities or
confidence intervals.

ST-to-Allen-CCF alignment
-------------------------

``load_allen_ccf_reference``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   atlas = spAlignDE.load_allen_ccf_reference(
       "data/allen_ccf_2022/annotation_10.nrrd",
       "data/allen_ccf_2022/voxel_count_and_differences.csv",
       slice_index=675,
   )

Returns an ``AllenCCFReference`` containing one annotation slice, its physical
x/y axes, voxel size and hierarchy table.

``build_st_cluster_hierarchy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Builds expression-based coarse-to-fine partitions from the selected
single-sample ``cluster`` labels. ``STAtlasAlignmentConfig`` controls the
number of levels, cumulative variance threshold, minimum genes, mask
pre-alignment, point filtering and continuation criteria. The validated Atlas
default uses three ST levels (7, 16 and 25 structures for S2R1). Atlas
candidates from hierarchy depths 2–10 remain eligible at every stage; only
the ST partition becomes finer. The same configuration exposes the five
structure-pairing weights, soft Dice/SDF/thickness gates, score threshold and
maximum average surface distance. The defaults use a normalized,
shape-balanced score (SDF 0.05, Chamfer 0.05, Dice 0.20, area 0.50 and
thickness 0.20) without any structure-name-specific override. Continuation
uses ``kernel_scale=200`` and ``velocity_grid_spacing=50``. The validated
automatic optimizer schedule is 100, 500 and 100 iterations across the three
coarse-to-fine stages, followed by 200 iterations per continuation round. All
stages retain the final optimization iterate.

``align_st_to_allen_atlas``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = spAlignDE.align_st_to_allen_atlas(
       clustered,
       atlas,
       config=spAlignDE.STAtlasAlignmentConfig(),
       cluster_key="cluster",
       output_dir="st_to_atlas_output",
   )

Runs whole-tissue IoU pre-alignment, hierarchical ST-to-atlas structure
pairing, coarse-to-fine S-LDDMM and per-cell Allen-label sampling. It returns
an ``STAtlasAlignmentResult`` with aligned AnnData, matched-pair table,
stage-level QC, selected atlas reference and pre-alignment parameters.

``load_st_atlas_alignment`` loads a previously saved run into the same result
object. ``plot_st_atlas_alignment`` compares globally pre-aligned and final
coordinates while assigning each matched ST cluster and Allen structure the
same fixed structure-name color. ``load_atlas_structure_color_map`` returns the
validated paper palette or loads a user CSV. ``load_atlas_label_color_map``
loads the separate fixed label-ID transfer palette; ``plot_atlas_label_transfer``
uses it for both the Allen annotation and labels sampled onto aligned ST cells.

``load_ui_atlas_pairing``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   pairing = spAlignDE.load_ui_atlas_pairing(
       "spalign_de_experimental_pairs.csv",
       expected_atlas_slice=atlas.slice_index,
   )

Loads the unedited CSV exported by the interactive region-pairing tool,
infers which UI panel is ST, validates the Allen slice and reconstructs raw or
custom selections. One ``group_id`` becomes one many-to-many deformation
channel formed from the union of its ST IDs and Allen IDs; individual CSV rows
are not treated as separate channels. The returned ``UIAtlasPairing`` retains
the raw export, grouped channels and per-ST-cluster source table.

``align_st_to_allen_atlas_from_ui_pairs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = spAlignDE.align_st_to_allen_atlas_from_ui_pairs(
       clustered,
       atlas,
       pairing,
       config=spAlignDE.UIAtlasAlignmentConfig(
           prealignment_mode="mask",
           kernel_scale=200,
           time_steps=5,
           velocity_grid_spacing=50,
           iterations=500,
           restore_best_checkpoint=False,
       ),
       cluster_key="cluster",
       output_dir="st_to_atlas_ui_output",
   )

Uses the UI export as the accepted pair specification. It does **not** run
automatic Atlas candidate discovery, pair metrics/gates, pair selection or
continuation-time rematching. The standard ``result.matched_pairs`` field is a
normalized UI-pair table in this workflow, not the output of a new
matching calculation.

The function still runs the required alignment preparation: global
initialization, point filtering, ST and Allen mask construction, mask cleanup,
signed-distance conversion, global channel weighting, S-LDDMM input/grid
construction, optimization, point mapping and Allen-label sampling. It returns the same
``STAtlasAlignmentResult`` and standardized AnnData coordinate/label fields
as automatic Atlas alignment. The ST cluster IDs must be exactly those used in
the UI session; a fresh clustering run may renumber otherwise equivalent
spatial structures. ``UIAtlasAlignmentConfig`` exposes the validated mask,
signed-distance, area-balancing and deformation settings. Weight changes apply
globally to every UI group; no named anatomy receives a special weight.

``prealignment_mode="mask"`` estimates the whole-tissue similarity transform.
For a separately completed manual pre-alignment, store its finite coordinates
in ``adata.obs["x_prealigned"]`` and ``adata.obs["y_prealigned"]`` and use:

.. code-block:: python

   manual_config = spAlignDE.UIAtlasAlignmentConfig(
       prealignment_mode="provided",
       provided_prealigned_x_key="x_prealigned",
       provided_prealigned_y_key="y_prealigned",
   )

Only transform estimation is skipped in this mode; filtering, mask processing,
S-LDDMM input construction and deformation still run.

ST-to-histology alignment
-------------------------

``extract_histology_features``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   features = spAlignDE.extract_histology_features(
       "data/histology/he_image.btf",
       "histology_output/feature_extraction",
       config=spAlignDE.HistologyFeatureConfig(
           target_microns_per_pixel=0.5,
           shifted_tiles=True,
       ),
   )

Accepts one high-resolution image, converts it to RGB, rescales from embedded
or explicit physical pixel size when available, pads to a multiple of 224 and
runs shifted-tile HIPT feature extraction. The histology-side dataset input is
the image only. Set ``SPALIGNDE_HIPT_DIR`` or ``extractor_dir`` to the HIPT
script and checkpoints.

``prepare_histology_image`` exposes image conversion, optional physical-scale
resampling and padding without running HIPT. ``load_histology_features``
constructs a ``HistologyFeatureResult`` from an already prepared image and
feature pickle, which is useful for resuming a checked run without repeating
feature extraction.

``cluster_histology_features``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   histology = spAlignDE.cluster_histology_features(
       features,
       "histology_output/clustering",
       config=spAlignDE.HistologyClusteringConfig(),
   )

Clusters HIPT, RGB and spatial feature blocks, fills tissue holes, merges
symmetry-compatible regions and cleans small disconnected islands. It returns
a ``HistologyClusteringResult`` with the tissue mask and raw, merged and
cleaned label rasters. ``plot_histology_feature_clusters`` displays the full
image-to-structure evidence chain. ``load_histology_clustering`` reloads the
saved compact clustering output into the same result type.

``build_st_histology_structures``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Builds expression-based coarse-to-fine partitions from the query's original
single-sample labels. ``STHistologyStructureConfig`` controls the number of
levels, cumulative variance fraction, minimum genes and blank-probe removal.
``plot_st_histology_structures`` compares all levels and marks the one selected
for image correspondence.

``prealign_st_to_histology``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prealignment = spAlignDE.prealign_st_to_histology(
       clustered,
       histology,
       config=spAlignDE.HistologyPrealignmentConfig(method="mask_overlap"),
   )

Estimates rotation, isotropic scale and translation by whole-tissue mask IoU.
Set ``method="manual"`` with a ``ManualPrealignmentConfig`` when automatic
overlap is anatomically unreliable. ``interactive_histology_prealignment``
provides notebook sliders and returns the same typed result; selected values
are persisted in ``uns["spAlignDE"]``.
``plot_histology_prealignment_preview`` is the non-interactive static preview
for a supplied ``ManualPrealignmentConfig``.

``align_st_to_histology``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = spAlignDE.align_st_to_histology(
       prealignment,
       structure_key="cluster_raw_level_k7",
       config=spAlignDE.STHistologyAlignmentConfig(),
       output_dir="histology_output/alignment",
   )

Rasterizes ST and image structures on the common feature grid, selects
non-overlapping geometric correspondences, converts accepted masks to
continuous signed-distance channels and fits S-LDDMM. The returned
``STHistologyAlignmentResult`` includes aligned AnnData, accepted pairs,
pre-alignment parameters, image structures and diagnostic fields.
``plot_st_histology_alignment`` compares global initialization and final
alignment over the original color image.
``plot_st_histology_pair_overlap(result, stage="after")`` displays every
accepted mask pair after deformation and returns long-form before/after Dice,
IoU and ASD metrics; the same table is saved as
``matched_structure_pair_overlap_metrics.csv``. ``STHistologyAlignmentConfig``
exposes the normalized SDF, Chamfer, Dice, area and thickness weights. Their
empirical defaults are 0.20, 0.40, 0.15, 0.25 and 0.00; raw ASD is excluded
from this score and used only by ``pair_asd_threshold`` as an independent QC
gate. The validated S-LDDMM defaults use ``kernel_scale=60`` and
``velocity_grid_spacing=6``. See :ref:`cross_modality_pairing_weights` before
changing these values.

Spatial ATAC-to-ST alignment
----------------------------

``prealign_atac_to_st``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prealigned = spAlignDE.prealign_atac_to_st(
       atac,
       st_reference,
       config=spAlignDE.ATACSTPrealignmentConfig(),
       atac_cluster_key="cluster_raw",
       st_cluster_key="cluster",
   )

Applies independently recorded global similarity transforms to the spatial
ATAC query and fixed ST reference, selects the corresponding ST half brain,
and places both modalities on one raster canvas. The original
``obsm["spatial"]`` coordinates remain unchanged. The returned
``ATACSTPrealignmentResult`` contains the ATAC AnnData, cropped ST AnnData,
canvas dimensions, and the recorded initialization settings.

``align_atac_to_st``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = spAlignDE.align_atac_to_st(
       prealigned,
       config=spAlignDE.ATACSTAlignmentConfig(),
       atac_cluster_key="cluster_raw",
       st_cluster_key="cluster",
       output_dir="atac_to_st_output",
   )

Rasterizes independently inferred structures, scores every cross-modality
candidate with globally shared SDF, Chamfer, area and Dice terms, selects a
one-to-one set, and runs area-balanced signed-distance S-LDDMM. The result
contains the aligned ATAC AnnData, fixed ST reference, accepted-pair table and
mask/deformation diagnostics. No anatomical label or region-specific weight
is required. ``ATACSTAlignmentConfig`` exposes normalized empirical weights of
0.35, 0.25, 0.30 and 0.10, respectively; see
:ref:`cross_modality_pairing_weights` before adapting them to new data.

``plot_atac_st_prealignment`` displays the fixed ST field, transformed ATAC
query and their initialization overlay. ``plot_atac_st_matched_structures``
assigns one shared color to each accepted ATAC/ST pair before and after
alignment. ``plot_atac_st_alignment`` compares the global initialization with
the final S-LDDMM coordinates.

Post-alignment local inference
------------------------------

The current inference entry points use the lowercase package namespace:

.. code-block:: python

   from spalignde.inference import (
       cluster_trajectories,
       fit_local_de,
       gene_level_acat_pvalue,
       gene_level_age_trend_acat,
       prepare_inference,
   )

``canonical_visium_barcodes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Signature::

   spalignde.datasets.canonical_visium_barcodes(
       values, *, source_name="barcode values"
   )

Extracts terminal 10x barcodes from plain, sample-prefixed or source-file
annotated observation identifiers. The kidney tutorial uses it to join raw
counts to aligned coordinates by identifier rather than row order.
``values`` accepts a sequence, Series or Index and returns a string Series in
the same order. ``source_name`` is used in validation messages. A
``ValueError`` is raised when any identifier lacks a terminal 10x barcode.

``build_visium_coordinate_table``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Signature::

   spalignde.datasets.build_visium_coordinate_table(
       tissue_positions, aligned_coordinates, *, sample_id,
       position_barcode_key="barcode", array_row_key="array_row",
       array_col_key="array_col", aligned_id_key="cell_id",
       aligned_sample_key="sample_id", aligned_coordinate_key=("x", "y")
   )

Matches one sample's Visium tissue-position table to its aligned coordinates
one-to-one by terminal 10x barcode, never by row order. The returned DataFrame
contains the standardized columns ``barcode``, ``cell_id``, ``sample_id``,
``x``, ``y``, ``x_aligned``, and ``y_aligned``. The kidney notebook then joins
raw expression and determines the broad risk-gene pool before calling
``prepare_inference``; these data-handoff steps are separate from calibration.

``prepare_inference``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prepared = spalignde.prepare_inference(
       inference_data,
       reference="NL3",
       genes=["Cbr1", "Cd44", "Myo5a"],
       risk_genes=risk_genes,
       aligned_coordinate_key=("x_aligned", "y_aligned"),
       density_energy_share=0.75,
       library_size=10_000,
       grid_n=None,
       cell_type_key=None,
       n_jobs=4,
       random_state=1,
   )

Accepts a long spot/cell table with sample identity, original and aligned
coordinates, batch identity and non-negative expression columns. It builds the
fixed shared grid, sample-specific local neighborhoods, technical
quality-control profiles, stable-gene/density mismatch-risk maps and optional
cell-type support maps. The returned ``PreparedInference`` is reusable across
genes and naive/mismatch-aware fits. With ``grid_n=None``, the R-driven
resolution is retained when the actual tissue-valid location count lies
between ``N_typ`` and ``2 * N_typ``; otherwise it is adjusted toward the
nearest bound. An explicit integer ``grid_n`` overrides this automatic rule.
The selected resolution, source, target interval and final location count are
stored in ``prepared.metadata``.

``n_jobs`` is the caller-requested worker count for the preparation stages that
can run in parallel. The two seeded per-sample auto-geometry passes—subsampling
and parameter estimation—always run with one worker, because sharing one RNG
across parallel samples would allow thread scheduling to change which sample
consumes each draw. All later preparation stages continue to use the requested
``n_jobs`` value; the complete analysis is therefore not described as
single-threaded. ``prepared.metadata`` records ``n_jobs``,
``auto_geometry_n_jobs=1`` and ``random_state``.

``fit_local_de``
~~~~~~~~~~~~~~~~

Signature::

   spalignde.fit_local_de(prepared, *, genes=None, contrast="vs_reference",
                          alpha=0.05, mismatch_aware=True,
                          technical_adjustment=True,
                          cell_type_adjustment=True, global_offset=True,
                          region_cleanup=False, n_jobs=1,
                          random_state=None, verbose=True)

.. code-block:: python

   result = spalignde.fit_local_de(
       prepared,
       contrast="vs_reference",
       mismatch_aware=True,
       technical_adjustment=True,
       cell_type_adjustment=False,
       global_offset=False,
       region_cleanup=False,
       random_state=1,
   )

Fits one local test per valid grid location and gene. ``contrast`` may be
``"vs_reference"`` or ``"sequential"``. ``mismatch_aware=False`` provides the
naive test on the same prepared geometry; enabling it inflates local variance
according to post-alignment risk without changing the estimated contrast.
Complete cell-type annotations are required before
``cell_type_adjustment=True`` can be used. BH adjustment is applied within
each gene and contrast across tested grid locations.

For every gene and contrast, mismatch-aware fitting starts from initial local
statistics obtained without mismatch inflation. It bins those statistics by
normalized local risk, removes each bin median, and divides the bin MAD by the
Student-t null MAD. Nonnegative excess dispersion is constrained to be
nondecreasing, fitted as a quadratic through the origin, and boundedly
rescaled at the risk bin nearest the 80th percentile to obtain a provisional
gene-by-contrast coefficient.

A provisional calibration is valid only after a successful within-contrast
fit with adequate usable locations, at least four distinct risk bins including
positive-risk support, finite calibration quantities, a finite nonnegative
capped local coefficient, and a zero global coefficient. A successful zero
coefficient remains valid, whereas failed calibrations are excluded. For
multiple contrasts, the valid provisional coefficients are combined with an
equal-weight Huber center. The resulting single robust gene-specific
coefficient :math:`\lambda_g` is shared across every fitted contrast, and the
applied mismatch factor is

.. math::

   \phi^{\mathrm{align}}_{ig}=1+\lambda_g r_i^2,
   \qquad \lambda_g \geq 0.

The through-origin relation leaves zero-risk locations at the base variance.
The comparison-level global risk score is retained only for diagnostics and
does not impose a spatially uniform variance penalty. With one
valid contrast, the Huber center is exactly that contrast's coefficient.
Provisional coefficients are diagnostics rather than contrast-specific final
coefficients. The calibration method, validity information, provisional
values, aggregation diagnostics, and final :math:`\lambda_g` are stored under
``result.fits[gene]["terrain_data"]["risk_calibration"]``.

``prepared`` must be returned by ``prepare_inference``. ``genes=None`` tests
all genes prepared earlier. ``contrast`` is ``"vs_reference"`` or
``"sequential"``; ``alpha`` is strictly between zero and one.
``mismatch_aware`` enables post-alignment risk weighting,
``technical_adjustment`` enables technical-quality weighting, and
``global_offset`` includes the global sample offset. ``region_cleanup`` only
post-processes the significant-region mask. ``n_jobs``, ``random_state`` and
``verbose`` control execution. The function returns ``LocalDEResult`` and
raises ``TypeError`` for a non-``PreparedInference`` input or ``ValueError``
for invalid contrasts, alpha, genes, or unavailable requested cell-type
adjustment.

The returned ``LocalDEResult`` stores fitted maps in ``result.fits`` and keeps
the originating ``PreparedInference``. ``plot_local_result`` displays aligned
reference/query expression beside the local statistic and FDR-significant
region contour.

``gene_level_acat_pvalue`` and ``acat_pvalue``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``gene_level_acat_pvalue(result, gene)`` is the public result-aware helper. It
combines dependent local P values within each contrast and, for multi-query
results, combines contrast-level evidence using valid-grid counts as weights.
Compact fitted results retain ``p_by_time`` so the helper strictly combines raw
local P values; adjusted q-values are never substituted. A missing raw-P map is
reported as an invalid result rather than reconstructed from another field.
``acat_pvalue`` is the lower-level array utility and uses a stable reciprocal
Cauchy-tail branch for extremely small P values. These summaries do not
replace the location-level q-value maps or genome-wide gene-level
multiple-testing control.

``gene_level_age_trend_acat``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Signature::

   spalignde.gene_level_age_trend_acat(result, gene, *, time_values=None,
                                        alpha=0.05)

Tests a spatially distributed linear age trend for one gene. At each retained
grid location it regresses unsmoothed adjusted local expression
``muA_adj_by_time`` on age with an intercept and uses the mismatch-aware
``Wv_by_time`` values as local precision. The two-sided local slope P values
are combined across space with ACAT. The reference section is not inserted as
an additional age observation, and this pre-trajectory test does not use
smoothed trajectories, cluster labels, or the selected cluster count.

The returned mapping contains ``summary``, ``per_contrast``, and ``per_grid``.
``summary["gene_level_trend_acat_p"]`` is the global P value for the multi-age
application. ``summary["legacy_any_signal_acat_p"]`` retains the older
per-contrast any-spatial-change result for diagnostic comparison only. When
``time_values`` is omitted, numeric values are parsed from contrast identifiers;
otherwise one numeric age must be supplied in the stored contrast order.

``cluster_trajectories``
~~~~~~~~~~~~~~~~~~~~~~~~

Signature::

   spalignde.cluster_trajectories(result, gene, *, n_clusters="auto",
                                  time_values=None, auto_k_options=None,
                                  random_state=None)

Clusters adjusted local-expression trajectories across ordered query samples.
With ``n_clusters="auto"``, every candidate K is first evaluated by its
held-out cluster-specific trajectory gain relative to a shared time trend. If
the best dynamic gain is not positive after subtracting its one-SE
uncertainty, the smallest candidate K is selected. Otherwise, candidates whose
dynamic gains are within one SE of the best gain are retained and examined
from fine to coarse resolution. At each step the diagnostic is the fraction of
grid locations lying in connected components smaller than one ``R_map``
footprint. Coarsening continues while that fraction decreases. If the next
coarser candidate fails to reduce fragmentation, the current finer-side local
minimum is retained. If fragmentation decreases throughout the retained scan
and supplies no elbow, the rule takes one conservative coarsening step from
the finest retained candidate. Fragmentation is therefore an elbow diagnostic,
not an objective that is globally minimized, and the no-elbow case does not
select the coarsest candidate. With four or five time points, the dynamic check
uses linear leave-one-time-out fits; with three or fewer, it returns the
smallest candidate.

``time_values`` follows the trajectory time-ID order stored in the fit. When
omitted, numeric suffixes are used when available and equal spacing otherwise.
The public selection record is available in
``trajectory.metadata["selection"]``. Its stable diagnostics are ``mode``,
``recommended_k``, ``best_dynamic_gain``,
``best_dynamic_gain_lower_1SE``, ``dynamic_candidates``,
``fine_to_coarse_order``, ``fine_to_coarse_scan``,
``fragmentation_stop_at_k``, ``rejected_coarser_k``,
``no_elbow_fallback_k``, ``reason`` and ``rule``.

Packaged examples and public result contracts
---------------------------------------------

``make_cross_sample_example`` returns the small two-sample AnnData object used
by the README smoke test. The fixed kidney handoff is exposed through
``KIDNEY_SAMPLES``, ``load_kidney_aligned_coordinates`` and
``kidney_alignment_metadata``. The compact five-section aging-brain example
is exposed through ``AGING_BRAIN_FIGURE5A_REFERENCE``,
``AGING_BRAIN_FIGURE5A_QUERIES``, ``AGING_BRAIN_FIGURE5A_SAMPLES``,
``load_aging_brain_figure5a``, ``aging_brain_figure5a_genes`` and
``aging_brain_figure5a_metadata``. These loaders return packaged data only;
they do not download the manuscript-scale raw inputs.

The 0.1 compatibility interface also retains the generic aging-brain aliases
``AGING_BRAIN_REFERENCE``, ``AGING_BRAIN_QUERIES``, ``AGING_BRAIN_SAMPLES``,
``load_aging_brain``, ``aging_brain_genes`` and ``aging_brain_metadata``.
For Visium data handoff it continues to expose ``VisiumInferenceInput``,
``summarize_raw_genes`` and ``build_visium_inference_table``. New code may use
the explicit coordinate-table workflow described above; these existing public
helpers remain documented here so older reproducible workflows keep a clear
API reference.

Workflow functions return typed objects so data, configuration settings and
diagnostics remain together. Relevant public result objects include
``ATACSTAlignmentResult``, ``HistologyFeatureResult``,
``HistologyPrealignmentResult``, ``HistologyPrealignmentUI`` and
``TrajectoryResult``. Their fields are populated by the corresponding
functions described above; users normally do not instantiate them directly.

``spAlignDE.__version__`` reports the installed release. The tuple
``REQUIRED_OUTPUT_COLUMNS`` lists the four standardized coordinate columns
required from a completed point-based alignment.

Visualization
-------------

``plot_prealignment_result(prealignment)`` displays raw and pre-aligned
query/reference scatter plots with the matched shared-cluster centroids.

``plot_rasterized_fields(fields)`` displays the query/reference
cluster-composition previews and density channels.

``plot_alignment_result(result)`` compares pre-aligned and final S-LDDMM
query/reference overlap without cluster colors.

``plot_cluster_alignment_result(result)`` repeats the pre-aligned versus final
comparison using a shared color mapping for joint-cluster identities.

Output schema
-------------

Every completed point-based alignment adds:

- ``adata.obs["x_prealigned"]``
- ``adata.obs["y_prealigned"]``
- ``adata.obs["x_aligned"]``
- ``adata.obs["y_aligned"]``

Original coordinates remain in ``adata.obsm["spatial"]``. Configuration,
grid metadata and diagnostic metrics are stored in ``adata.uns["spAlignDE"]``.

See :doc:`tutorials/cross_sample_alignment` and
:doc:`tutorials/cross_modality_atac_alignment` for alignment input formats,
and :doc:`tutorials/post_alignment_inference` for the complete local-inference
model, kidney input/output fields and executed notebook.
