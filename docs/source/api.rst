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
same combined-AnnData contract.

``read_cross_sample_csv``
~~~~~~~~~~~~~~~~~~~~~~~~~

Explicit CSV reader used by ``load_cross_sample_data``. Metadata files require
``cell_id``, ``x`` and ``y``; expression files require ``cell_id`` followed by
numeric gene columns. Rows are matched by ``cell_id`` and genes are aligned by
name across samples. The complete, canonical format is documented in
:ref:`cross-sample-csv-contract`.

``validate_cross_sample_anndata``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validates the combined AnnData input. By default it requires
``adata.obs["sample_id"]``, ``adata.obs["cluster"]``, globally unique
observation names, and a finite ``n_obs × 2`` coordinate matrix in
``adata.obsm["spatial"]``. Set ``require_cluster=False`` before joint
clustering.

``cluster_joint``
~~~~~~~~~~~~~~~~~

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
boundary refinement, and random state.

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
mismatch-aware EM and numeric-precision settings. CUDA is selected
automatically when available unless ``device`` is supplied.

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
pre-alignment, point filtering and continuation criteria. Its Atlas default is
four levels, which provides a more gradual deformation path for thin
structures than the earlier three-level workflow. By default the coarsest
partition retains at least seven structures before the remaining levels are
spaced through the final clustering. The same configuration exposes the five
structure-pairing weights, soft Dice/SDF/thickness gates, score threshold and
maximum average surface distance. The defaults use a normalized,
shape-balanced score (SDF 0.08, Chamfer 0.06, Dice 0.18, area 0.47 and
thickness 0.21) without any structure-name-specific override.

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
contract. ``plot_st_atlas_alignment`` compares globally pre-aligned and final
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
the raw export, grouped channels and per-ST-cluster provenance table.

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
normalized UI-pair provenance table in this workflow, not the output of a new
matching calculation.

The function still runs the required alignment preparation: global
initialization, point filtering, ST and Allen mask construction, mask cleanup,
signed-distance conversion, global channel weighting, S-LDDMM input/grid
construction, optimization, point mapping and Allen-label sampling. It returns the same
``STAtlasAlignmentResult`` and standardized AnnData coordinate/label contract
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
image-to-structure evidence chain.

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
canvas dimensions, and complete initialization provenance.

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
is required.

``plot_atac_st_prealignment`` displays the fixed ST field, transformed ATAC
query and their initialization overlay. ``plot_atac_st_matched_structures``
assigns one shared color to each accepted ATAC/ST pair before and after
alignment. ``plot_atac_st_alignment`` compares the global initialization with
the final S-LDDMM coordinates.

Post-alignment local inference
------------------------------

``canonical_visium_barcodes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts terminal 10x barcodes from plain, sample-prefixed or source-file
annotated observation identifiers. The kidney tutorial uses it to join raw
counts to the aligned H5AD by identifier rather than row order.

``build_visium_inference_table``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   visium_input = spAlignDE.build_visium_inference_table(
       "/path/to/aligned_samples.h5ad",
       {
           "reference": "/path/to/reference_filtered_feature_bc_matrix.h5",
           "query": "/path/to/query_filtered_feature_bc_matrix.h5",
       },
       genes=["gene_a", "gene_b"],
       min_detected_spots=10,
       min_total_counts=10,
       batch="matched_pair",
   )

Provides the complete Visium alignment-to-inference handoff. It validates
``sample_id``, original coordinates, ``x_aligned``/``y_aligned`` and terminal
barcodes in an aligned AnnData object; reads raw 10x HDF5 matrices; performs a
one-to-one barcode join; filters the broad mismatch-risk gene pool; and returns
a ``VisiumInferenceInput`` containing ``data``, ``coordinates``, ``genes``,
``risk_genes``, ``sample_sizes`` and ``n_common_genes``. Raw-count AnnData
objects can be supplied instead of HDF5 paths for programmatic workflows.

The function never assumes that ``aligned.X`` contains raw expression.

``prepare_inference``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   prepared = spAlignDE.prepare_inference(
       visium_input.data,
       reference="NL3",
       genes=["Cbr1", "Cd44", "Slc5a2"],
       risk_genes=visium_input.risk_genes,
       aligned_coordinate_key=("x_aligned", "y_aligned"),
       density_energy_share=0.25,
       library_size=10_000,
       cell_type_key=None,
       random_state=1,
   )

Accepts a long spot/cell table with sample identity, original and aligned
coordinates, batch identity and non-negative expression columns. It builds the
fixed shared grid, sample-specific local neighborhoods, technical
quality-control profiles, stable-gene/density mismatch-risk maps and optional
cell-type support maps. The returned ``PreparedInference`` is reusable across
genes and Naive/Mismatch-aware fits.

``fit_local_de``
~~~~~~~~~~~~~~~~

.. code-block:: python

   result = spAlignDE.fit_local_de(
       prepared,
       contrast="vs_reference",
       mismatch_aware=True,
       technical_adjustment=True,
       cell_type_adjustment=False,
       global_offset=False,
       region_cleanup=True,
       random_state=1,
   )

Fits one local test per valid grid location and gene. ``contrast`` may be
``"vs_reference"`` or ``"sequential"``. ``mismatch_aware=False`` provides the
Naive test on the same prepared geometry; enabling it inflates local variance
according to post-alignment risk without changing the estimated contrast.
Complete cell-type annotations are required before
``cell_type_adjustment=True`` can be used. BH adjustment is applied within
each gene and contrast across tested grid locations.

The returned ``LocalDEResult`` stores fitted maps in ``result.fits`` and keeps
the originating ``PreparedInference``. ``plot_local_result`` displays aligned
reference/query expression beside the local statistic and FDR-significant
region contour.

``acat_pvalue`` and ``cluster_trajectories``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``acat_pvalue`` combines dependent local P values when a gene-level summary is
needed. ``cluster_trajectories`` groups shared-grid locations with related
adjusted-expression trajectories across ordered query samples. These are
downstream summaries and do not replace the location-level q-value maps.

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
:doc:`tutorials/cross_modality_atac_alignment` for alignment input contracts,
and :doc:`tutorials/post_alignment_inference` for the complete local-inference
model, kidney input/output contract and executed notebook.
