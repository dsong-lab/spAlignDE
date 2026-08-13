Parameter Tuning Guide
======================

This page explains how to adapt spAlignDE to a new dataset. Start from the
configuration of the closest documented workflow and change one parameter
group at a time. A successful optimization cannot rescue incorrect query and
reference roles, a wrong atlas slice, a poor global pre-alignment, or
biologically implausible structure pairs.

Recommended tuning order
------------------------

Use the following order for every alignment workflow:

1. **Confirm the coordinate system and field of view.** Record coordinate
   units, tissue extent, image pixel size, query/reference roles, crop and
   atlas slice.
2. **Inspect spatial structures.** Tune clustering or image segmentation until
   structures are spatially coherent and cover both tissues. Do not compare
   cluster integer values between separate runs.
3. **Fix global pre-alignment.** Rotation, scale and translation should place
   homologous tissue compartments in approximate agreement before S-LDDMM.
4. **Inspect masks and candidate pairs.** Tune rasterization, mask construction
   and pairing gates before changing deformation parameters.
5. **Tune deformation scale and resolution.** Adjust ``kernel_scale`` (legacy
   ``a``) and ``velocity_grid_spacing`` (legacy ``grid_step``) together.
6. **Tune optimization only if needed.** Change momentum learning rate and
   iteration count after the geometry, masks and pairs are credible.
7. **Re-run quality control.** Compare pre-aligned and final overlays, accepted
   pairs, energy history and downstream label or cluster agreement.

Quick tuning map by pipeline section
------------------------------------

Use this table as the short entry point. Tune only one row at a time and keep
the remaining rows at the closest validated tutorial settings.

.. list-table::
   :header-rows: 1
   :widths: 20 31 49

   * - Section
     - Key parameters
     - Change them when
   * - Structure discovery
     - ``resolution``, ``banksy_lambda``, ``num_neighbors``
     - Regions are fragmented, over-merged or absent in one sample. Select by
       spatial coherence and shared coverage, not by cluster integers.
   * - Global pre-alignment
     - angle, scale, translation, reflection and crop/slice controls
     - Whole tissues have the wrong orientation, field of view or size. Fix
       this before pairing or S-LDDMM.
   * - Raster and masks
     - ``grid_spacing``, ``blur_sigma``, minimum area and cleanup size
     - Thin structures disappear, masks contain islands, or sparse points do
       not form continuous regions.
   * - Structure pairing
     - normalized shape weights, score threshold and independent Dice/ASD gates
     - Candidate ranking is systematically wrong or good pairs sit just below
       a justified QC gate. Inspect masks before loosening thresholds.
   * - Deformation scale
     - ``kernel_scale`` and ``velocity_grid_spacing``
     - Credible pairs align broad anatomy but miss local bends, or the fitted
       field is too wavy. Tune these two together.
   * - Optimizer
     - ``iterations``, ``momentum_lr`` and ``diffeomorphic_start``
     - Energy is still decreasing, converges too slowly, or oscillates after
       geometry, masks and pairs have passed QC.
   * - Post-alignment inference
     - ``risk_genes``, filtering, ``density_energy_share``, ``grid_n`` and
       adjustment/cleanup switches
     - Alignment already passes QC but mismatch risk, spatial resolution or
       reported connected regions are not appropriate for the study design.

For a controlled sweep, freeze the input files and row order, use one fixed
workflow seed, and change three nearby values from one parameter group. Reject
geometrically implausible runs before looking at downstream biological
statistics. After selecting a configuration, execute it twice in clean working
directories. Compare cluster partitions and pair identities exactly; compare
CUDA coordinates within the documented tolerance. Save the selected config,
input hashes, seed, accepted-pair table and overlays with the result.

Save the selected configuration under ``adata.uns["spAlignDE"]`` with the
aligned coordinates. Parameter choices are part of the result, not merely
runtime settings.

Coordinate units come first
---------------------------

``kernel_scale``, ``velocity_grid_spacing``, raster ``grid_spacing``, manual
translations and distance thresholds are expressed in the coordinate system
used by the corresponding stage. Their numeric values therefore cannot be
copied safely between micrometres, Visium array coordinates, image pixels and
histology feature-grid pixels.

Before tuning, calculate the query/reference tissue width and height after
pre-alignment and record the typical nearest-neighbor spacing. If coordinates
are multiplied or downsampled, scale every distance-valued parameter
consistently. The kidney tutorial, for example, multiplies coordinates by 50
internally and converts the final coordinates back to the original scale.

Legacy S-LDDMM symbols and public API names
-------------------------------------------

Older analysis notebooks use short mathematical names. Public package
configurations use descriptive names:

.. list-table::
   :header-rows: 1
   :widths: 18 35 47

   * - Legacy name
     - Public configuration field
     - Meaning
   * - ``a``
     - ``kernel_scale``
     - Spatial smoothness/correlation scale of the velocity kernel.
   * - ``p``
     - ``kernel_power``
     - Smoothness-operator power; normally keep the validated value 2.
   * - ``expand``
     - ``velocity_expand``
     - Expansion of the velocity domain beyond the source bounding box.
   * - ``nt``
     - ``time_steps``
     - Number of numerical steps used to integrate the diffeomorphic flow.
   * - ``grid_step``
     - ``velocity_grid_spacing``
     - Spacing between velocity/momentum control locations.
   * - ``niter``
     - ``iterations``
     - Maximum optimizer iterations.
   * - ``diffeo_start``
     - ``diffeomorphic_start``
     - Iteration at which momentum/deformation updates begin.
   * - ``lrL`` / ``lrT``
     - ``affine_linear_lr`` / ``affine_translation_lr``
     - Learning rates for affine linear and translation components.
   * - ``lrM``
     - ``momentum_lr``
     - Learning rate for the initial momentum field.
   * - ``lrM_min``
     - ``minimum_momentum_lr``
     - Lower bound when momentum learning-rate decay is active.
   * - ``sigmaM``
     - ``sigma_match`` or ``matching_scale``
     - Scale of the structure-matching term; smaller values increase matching
       pressure.
   * - ``sigmaR``
     - ``sigma_regularization`` or ``deformation_regularization``
     - Regularization denominator. **Larger** values weaken the deformation
       penalty and permit stronger warps; smaller values strengthen the
       penalty.

The three different grids
-------------------------

Do not confuse these resolutions:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Grid
     - Parameter/example
     - Role
   * - Observation coordinates
     - ``adata.obsm["spatial"]``
     - Original cell/spot locations and the units to which the final mapping is
       applied.
   * - Structural image grid
     - ``RasterizationConfig.grid_spacing`` or a histology/ATAC canvas
     - Converts point structures into continuous matching fields. A finer grid
       preserves more detail but increases image memory and sensitivity to
       sparse sampling.
   * - Velocity grid
     - ``velocity_grid_spacing`` / ``grid_step``
     - Controls the number of deformation degrees of freedom. It need not equal
       the structural image spacing.

S-LDDMM deformation parameters
------------------------------

``kernel_scale`` (``a``)
~~~~~~~~~~~~~~~~~~~~~~~~

This is the primary biological smoothness parameter.

- Increase it when the fitted deformation is too local, wavy or sensitive to
  small mask defects. The result becomes smoother and more global.
- Decrease it when broad structures align but reproducible local bends remain
  unresolved. The result can express more local deformation, with greater
  overfitting and stability risk.
- Interpret it relative to the pre-aligned tissue extent, not as a universal
  number. Start from the closest validated workflow.

``velocity_grid_spacing`` (``grid_step``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This parameter controls how densely the momentum field is represented.

- Decreasing it adds control locations, increases memory/runtime, and permits
  finer local changes.
- Increasing it reduces degrees of freedom, improves speed and regularizes the
  deformation, but may miss narrow or localized bending.
- If ``kernel_scale`` is reduced substantially, a smaller velocity-grid spacing
  may also be required to represent the shorter-scale field. A very dense grid
  cannot compensate for incorrect structure pairs.

A practical initial check is to visualize the velocity-grid dimensions. Fewer
than several control locations across a tissue axis is too coarse; an
extremely dense grid relative to sampling resolution wastes memory and can fit
noise.

``time_steps`` (``nt``)
~~~~~~~~~~~~~~~~~~~~~~~

This controls numerical integration accuracy, not the intended deformation
amplitude. Increase it when a strong deformation is poorly integrated or when
results change materially after increasing ``nt``. Runtime grows approximately
linearly with the number of steps. Values 3--8 cover the validated workflows;
do not increase ``nt`` merely because an alignment is geometrically wrong.

``iterations`` and ``momentum_lr``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the energy history to tune these together.

- If energy is still falling smoothly at the final iteration, increase
  ``iterations``.
- If energy plateaus early, more iterations alone are unlikely to help.
- Increase ``momentum_lr`` cautiously when progress is consistently slow.
- Reduce it when energy oscillates, becomes non-finite, produces implausible
  local motion, or repeatedly triggers rollback/gradient clipping.
- Coordinate rescaling changes gradient magnitudes, so a learning rate from one
  coordinate system may be inappropriate in another.

Keep the affine learning rates near the validated workflow values after a good
pre-alignment. Large affine updates during S-LDDMM often indicate that the
global initialization should be fixed instead.

By default, ``restore_best_checkpoint=False`` returns the final-iteration
transformation because EM intensity updates can make energies from different
optimization phases incomparable. Set it to ``True`` only when restoring the
lowest-energy recorded state is explicitly desired. Always inspect both the
energy history and the resulting geometry before changing this setting.

``diffeomorphic_start``
~~~~~~~~~~~~~~~~~~~~~~~

A positive value provides an affine-only warm-up before momentum updates.
Increase it when the initial global placement still needs mild affine
correction; use zero when global pre-alignment is already reliable. It is not
a substitute for correcting a wrong rotation, reflection or field-of-view
crop.

Validated S-LDDMM starting configurations
-----------------------------------------

These values reproduce the documented examples in their own coordinate
systems. They are starting profiles, not universal defaults.

.. list-table::
   :header-rows: 1
   :widths: 24 12 12 10 12 15 15

   * - Workflow
     - ``a``
     - ``grid_step``
     - ``nt``
     - ``niter``
     - ``lrM``
     - Coordinate frame
   * - MERFISH S2R3 → S2R2
     - 300
     - 100
     - 3
     - 500
     - 2,000
     - MERFISH spatial units
   * - Visium kidney IL3 → NL3
     - 500
     - 250
     - 5
     - 5,000
     - 50
     - coordinates multiplied by 50
   * - Xenium breast Rep2 → Rep1
     - 300
     - 100
     - 3
     - 500
     - 4,000
     - Xenium spatial units
   * - UI-curated MERFISH S2R1 → Allen CCF 675
     - 200
     - 50
     - 5
     - 500
     - 2,000
     - pre-aligned Allen physical canvas
   * - Xenium ST → H&E
     - 60
     - 6
     - 5
     - 300
     - 2,000
     - histology feature-grid pixels
   * - Spatial ATAC → MERFISH ST
     - 100
     - 50
     - 8
     - 500
     - 1,000
     - cropped raster canvas

Cross-sample structure discovery
--------------------------------

Tune clustering before alignment:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Parameter
     - Effect of increasing it
     - What to inspect
   * - ``num_neighbors``
     - Broader spatial-neighborhood summaries and smoother structures.
     - Thin domains should not disappear; isolated noise should decrease.
   * - ``banksy_lambda``
     - More weight on spatial-neighborhood information relative to each
       observation's own expression.
     - Expression-distinct regions should remain separated while spatial
       fragmentation decreases.
   * - ``resolution``
     - More Leiden clusters and finer structures.
     - Avoid both one broad cluster and many tiny sample-specific fragments.
   * - ``snn_neighbors``
     - Broader shared-nearest-neighbor connectivity.
     - Shared domains should occur in both samples without excessive merging.
   * - ``harmony_theta``
     - Stronger sample/batch correction.
     - Do not erase biological differences simply to maximize shared labels.

For a new dataset, first sweep a small grid of ``banksy_lambda`` and
``resolution`` while keeping the other parameters fixed. Select spatially
coherent structures with adequate coverage in both samples. Then inspect
``cluster_raw`` versus ``cluster_refined``. Cluster numbers are arbitrary;
compare maps and membership, not integer labels.

Cross-sample pre-alignment and rasterization
--------------------------------------------

- ``min_cluster_size`` removes tiny shared clusters from the centroid fit.
  Increase it when rare/noisy centroids pull the global transform; decrease it
  only when too few shared anchors remain.
- Disable ``allow_reflection`` unless section orientation or acquisition
  conventions make reflection biologically valid.
- ``grid_spacing`` controls the structural raster. Smaller values retain more
  detail and cost more memory; larger values smooth sparse spot data.
- ``blur_sigma`` is measured in raster pixels. Increase it for fragmented
  sparse counts and decrease it when adjacent thin domains merge.
- ``cluster_weight`` and ``density_weight`` set relative evidence. Density is
  helpful for cell-resolved tissues with informative sampling shape. Set or
  reduce ``density_weight`` when spot density is imposed by an approximately
  regular capture grid, as in the kidney example.

.. _cross_modality_pairing_weights:

Cross-modality pairing weights
------------------------------

For query structure :math:`i` and reference structure :math:`j`, spAlignDE
forms a composite score :math:`S_{ij}=\sum_m w_m C_{ij}^{(m)}` from
similarities in ``[0, 1]``. Active weights must be finite, non-negative and sum
to one. The public alignment configurations expose these weights so users can
adapt one global pairing rule to a new dataset. Do not assign different
weights to named anatomical regions.

The empirical paper profiles are:

.. list-table::
   :header-rows: 1
   :widths: 18 11 13 11 11 12 24

   * - Workflow
     - SDF
     - Chamfer
     - Dice
     - Area
     - Thickness
     - Acceptance QC
   * - Histology--ST
     - 0.20
     - 0.40
     - 0.15
     - 0.25
     - 0
     - score ≥ 0.40; ASD ≤ 30
   * - ATAC--ST
     - 0.35
     - 0.25
     - 0.10
     - 0.30
     - 0
     - final score ≥ 0.21; Dice ≥ 0.01
   * - Atlas--ST
     - 0.05
     - 0.05
     - 0.20
     - 0.50
     - 0.20
     - final score ≥ 0.50; ASD ≤ 50

ASD is not a weighted similarity in any of these composite scores. It is
reported in pairing-raster units and is used only as an independent absolute
boundary-distance QC threshold for histology--ST and atlas--ST. Chamfer already
provides the weighted bidirectional boundary-proximity component.

Tune the weights only after coordinate scale, field of view, global
pre-alignment and structure masks are credible:

1. Export the candidate-pair table and inspect overlays for both accepted and
   near-threshold rejected pairs. Identify a systematic failure mode rather
   than adjusting a single anatomy by name.
2. Increase Dice weight when broad, compact structures with reliable
   pre-alignment should be driven more strongly by common occupancy.
3. Increase SDF or Chamfer weight when elongated or thin structures have
   plausible boundary geometry but lose Dice after a small displacement. SDF
   gives broader signed-boundary agreement; Chamfer emphasizes nearest-boundary
   proximity.
4. Increase area weight when structures with clearly different spatial
   extents are being paired. Increase thickness weight when narrow atlas layers
   are confused with broader neighbors.
5. Change one component by a small amount, for example 0.05, redistribute the
   difference across the other active components, and renormalize to exactly
   one. Recompute candidate rankings and accepted masks before running
   S-LDDMM.
6. Adjust the minimum score or independent ASD/Dice QC gate only after the
   relative ranking is satisfactory. A higher score threshold or lower ASD
   threshold is stricter; the converse admits more candidates.

The H&E fields are ``pairing_weight_sdf``, ``pairing_weight_chamfer``,
``pairing_weight_dice``, ``pairing_weight_area`` and
``pairing_weight_thickness``. ATAC--ST uses
``sdf_weight``, ``chamfer_weight``, ``dice_weight`` and ``area_weight``.
Atlas--ST uses the corresponding ``pairing_weight_*`` fields, including
``pairing_weight_thickness``. If raster resolution changes, rescale the SDF
band, Chamfer decay/gate distances and ASD thresholds before interpreting a
weight sweep.

ST-to-Allen-CCF parameters
--------------------------

Tune this workflow in three groups.

Hierarchy
~~~~~~~~~

- ``n_levels`` adds intermediate coarse-to-fine ST stages. More levels can
  stabilize difficult thin structures but increase runtime and may propagate
  a poor coarse match. Three levels are used for the documented S2R1 run.
  Atlas candidates from hierarchy depths 2–10 remain eligible at every stage;
  only the ST resolution changes.
- ``minimum_coarse_structures`` controls the number of initial anchors. Too few
  provide weak anatomical support; too many make the first stage insufficiently
  coarse.
- ``variance_fraction`` and ``min_genes`` determine the expression evidence
  used to merge clusters. Confirm each hierarchy spatially rather than choosing
  levels only from expression dendrograms.

Whole-tissue pre-alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``prealign_angle_step_degrees`` gives angular search resolution. Smaller
  steps are slower and only useful after the correct broad orientation is in
  the search.
- ``prealign_scale_tweak`` and ``prealign_scale_steps`` widen/refine the scale
  search. Increase the range when tissue sizes remain mismatched, not when a
  local anatomical region is wrong.
- Confirm the atlas slice and orientation first. A neighboring slice can have
  good whole-mask IoU but incorrect internal anatomy.

Structure pairing
~~~~~~~~~~~~~~~~~

``STAtlasAlignmentConfig`` exposes hierarchy, pre-alignment, filtering,
pairing and continuation controls. The coarse-to-fine stages use a validated
schedule of ``stage_iterations=(100, 500, 100)`` with
``restore_best_checkpoint=False``; passing a cross-sample ``SLDDMMConfig`` does
not change Atlas alignment. The documented continuation uses
``continuation_kernel_scale=200``,
``continuation_velocity_grid_spacing=50`` and
``continuation_restore_best_checkpoint=False``. Tune structure stages and
pairs before changing these numerical controls.

The Atlas weights must be non-negative and sum to one. Start from the empirical
profile in :ref:`cross_modality_pairing_weights`; ``area`` and ``thickness``
distinguish narrow laminar structures from broader neighbors, whereas Dice and
SDF/Chamfer measure spatial agreement.

- Increase ``pairing_score_threshold`` for fewer, higher-confidence pairs;
  decrease it cautiously only after reviewing rejected candidate metrics.
- Decrease ``pairing_max_asd`` for a stricter boundary-distance gate.
- If a thin structure maps to a broader neighbor, first inspect mask quality,
  hierarchy level, and pre-alignment. Then compare area/thickness metrics and,
  if the failure is systematic across structures, adjust the global normalized
  weights.
- Use the interactive pairing/refinement notebook to document expert-curated
  correspondences when geometry alone remains ambiguous.

UI-curated Atlas pairing
~~~~~~~~~~~~~~~~~~~~~~~~

The CSV exported by the interactive tool is a correspondence specification,
not a cached transformation. ``align_st_to_allen_atlas_from_ui_pairs``
accepts those pairs directly. It skips automatic candidate discovery,
pair-score/gate calculation, non-overlap selection and subsequent pair
rematching. It does not skip the preparation required to turn the accepted
pairs into an alignment.

- Use the exact clustered AnnData shown in the UI. The selected ST IDs are
  categorical labels; rerunning clustering can renumber them even when the
  spatial structures look similar.
- Keep the export unedited. ``group_id`` defines one deformation channel; all
  ST IDs and Allen IDs in that group are unioned before signed-distance
  construction. Multiple rows in one group do not create extra weights.
- ``prealignment_mode="mask"`` computes whole-mask initialization. Use
  ``"provided"`` after manual pre-alignment has written finite
  ``x_prealigned``/``y_prealigned`` columns; this bypasses only transform
  estimation.
- Confirm ``atlas_z_slice`` matches the loaded reference. Display flip and
  rotation fields are provenance from the UI view, not a substitute for the
  package's whole-mask pre-alignment.
- Tune source/target mask smoothing and minimum areas before deformation
  settings when thin structures disappear. ``area_weight_power`` and the
  min/max channel weights balance groups globally; never add a named-region
  exception.
- In both pre-alignment modes, point filtering, ST/Atlas mask construction,
  mask processing, signed-distance and channel-weight construction, S-LDDMM
  input/grid building, optimization and point mapping still run.
- The validated S2R1 example uses ``kernel_scale=200``,
  ``velocity_grid_spacing=50``, ``time_steps=5`` and ``iterations=500``.
  ``restore_best_checkpoint=False`` reproduces its final-iteration paper
  result. Set it to ``True`` for exploratory runs when returning the
  lowest-energy checkpoint is preferable to matching that validated result.

ST-to-histology parameters
--------------------------

Feature extraction
~~~~~~~~~~~~~~~~~~

- ``target_microns_per_pixel`` establishes the physical analysis resolution.
  Supply ``source_microns_per_pixel`` when the image lacks reliable metadata.
  An incorrect value changes tissue size and invalidates downstream distance
  parameters.
- ``shifted_tiles=True`` averages 16 shifted HIPT views and reduces tile seams
  at substantially greater feature-extraction cost. Disable it only for rapid
  exploratory runs, then re-enable it for final results.
- Keep the 224-pixel padding multiple and validated HIPT checkpoints unless a
  new feature backend is being explicitly benchmarked.

Image-feature clustering
~~~~~~~~~~~~~~~~~~~~~~~~

- ``image_clusters`` controls the initial tissue partition. Increase it when
  distinct morphology is merged; decrease it when regions fragment into many
  visually indistinguishable pieces.
- ``rgb_weight`` increases the influence of color/stain; ``coordinate_weight``
  encourages spatial compactness. Excessive coordinate weight can split one
  morphology by location, while excessive RGB weight can follow staining
  artifacts.
- ``cleanup_min_size`` removes small disconnected islands in feature-grid
  pixels. Scale it if image resolution or tissue size changes.
- Symmetry merging is appropriate for approximately bilateral sections.
  Disable or relax it for asymmetric anatomy, partial sections or pathology.

Pre-alignment, pairing and S-LDDMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Accept automatic mask-overlap pre-alignment only after inspecting the overlay.
  IoU below ``minimum_recommended_iou`` is a warning to use interactive manual
  pre-alignment, not a reason to immediately loosen pairing thresholds.
- The empirical H&E pairing weights are SDF 0.20, Chamfer 0.40, Dice 0.15,
  area 0.25 and thickness 0.00. Change the five ``pairing_weight_*`` fields
  globally and keep their sum equal to one; see
  :ref:`cross_modality_pairing_weights`.
- Raising ``pair_score_threshold`` or lowering ``pair_asd_threshold`` makes
  structure acceptance stricter. Tune them from the candidate table, not only
  from the number of accepted pairs.
- ``global_shape_weight`` controls the additional whole-tissue channel. Raise
  it when local pairs pull the tissue outline apart; lower it when outline
  differences are genuine and prevent reproducible internal alignment.
- ``zoom_scale`` downsamples the S-LDDMM image field. Smaller values reduce
  memory and detail. The sampled coordinate axes still span the original
  feature-grid extent, so ``kernel_scale`` and ``velocity_grid_spacing`` remain
  in original feature-grid units; do not multiply them by ``zoom_scale``.

Spatial ATAC-to-ST parameters
-----------------------------

The ATAC query may cover only part of the ST reference, so tune the field of
view before matching.

- Adjust the two manual similarity transforms, then
  ``reference_crop_axis``, ``reference_crop_side`` and
  ``reference_crop_quantile`` until the same anatomical field is present.
- ``raster_scale`` sets the canvas resolution. Increasing it preserves more
  detail and increases memory; translations and distance-based settings must be
  interpreted in the resulting canvas units.
- ``density_neighbors`` and ``density_mad_multiplier`` remove sparse coordinate
  outliers before mask construction. A smaller MAD multiplier is stricter.
- Pairing weights for SDF, Chamfer, area and Dice describe one global rule.
  ``pair_score_threshold`` and ``pair_dice_threshold`` control acceptance;
  inspect rejected and accepted masks before loosening either.
- ``channel_area_power`` increases inverse-area balancing as it approaches one,
  giving narrow masks more influence. ``channel_uniform_mix`` pulls all channel
  weights toward equality. These are global channel rules, not anatomical
  special weights.

Post-alignment inference parameters
-----------------------------------

Inference tuning begins only after the alignment passes geometric QC.

- ``risk_genes`` should be a broad, prespecified set suitable for estimating
  residual mismatch; do not use only the genes being tested.
- ``min_detected_spots`` and ``min_total_counts`` filter the candidate risk-gene
  pool. Increase them when extremely sparse genes make risk unstable.
- ``density_energy_share`` controls how much local sampling-density discordance
  contributes to the standardized mismatch-feature energy and must lie
  strictly between zero and one. It is not a direct weight on the final risk
  map or variance multiplier. Increase it when density mismatch is known to be
  a major reliability signal; do not use it to conceal poor registration.
- Leave ``grid_n=None`` to retain the R-driven resolution when its actual
  tissue-valid location count lies between the median per-sample observation
  count, ``N_typ``, and ``2 * N_typ``; otherwise the automatic rule moves the
  resolution toward the nearest boundary. An explicit integer is the number of
  Cartesian points per axis. Larger values increase spatial resolution,
  runtime and memory. Report ``grid_n``, ``grid_n_source``,
  ``target_grid_locations`` and ``shared_grid_locations`` from
  ``prepared.metadata``.
- Enable ``cell_type_adjustment`` only when complete, validated cell-type labels
  are available in every sample. This support term is distinct from alignment
  mismatch calibration.
- ``region_cleanup=False`` reports connected components of the direct
  ``q < 0.05`` grid mask. Enabling cleanup removes isolated or unsupported
  fragments from the reported mask without changing statistics, P values or
  q-values. Report the choice when interpreting connected regions.
- Use ``random_state=WORKFLOW_SEED`` for both preparation and fitting. The
  public kidney and aging-brain notebooks also use ``n_jobs=1`` so their saved
  diagnostic streams and numerical outputs are reproducible exactly. Larger
  ``n_jobs`` values can shorten multi-query runs, but worker log order is not a
  scientific result and may differ between executions.

The gene-specific calibration has no public tuning knobs. It median-centers
first-pass statistics within normalized-risk bins, scales each MAD by the
Student-t null MAD, applies a monotone nonnegative excess-variance constraint,
and fits a through-origin quadratic with a bounded anchor near 80% risk. The
resulting factor is :math:`1+\lambda_{\mathrm{local},g}r_i^2`; the
gene-specific global coefficient is fixed at zero.

Failure-oriented checklist
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 31 34 35

   * - Observation
     - Check first
     - Parameter group to consider next
   * - Whole tissues are rotated or displaced
     - Query/reference roles, orientation, crop, global overlay
     - Pre-alignment scale/angle/translation; not S-LDDMM ``a``
   * - No structure pair is accepted
     - Mask overlap and candidate-score table
     - Mask/raster settings, then pairing thresholds
   * - Thin regions match broad neighbors
     - Hierarchy and mask thickness/area metrics
     - Global area/thickness balance or finer hierarchy
   * - Alignment is smooth but misses local bends
     - Whether pairs support the local bend
     - Smaller ``kernel_scale`` and possibly smaller ``grid_step``
   * - Alignment is wavy or follows mask noise
     - Fragmented masks and over-clustering
     - Larger ``kernel_scale`` and/or larger ``grid_step``
   * - Energy oscillates or becomes non-finite
     - Coordinate scale and pre-alignment
     - Smaller learning rates, clipping/rollback, coarser velocity grid
   * - GPU memory is excessive
     - Structural image and velocity-grid dimensions
     - Larger raster spacing, larger ``grid_step`` or smaller image zoom scale
   * - More iterations do not improve geometry
     - Energy plateau and incorrect pairs
     - Fix inputs/pairs; do not increase ``niter`` blindly

For every new dataset, retain the configuration, pre-alignment overlay,
structure/mask panels, accepted-pair table, final overlay and energy history.
These artifacts are the minimum evidence required to interpret an aligned
coordinate output.
