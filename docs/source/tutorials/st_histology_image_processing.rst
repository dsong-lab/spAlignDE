ST-to-Histology Alignment — Mouse Brain Xenium to H&E
======================================================

This workflow aligns a spatial-transcriptomic query to a histology image by
matching spatial structures rather than molecular features. The example maps
162,033 cells from Replicate 1 of the 10x Genomics Fresh Frozen Mouse Brain
Xenium dataset to the H&E image from control Replicate 1 of the Visium
CytAssist Post-Xenium Mouse Brain dataset.

The reference-side alignment input is only the high-resolution image. The
Visium count matrix and spot coordinates paired with that H&E section are not
used to estimate the transformation; in the paper they provide an independent
post-alignment molecular-concordance assessment.

The validated workflow uses seed ``0`` for feature extraction, K-means and
automatic pre-alignment. Call
``spAlignDE.set_random_seed(0, deterministic_torch=True)`` before the first
stage. The manuscript-grade S-LDDMM run uses float64: image structures, masks
and accepted pairs reproduce exactly, while CUDA coordinates are compared
with a declared numerical tolerance.

Data
----

Download the query and reference from:

* `Fresh Frozen Mouse Brain Replicates — Xenium
  <https://www.10xgenomics.com/datasets/fresh-frozen-mouse-brain-replicates-1-standard>`_;
* `Visium CytAssist Gene Expression Libraries of Post-Xenium Mouse Brain (FF)
  <https://www.10xgenomics.com/datasets/visium-cytassist-gene-expression-libraries-of-post-xenium-mouse-brain-ff-using-the-mouse-whole-transcriptome-probe-set-2-standard>`_.

Query AnnData
~~~~~~~~~~~~~

Run :doc:`Single Clustering (CSV or AnnData Input) <clustering_single>` first.
The alignment input requires:

.. list-table::
   :header-rows: 1

   * - Location
     - Required content
   * - ``adata.X``
     - Cell-by-gene expression matrix
   * - ``adata.obsm["spatial"]``
     - Finite ``n_obs × 2`` coordinates ordered x, y
   * - ``adata.obs["cluster_raw"]``
     - Original single-sample BANKSY labels used to build the paper hierarchy
   * - ``adata.obs["cluster"]``
     - Selected labels retained for display and provenance

Histology image
~~~~~~~~~~~~~~~

Provide one RGB or RGB-convertible JPEG, PNG, TIFF, OME-TIFF, or BTF image.
OME-TIFF/BTF physical pixel size is read automatically when present and the
image is rescaled to 0.5 μm per pixel. For an image without physical metadata,
spAlignDE preserves its current resolution unless
``source_microns_per_pixel`` is supplied explicitly. The image is padded with
white pixels to dimensions divisible by 224.

The downstream feature grid has origin at the upper left. Coordinates are
ordered x, y, the y-axis increases downward, and one feature-grid unit
represents a 16 × 16-pixel region of the prepared image.

Workflow
--------

1. Image preparation and HIPT feature extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`H&E image preparation and feature extraction
</source_notebooks/cross_modality/st_he_feature_extraction_nb>`

The package entry point is ``spAlignDE.extract_histology_features``. spAlignDE
implements the dense stitching, shifted-view averaging, smoothing, and output
serialization. The hierarchical model and original checkpoints come directly
from `MahmoodLab/HIPT <https://github.com/mahmoodlab/HIPT>`_. Clone that
official repository with Git LFS and set ``SPALIGNDE_HIPT_DIR`` to the clone.
The two required files are
``HIPT_4K/Checkpoints/vit256_small_dino.pth`` and
``HIPT_4K/Checkpoints/vit4k_xs_dino.pth``.

HIPT must run with a mutually compatible PyTorch/torchvision build. By default
it uses the notebook kernel's Python. If HIPT is installed in a separate
validated environment, set
``SPALIGNDE_HIPT_PYTHON=/path/to/hipt-environment/bin/python``; the notebook
still controls the seed and records the extractor route and checkpoint hashes.

HIPT divides the prepared image into 4,096-pixel context tiles, encodes local
256-pixel patches on a 16-pixel feature stride, and repeats inference at four
offsets per axis to average 16 shifted views. It writes:

.. list-table::
   :header-rows: 1

   * - Output
     - Content
   * - ``he.jpg``
     - RGB analysis image after physical-resolution handling and padding
   * - ``embeddings-hist-vit.pickle``
     - Dictionary containing 192 ``cls`` context maps, 384 ``sub`` local maps,
       and three ``rgb`` maps
   * - ``histology_image_preparation.json``
     - Source image, pixel scale, resizing, padding, grid, and image checksums
   * - ``histology_feature_manifest.json``
     - Extraction geometry, feature schema, runtime route, file size, and
       feature checksum

The fixed-seed vendor-BTF run produces a 1,288 × 798 feature grid. The BTF's
0.27381 μm-per-pixel metadata are used to rescale to 0.5 μm per pixel before
padding to a 12,768 × 20,608 analysis image. Every feature location
corresponds to one 16 × 16-pixel image region, and no expression matrix or
spatial-transcriptomic coordinates are stored in the feature pickle.

.. figure:: ../_static/tutorial_figures/histology_prepared_image.png
   :alt: Prepared H and E image supplied to HIPT feature extraction
   :width: 48%
   :align: center

   Prepared H&E image after physical-resolution handling and padding. This is
   the exact RGB image supplied to HIPT; the manifest records every resize and
   padding operation needed to map feature-grid coordinates back to pixels.

`UNI <https://github.com/mahmoodlab/UNI>`_ and its `gated Hugging Face weights
<https://huggingface.co/MahmoodLab/UNI>`_ provide an optional fine-grained
pathology representation. The canonical spAlignDE H&E result here uses
HIPT/ViT only; UNI is not required to reproduce the reported 26 image
structures.

2. Image-feature clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`H&E image-feature clustering into 26 structures
</source_notebooks/cross_modality/st_he_feature_clustering_nb>`

``spAlignDE.cluster_histology_features`` separates tissue from slide
background, clusters standardized HIPT, RGB, and coordinate blocks into 30
initial tissue regions, fills holes, performs feature- and symmetry-aware
merging, and removes small disconnected islands. The validated
``test0730/he_rep1`` configuration uses ``rgb_weight=0.25``,
``coordinate_weight=0.05``, ``random_state=0``, and produces 26 cleaned image
structures. These image labels do not represent cell types or atlas regions.

The complete image-only construction is:

1. separate tissue from slide background with a two-group brightness-supported
   mask and retain the main tissue support;
2. standardize the HIPT context, local, and RGB feature blocks independently;
3. concatenate HIPT features, RGB weighted by ``0.25``, and normalized x/y
   coordinates weighted by ``0.05``;
4. run deterministic K-means to obtain 30 initial tissue regions;
5. fill internal holes and merge regions supported by feature-centroid
   similarity;
6. accept at most two layout-aware reflection merges when overlap, feature
   cosine similarity, centroid distance, and symmetry-score gain pass their
   gates; and
7. remove disconnected islands smaller than 250 feature-grid pixels, keeping
   background encoded as −1.

Image orientation and symmetry axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prepared benchmark image contains **two coronal sections stacked
vertically** and is displayed exactly as stored with an upper-left image
origin. The paper setting ``symmetry_axis="ud"`` therefore compares
corresponding features between the upper and lower sections. It does not assume
that one section is anatomically symmetric along its dorsal--ventral axis.

For a single upright bilateral section with the two hemispheres arranged
left-to-right, use ``symmetry_axis="lr"``. Select the axis from the prepared
``he.jpg`` after all resizing, rotation, and cropping—not from the scanner's
original orientation. Partial or strongly asymmetric tissue requires direct
review of reflected-mask candidates before any symmetry-driven merge.

.. code-block:: python

   cluster_config = spAlignDE.HistologyClusteringConfig(
       background_clusters=2,
       image_clusters=30,
       merged_clusters=30,
       rgb_weight=0.25,
       coordinate_weight=0.05,
       cleanup_min_size=250,
       symmetry_axis="ud",  # benchmark: two sections stacked vertically
       symmetry_max_merges=2,
       random_state=0,
   )

   histology = spAlignDE.cluster_histology_features(
       features,
       "tutorials/cross_modality/histology/output/histology_clustering",
       config=cluster_config,
   )

.. figure:: ../_static/tutorial_figures/histology_feature_clustering.png
   :alt: Histology feature clustering and cleanup stages
   :width: 82%
   :align: center

   Image-to-structure quality-control sequence: prepared H&E, initial
   HIPT/RGB/coordinate feature clusters, symmetry-aware merging, and the final
   26 cleaned structures. The vertical arrangement is intentional for this
   two-section benchmark. Compare every stage with the prepared image before
   using the labels as alignment features.

3. ST structures, pre-alignment, pairing and S-LDDMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`Xenium Replicate 1 to H&E histology
</source_notebooks/cross_modality/st_he_alignment_nb>`

``spAlignDE.build_st_histology_structures`` reproduces the paper hierarchy
from the original BANKSY partition. With five levels and 75% retained gene
variance, this example produces k7, k14, k20, and k26 partitions and selects
k7 for histology correspondence.

.. figure:: ../_static/tutorial_figures/histology_st_hierarchy.png
   :alt: Coarse-to-fine Xenium spatial structure hierarchy
   :width: 100%
   :align: center

   Original BANKSY labels and expression-derived coarse-to-fine partitions.
   The k7 level is selected for H&E correspondence; the remaining panels show
   how anatomical detail changes with hierarchy resolution. The raw Xenium
   hierarchy is displayed with a lower-origin y axis so the dorsal cortex and
   hippocampal structures match the validated paper orientation; stored
   coordinates are unchanged.

Global pre-alignment uses whole-tissue mask overlap by default. It estimates
rotation, isotropic scale, and translation while disabling reflection. The
overlay must be inspected before deformation. Mask overlap can be unreliable
when the samples differ strongly in coverage, cropping, tears, or orientation;
in that case use ``spAlignDE.interactive_histology_prealignment`` to adjust
scale, rotation, and translation. The selected manual values are saved in the
AnnData metadata. The recorded paper initialization is scale 0.12, rotation
85°, x translation 787.84018089, and y translation −9.30873160 in the
feature-grid coordinate frame.

.. figure:: ../_static/tutorial_figures/histology_prealignment.png
   :alt: Xenium-to-histology global pre-alignment
   :width: 84%
   :align: center

   Tissue-mask initialization before local deformation. The blue Xenium
   outline should cover the same H&E anatomy; if it does not, use the
   interactive manual controls before changing structure-pair thresholds.

After pre-alignment, ST and H&E structures are converted to binary masks on a
shared grid. Candidate correspondences are scored using signed-distance,
Chamfer, Dice, area and thickness similarities with empirical weights 0.20,
0.40, 0.15, 0.25 and 0.00, respectively. The weights are configurable through
``STHistologyAlignmentConfig`` and must be non-negative and sum to one. ASD is
not part of the composite score; its raw value is retained only as an
independent QC measure. Pairs must have score at least 0.40 and ASD at most 30
pixels. Accepted pairs become continuous signed-distance channels for S-LDDMM.

Paired-feature overlap quality control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inspect the exact binary masks selected as S-LDDMM channels before interpreting
the deformation. The individual ST and H&E masks reveal whether a high score
comes from the intended morphology; the overlay separates ST-only area, H&E-only
area, and their intersection. Dice measures overlap, whereas ASD remains
sensitive to boundary displacement even for thin structures.

.. figure:: ../_static/tutorial_figures/histology_paired_feature_overlap.png
   :alt: Accepted Xenium and H&E feature masks with paired overlaps
   :width: 100%
   :align: center

   The two accepted k7/H&E feature pairs used by the documented S-LDDMM run.
   Blue is the pre-aligned ST mask, orange is the H&E structure mask, and
   purple is their intersection. Scores beneath each pair are calculated
   before deformation and provide direct QC of the selected alignment evidence.

.. figure:: ../_static/tutorial_figures/histology_alignment_before_after.png
   :alt: Xenium coordinates before and after histology-guided S-LDDMM
   :width: 100%
   :align: center

   Xenium cells overlaid on the reference H&E after global pre-alignment
   (left) and after structure-guided S-LDDMM (right). Review internal
   anatomical structures as well as the outer tissue contour.

Parameter adaptation
--------------------

Physical image resolution is the first parameter: use embedded metadata or set
``source_microns_per_pixel`` explicitly before extracting features.
``shifted_tiles=True`` reduces HIPT tile seams but costs substantially more
runtime. During clustering, ``image_clusters`` controls initial granularity,
``rgb_weight`` controls stain/color influence, ``coordinate_weight`` promotes
spatial compactness and ``cleanup_min_size`` removes small islands in feature
pixels.

For alignment, inspect the mask-overlap pre-alignment before changing pair
thresholds. A low-IoU or anatomically incorrect overlay should be corrected
with the interactive manual tool. ``pairing_weight_sdf``,
``pairing_weight_chamfer``, ``pairing_weight_dice``, ``pairing_weight_area``
and ``pairing_weight_thickness`` control the composite score; their paper
defaults are 0.20, 0.40, 0.15, 0.25 and 0.00. ``pair_score_threshold`` and
``pair_asd_threshold`` independently control accepted correspondences;
``global_shape_weight`` balances the whole-tissue outline against matched
internal structures. The histology S-LDDMM values ``a=50`` and
``grid_step=6`` are expressed in original feature-grid units. ``zoom_scale``
changes image sampling density but retains axes spanning that original grid;
do not rescale ``a`` or ``grid_step`` by ``zoom_scale``. See the
:ref:`cross-modality pairing-weight guide <cross_modality_pairing_weights>`
and :doc:`Parameter Tuning Guide <parameter_tuning>` before
transferring them to another image resolution.

Outputs
-------

The query AnnData preserves ``X``, observation order, metadata, and
``obsm["spatial"]``. The workflow adds:

.. list-table::
   :header-rows: 1

   * - Column
     - Meaning
   * - ``x_prealigned``, ``y_prealigned``
     - Query coordinates after global similarity pre-alignment
   * - ``x_aligned``, ``y_aligned``
     - Query coordinates after structure-guided S-LDDMM

Parameters, selected structure level, pre-alignment values, grid shape, and
pair count are stored under ``adata.uns["spAlignDE"]``. The alignment notebook
also writes the final H5AD, matched-pair table, filtering summary, JSON
manifest, and PNG plus editable SVG quality-control figures.

Troubleshooting
---------------

- If feature extraction cannot import torchvision or reports a missing
  ``torchvision::nms`` operator, PyTorch and torchvision are binary
  incompatible. Run ``tools/check_notebook_environment.py``, verify both
  checkpoint hashes, and point ``SPALIGNDE_HIPT_PYTHON`` to a compatible HIPT
  environment; a working ``import torch`` alone is insufficient.
- If tissue size is wrong, correct microns-per-pixel metadata and rerun feature
  extraction. Do not compensate later with ``a`` or manual scale.
- If feature clusters follow stain noise or image position, reduce
  ``rgb_weight`` or ``coordinate_weight`` respectively and review all
  clustering-stage panels.
- If whole masks overlap poorly, use interactive manual pre-alignment. If masks
  overlap but no pair passes, inspect candidate masks before changing score or
  ASD gates.
- If GPU memory is excessive, reduce ``zoom_scale`` or use a coarser velocity
  grid, while retaining enough resolution for the narrowest supported region.

Validation and interpretation
-----------------------------

Two August 11 fresh-process runs from the same vendor-BTF checksum reproduced
the H&E HIPT feature pickle and all four clustering-stage arrays exactly. The
fixed result contains 26 regions and two accepted ST/H&E pairs. CUDA S-LDDMM may still show
small floating-point differences across GPUs and PyTorch versions; anatomical
overlap and accepted structure pairs, rather than bitwise coordinate identity,
are the relevant checks.

The before/after overlay establishes geometric plausibility, not biological
ground truth. The paper's comparison against Visium expression is independent
because those measurements were withheld from transformation estimation.
