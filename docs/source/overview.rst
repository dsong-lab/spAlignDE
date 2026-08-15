Overview
========

spAlignDE is an integrated framework that connects **structure-guided spatial
alignment** with **mismatch-aware post-alignment local differential-expression
analysis**. It first places spatial datasets in a common coordinate system and
then uses those aligned coordinates for location-resolved comparisons while
accounting for residual correspondence errors.

The framework has two linked components:

1. alignment across spatial-omics samples and measurement modalities; and
2. post-alignment inference for aligned spatial transcriptomics samples.

The alignment component represents tissue organization as spatial structures
and estimates a smooth transformation with shooting-based large deformation
diffeomorphic metric mapping (S-LDDMM). The inference component constructs a
shared testing grid and reduces the precision of local comparisons when the
aligned neighborhoods remain poorly matched.

Why spAlignDE
-------------

Spatial datasets are rarely measured in the same coordinate system. Tissue
sections can differ in orientation, scale, sectioning plane, local deformation,
capture area and sampling density. Cross-modality comparisons introduce an
additional challenge because spatial transcriptomics, spatial ATAC-seq,
histology and anatomical atlases do not necessarily share a molecular feature
space.

spAlignDE addresses these challenges by aligning **tissue structures** rather
than requiring direct correspondence between individual observations or
features. This design provides three practical advantages:

- a common alignment strategy for samples and modalities with different feature
  spaces;
- a smooth, topology-preserving transformation that is applied to all original
  cells, spots or image positions; and
- downstream inference that does not treat the estimated alignment as
  error-free.

Alignment workflow
------------------

The query dataset is treated as the moving object and is aligned to a fixed
reference through five stages.

1. **Discover spatial structures.** For cross-sample spatial transcriptomics,
   joint clustering identifies shared spatial domains from expression and
   spatial-neighborhood information. For cross-modality analysis, structures
   are extracted independently from each modality.
2. **Estimate a global pre-alignment.** A similarity transformation brings the
   query and reference into approximate agreement using shared centroids,
   tissue masks or modality-specific spatial structures.
3. **Establish structure correspondences.** Shared cluster identities guide
   cross-sample matching, whereas geometric and relative-position agreement
   guide cross-modality pairing. Expert-guided pairs can be supplied when
   anatomical prior knowledge is available.
4. **Construct continuous structural fields.** Matched structures are converted
   into multichannel image-like representations, including composition,
   density or signed-distance-transform fields.
5. **Refine with S-LDDMM.** Shooting-based LDDMM estimates a smooth,
   affine--diffeomorphic query-to-reference transformation and maps the original
   observations into the reference coordinate frame.

The same registration principle is adapted to different analysis settings:

.. list-table::
   :header-rows: 1
   :widths: 23 31 46

   * - Setting
     - Structural representation
     - Alignment target
   * - Cross-sample ST
     - Jointly inferred spatial domains
     - Another cell- or spot-resolved tissue section
   * - ST to histology
     - Transcriptomic domains and image-derived morphological regions
     - H&E- or Nissl-stained tissue image
   * - ST to atlas
     - Transcriptomic domains and atlas anatomical compartments
     - Anatomical reference coordinate system
   * - Spatial ATAC to ST
     - Modality-specific epigenomic and transcriptomic domains
     - Spatial transcriptomics reference section

For repeated or multi-sample analyses, subsampling-based uncertainty
quantification can additionally summarize the positional stability of the
estimated transformation.

mismatch-aware post-alignment local inference
---------------------------------------------

For aligned transcriptomic samples, spAlignDE defines fixed locations on a
shared spatial grid and constructs kernel-weighted neighborhoods around each
location. These neighborhoods provide local expression estimates and
query-versus-reference contrasts.

Residual mismatch is spatially heterogeneous, so spAlignDE estimates the local
reliability of each comparison using:

- disagreement among prespecified stable genes;
- local sampling-density discordance;
- optional inconsistency in cell-type or anatomical-compartment support; and
- optional technical covariates such as local library size and detection rate.

The mismatch-aware model inflates local variance in poorly supported regions
without changing the estimated expression contrast. Those comparisons
therefore contribute less precise evidence instead of being interpreted as
confident biological differential expression. This adjustment complements
accurate registration; it is not a substitute for it.

Calibration is gene-specific and local-only. First-pass statistics are
median-centered within normalized-risk bins and scaled against the Student-t
null MAD. A monotone nonnegative excess-variance curve is fitted as a
through-origin quadratic, giving the final factor
:math:`1+\lambda_{\mathrm{local},g}r_i^2`. The gene-specific global coefficient
is fixed at zero, so a location with zero local risk receives no mismatch
inflation. Cell-type support, when available, remains a separate precision
adjustment.

The inference workflow returns local test statistics and raw P values,
false-discovery-rate-adjusted q-values, connected local DE regions, gene-level
ACAT summaries and inputs for spatial trajectory analysis across ordered
samples. ACAT combines the retained raw local P values; q-values are used only
for grid-level FDR masks.

Applications
------------

The documented workflows cover the main settings evaluated in the manuscript:

- cross-sample MERFISH mouse-brain alignment;
- spot-level alignment of normal and injured 10x Visium kidney sections;
- multi-sample alignment of 20 aging mouse-brain sections;
- Xenium breast-cancer technical-replicate alignment;
- alignment of spatial transcriptomics to histology, the Allen Common
  Coordinate Framework and spatial ATAC-seq; and
- mismatch-aware local inference for spatial expression trajectories,
  cell-neighborhood remodeling and compartment-specific programs.

Where to go next
----------------

- :doc:`Tutorials <tutorial>` organizes the workflows by analysis task.
- :doc:`Cross-Sample Alignment <tutorials/cross_sample_alignment>` introduces
  alignment between spatial transcriptomics sections.
- :doc:`Cross-Modality Alignment <tutorials/cross_modality_alignment>`
  introduces histology-, atlas- and spatial-ATAC-guided registration.
- :doc:`Parameter Tuning Guide <tutorials/parameter_tuning>` explains the
  deformation, rasterization, clustering and cross-modality matching settings.
- :doc:`Post-Alignment Inference <tutorials/post_alignment_inference>` describes
  shared-grid local testing and mismatch-aware adjustment.
- :doc:`Source Notebooks <source_notebooks>` provides executable workflow
  examples.
- :doc:`Python API <api>` lists the currently exposed package interfaces.
