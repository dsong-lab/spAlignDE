#!/usr/bin/env python3
"""Insert task-specific parameter guidance into public source notebooks.

The operation is idempotent and preserves all existing code outputs. Run this
after a notebook builder regenerates canonical notebooks and before copying
them into the Sphinx source tree.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "source_notebooks"
SPHINX_NOTEBOOK_ROOT = PROJECT_ROOT / "docs" / "source" / "source_notebooks"
ROLE_PREFIX = "spAlignDE_parameter_guidance:"


CLUSTERING_GUIDANCE = r"""
### How to adapt the clustering parameters

Tune clustering before changing alignment parameters. Keep the random seed and
all other settings fixed while comparing a small grid of candidate values.

- `num_neighbors`: increasing it summarizes a broader spatial neighborhood and
  usually smooths structures; reduce it if thin domains disappear.
- `banksy_lambda`: increasing it gives spatial-neighborhood information more
  influence relative to each observation's own expression.
- `resolution`: increasing it usually produces more Leiden clusters. Reject
  settings that create many tiny sample-specific fragments or merge clearly
  distinct spatial domains.
- `snn_neighbors` and `harmony_theta` (joint clustering only): the former
  broadens graph connectivity and the latter strengthens sample correction.
  Do not erase biological differences merely to maximize shared labels.
- `refine_boundaries`: compare `cluster_raw` and `cluster_refined` spatially.
  Refinement should remove isolated boundary noise without erasing narrow
  structures.

Select a configuration using spatial coherence, boundary preservation, cluster
size and coverage in every required sample. Cluster integer values are
arbitrary and may be renumbered across runs; compare memberships and maps, not
label numbers. The website **Parameter Tuning Guide** gives the complete
clustering-to-alignment tuning order.
"""


CROSS_SAMPLE_GUIDANCE = r"""
### How to adapt rasterization and S-LDDMM parameters

First confirm that the global pre-alignment is anatomically plausible and that
the rasterized shared structures overlap. S-LDDMM should refine an already
credible initialization, not repair a wrong rotation, scale, reflection or
cluster correspondence.

- `grid_spacing` is the structural-image resolution. Smaller values preserve
  detail but increase memory and sensitivity to sparse sampling.
- `blur_sigma` is measured in raster pixels. Increase it for fragmented sparse
  fields; decrease it if adjacent thin structures merge.
- `density_weight` is useful when sampling density describes tissue shape. It
  can be reduced or set to zero for approximately regular spot grids.
- `kernel_scale` is legacy `a`. Increasing it makes deformation smoother and
  more global; decreasing it permits more local bending and raises overfitting
  risk.
- `velocity_grid_spacing` is legacy `grid_step`. Decreasing it creates a denser
  velocity field with more memory/runtime and local flexibility; increasing it
  produces a coarser, more regularized field. Tune it together with `a`.
- `time_steps` (`nt`) controls integration accuracy rather than deformation
  amplitude. `iterations` (`niter`) controls optimization duration.
- Increase `momentum_lr` (`lrM`) only when energy decreases stably but too
  slowly; reduce it if energy oscillates, becomes non-finite or the warp is
  implausibly local.

All distance-valued settings use the coordinates passed to this stage. Do not
copy `a`, `grid_step`, raster spacing or translations between differently
scaled datasets without rescaling and rechecking the velocity-grid dimensions.
"""


ATLAS_GUIDANCE = r"""
### How to adapt the Atlas configuration

Tune this workflow in order: atlas slice/orientation, ST clustering, hierarchy,
whole-tissue pre-alignment, structure masks and pairs, then deformation.

- `n_levels` adds coarse-to-fine stages. More levels can stabilize narrow
  structures but increase runtime and can propagate a poor coarse match.
- `minimum_coarse_structures` controls initial anatomical anchors. Too few are
  weak; too many make the first stage insufficiently coarse.
- `variance_fraction` and `min_genes` control the expression evidence used to
  merge fine clusters. Inspect every hierarchy level spatially.
- A smaller `prealign_angle_step_degrees` refines angular search at higher
  cost. Expand the scale search only when whole-tissue size remains wrong.
- Pairing weights must be non-negative and sum to one. Area and thickness help
  distinguish narrow laminar structures from broad neighbors; Dice, SDF and
  Chamfer measure spatial agreement. Change weights globally, never for a
  named anatomical structure.
- The empirical Atlas profile is SDF 0.08, Chamfer 0.06, Dice 0.18, area 0.47,
  and thickness 0.21. Raw ASD is an independent QC gate, not a weighted score
  component. See the website's **Cross-modality pairing weights** section for
  a component-by-component tuning sequence.
- Raising `pairing_score_threshold` or lowering `pairing_max_asd` makes pair
  acceptance stricter. Inspect the candidate component metrics before changing
  either gate.

`STAtlasAlignmentConfig` does not expose one `kernel_scale`/`grid_step` pair:
the coarse-to-fine workflow uses a validated stage-specific S-LDDMM schedule
internally. Passing the cross-sample `SLDDMMConfig` does not affect Atlas
alignment. Tune the exposed hierarchy, pre-alignment, masks and pairing first;
a custom stage schedule requires an explicit API extension.

If a thin region matches a broader neighbor, check hierarchy level, mask
quality and pre-alignment first. Only then adjust the global area/thickness
balance. The four-level example is a validated starting profile, not a rule for
all cluster counts or atlas slices.
"""


HISTOLOGY_FEATURE_GUIDANCE = r"""
### How to adapt image preparation and feature extraction

- `target_microns_per_pixel` defines the physical analysis resolution. If the
  image lacks reliable metadata, set `source_microns_per_pixel` explicitly.
  An incorrect pixel size changes tissue scale and every downstream distance
  parameter.
- `shifted_tiles=True` averages 16 shifted HIPT views and reduces tile-boundary
  artifacts at substantially greater runtime. Use an unshifted run only for
  rapid exploration and restore shifted views for the final analysis.
- Keep the validated HIPT checkpoints and the 224-pixel preparation multiple
  unless explicitly benchmarking a new feature backend.

Inspect the prepared RGB image, feature-grid dimensions and feature maps before
clustering. A successful file write is not sufficient QC for an incorrect
resolution or a tiled feature artifact.
"""


HISTOLOGY_CLUSTER_GUIDANCE = r"""
### How to adapt image-feature clustering

- `image_clusters` controls initial tissue granularity. Increase it when
  distinct morphology is merged; decrease it when the image fragments into
  many visually indistinguishable regions.
- `rgb_weight` increases sensitivity to stain/color and
  `coordinate_weight` encourages spatial compactness. Excessive RGB weight can
  follow staining artifacts; excessive coordinate weight can split identical
  morphology solely by location.
- `cleanup_min_size` is measured in feature-grid pixels and must be reconsidered
  when image resolution or tissue size changes.
- Symmetry merging is designed for approximately bilateral sections. Relax or
  disable it for partial, asymmetric or strongly pathological tissue.

Choose settings from the full image-to-label QC sequence. The number of final
regions is an output, not a target that should be forced without morphological
support.
"""


HISTOLOGY_ALIGNMENT_GUIDANCE = r"""
### How to adapt ST-to-histology alignment

Inspect the whole-mask overlay first. If automatic pre-alignment has low IoU or
incorrect anatomy, use the interactive manual scale/rotation/translation tool
before changing structure-pairing or S-LDDMM parameters.

- Raising `pair_score_threshold` or lowering `pair_asd_threshold` makes pairing
  stricter. Review accepted and rejected masks, not only pair count.
- The empirical H&E profile is SDF 0.20, Chamfer 0.40, Dice 0.15, area 0.25,
  and thickness 0.00. Set these five `pairing_weight_*` fields globally; they
  must be non-negative and sum to one. ASD remains a raw-distance QC criterion
  and is not part of the composite score. See the website's **Cross-modality
  pairing weights** section before tuning this profile.
- `global_shape_weight` balances the whole-tissue outline against matched
  internal structures. Increase it if local pairs distort the overall outline;
  lower it when real outline differences block supported internal alignment.
- `zoom_scale` downsamples the matching field. Smaller values save memory and
  remove detail. `kernel_scale` (`a=50`) and `velocity_grid_spacing`
  (`grid_step=6`) remain in original histology feature-grid units because the
  sampled axes retain the full original extent. Do not multiply them by
  `zoom_scale`.
- Larger `a` or `grid_step` yields smoother/coarser deformation; smaller values
  permit more local motion and cost more memory or stability. Tune them
  together, then adjust `momentum_lr` and `iterations` from the energy history.

Do not transfer the paper distance values to a differently rescaled image
without re-evaluating the coordinate frame.
"""


ATAC_PREALIGN_GUIDANCE = r"""
### How to adapt the partial-field pre-alignment

Tune the global ST and ATAC similarity transforms before the reference crop.
Then choose `reference_crop_axis`, `reference_crop_side` and
`reference_crop_quantile` so both modalities contain the same anatomical field
of view. A crop with good point-cloud overlap but different anatomy is not a
valid initialization.

`raster_scale` converts transformed coordinates to the common canvas. A larger
value retains more spatial detail and increases memory. After changing it,
reinterpret every pixel-distance threshold, `kernel_scale` and velocity-grid
spacing in the new canvas units. Use `canvas_padding` only to keep supported
deformation away from the canvas boundary; it does not correct alignment.
"""


ATAC_ALIGNMENT_GUIDANCE = r"""
### How to adapt ATAC-to-ST pairing and S-LDDMM

- Inspect structure masks after outlier filtering before tuning pair scores.
  `density_neighbors` and `density_mad_multiplier` control sparse-coordinate
  filtering; a smaller MAD multiplier is stricter.
- SDF, Chamfer, area and Dice weights define one global pairing rule. The
  empirical profile is 0.35, 0.25, 0.30, and 0.10, respectively; the weights
  must be non-negative and sum to one. See the website's **Cross-modality
  pairing weights** section for tuning guidance.
  `pair_score_threshold` and `pair_dice_threshold` are acceptance gates. Do not
  loosen them until rejected candidate masks have been reviewed.
- `channel_area_power` strengthens inverse-area balancing as it approaches one;
  `channel_uniform_mix` moves weights toward equality. These settings can
  protect narrow masks without assigning an anatomy-specific weight.
- `kernel_scale=100` is legacy `a`; `velocity_grid_spacing=40` is legacy
  `grid_step`. Both use the cropped raster-canvas units. Larger values are
  smoother/coarser; smaller values are more local and more expensive.
- `time_steps=8` controls flow integration. Tune `iterations` and
  `momentum_lr` only after pre-alignment and accepted pairs pass QC; use the
  rollback and best-checkpoint diagnostics to detect instability.
"""


INFERENCE_GUIDANCE = r"""
### How to adapt the inference settings

- Select `risk_genes` independently from the genes being tested and retain a
  broad, adequately detected set. Do not estimate residual mismatch from only
  a few target genes.
- Increase `min_detected_spots` or `min_total_counts` when extremely sparse
  genes make the risk map unstable.
- `density_energy_share` must be between zero and one. Increase it only when
  sampling-density discordance is a meaningful reliability signal; it must not
  be used to conceal poor alignment.
- Enable cell-type adjustment only when complete, validated annotations exist
  for every sample. Record whether `region_cleanup` is enabled because it
  changes connected significant regions.

Inspect aligned geometry, local support and mismatch risk together before
interpreting differential-expression maps.
"""


UNCERTAINTY_GUIDANCE = r"""
### Adapting the subsampling design

The ten 80% replicates are an experimental design choice, not an S-LDDMM
default. For another dataset, choose the retained fraction high enough to
preserve all important spatial structures and use enough independent repeats
to stabilize the tail of the variability distribution. Re-run clustering,
pre-alignment, rasterization and S-LDDMM independently for every replicate.

Evaluate every learned transformation on one fixed support; otherwise changing
cell membership is confounded with transformation variability. Report the
number of repeats, retained fraction, random seeds and all alignment parameters.
The resulting spread is empirical subsampling stability, not a posterior
probability or confidence interval.
"""


GUIDANCE: dict[str, list[tuple[str, str]]] = {
    "clustering_joint_nb.ipynb": [("JointClusteringConfig(", CLUSTERING_GUIDANCE)],
    "clustering_single_nb.ipynb": [("SingleClusteringConfig(", CLUSTERING_GUIDANCE)],
    "cross_sample_alignment_mouse_kidney_clustering_nb.ipynb": [
        ("JointClusteringConfig(", CLUSTERING_GUIDANCE)
    ],
    "cross_sample_alignment_breast_cancer_clustering_nb.ipynb": [
        ("JointClusteringConfig(", CLUSTERING_GUIDANCE)
    ],
    "atac_st_single_clustering_nb.ipynb": [
        ("SingleClusteringConfig(", CLUSTERING_GUIDANCE)
    ],
    "cross_sample_alignment_nb.ipynb": [("SLDDMMConfig(", CROSS_SAMPLE_GUIDANCE)],
    "cross_sample_alignment_mouse_kidney_alignment_nb.ipynb": [
        ("SLDDMMConfig(", CROSS_SAMPLE_GUIDANCE)
    ],
    "cross_sample_alignment_breast_cancer_alignment_nb.ipynb": [
        ("SLDDMMConfig(", CROSS_SAMPLE_GUIDANCE)
    ],
    "cross_modal_atlas_alignment_nb.ipynb": [
        ("STAtlasAlignmentConfig(", ATLAS_GUIDANCE)
    ],
    "st_he_feature_extraction_nb.ipynb": [
        ("HistologyFeatureConfig(", HISTOLOGY_FEATURE_GUIDANCE)
    ],
    "st_he_feature_clustering_nb.ipynb": [
        ("HistologyClusteringConfig(", HISTOLOGY_CLUSTER_GUIDANCE)
    ],
    "st_he_alignment_nb.ipynb": [
        ("STHistologyAlignmentConfig(", HISTOLOGY_ALIGNMENT_GUIDANCE)
    ],
    "atac_st_alignment_nb.ipynb": [
        ("ATACSTPrealignmentConfig(", ATAC_PREALIGN_GUIDANCE),
        ("ATACSTAlignmentConfig(", ATAC_ALIGNMENT_GUIDANCE),
    ],
    "post_alignment_inference_nb.ipynb": [
        ("prepared = spAlignDE.prepare_inference(", INFERENCE_GUIDANCE)
    ],
    "cross_sample_uncertainty_report.ipynb": [
        ("## 1. Experimental design", UNCERTAINTY_GUIDANCE)
    ],
}


def add_guidance(path: Path) -> bool:
    additions = GUIDANCE.get(path.name)
    if not additions:
        return False

    notebook = nbformat.read(path, as_version=4)
    roles = {f"{ROLE_PREFIX}{index}" for index in range(len(additions))}
    notebook.cells = [
        cell
        for cell in notebook.cells
        if cell.get("metadata", {}).get("spAlignDE_role") not in roles
    ]

    for index, (marker, text) in enumerate(additions):
        matches = [
            position
            for position, cell in enumerate(notebook.cells)
            if marker in cell.source
        ]
        if not matches:
            raise RuntimeError(f"Could not find {marker!r} in {path}")
        insertion = matches[0] + 1
        cell = nbformat.v4.new_markdown_cell(text.strip())
        cell.metadata["spAlignDE_role"] = f"{ROLE_PREFIX}{index}"
        notebook.cells.insert(insertion, cell)

    nbformat.write(notebook, path)
    relative = path.relative_to(NOTEBOOK_ROOT)
    destination = SPHINX_NOTEBOOK_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    print(relative)
    return True


def main() -> None:
    seen: set[str] = set()
    for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
        if path.name in seen and path.name in GUIDANCE:
            raise RuntimeError(f"Duplicate public notebook filename: {path.name}")
        if add_guidance(path):
            seen.add(path.name)
    missing = sorted(set(GUIDANCE).difference(seen))
    if missing:
        raise RuntimeError("Public notebooks not found: " + ", ".join(missing))


if __name__ == "__main__":
    main()
