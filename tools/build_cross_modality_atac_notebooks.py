#!/usr/bin/env python3
"""Build the canonical spatial-ATAC clustering and ATAC-to-ST notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPHINX_SOURCE = PROJECT_ROOT / "docs" / "source"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def notebook(cells):
    result = nbf.v4.new_notebook(cells=cells)
    result.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    result.metadata["language_info"] = {"name": "python", "version": "3.10"}
    return result


COMMON_SETUP = r"""
%matplotlib inline

from pathlib import Path
import os
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from IPython.utils.capture import capture_output

import spAlignDE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook from the spAlignDE repository.")


PROJECT_ROOT = find_project_root(Path.cwd())
OUTPUT_DIR = PROJECT_ROOT / "tutorials" / "cross_modality" / "atac" / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
"""


def single_clustering_notebook():
    return notebook(
        [
            markdown(
                r"""
# Spatial ATAC single clustering — P22 mouse brain

This is notebook 1 of 2 in the spatial ATAC-to-ST workflow. It starts from a
P22 mouse-brain spatial ATAC **gene-activity matrix**, runs single-sample
BANKSY, and saves a clustered AnnData object for the alignment notebook.

The paper example contains 9,215 spatial observations measured at 20-µm
resolution. ATAC is clustered independently from the ST reference: matched
genes or a joint ATAC/ST latent space are not required.
"""
            ),
            markdown(
                r"""
## Installation, data, and input contract

Install the package from the repository root:

```bash
cd /path/to/spAlignDE
python -m pip install -e ".[clustering,atlas,tutorial]"
```

The P22 spatial ATAC–RNA dataset is available through the
[UCSC Cell Browser](https://brain-spatial-omics.cells.ucsc.edu/) and
[GEO GSE205055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055).
The original study and processing description are provided in the
[Nature article](https://www.nature.com/articles/s41586-023-05795-1).

spAlignDE accepts either of the following forms:

| Input | Required content |
|---|---|
| AnnData/H5AD | `X`: observations × genes, containing non-negative gene-activity values; `obsm["spatial"]`: finite `(n_obs, 2)` x/y coordinates; unique observation names. |
| Paired CSV | metadata CSV: `cell_id,x,y`; expression CSV: the same `cell_id` values followed by numeric, non-negative gene-activity columns. |

Use an ATAC-derived **gene-activity** matrix, not a peak-by-observation matrix.
Gene activity gives BANKSY a gene-level representation while the alignment
itself uses only independently inferred spatial structures. An R/Seurat RDS
can be used upstream, but it must first be exported to H5AD or the paired CSV
contract above.

Place `p22_atac_gene_activity.h5ad` under
`data/cross_modality/atac/`, or set `SPALIGNDE_ATAC_INPUT` to its path.
For CSV input, set both `SPALIGNDE_ATAC_METADATA_CSV` and
`SPALIGNDE_ATAC_EXPRESSION_CSV`.
"""
            ),
            code(COMMON_SETUP),
            code(
                r"""
default_input = PROJECT_ROOT / "data" / "cross_modality" / "atac" / "p22_atac_gene_activity.h5ad"
atac_h5ad = Path(os.environ.get("SPALIGNDE_ATAC_INPUT", default_input)).expanduser()
metadata_csv = os.environ.get("SPALIGNDE_ATAC_METADATA_CSV")
expression_csv = os.environ.get("SPALIGNDE_ATAC_EXPRESSION_CSV")

if metadata_csv or expression_csv:
    if not (metadata_csv and expression_csv):
        raise ValueError("Set both ATAC CSV environment variables.")
    atac = spAlignDE.load_single_sample_data(
        Path(metadata_csv).expanduser(),
        expression_csv=Path(expression_csv).expanduser(),
    )
    input_mode = "paired CSV"
else:
    atac = spAlignDE.load_single_sample_data(atac_h5ad)
    input_mode = "AnnData/H5AD"

print(f"Input mode: {input_mode}")
print(f"ATAC input: {atac.n_obs:,} observations × {atac.n_vars:,} gene-activity features")
print("Spatial array:", atac.obsm["spatial"].shape)
"""
            ),
            markdown(
                r"""
## Run single-sample BANKSY

The paper workflow uses 30 spatial neighbors, scaled-Gaussian neighbor
weighting, `λ = 0.6`, 20 principal components, and Leiden resolution 1.0.
The random seed is fixed. Optional boundary-aware refinement is retained as a
QC label layer; the paper ATAC-to-ST structure matching below uses the raw
BANKSY partition (`cluster_raw`) so it is not altered by a second local voting
step.

Dependency versions can cause small changes at cluster boundaries and may
renumber categorical labels. Reproduction should therefore be judged by the
spatial partition and final geometry rather than integer label identity.
"""
            ),
            code(
                r"""
cluster_config = spAlignDE.SingleClusteringConfig(
    num_neighbors=30,
    banksy_lambda=0.6,
    resolution=1.0,
    pca_dim=20,
    max_m=1,
    decay="scaled_gaussian",
    random_state=1234,
    refine_boundaries=True,
)

started = time.perf_counter()
with capture_output() as banksy_capture:
    clustered, details = spAlignDE.cluster_single(
        atac,
        config=cluster_config,
        banksy_output_dir=OUTPUT_DIR / "banksy_diagnostics",
        return_details=True,
    )
plt.close("all")
elapsed = time.perf_counter() - started

summary = pd.DataFrame(
    {
        "value": [
            f"{clustered.n_obs:,}",
            f"{clustered.n_vars:,}",
            clustered.obs["cluster_raw"].nunique(),
            clustered.obs["cluster_refined"].nunique(),
            f"{elapsed / 60:.2f} min",
        ]
    },
    index=["observations", "features", "raw clusters", "refined clusters", "runtime"],
)
display(summary)
"""
            ),
            markdown(
                r"""
## Inspect the spatial partition and boundary refinement

The two panels use the same color mapping. The raw labels define the structure
channels in notebook 2; refined labels remain available for checking whether
local voting changes narrow or boundary-adjacent regions too aggressively.
"""
            ),
            code(
                r"""
fig, axes = spAlignDE.plot_single_cluster_refinement(
    clustered,
    point_size=2.0,
    alpha=0.90,
    figsize=(12, 5.2),
)
fig.savefig(FIGURE_DIR / "atac_clusters_raw_vs_refined.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "atac_clusters_raw_vs_refined.svg", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## Save the explicit handoff to notebook 2

The output keeps the original gene-activity matrix and
`obsm["spatial"]`, and adds:

- `obs["cluster_raw"]`: the selected BANKSY partition used for ATAC-to-ST;
- `obs["cluster_refined"]`: boundary-aware QC labels; and
- `obs["cluster"]`: the package-selected canonical labels.

Parameters and provenance are stored under `adata.uns["spAlignDE"]`.
"""
            ),
            code(
                r"""
clustered_path = OUTPUT_DIR / "p22_atac_single_clustered.h5ad"
clustered.write_h5ad(clustered_path)

cluster_table = clustered.obs[["cluster_raw", "cluster_refined", "cluster"]].copy()
cluster_table.insert(0, "cell_id", clustered.obs_names.astype(str))
cluster_table[["x", "y"]] = clustered.obsm["spatial"]
cluster_table.to_csv(OUTPUT_DIR / "p22_atac_cluster_labels.csv", index=False)

print("Saved clustered AnnData:", clustered_path.relative_to(PROJECT_ROOT))
print("Next notebook cluster key: cluster_raw")
display(cluster_table.head())
"""
            ),
        ]
    )


def alignment_notebook():
    return notebook(
        [
            markdown(
                r"""
# P22 spatial ATAC to MERFISH S3R1

This is notebook 2 of 2. It registers the 9,215-observation P22 spatial ATAC
query to an independently clustered Vizgen MERFISH S3R1 reference. The fixed
reference begins with 70,844 cells; the matching half-brain crop contains
35,422 cells in the paper workflow.

The two modalities do not need shared measured features. spAlignDE converts
their independently inferred clusters into comparable masks, pairs structures
with global geometric criteria, and estimates a smooth ATAC-to-ST deformation.
"""
            ),
            markdown(
                r"""
## Notebook order, data, and input contract

Run [spatial ATAC single clustering](atac_st_single_clustering_nb.ipynb) first.
It writes `p22_atac_single_clustered.h5ad`. Prepare the fixed S3R1 reference
with the canonical single-clustering workflow so that it has one selected
cluster column and finite coordinates in `obsm["spatial"]`.

| Role | Dataset | Required AnnData content |
|---|---|---|
| Moving query | P22 spatial ATAC | gene-activity `X`; original x/y coordinates; `obs["cluster_raw"]` from notebook 1 |
| Fixed reference | Vizgen MERFISH S3R1 | cell-by-gene `X`; original x/y coordinates; independently inferred `obs["cluster"]` |

The P22 data are available from the
[UCSC Cell Browser](https://brain-spatial-omics.cells.ucsc.edu/) and
[GEO GSE205055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055).
Download S3R1 from the
[Vizgen MERFISH Mouse Brain Receptor Map](https://info.vizgen.com/mouse-brain-map).

Place the files under `tutorials/cross_modality/atac/output/` and
`data/cross_modality/atac/`, respectively, or set
`SPALIGNDE_ATAC_CLUSTERED_H5AD` and `SPALIGNDE_ST_CLUSTERED_H5AD`.
"""
            ),
            code(COMMON_SETUP),
            code(
                r"""
atac_path = Path(
    os.environ.get(
        "SPALIGNDE_ATAC_CLUSTERED_H5AD",
        OUTPUT_DIR / "p22_atac_single_clustered.h5ad",
    )
).expanduser()
st_path = Path(
    os.environ.get(
        "SPALIGNDE_ST_CLUSTERED_H5AD",
        PROJECT_ROOT / "data" / "cross_modality" / "atac" / "merfish_s3r1_clustered.h5ad",
    )
).expanduser()

atac = spAlignDE.load_single_sample_data(atac_path)
st_reference = spAlignDE.load_single_sample_data(st_path)
spAlignDE.validate_single_sample_anndata(atac, cluster_key="cluster_raw", require_cluster=True)
spAlignDE.validate_single_sample_anndata(st_reference, cluster_key="cluster", require_cluster=True)

inputs = pd.DataFrame(
    {
        "role": ["moving query", "fixed reference"],
        "dataset": ["P22 spatial ATAC", "MERFISH S3R1"],
        "observations": [atac.n_obs, st_reference.n_obs],
        "features": [atac.n_vars, st_reference.n_vars],
        "structures": [atac.obs["cluster_raw"].nunique(), st_reference.obs["cluster"].nunique()],
    }
)
display(inputs)
"""
            ),
            markdown(
                r"""
## Stage 1 — independent structures

ATAC and ST are clustered separately within their own molecular spaces. The
ATAC panel shows the raw BANKSY labels from notebook 1; the ST panel shows the
independently prepared S3R1 labels. Similar colors across panels do not imply
a match at this stage.
"""
            ),
            code(
                r"""
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), constrained_layout=True)
for axis, adata, key, title, size in [
    (axes[0], atac, "cluster_raw", "P22 spatial ATAC", 2.0),
    (axes[1], st_reference, "cluster", "MERFISH S3R1", 0.45),
]:
    labels = pd.Categorical(adata.obs[key].astype(str))
    axis.scatter(
        adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1],
        c=labels.codes, cmap="turbo", s=size, alpha=0.88,
        edgecolors="none", rasterized=True,
    )
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.axis("off")
fig.savefig(FIGURE_DIR / "atac_st_independent_structures.png", dpi=220, bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## Stage 2 — global pre-alignment and matching field of view

The paper experiment uses one global similarity transform for each modality,
then retains the left half of the transformed ST brain because the ATAC assay
covers a partial section. The shared analysis canvas is scaled by 0.25 and
padded by 10 pixels.

```text
ST:   rotation = -125°, scale = 1.0, translation = (0, 0)
ATAC: rotation =  -90°, scale = 1.6, translation = (-2600, -3300)
ST crop: x-axis left half at the 0.5 quantile
```

These are global, sample-level initialization parameters. They neither assign
region identities nor give a hippocampal or other anatomical structure a
special weight. If this initialization is unsuitable for a new sample, tune a
global transform with the package's interactive manual pre-alignment first,
then record the selected values in `ATACSTPrealignmentConfig`.
"""
            ),
            code(
                r"""
prealignment_config = spAlignDE.ATACSTPrealignmentConfig(
    st_transform=spAlignDE.ManualPrealignmentConfig(
        scale=1.0, theta_deg=-125.0, translation_x=0.0, translation_y=0.0
    ),
    atac_transform=spAlignDE.ManualPrealignmentConfig(
        scale=1.6, theta_deg=-90.0, translation_x=-2600.0, translation_y=-3300.0
    ),
    reference_crop_axis="x",
    reference_crop_side="left",
    reference_crop_quantile=0.5,
    raster_scale=0.25,
    canvas_padding=10,
)

prealigned = spAlignDE.prealign_atac_to_st(
    atac,
    st_reference,
    config=prealignment_config,
    atac_cluster_key="cluster_raw",
    st_cluster_key="cluster",
)

print("ATAC observations retained:", f"{prealigned.atac.n_obs:,}")
print("ST observations after half-brain crop:", f"{prealigned.st_reference.n_obs:,}")
print("Shared canvas (height, width):", prealigned.canvas_shape_hw)

fig, axes = spAlignDE.plot_atac_st_prealignment(
    prealigned,
    atac_cluster_key="cluster_raw",
    st_cluster_key="cluster",
)
fig.savefig(FIGURE_DIR / "atac_st_global_prealignment.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "atac_st_global_prealignment.svg", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## Stage 3 — rasterize and pair independently inferred structures

For every ATAC and ST cluster, spAlignDE removes sparse point outliers,
rasterizes the point cloud, smooths and closes the mask, fills holes, and
retains its principal connected components. Narrow structures receive a more
conservative mask recipe than broad structures.

Every cross-modality candidate pair is evaluated with one global score:

```text
0.35 × signed-distance correlation
+ 0.25 × Chamfer similarity
+ 0.30 × area similarity
+ 0.10 × Dice overlap
```

A global logistic gate downweights candidates with large boundary distance.
Pairs must have score ≥ 0.25 and Dice ≥ 0.01; greedy selection then enforces a
one-to-one mapping. These weights and gates are applied identically to every
structure—there are no region-specific labels or anatomical weights. For a new
dataset, follow the [cross-modality pairing-weight tuning guide](https://dsong-lab.github.io/spAlignDE/tutorials/parameter_tuning.html#cross-modality-pairing-weights),
change one component at a time, and renormalize all four weights to sum to one.
"""
            ),
            markdown(
                r"""
## Stage 4 — structure-guided S-LDDMM

Accepted masks are converted to clipped signed-distance-transform channels.
Inverse-area channel balancing prevents a large structure from dominating the
loss while remaining a global rule shared by every channel. S-LDDMM then
estimates a smooth diffeomorphic field and maps all ATAC observations, not only
the points belonging to matched masks.

The paper settings are `nt=8`, `niter=500`, diffeomorphic start 20,
`a=100`, `p=2`, velocity-grid spacing 40, `epL=2e-11`, `epT=2e-5`,
`epM=1e3`, `sigmaR=1e6`, and `sigmaM=0.5`.

The optimizer applies one global numerical-stability policy: clip the momentum
gradient, reduce the common step scale when total energy rises, reject a
non-finite or singular affine update, and restore the lowest-energy checkpoint.
These safeguards do not inspect cluster names or anatomical regions.
"""
            ),
            code(
                r"""
alignment_config = spAlignDE.ATACSTAlignmentConfig(
    sdf_weight=0.35,
    chamfer_weight=0.25,
    area_weight=0.30,
    dice_weight=0.10,
    chamfer_gate_center=16.0,
    chamfer_gate_scale=6.0,
    pair_score_threshold=0.25,
    pair_dice_threshold=0.01,
    maximum_pairs=20,
    time_steps=8,
    iterations=500,
    diffeomorphic_start=20,
    kernel_scale=100.0,
    kernel_power=2.0,
    velocity_grid_spacing=40.0,
    affine_linear_lr=2e-11,
    affine_translation_lr=2e-5,
    momentum_lr=1e3,
    deformation_regularization=1e6,
    matching_scale=0.5,
)

started = time.perf_counter()
result = spAlignDE.align_atac_to_st(
    prealigned,
    config=alignment_config,
    atac_cluster_key="cluster_raw",
    st_cluster_key="cluster",
    output_dir=OUTPUT_DIR / "alignment",
    verbose=False,
)
elapsed = time.perf_counter() - started

print(f"Alignment runtime: {elapsed / 60:.2f} min")
print("Accepted structure pairs:", len(result.matched_pairs))
display(
    result.matched_pairs[
        [
            "st_structure", "atac_structure", "align_score", "sdf_corr",
            "chamfer_distance", "area_sim", "dice",
        ]
    ].round(4)
)
"""
            ),
            markdown(
                r"""
## Stage 5 — visual quality control

The first figure assigns one shared color to each accepted ST/ATAC structure
pair. Gray points were not used as matched channels but are still carried by
the smooth deformation. The second figure compares the global initialization
with the final ATAC coordinates over the fixed ST reference.
"""
            ),
            code(
                r"""
fig, axes = spAlignDE.plot_atac_st_matched_structures(
    result,
    atac_cluster_key="cluster_raw",
    st_cluster_key="cluster",
)
fig.savefig(FIGURE_DIR / "atac_st_matched_structure_pairs.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "atac_st_matched_structure_pairs.svg", bbox_inches="tight")
plt.show()

fig, axes = spAlignDE.plot_atac_st_alignment(
    result,
    atac_cluster_key="cluster_raw",
)
fig.savefig(FIGURE_DIR / "atac_st_alignment_before_after.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "atac_st_alignment_before_after.svg", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## Outputs and interpretation

The aligned ATAC AnnData preserves the original assay and coordinates and
adds the package-wide coordinate contract:

- `x_prealigned`, `y_prealigned`: ATAC after the recorded global transform;
- `x_aligned`, `y_aligned`: final ATAC coordinates in the ST analysis frame.

The cropped ST reference receives the same four columns; because it is fixed,
its prealigned and aligned values are identical. Provenance is stored under
`uns["spAlignDE"]["atac_to_st"]`.

`alignment/` also contains the aligned/query and fixed/reference H5AD files,
all candidate scores, accepted one-to-one pairs, mask summaries, and a JSON
manifest. The accepted masks are optimization inputs; downstream label
agreement or local-neighborhood preservation is an independent evaluation and
is not used to tune the deformation.

For a new section, first inspect the global pre-alignment and pair overlay. If
the tissue fields do not correspond, revise the global initialization or crop
before changing the globally shared score weights.
"""
            ),
            code(
                r"""
output_columns = ["x_prealigned", "y_prealigned", "x_aligned", "y_aligned"]
assert all(column in result.atac.obs for column in output_columns)
assert result.atac.n_obs == atac.n_obs
assert result.atac.obs[output_columns].notna().all().all()

output_summary = pd.DataFrame(
    {
        "file": [
            "atac_to_st_aligned.h5ad",
            "st_reference_analysis_frame.h5ad",
            "matched_structure_pairs.csv",
            "candidate_structure_pairs.csv",
            "atac_mask_summary.csv / st_mask_summary.csv",
            "alignment_manifest.json",
        ],
        "content": [
            "ATAC assay plus standardized coordinates",
            "fixed cropped ST reference and analysis-frame coordinates",
            "accepted one-to-one structure pairs",
            "all global geometric pair scores",
            "mask construction diagnostics",
            "pre-alignment, pairing, and S-LDDMM provenance",
        ],
    }
)
display(output_summary)
print("Validated all", f"{result.atac.n_obs:,}", "ATAC observations.")
"""
            ),
        ]
    )


def write_notebook(result, *paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(result, path)
        print(path)


def main() -> None:
    canonical = PROJECT_ROOT / "source_notebooks" / "cross_modality"
    tutorial = PROJECT_ROOT / "tutorials" / "cross_modality" / "atac"
    sphinx = SPHINX_SOURCE / "source_notebooks" / "cross_modality"

    write_notebook(
        single_clustering_notebook(),
        canonical / "atac_st_single_clustering_nb.ipynb",
        tutorial / "01_atac_single_clustering.ipynb",
        sphinx / "atac_st_single_clustering_nb.ipynb",
    )
    write_notebook(
        alignment_notebook(),
        canonical / "atac_st_alignment_nb.ipynb",
        tutorial / "02_atac_to_st_alignment.ipynb",
        sphinx / "atac_st_alignment_nb.ipynb",
    )


if __name__ == "__main__":
    main()
