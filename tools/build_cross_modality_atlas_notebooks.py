#!/usr/bin/env python3
"""Build the canonical single-clustering and ST-to-Allen-CCF notebooks."""

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


def single_clustering_notebook():
    return notebook(
        [
            markdown(
                r"""
# Single clustering for cross-modality alignment

This notebook prepares one spatial transcriptomics sample for cross-modality
alignment. It accepts either AnnData/H5AD or a paired metadata/expression CSV
input, runs single-sample BANKSY, optionally applies boundary-aware refinement,
and writes the canonical labels `cluster_raw`, `cluster_refined`, and `cluster`.

The example is the adult mouse brain MERFISH S2R1 sample (83,546 cells and 649
measured genes). The resulting `AnnData` is the direct input to the ST-to-Allen
CCF notebook; there is no intermediate research-script CSV contract.
"""
            ),
            markdown(
                r"""
## Installation and data

From the repository root:

```bash
cd /path/to/spAlignDE
python -m pip install -e ".[clustering,atlas,tutorial]"
```

Download the MERFISH Mouse Brain Receptor Map from
[Vizgen](https://info.vizgen.com/mouse-brain-map). Put either input form in
`data/cross_modality/mouse_brain/merfishS2/`, or set
`SPALIGNDE_ST_DATA_DIR` to another directory.

| Input form | Required files | Required fields |
|---|---|---|
| AnnData | `merfishS2_single.h5ad` | cell-by-gene `X`; finite `(n_obs, 2)` coordinates in `obsm["spatial"]`, or numeric `obs["x"]` and `obs["y"]` |
| CSV | `cell_metadata_S2R1.csv` and `cell_by_gene_S2R1.csv` | metadata: unique `cell_id`, numeric `x`, `y`; expression: the same `cell_id` values plus numeric, non-negative gene columns |

Set `SPALIGNDE_SINGLE_INPUT=csv` to exercise the CSV route. Both routes are
normalized by `spAlignDE.load_single_sample_data` to the same AnnData contract.
"""
            ),
            code(
                r"""
%matplotlib inline

from pathlib import Path
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import spAlignDE

WORKFLOW_SEED = 1234
seed_controls = spAlignDE.set_random_seed(
    WORKFLOW_SEED,
    deterministic_torch=True,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook from the spAlignDE repository.")


PROJECT_ROOT = find_project_root(Path.cwd())
public_data_dir = PROJECT_ROOT / "data" / "cross_modality" / "mouse_brain" / "merfishS2"
DATA_DIR = Path(os.environ.get("SPALIGNDE_ST_DATA_DIR", public_data_dir)).expanduser()

OUTPUT_DIR = PROJECT_ROOT / "tutorials" / "cross_modality" / "atlas" / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_MODE = os.environ.get("SPALIGNDE_SINGLE_INPUT", "h5ad").strip().lower()

if INPUT_MODE in {"h5ad", "anndata"}:
    input_adata = spAlignDE.load_single_sample_data(DATA_DIR / "merfishS2_single.h5ad")
elif INPUT_MODE == "csv":
    input_adata = spAlignDE.load_single_sample_data(
        DATA_DIR / "cell_metadata_S2R1.csv",
        expression_csv=DATA_DIR / "cell_by_gene_S2R1.csv",
    )
else:
    raise ValueError("SPALIGNDE_SINGLE_INPUT must be 'h5ad' or 'csv'.")

print(f"Input mode: {INPUT_MODE}")
print(f"Input AnnData: {input_adata.n_obs:,} cells × {input_adata.n_vars:,} genes")
"""
            ),
            markdown(
                r"""
## Run single-sample BANKSY

`banksy_lambda=0.8` and `resolution=1.2` reproduce the setting used for the
paper's S2R1 atlas experiment. BANKSY incorporates each cell's expression and
spatial-neighborhood features. Boundary-aware refinement uses fewer neighbors
near the tissue edge so thin anatomical structures are not erased by a global
majority vote.

This is a fresh package run. The notebook writes its clustered AnnData once;
the following alignment notebook reads that file as the explicit handoff
between the two workflow steps.
"""
            ),
            code(
                r"""
config = spAlignDE.SingleClusteringConfig(
    num_neighbors=30,
    banksy_lambda=0.8,
    resolution=1.2,
    max_m=1,
    decay="scaled_gaussian",
    random_state=WORKFLOW_SEED,
    refine_boundaries=True,
)

clustered = spAlignDE.cluster_single(
    input_adata,
    config=config,
    banksy_output_dir=OUTPUT_DIR / "banksy_diagnostics",
)

spAlignDE.validate_single_sample_anndata(
    clustered,
    cluster_key="cluster",
    require_cluster=True,
)
print("Mode: fresh package run")
print("Raw clusters:", clustered.obs["cluster_raw"].nunique())
print("Refined clusters:", clustered.obs["cluster_refined"].nunique())
"""
            ),
            markdown(
                r"""
## Inspect raw and refined spatial structures

Both panels share one cluster-color map. This makes boundary corrections
visible without changing cluster identity colors. For paper-scale point
clouds, the scatter layer is rasterized inside the SVG to keep the vector file
responsive while retaining editable axes and text.
"""
            ),
            code(
                r"""
fig, axes = spAlignDE.plot_single_cluster_refinement(
    clustered,
    point_size=0.35,
    alpha=0.85,
    figsize=(12, 5.5),
)
fig.savefig(FIGURE_DIR / "single_clustering_raw_vs_refined.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "single_clustering_raw_vs_refined.svg", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## Save the canonical clustered AnnData

The original expression matrix and `obsm["spatial"]` remain unchanged. The
three package-owned columns are:

- `cluster_raw`: selected BANKSY labels;
- `cluster_refined`: boundary-refined labels; and
- `cluster`: labels selected for downstream alignment.

Package provenance is stored only under `adata.uns["spAlignDE"]`.
"""
            ),
            code(
                r"""
clustered_path = OUTPUT_DIR / "merfish_S2R1_single_clustered.h5ad"
clustered.write_h5ad(clustered_path)

summary = pd.DataFrame(
    {
        "value": [
            f"{clustered.n_obs:,}",
            f"{clustered.n_vars:,}",
            clustered.obs["cluster_raw"].nunique(),
            clustered.obs["cluster_refined"].nunique(),
            "cluster_raw, cluster_refined, cluster",
        ]
    },
    index=["cells", "genes", "raw clusters", "refined clusters", "output labels"],
)
display(summary)
print("Saved: tutorials/cross_modality/atlas/output/merfish_S2R1_single_clustered.h5ad")
"""
            ),
            markdown(
                r"""
## Next step

Open **ST-to-Allen-CCF alignment — MERFISH adult mouse brain**. It reads this
clustered AnnData, builds expression-based coarse-to-fine structure levels,
aligns S2R1 to Allen CCF slice 675, and transfers atlas labels to every cell.
"""
            ),
        ]
    )


def atlas_alignment_notebook():
    return notebook(
        [
            markdown(
                r"""
# MERFISH S2R1 to Allen CCF slice 675

This notebook aligns the adult mouse brain MERFISH S2R1 sample (83,546 cells)
to annotated Allen Common Coordinate Framework (CCF) coronal slice 675. The
reference has no gene-expression features. spAlignDE therefore discovers
correspondences between ST spatial structures and hierarchical atlas regions,
then refines those correspondences with coarse-to-fine S-LDDMM.

The analysis starts from the canonical AnnData produced by the single-
clustering notebook and returns the same AnnData with pre-aligned coordinates,
final aligned coordinates, Allen labels, and package provenance.
"""
            ),
            markdown(
                r"""
## Installation and input data

From the repository root:

```bash
cd /path/to/spAlignDE
python -m pip install -e ".[clustering,atlas,tutorial]"
```

| Role | Input | Contract |
|---|---|---|
| Query ST | `tutorials/cross_modality/atlas/output/merfish_S2R1_single_clustered.h5ad` | cell-by-gene `X`; coordinates in `obsm["spatial"]`; selected labels in `obs["cluster"]` |
| Atlas annotation | `annotation_10.nrrd` | Allen CCF 2022 annotation volume; this example selects coronal slice 675 |
| Atlas hierarchy | `voxel_count_and_differences.csv` | annotation ID index plus structure name/acronym, hierarchy path, and display color |

Download the Allen CCF 2022 files from the
[Allen Institute annotation release](https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2022/)
and place them in `data/allen_ccf_2022/`, or set `SPALIGNDE_ALLEN_CCF_DIR`.
Run the single-clustering notebook first, or set
`SPALIGNDE_CLUSTERED_ST_H5AD` to an equivalent clustered AnnData file.
"""
            ),
            code(
                r"""
%matplotlib inline

from pathlib import Path
import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import display

import spAlignDE

WORKFLOW_SEED = 1234
seed_controls = spAlignDE.set_random_seed(
    WORKFLOW_SEED,
    deterministic_torch=True,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook from the spAlignDE repository.")


PROJECT_ROOT = find_project_root(Path.cwd())
OUTPUT_DIR = PROJECT_ROOT / "tutorials" / "cross_modality" / "atlas" / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

clustered_default = OUTPUT_DIR / "merfish_S2R1_single_clustered.h5ad"
CLUSTERED_PATH = Path(os.environ.get("SPALIGNDE_CLUSTERED_ST_H5AD", clustered_default)).expanduser()

public_atlas_dir = PROJECT_ROOT / "data" / "allen_ccf_2022"
ATLAS_DIR = Path(os.environ.get("SPALIGNDE_ALLEN_CCF_DIR", public_atlas_dir)).expanduser()

required_atlas_files = [
    ATLAS_DIR / "annotation_10.nrrd",
    ATLAS_DIR / "voxel_count_and_differences.csv",
]
missing_atlas_files = [str(path) for path in required_atlas_files if not path.is_file()]
if missing_atlas_files:
    raise FileNotFoundError(
        "Allen CCF inputs are missing. Place them under data/allen_ccf_2022 "
        "or set SPALIGNDE_ALLEN_CCF_DIR. Missing: " + ", ".join(missing_atlas_files)
    )

adata = spAlignDE.load_single_sample_data(CLUSTERED_PATH)
if "cluster" not in adata.obs and "banksy_cluster_refined" in adata.obs:
    adata.obs["cluster_raw"] = adata.obs["banksy_cluster"].astype(str).astype("category")
    adata.obs["cluster_refined"] = adata.obs["banksy_cluster_refined"].astype(str).astype("category")
    adata.obs["cluster"] = adata.obs["cluster_refined"].copy()

atlas = spAlignDE.load_allen_ccf_reference(
    ATLAS_DIR / "annotation_10.nrrd",
    ATLAS_DIR / "voxel_count_and_differences.csv",
    slice_index=675,
)

print(f"Query: {adata.n_obs:,} cells × {adata.n_vars:,} genes; {adata.obs['cluster'].nunique()} clusters")
print(f"Reference: Allen CCF slice {atlas.slice_index}; image shape {atlas.shape}")
"""
            ),
            markdown(
                r"""
## 1. Build coarse-to-fine ST structure levels

Single-sample BANKSY supplies the finest ST labels. spAlignDE then aggregates
log-transformed expression within each cluster, retains genes explaining 80%
of total gene-wise variance (at least 50 genes), standardizes cluster-average
profiles, and applies Ward linkage. Two coarser partitions precede the final
BANKSY partition. Three levels provide the validated coarse-to-fine path. The
structure-pairing score also treats area and thickness as first-class shape
evidence, which helps distinguish narrow laminar regions from broader adjacent
structures. No CA3-specific matching rule is used. The executed output below
is authoritative for the current locked environment.

These levels are ST-only: Allen labels and coordinates are not used to create
them. Atlas hierarchy candidates from depths 2–10 remain eligible at every
stage; only the ST partition becomes finer.
"""
            ),
            code(
                r"""
config = spAlignDE.STAtlasAlignmentConfig(
    n_levels=3,
    minimum_coarse_structures=7,
    variance_fraction=0.8,
    min_genes=50,
    continue_alignment=True,
    continue_max_iterations=10,
    continue_min_pair_gain=1,
    pairing_weight_sdf=0.05,
    pairing_weight_chamfer=0.05,
    pairing_weight_dice=0.20,
    pairing_weight_area=0.50,
    pairing_weight_thickness=0.20,
    pairing_dice_soft=0.25,
    continuation_kernel_scale=200,
    continuation_velocity_grid_spacing=50,
    continuation_restore_best_checkpoint=False,
    device=None,  # automatically uses CUDA when available
)

adata, hierarchy_columns = spAlignDE.build_st_cluster_hierarchy(
    adata,
    config=config,
    cluster_key="cluster",
    copy=False,
)
print("Coarse-to-fine alignment labels:", [*hierarchy_columns, "cluster"])
display(
    pd.DataFrame(
        {
            "label column": [*hierarchy_columns, "cluster"],
            "number of structures": [
                *[adata.obs[column].nunique() for column in hierarchy_columns],
                adata.obs["cluster"].nunique(),
            ],
        }
    )
)
"""
            ),
            markdown(
                r"""
## 2. Align ST to the Allen CCF slice

The package performs four linked operations:

1. Estimate rotation, isotropic scale, and translation by maximizing IoU
   between the whole-ST mask and atlas brain mask; reflection is disabled.
2. Rasterize each ST structure as an adaptive smoothed binary mask and compare
   it with atlas structures derived directly from the annotation hierarchy.
3. Select non-overlapping one-to-one correspondences using a gated composite
   of Dice, signed-distance correlation, Chamfer similarity, area similarity,
   and thickness similarity. Average surface distance is retained as QC.
4. Convert accepted mask pairs to matched signed-distance channels and apply
   coarse-to-fine shooting-based LDDMM.

This notebook always performs a fresh alignment from the clustered AnnData.
No validated-result loader or alignment cache is involved.

The default pairing score is

`0.05 × SDF + 0.05 × Chamfer + 0.20 × Dice + 0.50 × area + 0.20 × thickness`.

These weights sum to one. They emphasize global size and laminar thickness
without a structure-name override; Dice, SDF, and Chamfer still require spatial
agreement. Raw ASD is retained only as an independent QC gate. For new data,
follow the [cross-modality pairing-weight tuning guide](https://dsong-lab.github.io/spAlignDE/tutorials/parameter_tuning.html#cross-modality-pairing-weights)
and change the five weights globally while keeping them non-negative and
normalized to one.
"""
            ),
            code(
                r"""
alignment_output = OUTPUT_DIR / "alignment"
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
alignment_start = time.perf_counter()
result = spAlignDE.align_st_to_allen_atlas(
    adata,
    atlas,
    config=config,
    cluster_key="cluster",
    output_dir=alignment_output,
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed_seconds = time.perf_counter() - alignment_start
peak_cuda_gib = (
    torch.cuda.max_memory_allocated() / (1024**3)
    if torch.cuda.is_available()
    else 0.0
)
runtime_metrics = {
    "elapsed_seconds": elapsed_seconds,
    "elapsed_minutes": elapsed_seconds / 60.0,
    "peak_cuda_memory_allocated_gib": peak_cuda_gib,
    "cuda_available": torch.cuda.is_available(),
}
(alignment_output / "end_to_end_runtime_cuda_metrics.json").write_text(
    json.dumps(runtime_metrics, indent=2)
)

print("Mode: fresh package run")
print("Aligned observations:", f"{result.adata.n_obs:,}")
print("Final matched structure pairs:", len(result.matched_pairs))
print(f"Alignment runtime: {runtime_metrics['elapsed_minutes']:.1f} minutes")
print(f"Peak GPU memory allocation: {runtime_metrics['peak_cuda_memory_allocated_gib']:.3f} GiB")
"""
            ),
            markdown(
                r"""
## 3. Inspect structure correspondences

The three-level package workflow starts with at least seven coarse structures
and then makes only the ST partition progressively finer. The executed output
reports the exact scheduled-stage and post-continuation pair counts. Automatic
pairing uses spatial masks and the full eligible Atlas hierarchy at every
stage; no shared molecular features or structure-specific override is used.
"""
            ),
            code(
                r"""
display(result.stage_summary)

pair_columns = [
    column
    for column in (
        "cluster",
        "candidate_name",
        "pair_type",
        "align_score_gated",
        "dice",
        "area_sim",
        "chamfer_dist",
        "asd",
    )
    if column in result.matched_pairs
]
display(result.matched_pairs[pair_columns].head(20))
"""
            ),
            markdown(
                r"""
## 4. Before and after structure-guided S-LDDMM

The left panel shows the globally pre-aligned ST point cloud. The right panel
shows the final coordinates after coarse-to-fine S-LDDMM. Every matched ST
cluster and its paired Allen structure use the same fixed, structure-name
color. The palette preserves the clearer colors from the validated paper
notebook instead of changing when cluster numbers or pair-table order change.
Unmatched atlas regions are light gray and unmatched ST cells are dark gray.
"""
            ),
            code(
                r"""
structure_color_map = spAlignDE.load_atlas_structure_color_map()
fig, axes = spAlignDE.plot_st_atlas_alignment(
    result,
    cluster_key="cluster",
    structure_color_map=structure_color_map,
    point_size=1.0,
    figsize=(14, 7),
)
fig.savefig(FIGURE_DIR / "st_to_allen_before_after.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "st_to_allen_before_after.svg", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "st_to_allen_before_after.pdf", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## 5. Transfer Allen labels to aligned ST cells

After alignment, each final `(x_aligned, y_aligned)` location samples the
annotation image. Atlas ID 0 is retained as background/unlabeled instead of
being replaced by a nearby region. The executed summary below reports the
fresh package result rather than loading cached paper coordinates.

The two panels use the original notebook's fixed label-color table: label 0 is
white, every Allen label has one reproducible color, and transferred ST labels
reuse exactly those colors.
"""
            ),
            code(
                r"""
transfer_summary = pd.Series(
    {
        "all ST cells": result.adata.n_obs,
        "non-background Allen label": int(result.adata.obs["atlas_label_transferred"].sum()),
        "background / unlabeled": int((~result.adata.obs["atlas_label_transferred"]).sum()),
        "unlabeled fraction": float((~result.adata.obs["atlas_label_transferred"]).mean()),
    },
    name="value",
)
display(transfer_summary.to_frame())

color_map_path = result.output_dir / (
    f"atlas_z{result.atlas.slice_index}_white_label_color_map_for_transfer_labels.csv"
)
label_color_map = spAlignDE.load_atlas_label_color_map(
    color_map_path,
    atlas=result.atlas,
)
fig, axes = spAlignDE.plot_atlas_label_transfer(
    result,
    color_map=label_color_map,
    point_size=2.0,
    point_alpha=0.8,
    figsize=(14, 7),
)
fig.savefig(FIGURE_DIR / "st_to_allen_label_transfer.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "st_to_allen_label_transfer.svg", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "st_to_allen_label_transfer.pdf", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## 6. Output contract

The returned AnnData preserves `X` and `obsm["spatial"]` and adds:

- `obs["x_prealigned"]`, `obs["y_prealigned"]` — whole-mask IoU result;
- `obs["x_aligned"]`, `obs["y_aligned"]` — final S-LDDMM coordinates;
- `obs["atlas_label_id"]`, `obs["atlas_label_acronym"]`,
  `obs["atlas_label_name"]`, `obs["atlas_label_transferred"]` — sampled Allen
  annotation; and
- `uns["spAlignDE"]["st_to_allen_atlas"]` — parameters and provenance for a
  fresh package run.

`result.matched_pairs` and `result.stage_summary` retain the structure-level
QC tables. A fresh run also writes the aligned H5AD, pair tables, hierarchy
table, and PNG/PDF alignment QC files under the chosen `output_dir`.
"""
            ),
            code(
                r"""
coordinate_columns = ["x_prealigned", "y_prealigned", "x_aligned", "y_aligned"]
label_columns = [
    "atlas_label_id",
    "atlas_label_acronym",
    "atlas_label_name",
    "atlas_label_transferred",
]
display(result.adata.obs[coordinate_columns + label_columns].head())

metrics_path = result.output_dir / "end_to_end_runtime_cuda_metrics.json"
if metrics_path.is_file():
    metrics = json.loads(metrics_path.read_text())
    print(f"Fresh end-to-end runtime: {metrics['elapsed_minutes']:.1f} minutes")
    print(f"Peak GPU memory allocation: {metrics['peak_cuda_memory_allocated_gib']:.3f} GiB")
"""
            ),
        ]
    )


def write_notebook(result, destinations):
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(result, destination)
        print(destination)


def main():
    single = single_clustering_notebook()
    atlas = atlas_alignment_notebook()
    write_notebook(
        single,
        (
            PROJECT_ROOT / "source_notebooks" / "clustering" / "clustering_single_nb.ipynb",
            PROJECT_ROOT / "tutorials" / "cross_modality" / "atlas" / "01_single_clustering.ipynb",
            SPHINX_SOURCE / "source_notebooks" / "clustering" / "clustering_single_nb.ipynb",
        ),
    )
    write_notebook(
        atlas,
        (
            PROJECT_ROOT / "source_notebooks" / "cross_modal_atlas_alignment_nb.ipynb",
            PROJECT_ROOT / "tutorials" / "cross_modality" / "atlas" / "02_st_to_allen_atlas.ipynb",
            SPHINX_SOURCE / "source_notebooks" / "cross_modal_atlas_alignment_nb.ipynb",
        ),
    )


if __name__ == "__main__":
    main()
