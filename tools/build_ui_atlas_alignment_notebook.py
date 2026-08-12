#!/usr/bin/env python3
"""Build the UI-export-to-Allen-CCF alignment source notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPHINX_SOURCE = PROJECT_ROOT / "docs" / "source"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook():
    result = nbf.v4.new_notebook(
        cells=[
            markdown(
                r"""
# UI-curated ST-to-Allen-CCF alignment — MERFISH S2R1

This notebook consumes the pairing CSV downloaded from the spAlignDE
interactive region-pairing tool and runs ST-to-Allen alignment directly. The
example aligns adult mouse-brain MERFISH S2R1 (83,546 cells) to Allen CCF
coronal slice 675.

The UI output is the authoritative pair specification. This interface does
**not** rediscover candidates, calculate pair scores/gates or run pair
matching: each `group_id` is accepted directly as one many-to-many S-LDDMM
channel formed by the union of its selected ST clusters and Allen labels.
Individual CSV rows are selection records, not independent deformation
channels.

| Skipped because the UI already supplies pairs | Still performed for alignment |
|---|---|
| automatic Atlas candidate discovery | validate the CSV, slice and cluster IDs |
| pair metrics, score thresholds and gating | whole-mask pre-alignment, or consume separately prepared manual pre-alignment coordinates |
| greedy/non-overlap pair selection | point filtering and ST/Atlas mask construction |
| continuation-time pair rematching | mask cleanup, signed-distance conversion and global channel weighting |
| — | S-LDDMM input/grid construction, optimization, point mapping and Allen-label transfer |

The default example computes whole-mask pre-alignment. If manual
pre-alignment has already written `obs["x_prealigned"]` and
`obs["y_prealigned"]`, set `prealignment_mode="provided"`; every downstream
mask, processing and S-LDDMM stage remains unchanged.
"""
            ),
            markdown(
                r"""
## Installation and required inputs

From the repository root:

```bash
cd /path/to/spAlignDE
python -m pip install -e ".[clustering,atlas,tutorial]"
```

| Role | Input | Required contract |
|---|---|---|
| Query ST | clustered AnnData/H5AD | `obsm["spatial"]` and `obs["cluster"]`; the paper example is the seed-1234 output of the preceding single-clustering notebook |
| Curated correspondences | `spalign_de_experimental_pairs.csv` | the validated CSV for those fixed-seed labels; retain `group_id`, panel dataset kinds, selected IDs and `atlas_z_slice` |
| Atlas annotation | `annotation_10.nrrd` | the same Allen CCF release and slice used in the UI |
| Atlas hierarchy | `voxel_count_and_differences.csv` | Allen annotation IDs, names, hierarchy paths and colors |

Create the pairing CSV first with the **Interactive region pairing and
refinement** source notebook linked from the Cross-Modality documentation. The
included UI guide explains how to select raw or custom regions, place several
selections in one group and export the result.

The example files under `tutorials/cross_modality/atlas/data/` preserve the
Allen selections from the paper run, with ST IDs remapped by cell overlap to
the seed-1234 clustering output. For a new dataset, set `SPALIGNDE_CLUSTERED_ST_H5AD`,
`SPALIGNDE_UI_PAIRING_CSV` and `SPALIGNDE_ALLEN_CCF_DIR`. The UI display's
flip/rotation fields are retained as provenance. They do not replace the
chosen global initialization: use `prealignment_mode="mask"` to recompute it,
or `"provided"` to consume manual pre-aligned coordinates.
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook from the spAlignDE repository.")


PROJECT_ROOT = find_project_root(Path.cwd())
ATLAS_TUTORIAL_DIR = PROJECT_ROOT / "tutorials" / "cross_modality" / "atlas"
DATA_DIR = ATLAS_TUTORIAL_DIR / "data"
OUTPUT_DIR = ATLAS_TUTORIAL_DIR / "output" / "ui_paired_alignment"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

clustered_default = ATLAS_TUTORIAL_DIR / "output" / "merfish_S2R1_single_clustered.h5ad"
CLUSTERED_PATH = Path(
    os.environ.get("SPALIGNDE_CLUSTERED_ST_H5AD", clustered_default)
).expanduser()
PAIRING_PATH = Path(
    os.environ.get(
        "SPALIGNDE_UI_PAIRING_CSV",
        DATA_DIR / "spalign_de_experimental_pairs.csv",
    )
).expanduser()

public_atlas_dir = PROJECT_ROOT / "data" / "allen_ccf_2022"
ATLAS_DIR = Path(
    os.environ.get("SPALIGNDE_ALLEN_CCF_DIR", public_atlas_dir)
).expanduser()

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

print("Clustered ST file:", CLUSTERED_PATH.name)
print("UI pairing file:", PAIRING_PATH.name)
print("Allen CCF inputs:", [path.name for path in required_atlas_files])
"""
            ),
            markdown(
                r"""
## Load the fixed-seed ST labels used by this tutorial

Pairing IDs are categorical identifiers, not transferable biological names.
Changing the clustering seed, refinement or resolution can renumber clusters
even when cell coordinates are unchanged. The checked-in pairing CSV was
therefore revalidated against the seed-1234 clustering output: Allen regions
and deformation groups are unchanged, while ST IDs were mapped by cell
overlap. The mapping retains 94.7--99.8% of the cells in each original UI
group; one split group is represented by the union of fixed-seed clusters 2
and 13. A user-supplied H5AD must likewise use a pairing CSV created or
revalidated for its own labels.
"""
            ),
            code(
                r"""
adata = spAlignDE.load_single_sample_data(CLUSTERED_PATH)
spAlignDE.validate_single_sample_anndata(
    adata,
    cluster_key="cluster",
    require_cluster=True,
)
print("Using fixed-seed cluster labels directly from the supplied AnnData.")

atlas = spAlignDE.load_allen_ccf_reference(
    ATLAS_DIR / "annotation_10.nrrd",
    ATLAS_DIR / "voxel_count_and_differences.csv",
    slice_index=675,
)

print(f"Query: {adata.n_obs:,} cells; {adata.obs['cluster'].nunique()} clusters")
print(f"Reference: Allen CCF slice {atlas.slice_index}; image shape {atlas.shape}")
"""
            ),
            markdown(
                r"""
## Inspect and validate the UI export

`load_ui_atlas_pairing` determines which panel is ST from the retained
`*_dataset_kind` columns, checks the Allen slice, expands custom regions back
to their original selected IDs and reconstructs the many-to-many groups. This
is parsing and validation only; it does not score or rematch a pair.

This example contains 54 selection rows, 9 deformation groups and 12 ST
cluster-level records for plotting and provenance. Several rows may describe
the same group because a group can contain multiple ST clusters and multiple
Allen structures.
"""
            ),
            code(
                r"""
pairing = spAlignDE.load_ui_atlas_pairing(
    PAIRING_PATH,
    expected_atlas_slice=atlas.slice_index,
)

ui_cluster_ids = {
    str(cluster_id)
    for values in pairing.deformation_groups["st_cluster_ids"]
    for cluster_id in values
}
available_cluster_ids = set(adata.obs["cluster"].astype(str))
missing_cluster_ids = sorted(ui_cluster_ids.difference(available_cluster_ids))
if missing_cluster_ids:
    raise ValueError(
        "The AnnData cluster labels do not match the UI export. "
        f"Missing IDs: {missing_cluster_ids}"
    )

pairing_summary = pd.Series(
    {
        "raw UI rows": len(pairing.raw),
        "S-LDDMM deformation groups": len(pairing.deformation_groups),
        "ST cluster-level output pairs": len(pairing.matched_pairs),
        "ST panel": pairing.st_side,
        "Allen slice": pairing.atlas_slice_index,
    },
    name="value",
)
display(pairing_summary.to_frame())
display(
    pairing.deformation_groups[
        ["group_id", "st_cluster_ids", "atlas_labels_union", "candidate_name"]
    ]
)
"""
            ),
            markdown(
                r"""
## Configure the UI-curated S-LDDMM run

The values below reproduce the validated UI-based example. Public names are
shown next to their legacy notebook symbols:

- `kernel_scale=200` (`a`) controls the spatial scale and smoothness of the
  deformation. Increase it for a broader, smoother field; decrease it only
  when reproducible local bends remain unresolved.
- `velocity_grid_spacing=50` (`grid_step`) controls deformation resolution.
  Smaller spacing adds control points and costs more GPU memory; larger
  spacing is faster and more regularized.
- `time_steps=5` (`nt`) controls numerical integration accuracy.
- `iterations=500` (`niter`) is increased only when the energy is still
  falling at the final iteration.
- `restore_best_checkpoint=False` preserves the transformation at iteration
  500 and reproduces the validated paper run. For exploratory data, setting it
  to `True` returns the lowest-energy checkpoint instead.

Mask preprocessing, signed-distance scaling and area balancing are global:
they apply identically to every region group. There is no hippocampus- or
structure-name-specific weight. Inspect group masks and whole-tissue
pre-alignment before changing deformation parameters.
"""
            ),
            code(
                r"""
config = spAlignDE.UIAtlasAlignmentConfig(
    prealignment_mode="mask",  # use "provided" for manual x/y_prealigned columns
    kernel_scale=200,
    time_steps=5,
    velocity_grid_spacing=50,
    iterations=500,
    momentum_learning_rate=2000,
    raster_zoom_scale=0.6,
    restore_best_checkpoint=False,
    device=None,  # CUDA when available, otherwise CPU
    verbose=True,
)

display(
    pd.Series(
        {
            "prealignment_mode": config.prealignment_mode,
            "kernel_scale (a)": config.kernel_scale,
            "velocity_grid_spacing (grid_step)": config.velocity_grid_spacing,
            "time_steps (nt)": config.time_steps,
            "iterations (niter)": config.iterations,
            "momentum_learning_rate (lrM)": config.momentum_learning_rate,
            "restore_best_checkpoint": config.restore_best_checkpoint,
        },
        name="value",
    ).to_frame()
)
"""
            ),
            markdown(
                r"""
### Optional manual pre-alignment handoff

Manual pre-alignment is an alternative global initialization, not another pair
matching step. After an interactive/manual transform has populated the
standard coordinate columns, change only the configuration:

```python
manual_config = spAlignDE.UIAtlasAlignmentConfig(
    prealignment_mode="provided",
    provided_prealigned_x_key="x_prealigned",
    provided_prealigned_y_key="y_prealigned",
    kernel_scale=200,
    velocity_grid_spacing=50,
    iterations=500,
)
```

The function then skips whole-mask transform estimation but still filters
points, rebuilds ST and Allen masks, applies mask processing, constructs the
S-LDDMM input channels and runs the final deformation. The executable paper
example below continues with `prealignment_mode="mask"`.
"""
            ),
            markdown(
                r"""
## Run alignment directly from the exported CSV

No notebook-local alignment function or subprocess is required. The package
uses the selected global initialization, then constructs one unioned
signed-distance channel per valid UI group, performs one S-LDDMM stage, maps
all cells and samples Allen labels at the final coordinates. It never invokes
the automatic pair-matching pipeline.
"""
            ),
            code(
                r"""
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

started = time.perf_counter()
result = spAlignDE.align_st_to_allen_atlas_from_ui_pairs(
    adata,
    atlas,
    pairing,
    config=config,
    cluster_key="cluster",
    spatial_key="spatial",
    output_dir=OUTPUT_DIR,
    save_outputs=True,
)

if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed_seconds = time.perf_counter() - started
peak_gib = (
    torch.cuda.max_memory_allocated() / 1024**3
    if torch.cuda.is_available()
    else None
)
runtime_metrics = {
    "elapsed_seconds": elapsed_seconds,
    "elapsed_minutes": elapsed_seconds / 60,
    "peak_cuda_memory_allocated_gib": peak_gib,
    "n_cells": result.adata.n_obs,
    "n_ui_rows": len(pairing.raw),
    "n_ui_groups": len(pairing.deformation_groups),
    "n_output_pairs": len(result.matched_pairs),
}
(OUTPUT_DIR / "notebook_runtime_metrics.json").write_text(
    json.dumps(runtime_metrics, indent=2)
)

display(result.stage_summary)
print(f"Aligned observations: {result.adata.n_obs:,}")
print(f"Wall-clock runtime: {elapsed_seconds / 60:.2f} minutes")
if peak_gib is not None:
    print(f"Peak CUDA memory allocated: {peak_gib:.3f} GiB")
"""
            ),
            markdown(
                r"""
## Review the curated correspondence table

The result table has one row per selected ST cluster, while
`n_manual_groups` reports the actual number of S-LDDMM channels. Use the saved
`manual_pairs_grouped_valid_for_lddmm.csv` to audit which requested ST and
Allen IDs were present in this run. `result.matched_pairs` is retained as the
standard result-field name, but here it is a normalized copy of UI-accepted
pairs—not the output of a new matching calculation. The stage summary records
`pair_matching_ran=False` explicitly.
"""
            ),
            code(
                r"""
pair_columns = [
    "group_id",
    "cluster",
    "st_cluster_ids",
    "atlas_labels_union",
    "candidate_name",
    "pair_type",
]
display(result.matched_pairs[pair_columns])
"""
            ),
            markdown(
                r"""
## Before and after UI-curated alignment

The left panel is the package-recomputed whole-mask pre-alignment. The right
panel is the final S-LDDMM result. ST clusters that share one UI group undergo
one common union-mask constraint; the plot retains cluster-level rows so the
selected structures remain interpretable.
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
fig.savefig(FIGURE_DIR / "ui_st_to_allen_before_after.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "ui_st_to_allen_before_after.svg", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## Transfer Allen labels to aligned cells

Each final `(x_aligned, y_aligned)` point samples Allen slice 675. Atlas label
0 remains background/unlabeled. The annotation and transferred-cell panels
use the same deterministic Allen label-ID palette.
"""
            ),
            code(
                r"""
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
fig.savefig(FIGURE_DIR / "ui_st_to_allen_label_transfer.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "ui_st_to_allen_label_transfer.svg", bbox_inches="tight")
plt.show()

transfer_summary = pd.Series(
    {
        "all ST cells": result.adata.n_obs,
        "non-background Allen label": int(
            result.adata.obs["atlas_label_transferred"].sum()
        ),
        "background / unlabeled": int(
            (~result.adata.obs["atlas_label_transferred"]).sum()
        ),
    },
    name="value",
)
display(transfer_summary.to_frame())
"""
            ),
            markdown(
                r"""
## Output contract and troubleshooting

The returned AnnData preserves `X` and `obsm["spatial"]` and adds
`x_prealigned`, `y_prealigned`, `x_aligned`, `y_aligned` plus Allen label
columns to `obs`. Provenance is stored under
`uns["spAlignDE"]["st_to_allen_atlas"]` with
`pairing_mode="ui_curated"`. The output directory also contains the aligned
H5AD, raw/grouped/valid pair tables, coordinate table, filtering statistics
and alignment figures.

Common failures:

- **Missing ST IDs:** use the exact clustered AnnData uploaded to the UI; do
  not assume a rerun preserves integer cluster labels.
- **Atlas slice mismatch:** reload the slice recorded in `atlas_z_slice`; do
  not silently edit the CSV.
- **A group is skipped:** inspect
  `manual_pairs_grouped_valid_for_lddmm.csv`; a requested ST mask or Allen ID
  may be absent after filtering/on the selected slice.
- **Poor global placement:** improve whole-mask pre-alignment or provide
  manually pre-aligned coordinates before decreasing `kernel_scale` or grid
  spacing.
- **Overly local deformation:** increase `kernel_scale` or
  `velocity_grid_spacing`; keep weighting changes global rather than assigning
  anatomy-specific exceptions.
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
print("Pairing mode:", result.adata.uns["spAlignDE"]["st_to_allen_atlas"]["pairing_mode"])
print("Saved output:", result.output_dir)
"""
            ),
        ]
    )
    result.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    result.metadata["language_info"] = {"name": "python", "version": "3.10"}
    return result


def main():
    result = build_notebook()
    destinations = (
        PROJECT_ROOT / "source_notebooks" / "cross_modality" / "ui_paired_atlas_alignment_nb.ipynb",
        PROJECT_ROOT / "tutorials" / "cross_modality" / "atlas" / "03_ui_paired_alignment.ipynb",
        SPHINX_SOURCE / "source_notebooks" / "cross_modality" / "ui_paired_atlas_alignment_nb.ipynb",
    )
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(result, destination)
        print(destination)


if __name__ == "__main__":
    main()
