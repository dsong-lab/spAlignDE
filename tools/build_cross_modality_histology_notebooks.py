#!/usr/bin/env python3
"""Build the ST-to-histology executable notebooks."""

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
    result.metadata["spAlignDE_reproducibility"] = {
        "workflow_seed": 0,
        "seed_scope": "Python, NumPy, Torch and configured stochastic methods",
    }
    return result


COMMON_SETUP = r"""
%matplotlib inline

from pathlib import Path
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from PIL import Image

import spAlignDE

WORKFLOW_SEED = 0
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
DATA_DIR = Path(
    os.environ.get(
        "SPALIGNDE_HISTOLOGY_DATA_DIR",
        PROJECT_ROOT / "data" / "cross_modality" / "histology" / "mouse_brain",
    )
).expanduser()
OUTPUT_DIR = PROJECT_ROOT / "tutorials" / "cross_modality" / "histology" / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
"""


def feature_extraction_notebook():
    return notebook(
        [
            markdown(
                r"""
# H&E image preparation and feature extraction

This is the first notebook in the ST-to-histology workflow. The histology-side
dataset input is **one high-resolution image**. Spatial coordinates, a Visium
matrix, spot locations, and gene expression from the reference section are not
used to estimate the alignment.

By default this notebook performs feature extraction itself. spAlignDE prepares
the image, runs the hierarchical HIPT encoders with shifted tilings, smooths
the image-feature fields, and saves the result consumed by the clustering
notebook. Documentation maintainers may set
`SPALIGNDE_DOCUMENTATION_FEATURE_DIR` to re-render the page from a previously
completed and validated feature-extraction result; public users do not need
that variable.
"""
            ),
            markdown(
                r"""
## Pipeline implemented here

1. **Standardize the image.** Convert to RGB, read OME physical pixel size when
   available, rescale to 0.5 μm per pixel, and add white padding on the right
   and bottom to a multiple of 224 pixels. A JPEG already at the requested
   scale and compatible size is copied byte-for-byte to avoid re-encoding.
2. **Encode local morphology.** Split each 4,096-pixel context tile into
   256 × 256-pixel patches. ViT-256 produces 384-dimensional tokens on a
   16 × 16-pixel image grid.
3. **Encode larger tissue context.** Stitch the ViT-256 patch representations
   and pass them through ViT-4K, producing 192 context channels on the same
   feature grid.
4. **Reduce tile-boundary artifacts.** Repeat extraction at offsets
   0, 64, 128, and 192 pixels along both axes (16 shifted views) and average
   the overlapping fields. Apply 16 × 16 and 4 × 4 uniform smoothing windows
   to the context and subpatch channels, respectively.
5. **Add image color and save.** Mean-downsample RGB by 16 and write one
   dictionary containing `cls`, `sub`, and `rgb` feature groups.

Thus, one output coordinate corresponds to a 16 × 16-pixel region of the
prepared image. All later histology clustering is performed on this image-only
feature field.
"""
            ),
            markdown(
                r"""
## Data, source code, and pretrained models

The following resources are sufficient to reproduce this feature-extraction
stage. The Xenium query is listed for completeness but is **not read in this
notebook**.

| Resource | Source | What to obtain |
|---|---|---|
| Reference H&E image | [10x Visium CytAssist Post-Xenium Mouse Brain](https://www.10xgenomics.com/datasets/visium-cytassist-gene-expression-libraries-of-post-xenium-mouse-brain-ff-using-the-mouse-whole-transcriptome-probe-set-2-standard) | Control Replicate 1 full-resolution tissue image; the vendor file used here is `CytAssist_FreshFrozen_Mouse_Brain_Rep1_tissue_image.btf`. |
| Query ST data, used in notebook 3 | [10x Fresh Frozen Mouse Brain Xenium Replicates](https://www.10xgenomics.com/datasets/fresh-frozen-mouse-brain-replicates-1-standard) | Replicate 1 Xenium output bundle. It is not an input to image feature extraction. |
| spAlignDE feature extractor | This installed package | `spAlignDE.extract_histology_features()` implements dense stitching, shifted-view averaging, smoothing, and output serialization. No separate extractor package is required. |
| HIPT model and method | [MahmoodLab/HIPT](https://github.com/mahmoodlab/HIPT) | Official hierarchical ViT implementation and model description. |
| HIPT ViT-256 weights | [official `vit256_small_dino.pth`](https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit256_small_dino.pth) | DINO-pretrained ViT-S/16 checkpoint, approximately 672 MB. |
| HIPT ViT-4K weights | [official `vit4k_xs_dino.pth`](https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/Checkpoints/vit4k_xs_dino.pth) | DINO-pretrained ViT-XS/256 checkpoint, approximately 378 MB. |
| Optional UNI model | [MahmoodLab/UNI code](https://github.com/mahmoodlab/UNI) and [gated Hugging Face weights](https://huggingface.co/MahmoodLab/UNI) | Optional ViT-L/16 feature model. UNI access requires accepting its terms with an eligible individual account. It is not required for the paper H&E alignment. |

HIPT and UNI have their own licenses and model-use terms. Review those terms
before downloading or redistributing code, checkpoints, or derived models.

### Install spAlignDE and prepare HIPT

From the repository root:

```bash
cd /path/to/spAlignDE
python -m pip install -e ".[histology,tutorial]"

mkdir -p external
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/mahmoodlab/HIPT.git external/HIPT
cd external/HIPT
git lfs pull --include "*.pth"
cd /path/to/spAlignDE

export SPALIGNDE_HIPT_DIR=/path/to/spAlignDE/external/HIPT
```

Git LFS retrieves the two pretrained checkpoints released in the official HIPT
repository. Before running the notebook, verify this layout:

```text
/path/to/spAlignDE/
├── data/cross_modality/histology/mouse_brain/
│   └── CytAssist_FreshFrozen_Mouse_Brain_Rep1_tissue_image.btf
└── external/HIPT/
    └── HIPT_4K/
        ├── hipt_4k.py
        ├── hipt_model_utils.py
        ├── vision_transformer.py
        ├── vision_transformer4k.py
        └── Checkpoints/
            ├── vit256_small_dino.pth
            └── vit4k_xs_dino.pth
```

The HIPT environment requires compatible PyTorch and torchvision builds. By
default the extractor uses the notebook's Python executable. If HIPT is
installed in a separate compatible environment, set:

```bash
export SPALIGNDE_HIPT_PYTHON=/path/to/hipt-environment/bin/python
```

A CUDA-capable GPU is strongly recommended because shifted extraction runs 16
HIPT views and the BTF-derived feature file is approximately 2.38 GB.

### Place the input image

Place the vendor image at
`data/cross_modality/histology/mouse_brain/CytAssist_FreshFrozen_Mouse_Brain_Rep1_tissue_image.btf`,
or point to any supported high-resolution image:

```bash
export SPALIGNDE_HISTOLOGY_IMAGE=/path/to/high_resolution_image.btf
```

Accepted inputs are RGB or RGB-convertible JPEG, PNG, TIFF, OME-TIFF, and BTF
images readable by Pillow/tifffile. OME physical pixel size is used when
available. Different source crops or white canvas sizes produce different
feature-grid dimensions. The fixed-seed release run reads the vendor BTF's
0.27381 μm-per-pixel metadata, rescales it to 0.5 μm per pixel, and pads to a
12,768 × 20,608 analysis image, giving a 1,288 × 798 grid.

Coordinates in all downstream image outputs are `(x, y)` feature-grid
coordinates with origin at the upper left. One feature-grid unit represents a
16 × 16-pixel region of the prepared image.
"""
            ),
            code(COMMON_SETUP),
            code(
                r"""
IMAGE_PATH = Path(
    os.environ.get(
        "SPALIGNDE_HISTOLOGY_IMAGE",
        DATA_DIR / "CytAssist_FreshFrozen_Mouse_Brain_Rep1_tissue_image.btf",
    )
).expanduser()
FEATURE_DIR = OUTPUT_DIR / "feature_extraction"

if not IMAGE_PATH.is_file():
    raise FileNotFoundError(
        f"Histology image not found: {IMAGE_PATH}\n"
        "Download the public Replicate 1 image or set SPALIGNDE_HISTOLOGY_IMAGE."
    )

print("Histology input file:", IMAGE_PATH.name)
print("Only the image is used on the reference side.")
"""
            ),
            markdown(
                r"""
## Run image preparation and HIPT feature extraction

The public function below executes all five stages described above. It creates
the feature file from `IMAGE_PATH` on every fresh run; there is no input option
for a precomputed feature file in this notebook.
"""
            ),
            code(
                r"""
completed_feature_dir = os.environ.get("SPALIGNDE_DOCUMENTATION_FEATURE_DIR")
if completed_feature_dir:
    import json

    completed_feature_dir = Path(completed_feature_dir).expanduser()
    features = spAlignDE.load_histology_features(
        completed_feature_dir / "he.jpg",
        completed_feature_dir / "embeddings-hist-vit.pickle",
    )
    manifest_path = completed_feature_dir / "histology_feature_manifest.json"
    features.manifest.update(json.loads(manifest_path.read_text()))
    execution_mode = "validated feature-stage output"
else:
    feature_config = spAlignDE.HistologyFeatureConfig(
        target_microns_per_pixel=0.5,
        shifted_tiles=True,
        device=None,  # automatically select CUDA when available
    )
    features = spAlignDE.extract_histology_features(
        IMAGE_PATH,
        FEATURE_DIR,
        config=feature_config,
    )
    execution_mode = "fresh package extraction"

display(
    pd.Series(
        {
            "mode": execution_mode,
            "input image": features.source_image_path.name,
            "prepared image": features.prepared_image_path.name,
            "prepared image size (width, height)": features.image_size_wh,
            "feature grid (height, width)": features.feature_grid_shape_hw,
            "feature file": features.feature_path.name,
            "feature size (GB)": round(features.feature_path.stat().st_size / 1e9, 3),
        },
        name="value",
    ).to_frame()
)
"""
            ),
            markdown(
                r"""
## Inspect image preparation

The preparation information describes how the downloaded image was converted
to `he.jpg`. If the source provides physical pixel size, `resize_scale` records
the conversion to 0.5 μm per pixel; otherwise spAlignDE preserves the native
resolution and reports the output physical scale as unavailable.
"""
            ),
            code(
                r"""
preparation_fields = [
    "preparation_mode",
    "original_size_wh",
    "resized_size_wh",
    "padded_size_wh",
    "source_microns_per_pixel",
    "output_microns_per_pixel",
    "resize_scale",
    "pad_right_px",
    "pad_bottom_px",
]
display(
    pd.Series(
        {field: features.manifest[field] for field in preparation_fields},
        name="value",
    ).to_frame()
)

with Image.open(features.prepared_image_path) as image:
    image.draft("RGB", (1400, 1400))
    prepared_preview = image.convert("RGB")
    prepared_preview.thumbnail((1400, 1400))

fig, ax = plt.subplots(figsize=(7, 8))
ax.imshow(prepared_preview)
ax.set_title("Prepared H&E image used by HIPT")
ax.axis("off")
plt.show()
"""
            ),
            markdown(
                r"""
## Inspect the extracted feature field

The table below is generated from the saved extraction information rather than
from an expected hard-coded display. For the vendor BTF release input, each channel
has shape 1,288 × 798. `cls` is the 192-channel ViT-4K context field, `sub` is the
384-channel ViT-256 local-morphology field, and `rgb` contains three
mean-downsampled color channels. The next notebook reads these 579
image-derived channels and clusters them into spatial structures.
"""
            ),
            code(
                r"""
geometry = features.manifest["extraction_geometry"]
display(pd.Series(geometry, name="value").to_frame())

feature_schema = pd.DataFrame(features.manifest["feature_schema"]).T
feature_schema.index.name = "feature group"
display(feature_schema)
"""
            ),
            markdown(
                r"""
## What is stored in `embeddings-hist-vit.pickle`?

The file is a Python dictionary. It contains dense image-feature maps, not a
cell-by-feature table and not an AnnData object:

```text
{
  "cls": list of 192 float32 arrays, each shaped (H, W),
  "sub": list of 384 float32 arrays, each shaped (H, W),
  "rgb": one float32 array shaped (3, H, W)
}
```

For the executed vendor BTF, `(H, W) = (1288, 798)`. Feature location
`[y, x]` summarizes prepared-image pixels
`y*16:(y+1)*16, x*16:(x+1)*16`. The pickle therefore contains spatially
dense morphology and color fields, but no gene expression, cell identity,
Visium spot coordinates, or Xenium coordinates.

Because the pickle is large, inspect `histology_feature_manifest.json` first;
loading the pickle with `pickle.load` reads the full object into memory. The
accompanying record lists the schema, grid size, preparation settings,
checkpoints, execution route and file size.

### Optional UNI and merged features

spAlignDE can also be extended with UNI features after separate dimension
reduction:

```text
embeddings-hist-uni.pickle
  his: 1024 UNI channels on the H × W grid
  rgb: 3 RGB channels on the H × W grid
  pos: 2 positional channels on the H × W grid

embeddings-hist-merged.pickle
  vit: reduced HIPT/ViT tensor
  uni: reduced UNI tensor
```

UNI is an **optional advanced representation**. The paper result reproduced
by these notebooks intentionally uses the unmasked HIPT/ViT-only
`embeddings-hist-vit.pickle`; no Hugging Face login or UNI weights are needed.
Do not replace the input to notebook 2 with a merged feature file when trying
to reproduce the reported 26-cluster merge target and 21 final cleaned
structures.
"""
            ),
            markdown(
                r"""
## Saved output and next step

The stage writes:

- `he.jpg`: RGB image at the analysis resolution, padded to a multiple of 224;
- `embeddings-hist-vit.pickle`: the newly extracted image-only HIPT feature
  field (`192 cls + 384 sub + 3 rgb` channels); and
- `histology_image_preparation.json`: source-image, physical-resolution,
  resizing, padding, and prepared-image details; and
- `histology_feature_manifest.json`: model route, tiling/shift/smoothing
  parameters, feature schema and file information.

No spatial-transcriptomic values are present in these outputs. Continue with
**H&E image-feature clustering — 21 final histology structures**.
"""
            ),
        ]
    )


def feature_clustering_notebook():
    return notebook(
        [
            markdown(
                r"""
# H&E image-feature clustering into 21 final structures

This second notebook converts the HIPT feature field into coherent
histology-derived spatial structures. It reproduces the H&E branch validated
in `test0730/he_rep1`: 30 initial tissue clusters are consolidated first to a
26-cluster merge target and then by reflection-aware and post-symmetry merging
into 21 final cleaned regions. These labels are derived from the image only.
"""
            ),
            markdown(
                r"""
## Inputs from notebook 1

Run the image-feature notebook first. Its default outputs are
`tutorials/cross_modality/histology/output/feature_extraction/he.jpg` and
`embeddings-hist-vit.pickle`. During documentation validation, explicit image,
feature, or completed clustering-stage paths can be supplied with environment
variables; public users do not need those variables.
"""
            ),
            code(COMMON_SETUP),
            code(
                r"""
FEATURE_DIR = OUTPUT_DIR / "feature_extraction"
IMAGE_PATH = Path(
    os.environ.get("SPALIGNDE_HISTOLOGY_IMAGE", FEATURE_DIR / "he.jpg")
).expanduser()
FEATURE_PATH = Path(
    os.environ.get(
        "SPALIGNDE_DOCUMENTATION_FEATURE_PATH",
        FEATURE_DIR / "embeddings-hist-vit.pickle",
    )
).expanduser()
CLUSTER_DIR = OUTPUT_DIR / "histology_clustering"

features = spAlignDE.load_histology_features(IMAGE_PATH, FEATURE_PATH)
print("Image:", features.prepared_image_path.name)
print("Feature field:", features.feature_path.name)
print("Feature grid:", features.feature_grid_shape_hw)
"""
            ),
            markdown(
                r"""
## Cluster, merge, and spatially clean image features

The paper profile uses `k_bg=2`, `k_slide=30`, `k_merge=26`,
`rgb_weight=0.25`, `coordinate_weight=0.05`, `random_state=0`, and a minimum
cleaned component size of 250 feature pixels. Reflection merge uses a minimum
area of 200, IoU of 0.20 and Dice of 0.30; at most three additional
symmetry-supported merges are accepted. Background is encoded as −1 and
excluded from alignment.
"""
            ),
            code(
                r"""
completed_stage = os.environ.get("SPALIGNDE_DOCUMENTATION_CLUSTER_DIR")
if completed_stage:
    histology = spAlignDE.load_histology_clustering(completed_stage)
    execution_mode = "validated clustering-stage output"
else:
    histology = spAlignDE.cluster_histology_features(
        features,
        CLUSTER_DIR,
        config=spAlignDE.HistologyClusteringConfig(
            merged_clusters=26,
            do_reflection_merge=True,
            symmetry_axis="ud",
            reflection_min_area=200,
            reflection_min_iou=0.20,
            reflection_min_dice=0.30,
            symmetry_max_merges=3,
            symmetry_min_score_gain=0.02,
            symmetry_min_reflected_dice=0.15,
            symmetry_max_centroid_distance=0.15,
            symmetry_min_feature_cosine=0.30,
            cleanup_min_size=250,
            random_state=WORKFLOW_SEED,
        ),
    )
    execution_mode = "fresh package clustering"

display(
    pd.Series(
        {
            "mode": execution_mode,
            "feature grid (height, width)": histology.shape,
            "tissue pixels": int(histology.tissue_mask.sum()),
            "initial regions": int(histology.summary["regions"]["labels_full_raw"]),
            "final cleaned regions": histology.n_structures,
        },
        name="value",
    ).to_frame()
)
"""
            ),
            markdown(
                r"""
## Quality control: image to final structures

The four panels show the evidence chain used in Supplementary Figure 8:
original H&E morphology, 30 initial feature assignments, symmetry-aware
merging, and the cleaned 21-region structure map. Colors identify image
structures and do not represent cell types or anatomical labels.
"""
            ),
            code(
                r"""
fig, axes = spAlignDE.plot_histology_feature_clusters(
    histology,
    figsize=(13, 8),
)
fig.savefig(FIGURE_DIR / "he_feature_clustering_21_structures.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "he_feature_clustering_21_structures.svg", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## Saved output and next step

The compact result stores the tissue mask and raw, merged, and cleaned integer
label rasters at feature-grid resolution. Continue with **Xenium Replicate 1
to H&E — pre-alignment and S-LDDMM**. The clustering output is the fixed image
reference for that notebook.
"""
            ),
        ]
    )


def alignment_notebook():
    return notebook(
        [
            markdown(
                r"""
# Xenium Replicate 1 to H&E histology

This final notebook aligns 162,033 Xenium mouse-brain cells (query) to the
21-region H&E feature map (reference). The query is Replicate 1 from the
[10x Genomics Fresh Frozen Mouse Brain Xenium dataset](https://www.10xgenomics.com/datasets/fresh-frozen-mouse-brain-replicates-1-standard).
The reference image is control Replicate 1 from the
[Visium CytAssist Post-Xenium Mouse Brain dataset](https://www.10xgenomics.com/datasets/visium-cytassist-gene-expression-libraries-of-post-xenium-mouse-brain-ff-using-the-mouse-whole-transcriptome-probe-set-2-standard).

Only the H&E image-derived structures estimate the transformation. The Visium
expression measurements paired with the image are not alignment inputs; the
paper reserves them for independent molecular-concordance evaluation.
"""
            ),
            markdown(
                r"""
## Input format and notebook order

Run these steps in order:

1. **Single clustering (CSV or AnnData input)** creates a clustered query
   AnnData with `X`, `obsm["spatial"]`, `obs["cluster_raw"]`, and
   `obs["cluster"]`.
2. **H&E image preparation and feature extraction** starts from one image.
3. **H&E image-feature clustering** creates the fixed 21-region raster.
4. This notebook constructs coarse-to-fine ST structures, estimates a global
   pre-alignment, pairs geometrically compatible masks, and runs S-LDDMM.

Set `SPALIGNDE_CLUSTERED_ST_H5AD` and
`SPALIGNDE_DOCUMENTATION_CLUSTER_DIR` only when the outputs live outside the
default tutorial directories.
"""
            ),
            code(COMMON_SETUP),
            code(
                r"""
clustered_default = DATA_DIR / "xenium_rep1_single_clustered.h5ad"
ST_PATH = Path(
    os.environ.get("SPALIGNDE_CLUSTERED_ST_H5AD", clustered_default)
).expanduser()
CLUSTER_DIR = Path(
    os.environ.get(
        "SPALIGNDE_DOCUMENTATION_CLUSTER_DIR",
        OUTPUT_DIR / "histology_clustering",
    )
).expanduser()

adata = spAlignDE.load_single_sample_data(ST_PATH)
histology = spAlignDE.load_histology_clustering(CLUSTER_DIR)
spAlignDE.validate_single_sample_anndata(
    adata,
    cluster_key="cluster",
    require_cluster=True,
)

print(f"Query: {adata.n_obs:,} Xenium cells × {adata.n_vars:,} genes")
print(f"Reference: {histology.n_structures} image-derived structures on a {histology.shape} grid")
"""
            ),
            markdown(
                r"""
## 1. Build expression-based ST structure levels

The paper hierarchy is built from the original BANKSY labels
(`cluster_raw`), not from H&E pixels or Visium expression. Cluster-average
profiles are compared after variance-based gene selection and Ward linkage.
For this dataset the resulting levels contain 7, 14, 20, and 26 structures;
the 7-structure level is used for H&E correspondence.
"""
            ),
            code(
                r"""
hierarchy_base = "cluster_raw" if "cluster_raw" in adata.obs else "cluster"
adata, hierarchy_columns = spAlignDE.build_st_histology_structures(
    adata,
    config=spAlignDE.STHistologyStructureConfig(
        n_levels=5,
        variance_fraction=0.75,
        min_genes=50,
    ),
    cluster_key=hierarchy_base,
    copy=False,
)
structure_key = next(
    (column for column in hierarchy_columns if column.endswith("_k7")),
    hierarchy_columns[0],
)

display(
    pd.DataFrame(
        {
            "structure column": hierarchy_columns,
            "number of structures": [adata.obs[column].nunique() for column in hierarchy_columns],
            "selected": [column == structure_key for column in hierarchy_columns],
        }
    )
)

fig, axes = spAlignDE.plot_st_histology_structures(
    adata,
    hierarchy_columns,
    cluster_key=hierarchy_base,
    selected_key=structure_key,
)
fig.savefig(FIGURE_DIR / "xenium_st_structure_hierarchy.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "xenium_st_structure_hierarchy.svg", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## 2. Global pre-alignment by tissue-mask overlap

The default searches rotation, isotropic scale, and translation to maximize
whole-tissue mask IoU; reflection is disabled. Always inspect the overlay.
Mask overlap may fail when tissue coverage, orientation, cropping, or tears
differ strongly between modalities. In that case, use the package's
interactive manual panel and save the selected similarity transform before
running S-LDDMM.

Set `SPALIGNDE_HISTOLOGY_PREALIGN=manual` to reproduce the paper's recorded
manual initialization (`scale=0.12`, `theta=85°`).
"""
            ),
            code(
                r"""
prealign_mode = os.environ.get("SPALIGNDE_HISTOLOGY_PREALIGN", "mask_overlap").lower()
paper_manual = spAlignDE.ManualPrealignmentConfig(
    scale=0.12,
    theta_deg=85.0,
    translation_x=787.84018089,
    translation_y=-9.30873160,
)
prealignment = spAlignDE.prealign_st_to_histology(
    adata,
    histology,
    config=spAlignDE.HistologyPrealignmentConfig(
        method=prealign_mode,
        manual=paper_manual,
    ),
    cluster_key="cluster",
)

display(pd.Series(prealignment.params, name="value").drop(labels="matrix").to_frame())

preview_config = spAlignDE.ManualPrealignmentConfig(
    scale=prealignment.params["scale"],
    theta_deg=prealignment.params["theta_deg"],
    translation_x=prealignment.params["translation_x"],
    translation_y=prealignment.params["translation_y"],
)
fig, axes = spAlignDE.plot_histology_prealignment_preview(
    adata,
    histology,
    config=preview_config,
    transformed_title=(
        "Mask-overlap pre-alignment"
        if prealignment.params["method"] == "whole_tissue_mask_overlap"
        else "Manual pre-alignment"
    ),
)
fig.savefig(FIGURE_DIR / "xenium_to_he_global_prealignment.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "xenium_to_he_global_prealignment.svg", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
### Optional interactive manual pre-alignment

If the automatic overlay is anatomically implausible, run:

```python
ui = spAlignDE.interactive_histology_prealignment(
    adata,
    histology,
    initial_config=preview_config,
)
manual_prealignment = ui.apply()
```

The sliders adjust scale, rotation, x translation, and y translation. The
selected values are written under
`adata.uns["spAlignDE"]["st_to_histology"]["prealignment"]`, making the
manual decision reproducible rather than leaving it only in a figure.
"""
            ),
            code(
                r"""
if prealignment.params.get("iou") is not None and prealignment.params["iou"] < 0.20:
    print("Low mask IoU: run the interactive manual pre-alignment cell above before continuing.")
else:
    print("Global initialization passed the default mask-overlap QC check.")
"""
            ),
            markdown(
                r"""
## 3. Structure pairing and S-LDDMM

ST and H&E structures are rasterized on the shared feature grid. Candidate
pairs use the paper weights for SDF agreement, Chamfer similarity, Dice, area,
and thickness (`0.20, 0.40, 0.15, 0.25, 0.00`). ASD is not included in the
composite score; its raw raster-space value is used only as an independent QC
gate. Pairs must have score ≥ 0.40 and ASD ≤ 30 pixels. Accepted pairs become
 continuous signed-distance channels for a single-stage S-LDDMM fit.

The run retains the final optimizer iterate (`restore_best_checkpoint=False`),
which is the package-wide default and avoids comparing energies across EM
intensity updates with different objective scales.

Users can change the five normalized weights in
`STHistologyAlignmentConfig`. Tune them only after checking the coordinate
scale, global pre-alignment, structure masks, and candidate-pair overlays. See
the website [Cross-modality pairing weights](https://dsong-lab.github.io/spAlignDE/tutorials/parameter_tuning.html#cross-modality-pairing-weights)
section for the effect of each component and a recommended tuning sequence.
"""
            ),
            code(
                r"""
result = spAlignDE.align_st_to_histology(
    prealignment,
    config=spAlignDE.STHistologyAlignmentConfig(
        pairing_weight_sdf=0.20,
        pairing_weight_chamfer=0.40,
        pairing_weight_dice=0.15,
        pairing_weight_area=0.25,
        pairing_weight_thickness=0.00,
        pair_score_threshold=0.40,
        pair_asd_threshold=30.0,
        time_steps=5,
        kernel_scale=60.0,
        velocity_grid_spacing=6.0,
        iterations=300,
        momentum_lr=2e3,
        restore_best_checkpoint=False,
        device=None,
        dtype="float64",
    ),
    cluster_key="cluster",
    structure_key=structure_key,
    output_dir=OUTPUT_DIR / "alignment",
)

pair_columns = [
    column
    for column in ("st", "he", "align_score", "dice", "area_sim", "asd")
    if column in result.matched_pairs
]
display(result.matched_pairs[pair_columns])

# Plot the exact cleaned binary channels supplied to S-LDDMM.  The final
# channel is the whole-tissue support and is intentionally omitted here.
from matplotlib.patches import Patch

source_masks = result.context["source_binary"][:-1].astype(bool)
target_masks = result.context["target_binary"][:-1].astype(bool)
source_color = (0.00, 0.45, 0.70, 1.00)
target_color = (0.90, 0.60, 0.00, 1.00)
overlap_color = (0.80, 0.47, 0.65, 1.00)

fig, axes = plt.subplots(
    len(result.matched_pairs),
    3,
    figsize=(11, 3.4 * len(result.matched_pairs)),
    constrained_layout=True,
    squeeze=False,
)
for row_index, ((_, pair), source_mask, target_mask) in enumerate(
    zip(result.matched_pairs.iterrows(), source_masks, target_masks, strict=True)
):
    union = source_mask | target_mask
    yy, xx = union.nonzero()
    pad = 18
    y0, y1 = max(0, yy.min() - pad), min(union.shape[0], yy.max() + pad + 1)
    x0, x1 = max(0, xx.min() - pad), min(union.shape[1], xx.max() + pad + 1)

    source_rgba = plt.cm.colors.to_rgba_array([source_color])[0]
    target_rgba = plt.cm.colors.to_rgba_array([target_color])[0]
    panels = []
    for mask, color in ((source_mask, source_rgba), (target_mask, target_rgba)):
        image = 1.0 * (~mask)[..., None] * (1.0, 1.0, 1.0, 1.0)
        image[mask] = color
        panels.append(image)
    overlap_image = panels[1].copy()
    overlap_image[source_mask] = source_color
    overlap_image[source_mask & target_mask] = overlap_color
    panels.append(overlap_image)

    titles = (
        "ST structure",
        "H&E structure",
        f"Paired overlap\nST {pair['st']} ↔ H&E {pair['he']}",
    )
    for axis, image, title in zip(axes[row_index], panels, titles, strict=True):
        axis.imshow(image[y0:y1, x0:x1], origin="lower", interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    axes[row_index, 2].text(
        0.5,
        -0.04,
        f"score={pair['align_score']:.3f}; Dice={pair['dice']:.3f}; ASD={pair['asd']:.1f}",
        transform=axes[row_index, 2].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

fig.legend(
    handles=(
        Patch(facecolor=source_color, label="ST mask"),
        Patch(facecolor=target_color, label="H&E mask"),
        Patch(facecolor=overlap_color, label="overlap"),
    ),
    loc="lower center",
    ncol=3,
    frameon=False,
)
fig.savefig(FIGURE_DIR / "xenium_he_paired_feature_overlap.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "xenium_he_paired_feature_overlap.svg", bbox_inches="tight")
plt.show()

# Re-rasterize the accepted ST structures after the final deformation and
# compare pair-level Dice, IoU, and ASD against the pre-aligned masks above.
fig, axes, pair_overlap_metrics = spAlignDE.plot_st_histology_pair_overlap(
    result,
    stage="after",
)
display(pair_overlap_metrics[
    ["st", "he", "stage", "dice", "iou", "asd", "hd"]
])
fig.savefig(
    FIGURE_DIR / "xenium_he_paired_feature_overlap_after_lddmm.png",
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    FIGURE_DIR / "xenium_he_paired_feature_overlap_after_lddmm.svg",
    bbox_inches="tight",
)
plt.show()
"""
            ),
            markdown(
                r"""
## 4. Before and after histology-guided alignment

Both panels show all Xenium cells in blue over the same original color H&E
image, matching the paper visualization. The first panel is the global
initialization; the second is the final structure-guided S-LDDMM result.
"""
            ),
            code(
                r"""
fig, axes = spAlignDE.plot_st_histology_alignment(
    result,
    cluster_key="cluster",
    point_size=1.0,
    alpha=0.8,
    color_by_cluster=False,
    point_color="#0066CC",
    figsize=(13, 6.5),
)
fig.savefig(FIGURE_DIR / "xenium_to_he_before_after.png", dpi=220, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "xenium_to_he_before_after.svg", bbox_inches="tight")
plt.show()
"""
            ),
            markdown(
                r"""
## 5. Standardized output

The original `X`, observation order, metadata, and `obsm["spatial"]` are
preserved. Alignment adds `x_prealigned`, `y_prealigned`, `x_aligned`, and
`y_aligned`. Run parameters are stored only under
`adata.uns["spAlignDE"]`. The final AnnData, pair table, filter summary and
JSON run summary are written to `tutorials/cross_modality/histology/output/alignment/`.
"""
            ),
            code(
                r"""
display(result.adata.obs[["x_prealigned", "y_prealigned", "x_aligned", "y_aligned"]].describe())
display(pd.Series(result.adata.uns["spAlignDE"]["st_to_histology"]["alignment"], name="value").to_frame())
print("Saved: tutorials/cross_modality/histology/output/alignment/st_to_histology_aligned.h5ad")
"""
            ),
        ]
    )


def write_notebook(result, paths):
    validated_path = paths[0]
    if validated_path.is_file():
        validated = nbf.read(validated_path, as_version=4)
        result_code = [cell for cell in result.cells if cell.cell_type == "code"]
        validated_code = [cell for cell in validated.cells if cell.cell_type == "code"]
        if len(result_code) == len(validated_code):
            for destination, source in zip(result_code, validated_code):
                destination["execution_count"] = source.get("execution_count")
                destination["outputs"] = source.get("outputs", [])
                destination["id"] = source.get("id", destination.get("id"))
                destination["metadata"] = source.get("metadata", {})
            for destination, source in zip(result.cells, validated.cells):
                if destination.cell_type == source.cell_type:
                    destination["id"] = source.get("id", destination.get("id"))
            if "spAlignDE_execution" in validated.metadata:
                result.metadata["spAlignDE_execution"] = validated.metadata[
                    "spAlignDE_execution"
                ]
            if "language_info" in validated.metadata:
                result.metadata["language_info"] = validated.metadata["language_info"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(result, path)


def main():
    source_dir = PROJECT_ROOT / "source_notebooks" / "cross_modality"
    sphinx_dir = SPHINX_SOURCE / "source_notebooks" / "cross_modality"

    write_notebook(
        feature_extraction_notebook(),
        (
            source_dir / "st_he_feature_extraction_nb.ipynb",
            sphinx_dir / "st_he_feature_extraction_nb.ipynb",
        ),
    )
    write_notebook(
        feature_clustering_notebook(),
        (
            source_dir / "st_he_feature_clustering_nb.ipynb",
            sphinx_dir / "st_he_feature_clustering_nb.ipynb",
        ),
    )
    write_notebook(
        alignment_notebook(),
        (
            source_dir / "st_he_alignment_nb.ipynb",
            sphinx_dir / "st_he_alignment_nb.ipynb",
        ),
    )


if __name__ == "__main__":
    main()
