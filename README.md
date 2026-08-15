# spAlignDE

[![Tests](https://github.com/dsong-lab/spAlignDE/actions/workflows/tests.yml/badge.svg)](https://github.com/dsong-lab/spAlignDE/actions/workflows/tests.yml)
[![Documentation](https://github.com/dsong-lab/spAlignDE/actions/workflows/docs.yml/badge.svg)](https://github.com/dsong-lab/spAlignDE/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

spAlignDE is a structure-guided framework for spatial alignment and
mismatch-aware local inference. It accepts AnnData/H5AD and standardized CSV
inputs and provides reproducible workflows across samples, modalities and
post-alignment analyses.

**[Documentation](https://dsong-lab.github.io/spAlignDE/) · [Tutorials](https://dsong-lab.github.io/spAlignDE/tutorial.html) · [Executable notebooks](https://dsong-lab.github.io/spAlignDE/source_notebooks.html)**

<p align="center">
  <a href="assets/Figure_1_08132026_DS.png">
    <img src="assets/Figure_1_08132026_DS.png" alt="Overview of the spAlignDE spatial alignment and mismatch-aware local inference framework" width="600">
  </a>
</p>
<p align="center"><em>Overview of the spAlignDE framework. Click the figure to enlarge.</em></p>

## What spAlignDE does

- cross-sample spatial-transcriptomics alignment;
- ST-to-Allen-CCF alignment and label transfer;
- ST-to-histology-image alignment through HIPT image features;
- spatial-ATAC-to-ST alignment;
- interactive many-to-many region pairing and refinement; and
- post-alignment local differential-expression inference.

## Quick start

The public import is case preserving. This deterministic CPU example checks
the installation and the complete cross-sample wrapper on a bundled synthetic
dataset:

```python
import spAlignDE

spAlignDE.set_random_seed(0)
adata = spAlignDE.make_cross_sample_example(n_per_cluster=10, random_state=0)

result = spAlignDE.align_cross_sample(
    adata,
    query_sample="query",
    reference_sample="reference",
    rasterization_config=spAlignDE.RasterizationConfig(grid_spacing=0.3),
    slddmm_config=spAlignDE.SLDDMMConfig(
        iterations=2,
        kernel_scale=1.0,
        velocity_grid_spacing=0.5,
        momentum_lr=1.0,
        minimum_momentum_lr=1.0,
        sigma_regularization=100.0,
    ),
    device="cpu",
    verbose=False,
    return_result=True,
)
result.adata.write_h5ad("cross_sample_example_aligned.h5ad")
print(result.metrics)
```

This is an API smoke test, not a parameter recommendation for biological data.
Choose an executed workflow below for dataset-scale configuration, diagnostics
and interpretation.

Single-sample input can be AnnData/H5AD or one metadata/expression CSV pair.
Cross-sample input can be a combined AnnData/H5AD object or a directory of
per-sample CSV pairs. Alignment writes `x_prealigned`, `y_prealigned`,
`x_aligned` and `y_aligned` while preserving the original expression matrix
and `obsm["spatial"]`.

## Installation

### Package and workflow dependencies

```bash
git clone https://github.com/dsong-lab/spAlignDE.git
cd spAlignDE
python -m pip install -e ".[clustering,atlas,histology,ui,tutorial]"
```

### Validated notebook environment

The reference environment contains the complete package, notebook, UI and
documentation dependency set. Run from the cloned repository root:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
conda env create -f environment.yml
conda activate spAlignDE-notebooks
python -m pip install --no-deps --no-build-isolation -e .
python -m ipykernel install --user \
  --name spAlignDE-notebooks \
  --display-name "Python (spAlignDE-notebooks)"
python tools/check_notebook_environment.py
```

On a GPU workstation, require the CUDA compatibility check:

```bash
python tools/check_notebook_environment.py --require-cuda
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for the validated CUDA/CPU variants,
external HIPT and Allen CCF assets, reproducibility expectations and update
policy.

## Workflow tutorials

Run notebooks in workflow order. The canonical, executed copies are under
[`source_notebooks/`](source_notebooks/); the documentation contains the same
saved outputs. The public notebook collection contains only the full data
analysis workflows listed below.

| Workflow | Notebook order |
|---|---|
| MERFISH mouse-brain cross-sample | [`clustering_joint_nb.ipynb`](source_notebooks/clustering/clustering_joint_nb.ipynb) → [`cross_sample_alignment_nb.ipynb`](source_notebooks/cross_sample_alignment_nb.ipynb) |
| Mouse-kidney cross-sample | [`cross_sample_alignment_mouse_kidney_clustering_nb.ipynb`](source_notebooks/cross_sample_alignment_mouse_kidney_clustering_nb.ipynb) → [`cross_sample_alignment_mouse_kidney_alignment_nb.ipynb`](source_notebooks/cross_sample_alignment_mouse_kidney_alignment_nb.ipynb) |
| Breast-cancer cross-sample | [`cross_sample_alignment_breast_cancer_clustering_nb.ipynb`](source_notebooks/cross_sample_alignment_breast_cancer_clustering_nb.ipynb) → [`cross_sample_alignment_breast_cancer_alignment_nb.ipynb`](source_notebooks/cross_sample_alignment_breast_cancer_alignment_nb.ipynb) |
| Transformation uncertainty | [`cross_sample_uncertainty_report.ipynb`](source_notebooks/cross_sample_uncertainty_report.ipynb), after the MERFISH workflow |
| ST to Allen CCF | [`clustering_single_nb.ipynb`](source_notebooks/clustering/clustering_single_nb.ipynb) → [`cross_modal_atlas_alignment_nb.ipynb`](source_notebooks/cross_modal_atlas_alignment_nb.ipynb) |
| UI-curated ST to Allen CCF | [`interactive_region_pairing_nb.ipynb`](source_notebooks/cross_modality/interactive_region_pairing_nb.ipynb) → [`ui_paired_atlas_alignment_nb.ipynb`](source_notebooks/cross_modality/ui_paired_atlas_alignment_nb.ipynb) |
| ST to histology image | [`st_he_feature_extraction_nb.ipynb`](source_notebooks/cross_modality/st_he_feature_extraction_nb.ipynb) → [`st_he_feature_clustering_nb.ipynb`](source_notebooks/cross_modality/st_he_feature_clustering_nb.ipynb) → [`st_he_alignment_nb.ipynb`](source_notebooks/cross_modality/st_he_alignment_nb.ipynb) |
| Spatial ATAC to ST | [`atac_st_single_clustering_nb.ipynb`](source_notebooks/cross_modality/atac_st_single_clustering_nb.ipynb) → [`atac_st_alignment_nb.ipynb`](source_notebooks/cross_modality/atac_st_alignment_nb.ipynb) |
| Post-alignment inference | [`post_alignment_inference_nb.ipynb`](source_notebooks/post_alignment_inference_nb.ipynb) for injured kidney; [`post_alignment_inference_aging_brain_nb.ipynb`](source_notebooks/post_alignment_inference_aging_brain_nb.ipynb) for aging brain |

### Alignment-to-inference handoff

The kidney inference notebook uses packaged coordinates from the validated
fixed-seed kidney alignment by default. Set `SPALIGNDE_KIDNEY_ALIGNED_H5AD` to
the H5AD written by the kidney alignment notebook to continue directly from
your own `x_aligned` and `y_aligned` coordinates. The aging-brain notebook uses
five sections and their packaged fixed-seed aligned coordinates to reproduce
the compact website example; it is not the manuscript's full 20-section
analysis. See the [post-alignment inference guide](docs/source/tutorials/post_alignment_inference.rst)
for grid construction, mismatch calibration, local testing and gene-level
aggregation.

## Reproducibility

The public workflows set every stochastic stage explicitly: seed `1234` for
single-sample BANKSY, seed `1000` for joint cross-sample clustering, seed `0`
for histology feature processing and clustering, and seed `1` for
post-alignment reference subsampling. `spAlignDE.set_random_seed()` resets
Python, NumPy and Torch before randomized PCA.

Discrete outputs are expected to reproduce exactly. Continuous CUDA
deformation coordinates are validated within a declared numerical tolerance
because some GPU operators are not bitwise deterministic. See the
[reproducibility guide](docs/source/tutorials/reproducibility.rst) for
launch-time controls, saved provenance and the validation contract.

## Interactive region-pairing UI

The complete Streamlit source and custom Plotly component are included under
[`ui/`](ui/README.md). The Allen annotation volume is an external input and is
not stored in GitHub:

```bash
export SPALIGNDE_ALLEN_CCF_DIR=/path/to/allen_ccf_2022
streamlit run ui/app.py
```

The exported pairing CSV is authoritative: UI-based alignment skips automatic
candidate discovery, scoring and pair matching. It still performs pairing-file
validation, whole-mask or provided manual pre-alignment, point filtering, mask
processing, signed-distance construction, global channel weighting, S-LDDMM
optimization and label transfer.

## Documentation and parameter tuning

The complete Sphinx website is published at
**https://dsong-lab.github.io/spAlignDE/**. Start with the
[tutorial index](https://dsong-lab.github.io/spAlignDE/tutorial.html), then open
the [executable source notebooks](https://dsong-lab.github.io/spAlignDE/source_notebooks.html)
for the corresponding end-to-end workflows. The website source is versioned
under [`docs/source/`](docs/source/).
Build it locally with:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs/source docs/build/html
python tools/audit_built_html.py docs/build/html
```

Open `docs/build/html/index.html` after the build. The repository contains both
`.readthedocs.yaml` and a GitHub Pages workflow:

- Read the Docs detects `.readthedocs.yaml` after the repository is imported.
- GitHub Pages builds, audits and deploys the website after every push to
  `main` once **Settings → Pages → Source: GitHub Actions** is enabled.

The [Parameter Tuning Guide](docs/source/tutorials/parameter_tuning.rst)
explains coordinate units, clustering/refinement, pre-alignment, the three
grids, `kernel_scale` (legacy `a`), `velocity_grid_spacing` (legacy
`grid_step`), time steps, iterations, momentum learning rate and
workflow-specific pairing controls.

## Data and pretrained models

Large datasets, generated H5AD files, HIPT checkpoints and the Allen annotation
volume are deliberately excluded from Git. Each workflow page links its public
data source and defines the required environment variables. In particular:

- HIPT source/checkpoints: set `SPALIGNDE_HIPT_DIR`;
- Allen CCF: set `SPALIGNDE_ALLEN_CCF_DIR`;
- uncertainty inputs: set `SPALIGNDE_UNCERTAINTY_INPUT_DIR`; and
- other workflow inputs: use the `SPALIGNDE_*` variables documented by the
  corresponding notebook.

## Validation

```bash
python -m pytest -q
python tools/audit_source_notebooks.py source_notebooks
python tools/audit_tutorial_reproducibility.py
python tools/audit_public_references.py
sphinx-build -W --keep-going -b html docs/source docs/build/html
python tools/audit_built_html.py docs/build/html
```

These checks validate package contracts, the CPU alignment implementation,
notebook portability and saved execution state, public notebook paths and
mirrors, strict Sphinx construction, and every generated local link, fragment
and image reference.

## Citation


## License

spAlignDE is released under the [MIT License](LICENSE).
