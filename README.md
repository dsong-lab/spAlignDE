# spAlignDE

[![Tests](https://github.com/dsong-lab/spAlignDE/actions/workflows/tests.yml/badge.svg)](https://github.com/dsong-lab/spAlignDE/actions/workflows/tests.yml)
[![Documentation](https://github.com/dsong-lab/spAlignDE/actions/workflows/docs.yml/badge.svg)](https://github.com/dsong-lab/spAlignDE/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**spAlignDE** uses tissue structure to align spatial omics datasets across
samples and modalities. For aligned spatial transcriptomics samples, it tests
local differences in gene expression while accounting for residual alignment
mismatch.

It supports alignment between tissue sections, registration with histology
images and anatomical atlases, spatial ATAC-seq to spatial transcriptomics
alignment, and interactive region pairing.

**[Documentation](https://dsong-lab.github.io/spAlignDE/) · [Tutorials](https://dsong-lab.github.io/spAlignDE/tutorial.html) · [Notebooks](https://dsong-lab.github.io/spAlignDE/source_notebooks.html)**

<p align="center">
  <a href="assets/Figure_1_08132026_DS.png">
    <img src="assets/Figure_1_08132026_DS.png" alt="Overview of spatial alignment and mismatch-aware local differential expression with spAlignDE" width="600">
  </a>
</p>

## Installation

To run the tutorials and interactive interface, use the Conda environment
below. It pins the dependencies used by the published examples. The reference
environment uses CUDA 12.8; for a CPU-only setup, follow the
[installation guide](https://dsong-lab.github.io/spAlignDE/installation.html#gpu-and-cpu-variants)
before creating the environment.

```bash
git clone https://github.com/dsong-lab/spAlignDE.git
cd spAlignDE
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

Select **Python (spAlignDE-notebooks)** as the Jupyter kernel. On a GPU
workstation, also run `python tools/check_notebook_environment.py --require-cuda`.
See the [installation guide](https://dsong-lab.github.io/spAlignDE/installation.html)
for hardware requirements and installation into an existing environment.

## Choose a workflow

Read the guide for the method and input requirements, then open the notebooks
for the complete example. Each notebook page lists the order to run them.

| Task | Guide | Notebooks |
|---|---|---|
| Align spatial transcriptomics samples | [Cross-sample alignment](https://dsong-lab.github.io/spAlignDE/tutorials/cross_sample_alignment.html) | [Mouse brain](https://dsong-lab.github.io/spAlignDE/source_notebooks/cross_sample_alignment_mouse_brain.html) · [Mouse kidney](https://dsong-lab.github.io/spAlignDE/source_notebooks/cross_sample_alignment_mouse_kidney.html) · [Breast cancer](https://dsong-lab.github.io/spAlignDE/source_notebooks/cross_sample_alignment_breast_cancer.html) |
| Align to the Allen brain atlas | [Atlas alignment](https://dsong-lab.github.io/spAlignDE/tutorials/cross_modality_atlas_alignment.html) | [Automatic and interactive pairing](https://dsong-lab.github.io/spAlignDE/source_notebooks/cross_modality_atlas_alignment.html) |
| Align to a histology image | [Histology alignment](https://dsong-lab.github.io/spAlignDE/tutorials/st_histology_image_processing.html) | [Mouse brain H&E](https://dsong-lab.github.io/spAlignDE/source_notebooks/cross_modality_he_alignment.html) |
| Align spatial ATAC-seq to spatial transcriptomics | [ATAC alignment](https://dsong-lab.github.io/spAlignDE/tutorials/cross_modality_atac_alignment.html) | [Mouse brain ATAC](https://dsong-lab.github.io/spAlignDE/source_notebooks/cross_modality_atac_alignment.html) |
| Assess alignment stability | [Subsampling and variability](https://dsong-lab.github.io/spAlignDE/source_notebooks/cross_sample_uncertainty_qualification.html) | [Stability notebook](source_notebooks/cross_sample_uncertainty_report.ipynb) |
| Test local differential expression after alignment | [Local inference](https://dsong-lab.github.io/spAlignDE/tutorials/post_alignment_inference.html) | [Kidney and aging brain](https://dsong-lab.github.io/spAlignDE/source_notebooks/post_alignment_inference.html) |

The kidney inference example uses the saved coordinates from the kidney
alignment workflow. The aging-brain inference example uses precomputed
coordinates for **five sections**; the manuscript's complete 20-section
analysis and Nissl analysis are not provided as complete public notebooks.

Large datasets, Allen atlas volumes and pretrained HIPT image-model weights
must be downloaded separately. Each workflow links its inputs and explains
where to store them. See [external inputs](https://dsong-lab.github.io/spAlignDE/installation.html#inputs-outside-the-python-environment).

<a id="quick-start"></a>

## Try a small example

After installation, run this synthetic example to check that alignment works
on the CPU. It uses only two optimization iterations to finish quickly. Use
the workflow notebooks above for real-data settings.

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

The example saves aligned data to `cross_sample_example_aligned.h5ad`.
Alignment adds `x_prealigned`, `y_prealigned`, `x_aligned` and `y_aligned`,
preserving the expression matrix and original `obsm["spatial"]` coordinates.
The [Python API](https://dsong-lab.github.io/spAlignDE/api.html) describes the
available functions and output fields.

## Interactive region pairing

Use the Streamlit interface to review tissue regions, choose their
correspondences and export a pairing CSV for alignment. For Allen atlas data:

```bash
export SPALIGNDE_ALLEN_CCF_DIR=/path/to/allen_ccf_2022
streamlit run ui/app.py
```

See the [UI guide](ui/README.md) for setup and the
[atlas notebooks](https://dsong-lab.github.io/spAlignDE/source_notebooks/cross_modality_atlas_alignment.html)
for alignment with your exported pairs.

## Adapting and reproducing an analysis

Use the [parameter guide](https://dsong-lab.github.io/spAlignDE/tutorials/parameter_tuning.html)
when changing datasets or coordinate units. To reproduce a tutorial, keep the
same inputs, observation order, parameters and random seeds. The
[reproducibility guide](https://dsong-lab.github.io/spAlignDE/tutorials/reproducibility.html)
lists the seeds and explains how to compare results across runs.

For tests, documentation builds and release checks, see the
[maintainer guide](tools/README.md).

## Citation

If you use **spAlignDE** in your research, please cite our preprint:

Xu, S., Wang, Y., Meng, L., Dalal, A., Yin, Y., & Song, D. (2026). **spAlignDE unifies cross-sample and cross-modal spatial alignment with mismatch-aware differential expression.** *bioRxiv*. [doi:10.64898/2026.09.05.749632](https://doi.org/10.64898/2026.09.05.749632)

## License

spAlignDE is released under the [MIT License](LICENSE).
