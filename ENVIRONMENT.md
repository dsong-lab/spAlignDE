# Notebook environment

`environment.yml` is the public reference environment for executing the
spAlignDE source notebooks, rebuilding the documentation and running the
interactive region-pairing interface. It is a cleaned, portable specification
of the validated development environment: unrelated packages, host-specific
paths and a local editable install are deliberately excluded.

## Create the environment

Run these commands from the spAlignDE package root—the directory containing
`pyproject.toml` and `environment.yml`:

```bash
cd /path/to/spAlignDE
unset PYTHONPATH
export PYTHONNOUSERSITE=1
conda env create -f environment.yml
conda activate spAlignDE-notebooks
python -m pip install --no-deps --no-build-isolation -e .
python -m ipykernel install --user \
  --name spAlignDE-notebooks \
  --display-name "Python (spAlignDE-notebooks)"
```

Use **Python (spAlignDE-notebooks)** as the Jupyter kernel. Verify the
installation before running a full dataset:

```bash
python tools/check_notebook_environment.py
python tools/check_notebook_environment.py --require-cuda  # GPU workstation
python -m pytest -q
```

The first checker command also supports CPU-only hosts. Use
`--require-cuda` on the GPU workstation so a hidden/incompatible GPU cannot
silently turn validation into a CPU-only pass. The editable package
installation is a separate command so the environment
file contains no local filesystem path. `--no-deps` is intentional because all
public dependencies are pinned in `environment.yml`; `--no-build-isolation`
uses its pinned Setuptools instead of downloading a second build environment.

Clearing `PYTHONPATH` before environment creation is important. Otherwise Pip
may see packages from an unrelated shared directory, report them as already
installed, and omit them from the new environment. `PYTHONNOUSERSITE=1`
prevents the same problem through the per-user site-packages directory. The
checker fails when either isolation condition is not satisfied.

## GPU and CPU execution

The reference file installs the matched PyTorch 2.10.0 and torchvision 0.25.0
CUDA 12.8 wheels. This environment has passed the package, compiled-operator
and CUDA smoke tests on an NVIDIA RTX PRO 6000 Blackwell GPU. PyTorch 2.7 was the first
stable release with Blackwell support in CUDA 12.8 wheels; the earlier
PyTorch 2.5.1/CUDA 12.4 combination must not be used on ``sm_120`` hardware.
See the official [PyTorch 2.7 release
notes](https://pytorch.org/blog/pytorch-2-7/) and [versioned installation
commands](https://pytorch.org/get-started/previous-versions/). The NVIDIA
driver is not installed by Conda. Check the environment with:

```bash
nvidia-smi
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"
```

S-LDDMM selects CUDA when available. CPU execution is supported for small tests
but is substantially slower for full cell-resolution tutorials. For CPU-only
installation, copy `environment.yml`, remove the PyTorch CUDA extra-index line
and replace the two packages with the matched CPU wheels:

```yaml
- --extra-index-url https://download.pytorch.org/whl/cpu
- torch==2.10.0+cpu
- torchvision==0.25.0+cpu
```

Do not install a second or mismatched PyTorch/torchvision stack on top of this
environment. A metadata version that differs from the actually imported
package, or a torch/torchvision release mismatch, can allow basic Torch imports
while failing later inside histology feature extraction. The environment check
imports both packages, runs a compiled torchvision NMS operator, and executes
a small CUDA kernel when a GPU is visible.

## External assets not stored in the environment

The following inputs are versioned separately from Python packages:

- **HIPT source and checkpoints** for histology feature extraction. Set
  `SPALIGNDE_HIPT_DIR` to the official HIPT clone containing
  `HIPT_4K/Checkpoints/vit256_small_dino.pth` and
  `HIPT_4K/Checkpoints/vit4k_xs_dino.pth`. The feature-extraction manifest
  records both checkpoint SHA-256 hashes.
- **Allen CCF 2022** `annotation_10.nrrd` and
  `voxel_count_and_differences.csv`.
- Dataset files downloaded from Vizgen, 10x Genomics, GEO/UCSC or the Zenodo
  records linked by each tutorial.

The notebook environment does not bundle data, pretrained model files or an
NVIDIA driver.

## Reproducibility levels

Launch Jupyter with `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG` and the thread
limits documented in `docs/source/tutorials/reproducibility.rst`. Each
notebook then calls `spAlignDE.set_random_seed()` before randomized work.

The environment file locks the direct packages that define results. With
fixed input checksums and observation order, discrete clustering labels,
masks, hierarchy memberships and accepted pair tables must reproduce exactly.
Continuous CUDA coordinates are different: PyTorch reports no deterministic
CUDA implementation for one grid-sampling backward operation used by S-LDDMM.
They must therefore pass the workflow's declared absolute or subpixel
tolerance; a random seed alone does not imply bitwise-identical deformation
fields.

The public August 2026 validation used float64 for Atlas, H&E and ATAC
manuscript-grade alignments and observed repeat differences below `1e-6`
(below `1e-12` for H&E and ATAC). Large cross-sample float32 runs were accepted
within one coordinate unit, no more than one thirtieth of their 30-unit raster
spacing. Large changes in discrete pairs, tissue orientation or local geometry
are failures, not acceptable numeric drift.

## Updating the environment

When a dependency is intentionally changed:

1. update `environment.yml`;
2. create a fresh environment rather than modifying the old one in place;
3. run `tools/check_notebook_environment.py` and the package tests;
4. execute the affected source notebooks;
5. rebuild Sphinx; and
6. compare structure maps, accepted pairs, output coordinates and key metrics
   with the previous environment.

Do not regenerate this file from an unrestricted long-lived developer
environment without reviewing local paths, unrelated packages and conflicting
CUDA runtimes.
