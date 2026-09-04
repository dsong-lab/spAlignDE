Installation and Notebook Environment
=====================================

The source notebooks use one validated, case-preserving ``spAlignDE`` Python
environment. The public environment specification is cleaned of local paths
and unrelated development packages, and pins the direct dependencies that can
change clustering, pairing, rasterization or S-LDDMM results.

Download the current files:

* :download:`environment.yml <_static/environment/environment.yml>`
* :download:`environment notes <_static/environment/ENVIRONMENT.md>`
* :download:`environment checker <_static/environment/check_notebook_environment.py>`

Create the environment
----------------------

Run the following commands from the package root, which contains
``pyproject.toml`` and ``environment.yml``:

.. code-block:: bash

   cd /path/to/spAlignDE
   unset PYTHONPATH
   export PYTHONNOUSERSITE=1
   conda env create -f environment.yml
   conda activate spAlignDE-notebooks
   python -m pip install --no-deps --no-build-isolation -e .
   python -m ipykernel install --user \
     --name spAlignDE-notebooks \
     --display-name "Python (spAlignDE-notebooks)"

Select **Python (spAlignDE-notebooks)** as the Jupyter kernel. The editable
installation is kept outside ``environment.yml`` so that the exported file
does not contain a developer's local filesystem path. ``--no-deps`` is
intentional because the notebook dependencies are already pinned, while
``--no-build-isolation`` uses the pinned Setuptools instead of downloading a
second build environment.

Clearing ``PYTHONPATH`` before creation prevents Pip from treating packages in
an unrelated shared directory as if they were installed in the new
environment. ``PYTHONNOUSERSITE=1`` also prevents per-user packages from
shadowing pinned dependencies. Keep both settings during validation and
notebook execution; the checker reports either source of contamination.

Verify before running data
--------------------------

.. code-block:: bash

   python tools/check_notebook_environment.py
   python tools/check_notebook_environment.py --require-cuda  # GPU workstation

The checker prints Python, platform, package, imported PyTorch, CUDA runtime
and visible GPU information as JSON. The first command also supports CPU-only
hosts; ``--require-cuda`` additionally requires a visible GPU to complete the
kernel smoke test. A PyTorch metadata/import mismatch is a
sign of an environment modified in place; create a fresh environment before
executing release notebooks.

The package test suite is a maintainer check and is not required before using
a tutorial notebook.

GPU and CPU variants
--------------------

The reference environment uses Python 3.10.14 and matched PyTorch 2.10.0 /
torchvision 0.25.0 wheels for CUDA 12.8. This stack has passed the package,
compiled-operator and CUDA smoke tests on an NVIDIA RTX PRO 6000 Blackwell
GPU. PyTorch 2.7 was the
first stable release to support Blackwell in CUDA 12.8 wheels; the older
PyTorch 2.5.1/CUDA 12.4 stack must not be used on ``sm_120`` hardware. See the
official `PyTorch 2.7 release notes <https://pytorch.org/blog/pytorch-2-7/>`__
and `versioned installation commands
<https://pytorch.org/get-started/previous-versions/>`__. The host must provide
a compatible NVIDIA driver; Conda does not install the driver. Check the full
binary-extension stack, not only Torch, with:

.. code-block:: bash

   nvidia-smi
   python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available())"

S-LDDMM uses CUDA when available. CPU execution is suitable for small tests
but is substantially slower for the full cell-resolution tutorials. For a
CPU-only installation, copy ``environment.yml``, remove the PyTorch CUDA
extra-index line and replace the two wheel specifications with:

.. code-block:: yaml

   - --extra-index-url https://download.pytorch.org/whl/cpu
   - torch==2.10.0+cpu
   - torchvision==0.25.0+cpu

Do not install a second or mismatched Torch/torchvision stack into the same
environment. Basic Torch operations can succeed while torchvision fails later
during histology feature extraction. The environment checker imports both,
compares installed and imported versions, runs a compiled torchvision NMS
operator, verifies the visible GPU architecture and runs a small CUDA
``grid_sample`` kernel.

Inputs outside the Python environment
-------------------------------------

Data, pretrained weights and GPU drivers are not bundled with the environment.
The relevant tutorials link their public datasets. In addition:

* Histology feature extraction requires the HIPT source and the
  ``vit256_small_dino.pth`` and ``vit4k_xs_dino.pth`` checkpoints. Set
  ``SPALIGNDE_HIPT_DIR`` to that HIPT directory.
* ST-to-atlas alignment requires Allen CCF 2022 ``annotation_10.nrrd`` and
  ``voxel_count_and_differences.csv`` inputs.

Run notebooks in the order shown on each workflow page because downstream
notebooks consume the AnnData and coordinate outputs generated upstream.
Small floating-point differences between GPU models are expected; changes in
tissue orientation, matched structures or local geometry require review.

What the installation includes
------------------------------

The Python wheel contains the public clustering, alignment, uncertainty and
post-alignment inference modules, together with compact kidney, aging-brain
and synthetic example data. The optional dependency groups install the
libraries required by those modules; they do not download large biological
datasets, Allen CCF volumes or pretrained HIPT weights.

The executed notebooks, Sphinx source, Streamlit application, tutorial figures
and validation tools are repository resources rather than wheel contents.
Clone the repository and use the editable installation above when following
the published workflows or running the UI. A wheel-only installation is
sufficient when calling the Python API with user-provided inputs.

The manuscript-scale Nissl and full 20-section aging-brain analyses are not
bundled as complete public notebooks. The website provides the H&E,
automatic/UI Atlas, spatial-ATAC and kidney workflows, plus a five-section
aging-brain alignment-to-inference example. This distinction keeps the public
workflow scope clear without implying that large third-party data are shipped
inside the package.
