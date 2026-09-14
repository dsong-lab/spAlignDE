# Maintainer tools

The scripts in this directory cover recurring repository maintenance. The
canonical tutorial files live only in `source_notebooks/`; documentation inputs
are generated and ignored by Git.

| Task | Script |
|---|---|
| Execute computational tutorials | `execute_fixed_seed_tutorials.py` |
| Run the breast-cancer Xenium clustering notebook workflow | `run_breast_cancer_xenium_clustering.py` |
| Inspect saved notebook state | `audit_source_notebooks.py` |
| Remove nonportable saved output | `sanitize_notebook_outputs.py` |
| Stage notebooks and downloads for Sphinx | `stage_documentation.py` |
| Check public paths, shared pins and versions | `audit_public_references.py` |
| Check API documentation, built HTML and wheel contents | `audit_api_documentation.py`, `audit_built_html.py`, `audit_distribution_contents.py` |
| Diagnose the notebook environment | `check_notebook_environment.py` |
| Refresh packaged post-alignment coordinates for a release | `refresh_post_alignment_fixed_coordinates.py` |

## Build the documentation

Run these commands from the repository root:

```bash
python -m pip install -r docs/requirements.txt
python tools/stage_documentation.py
sphinx-build -W --keep-going -b html docs/source docs/build/html
python tools/audit_built_html.py docs/build/html
```

Open `docs/build/html/index.html` to preview the site. The build renders the
saved notebook outputs without executing the notebooks. GitHub Actions
publishes the same site from `main`.

## Check a release

In the notebook environment, run the package tests and source checks:

```bash
python -m pytest -q
python tools/audit_source_notebooks.py source_notebooks
python tools/audit_public_references.py
python tools/audit_api_documentation.py
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
python tools/audit_distribution_contents.py dist/*.whl
```

Then build and audit the documentation using the commands above.
The wheel includes the Python API and compact example data. The repository
also contains the notebooks, documentation, UI, figures and these tools.

Re-execute the notebooks affected by a scientific change. If the change affects
a reported result, repeat the workflow independently and compare its outputs
as described in the
[reproducibility guide](../docs/source/tutorials/reproducibility.rst).
