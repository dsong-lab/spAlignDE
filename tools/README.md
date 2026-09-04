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

Run package tests and the audit scripts before a release. Re-execute only the
notebooks affected by a scientific change; independent repeat runs are reserved
for changes to reported results.
