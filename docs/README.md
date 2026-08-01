# Documentation build and deployment

The public website is built from `docs/source` with Sphinx and MyST-NB. Saved
notebook outputs are rendered without re-executing the data-heavy workflows.

## Local preview

From the repository root:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs/source docs/build/html
python tools/audit_built_html.py docs/build/html
python docs/serve_docs_nocache.py
```

Open `http://localhost:8000`. The preview server disables browser caching and
redirects the removed legacy H&E/Nissl URL to the current histology workflow.

## GitHub Pages

`.github/workflows/docs.yml` builds pull requests strictly and deploys pushes
to `main`. After the first push, select **Settings → Pages → Source: GitHub
Actions**. No generated HTML needs to be committed.

## Read the Docs

Import the GitHub repository into Read the Docs. The root
`.readthedocs.yaml` selects Python 3.10, installs `docs/requirements.txt`, uses
`docs/source/conf.py`, and treats warnings as build failures.

## Updating a notebook page

1. regenerate or edit the canonical notebook under `source_notebooks/`;
2. execute it with the `spAlignDE-notebooks` kernel;
3. run `tools/add_parameter_guidance.py` when a builder replaced tuning cells;
4. sanitize nonportable output paths with `tools/sanitize_notebook_outputs.py`;
5. copy it to the matching `docs/source/source_notebooks/` path;
6. run `tools/audit_source_notebooks.py source_notebooks`; and
7. rebuild and audit the website strictly.

The website intentionally contains no dataset, Allen annotation, HIPT
checkpoint or generated H5AD dependency. Those assets remain linked external
inputs to the notebooks.
