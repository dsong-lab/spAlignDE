# Executed source notebooks

This directory is the package-workspace source of truth for every executable
notebook published under the documentation site's **Source Notebooks** section.
Its folder structure mirrors `docs/source/source_notebooks`:

- `clustering/`: joint and single-sample clustering;
- `cross_modality/`: ATAC–ST, H&E–ST, and interactive region-pairing workflows;
- top-level cross-sample, atlas, uncertainty, and post-alignment notebooks,
  including kidney and aging-brain inference workflows.

Notebooks are executed here first and then copied to the matching Sphinx source
path. Public Python examples use the case-preserving `spAlignDE` name, and
package-generated AnnData metadata is stored under `adata.uns["spAlignDE"]`.

Release outputs are regenerated from the repository root with:

```bash
python tools/execute_fixed_seed_tutorials.py
python tools/audit_tutorial_reproducibility.py
```

The executor uses a fresh kernel per notebook, imports the current checkout
through an absolute `src` path, stops on cell errors, sanitizes local paths and
records source/output hashes only after full execution. The public execution
set contains 17 computational notebooks. External input checksums are stored in
`docs/source/_static/tutorial_execution_manifest.json`.

The task-specific **How to adapt...** cells are maintained by
`tools/add_parameter_guidance.py`. They preserve executed outputs and should be
reapplied after a notebook builder regenerates a canonical notebook. The full
cross-workflow reference is the website `Parameter Tuning Guide`.
