# Executed source notebooks

This directory is the package-workspace source of truth for every executable
notebook published under the documentation site's **Source Notebooks** section.
Its folder structure mirrors `docs/source/source_notebooks`:

- `clustering/`: joint and single-sample clustering;
- `cross_modality/`: ATAC–ST, H&E–ST, and interactive region-pairing workflows;
- top-level cross-sample, atlas, uncertainty, and post-alignment notebooks.

Notebooks are executed here first and then copied to the matching Sphinx source
path. Public Python examples use the case-preserving `spAlignDE` name, and
package-generated AnnData metadata is stored under `adata.uns["spAlignDE"]`.

The task-specific **How to adapt...** cells are maintained by
`tools/add_parameter_guidance.py`. They preserve executed outputs and should be
reapplied after a notebook builder regenerates a canonical notebook. The full
cross-workflow reference is the website `Parameter Tuning Guide`.
