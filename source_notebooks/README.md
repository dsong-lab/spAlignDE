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
checks that saved outputs and documentation copies are complete and consistent.
The public execution set contains 17 computational notebooks. External inputs
used for these examples are listed in the documentation by role and filename.

The task-specific **How to adapt...** cells are maintained by
`tools/add_parameter_guidance.py`. They preserve executed outputs and should be
reapplied after a notebook builder regenerates a canonical notebook. The full
cross-workflow reference is the website `Parameter Tuning Guide`.

## Recommended notebook routes

Use the shortest route matching the task; each arrow denotes a saved AnnData
or feature hand-off consumed by the next notebook.

| Goal | Notebook route | Seed | Validated fixed result |
|---|---|---:|---|
| Single ST clustering | [`clustering_single_nb.ipynb`](clustering/clustering_single_nb.ipynb) | 1234 | 25 S2R1 structures |
| ST to Allen CCF | single clustering → [`cross_modal_atlas_alignment_nb.ipynb`](cross_modal_atlas_alignment_nb.ipynb) | 1234 | 18 final pairs |
| Spatial ATAC to ST | [`atac_st_single_clustering_nb.ipynb`](cross_modality/atac_st_single_clustering_nb.ipynb) → [`atac_st_alignment_nb.ipynb`](cross_modality/atac_st_alignment_nb.ipynb) | 1234 | 17 ATAC structures, 8 pairs |
| ST to H&E | [image extraction](cross_modality/st_he_feature_extraction_nb.ipynb) → [image clustering](cross_modality/st_he_feature_clustering_nb.ipynb) → [`st_he_alignment_nb.ipynb`](cross_modality/st_he_alignment_nb.ipynb) | 0 | 21 image structures, 2 pairs |
| Cross-sample ST | joint clustering → the matching alignment notebook | 1000 | exact discrete labels/pairs |
| Transformation variability | [`cross_sample_uncertainty_report.ipynb`](cross_sample_uncertainty_report.ipynb) | 1000 | 10 repeats; median `dist_var` 429.54 |
| Post-alignment inference | kidney or aging-brain inference notebook | 1 | serial deterministic fit |

Every notebook begins with its input contract and ends with its output contract.
For external data, use the documented `SPALIGNDE_*` environment variables rather
than editing workstation-specific paths into the notebook. Alignment notebooks
return the final optimizer iterate (`restore_best_checkpoint=False`) unless a
user explicitly chooses a different checkpoint policy.
