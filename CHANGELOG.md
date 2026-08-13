# Changelog

## 0.1.0 — 2026-08-04

- Updated the fixed-seed H&E image-clustering profile to a 26-cluster merge
  target followed by the selected reflection and post-symmetry gates, yielding
  21 final cleaned structures for the documented mouse-brain example.
- Exposed reflection-merge thresholds in ``HistologyClusteringConfig`` and
  refreshed the ST-to-H&E tutorial outputs from the resulting two-pair,
  final-iterate S-LDDMM alignment.
- Replaced the former tail/intercept mismatch calibration with the promoted
  gene-specific MAD-null1-A0 design: within-risk-bin median centering,
  Student-t null MAD scaling, monotone nonnegative scale excess, and a
  quadratic local-risk fit through the origin with bounded anchor rescaling.
- Fixed the gene-specific global mismatch coefficient at zero. The
  comparison-level global risk score remains available as provenance but no
  longer applies a spatially uniform variance factor.
- Preserved local P-value arrays in compact fitted results so
  `gene_level_acat_pvalue` works on normal public `fit_local_de` output.
- Matched the manuscript ACAT numerical small-P branch and made the public
  summary strict about using raw local P values rather than adjusted q-values.
- Added actual per-gene calibration modes and coefficients to result metadata.
- Clarified density feature-energy weighting, stable-gene candidate screening,
  shared-grid sizing, cell-type support, and optional region cleanup in the
  README and tutorials.
- Updated the kidney tutorial to use the manuscript-comparable uncleaned
  `q < 0.05` region mask.
- Promoted raw-gene support summaries to `summarize_raw_genes`, extended
  `build_visium_inference_table` to accept standardized coordinate DataFrames,
  and removed notebook-local data-loader and table-construction functions from
  the executed kidney tutorial.

Package publication remains a separate release action. The documentation is
deployed by the GitHub Pages workflow after this update reaches ``main``.
