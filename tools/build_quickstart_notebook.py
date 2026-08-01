#!/usr/bin/env python3
"""Build the self-contained CPU quickstart notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "source_notebooks/quickstart_nb.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook():
    cells = [
        markdown(
            """
            # CPU quickstart — synthetic cross-sample alignment

            This self-contained notebook verifies a spAlignDE installation
            without downloading external data. It creates a deterministic
            two-sample AnnData object with three known shared structures, runs
            pre-alignment, rasterization and a short S-LDDMM fit, and validates
            the standardized output columns.

            The data are synthetic and intended only for software validation;
            do not use this example for biological interpretation. Full-scale
            MERFISH, kidney and breast-cancer workflows are documented in the
            cross-sample tutorial.
            """
        ),
        code(
            """
            import numpy as np
            import spAlignDE

            print("spAlignDE", spAlignDE.__version__)
            """
        ),
        markdown(
            """
            ## 1. Create a ready-to-align AnnData input

            Real inputs require `obs["sample_id"]`, `obs["cluster"]`, unique
            observation names and finite coordinates in `obsm["spatial"]`.
            The helper below supplies that exact contract with known labels so
            this quickstart can focus on alignment. Full workflows infer the
            cluster labels in their preceding clustering notebook.
            """
        ),
        code(
            """
            adata = spAlignDE.make_cross_sample_example(
                n_per_cluster=20,
                n_genes=12,
                random_state=0,
            )
            spAlignDE.validate_cross_sample_anndata(adata)
            print(adata)
            print(adata.obs.groupby(["sample_id", "cluster"], observed=True).size())
            """
        ),
        markdown(
            """
            ## 2. Run the alignment on CPU

            Two optimization iterations keep the CI smoke test short. They are
            not a scientific recommendation. Use the validated dataset-specific
            profiles and the Parameter Tuning Guide for real tissue sections.
            """
        ),
        code(
            """
            result = spAlignDE.align_cross_sample(
                adata,
                query_sample="query",
                reference_sample="reference",
                rasterization_config=spAlignDE.RasterizationConfig(
                    grid_spacing=0.30,
                ),
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
            print(result.metrics)
            """
        ),
        markdown(
            """
            ## 3. Validate and inspect the output

            spAlignDE preserves `X`, observation order and the original
            coordinates. It adds pre-aligned and aligned x/y columns to `obs`;
            reference coordinates remain fixed.
            """
        ),
        code(
            """
            required = list(spAlignDE.REQUIRED_OUTPUT_COLUMNS)
            assert all(column in result.adata.obs for column in required)
            assert result.adata.shape == adata.shape
            assert np.array_equal(result.adata.X.toarray(), adata.X.toarray())

            reference = result.adata.obs["sample_id"].astype(str) == "reference"
            np.testing.assert_allclose(
                result.adata.obs.loc[reference, ["x_aligned", "y_aligned"]],
                adata.obsm["spatial"][reference],
            )
            result.adata.obs[["sample_id", "cluster", *required]].head()
            """
        ),
        code(
            """
            figure, axes = spAlignDE.plot_alignment_result(result)
            figure.suptitle("Synthetic query-to-reference quickstart", y=1.02)
            """
        ),
        markdown(
            """
            ## Next steps

            Continue with the cross-sample tutorial to prepare public MERFISH
            or Visium inputs, infer joint structures and use a full validated
            S-LDDMM configuration. Save the aligned AnnData with
            `result.adata.write_h5ad("aligned.h5ad")` when adapting this code.
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (spAlignDE-notebooks)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
    )
    return notebook


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), TARGET)
    print(TARGET.relative_to(ROOT))


if __name__ == "__main__":
    main()
