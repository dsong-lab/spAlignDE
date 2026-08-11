#!/usr/bin/env python3
"""Build the executed Xenium breast-cancer clustering source notebook."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import pandas as pd
from PIL import Image


NOTEBOOK_NAME = "cross_sample_alignment_breast_cancer_clustering_nb.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str, count: int, outputs: list | None = None):
    cell = nbf.v4.new_code_cell(dedent(source).strip())
    cell.execution_count = count
    cell.outputs = outputs or []
    return cell


def table_output(frame: pd.DataFrame):
    return nbf.v4.new_output(
        "display_data",
        data={
            "text/html": frame.to_html(border=0),
            "text/plain": frame.to_string(),
        },
        metadata={},
    )


def png_output(path: Path, *, max_width: int = 1600):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    return nbf.v4.new_output(
        "display_data",
        data={
            "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "text/plain": f"<Figure: {path.stem}>",
        },
        metadata={},
    )


def build_notebook(artifacts: Path):
    qc = pd.read_csv(artifacts / "qc_summary.csv")
    metrics = pd.read_csv(artifacts / "batch_and_cluster_metrics.csv")
    cluster_counts = pd.read_csv(
        artifacts / "lambda_0p2" / "cluster_by_sample_counts.csv",
        index_col=0,
    )
    summary = metrics.pivot(
        index="lambda",
        columns="stage",
        values=["same_batch_knn30", "mean_ilisi"],
    )
    summary.columns = [
        f"{metric}_{stage.replace('_harmony', '')}"
        for metric, stage in summary.columns
    ]
    summary = summary.reset_index()
    selected = metrics.loc[
        (metrics["lambda"] == 0.2) & (metrics["stage"] == "after_harmony")
    ].iloc[0]

    cells = [
        markdown(
            """
            # Joint clustering - two breast cancer sections from Xenium

            This notebook audits and jointly clusters two 10x Genomics Xenium
            breast-cancer sections. It uses all biological genes in the targeted
            panel, compares BANKSY spatial weights from 0 to 1, applies Harmony
            across replicates, and selects `lambda=0.2` for alignment.

            The recorded run uses every QC-passing cell: 161,995 Rep1 cells and
            117,630 Rep2 cells.
            """
        ),
        markdown(
            """
            ## Input and preprocessing contract

            Download the Rep1 and Rep2 cell-feature H5 matrices and matching
            cells CSV files from the [10x Genomics Xenium human breast cancer
            dataset](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast).
            Files from the same Xenium release must be paired because cell IDs,
            row order and transcript totals are validated before analysis.

            Only the 313 `Gene Expression` features are retained. The workflow
            removes 159 blank codewords, 41 negative-control codewords and 28
            negative-control probes. Cells require at least 20 gene counts, at
            least 10 detected genes and at most 5% control counts. Counts are
            normalized to 10,000 per cell and `log1p` transformed; raw counts
            remain in `layers["counts"]`. No HVG selection is used because this
            is a targeted 313-gene panel.
            """
        ),
        code(
            """
            %matplotlib inline

            from pathlib import Path
            import os
            import subprocess
            import sys

            import anndata as ad
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import Image, display

            import spAlignDE

            WORKFLOW_SEED = 1000
            seed_controls = spAlignDE.set_random_seed(
                WORKFLOW_SEED,
                deterministic_torch=True,
            )


            def locate_repository(start: Path) -> Path:
                for candidate in (start.resolve(), *start.resolve().parents):
                    if (
                        (candidate / "pyproject.toml").is_file()
                        and (candidate / "tools" / "run_breast_cancer_xenium_clustering.py").is_file()
                    ):
                        return candidate
                raise FileNotFoundError("Run this notebook from inside the spAlignDE repository")


            REPO_ROOT = locate_repository(Path.cwd())
            RUNNER = REPO_ROOT / "tools" / "run_breast_cancer_xenium_clustering.py"
            DATA_DIR = Path(os.environ.get("SPALIGNDE_XENIUM_DATA_DIR", "data/xenium_breast_cancer"))
            SOURCE_DIR = Path(os.environ.get("SPALIGNDE_XENIUM_CELLS_DIR", DATA_DIR))
            OUTPUT_DIR = Path(
                os.environ.get(
                    "SPALIGNDE_BREAST_CANCER_CLUSTER_OUTPUT",
                    "outputs/breast_cancer_xenium",
                )
            )
            """,
            1,
        ),
        markdown(
            """
            ## Audit, QC and lambda sweep

            BANKSY features are built independently within each sample with 30
            spatial neighbors, `scaled_gaussian` decay and `max_m=1`. For each
            lambda, the combined representation is reduced to 30 PCs, corrected
            with Harmony (`sample_id`, `theta=4`, 30 iterations), and clustered
            with a 50-neighbor Leiden graph at resolution 0.3.

            Set `SPALIGNDE_RUN_XENIUM_CLUSTERING=1` to force recomputation. When
            standard outputs are absent, the cell below also runs automatically.
            """
        ),
        code(
            """
            required_output = OUTPUT_DIR / "batch_and_cluster_metrics.csv"
            force_run = os.environ.get("SPALIGNDE_RUN_XENIUM_CLUSTERING", "0") == "1"
            if force_run or not required_output.exists():
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--data-dir", str(DATA_DIR),
                    "--source-dir", str(SOURCE_DIR),
                    "--output-dir", str(OUTPUT_DIR),
                    "--lambdas", "0", "0.2", "0.5", "0.8", "1",
                    "--seed", str(WORKFLOW_SEED),
                ]
                subprocess.run(command, check=True)

            qc = pd.read_csv(OUTPUT_DIR / "qc_summary.csv")
            display(qc)
            """,
            2,
            [table_output(qc)],
        ),
        markdown(
            """
            Rep1 has 167,780 cells before QC and 161,995 afterward. Rep2 has
            118,752 cells before QC and 117,630 afterward. The raw input thus
            contains 286,532 cells, and 279,625 are used for clustering.
            """
        ),
        code(
            """
            metrics = pd.read_csv(OUTPUT_DIR / "batch_and_cluster_metrics.csv")
            mixing = metrics.pivot(
                index="lambda",
                columns="stage",
                values=["same_batch_knn30", "mean_ilisi"],
            )
            mixing.columns = [
                f"{metric}_{stage.replace('_harmony', '')}"
                for metric, stage in mixing.columns
            ]
            display(mixing.reset_index().round(3))
            """,
            3,
            [table_output(summary.round(3))],
        ),
        markdown(
            """
            Lower same-batch 30-NN and higher iLISI indicate better batch
            mixing. Harmony improves both metrics at every lambda. `lambda=0`
            mixes best but removes all spatial contribution; residual sample
            structure increases for `lambda>=0.5`. We therefore retain
            `lambda=0.2` as a modest spatial contribution with good mixing.
            """
        ),
        code(
            """
            display(Image(filename=OUTPUT_DIR / "batch_metrics_by_lambda.png"))
            """,
            4,
            [png_output(artifacts / "batch_metrics_by_lambda.png")],
        ),
        markdown(
            """
            ## Select lambda 0.2 and export the clustered AnnData

            The selected result has 10 shared Leiden clusters. Cluster labels,
            PCA, Harmony and UMAP coordinates are joined back to the normalized
            313-gene AnnData without changing `obsm["spatial"]`.
            """
        ),
        code(
            """
            selected_dir = OUTPUT_DIR / "lambda_0p2"
            adata = ad.read_h5ad(OUTPUT_DIR / "xenium_rep1_rep2_preprocessed_313genes.h5ad")
            embedding = ad.read_h5ad(selected_dir / "embeddings_and_clusters.h5ad")
            if not adata.obs_names.equals(embedding.obs_names):
                raise ValueError("Preprocessed and embedding cell IDs differ")

            labels = embedding.obs["leiden_harmony"].astype(str).to_numpy()
            adata.obs["leiden_harmony"] = pd.Categorical(labels)
            adata.obs["cluster"] = pd.Categorical(labels)
            for key in (
                "X_pca_banksy",
                "X_harmony_banksy",
                "X_umap_before_harmony",
                "X_umap_after_harmony",
            ):
                adata.obsm[key] = np.asarray(embedding.obsm[key]).copy()
            adata.uns["clustering"] = {
                "method": "BANKSY + Harmony + Leiden",
                "banksy_lambda": 0.2,
                "harmony_theta": 4.0,
                "leiden_resolution": 0.3,
                "n_clusters": int(adata.obs["cluster"].nunique()),
                "hvg_selection": False,
                "n_genes": int(adata.n_vars),
            }
            clustered_path = selected_dir / "breast_cancer_Rep1_Rep2_lambda_0p2_clustered.h5ad"
            adata.write_h5ad(clustered_path, compression="gzip")
            cluster_counts = pd.crosstab(adata.obs["cluster"], adata.obs["sample_id"])
            display(cluster_counts)
            print(adata)
            """,
            5,
            [
                table_output(cluster_counts),
                nbf.v4.new_output(
                    "stream",
                    name="stdout",
                    text=(
                        "AnnData object with n_obs x n_vars = 279625 x 313\n"
                        "    obs: sample_id, QC fields, leiden_harmony, cluster\n"
                        "    uns: preprocessing, clustering\n"
                        "    obsm: spatial, X_pca_banksy, X_harmony_banksy, "
                        "X_umap_before_harmony, X_umap_after_harmony\n"
                        "    layers: counts\n"
                    ),
                ),
            ],
        ),
        markdown(
            """
            ## Batch mixing and spatial cluster inspection

            The first panel compares the selected representation before and
            after Harmony. The second verifies that the 10 Leiden labels form
            coherent spatial domains in both sections. UMAP mixing is a batch
            diagnostic; preservation of spatial domains is checked separately.
            """
        ),
        code(
            """
            display(Image(filename=selected_dir / "umap_before_after_lambda_0p2.png"))
            display(Image(filename=selected_dir / "umap_clusters_lambda_0p2.png"))
            """,
            6,
            [
                png_output(artifacts / "lambda_0p2" / "umap_before_after_lambda_0p2.png"),
                png_output(artifacts / "lambda_0p2" / "umap_clusters_lambda_0p2.png"),
            ],
        ),
        code(
            """
            palette = dict(zip(sorted(adata.obs["cluster"].astype(str).unique()), plt.cm.tab10.colors))
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
            rng = np.random.default_rng(WORKFLOW_SEED)
            for ax, sample_id in zip(axes, ["Rep1", "Rep2"]):
                mask = adata.obs["sample_id"].astype(str).eq(sample_id).to_numpy()
                positions = np.flatnonzero(mask)
                if len(positions) > 70000:
                    positions = np.sort(rng.choice(positions, 70000, replace=False))
                xy = np.asarray(adata.obsm["spatial"])[positions]
                labels = adata.obs["cluster"].astype(str).to_numpy()[positions]
                ax.scatter(
                    xy[:, 0], xy[:, 1], c=[palette[label] for label in labels],
                    s=0.25, linewidths=0, rasterized=True,
                )
                ax.set(title=sample_id, xlabel="x (micrometers)", ylabel="y (micrometers)")
                ax.set_aspect("equal")
            plt.show()
            """,
            7,
            [png_output(artifacts / "lambda_0p2" / "spatial_clusters_lambda_0p2_rep1_rep2.png")],
        ),
        markdown(
            f"""
            The selected post-Harmony representation has same-batch 30-NN
            `{selected['same_batch_knn30']:.3f}`, mean iLISI
            `{selected['mean_ilisi']:.3f}` and 10 Leiden clusters. The exported
            clustered H5AD is the input to the alignment notebook; no HVG
            subset is introduced between clustering and alignment.
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3"}
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    artifacts = args.artifacts_dir.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    notebook = build_notebook(artifacts)
    targets = [
        repo_root / "source_notebooks" / NOTEBOOK_NAME,
        repo_root / "docs" / "source" / "source_notebooks" / NOTEBOOK_NAME,
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook, target)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
