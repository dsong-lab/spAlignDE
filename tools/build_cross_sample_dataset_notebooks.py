"""Build the dataset-specific cross-sample tutorial notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_ROOT = ROOT / "tutorials" / "cross_sample"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def write_notebook(path: Path, cells: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
    )
    nbf.write(notebook, path)
    print(f"Wrote {path.relative_to(ROOT)}")


def kidney_clustering_cells() -> list:
    return [
        markdown(
            """
            # Joint clustering - two mouse kidney sections from Visium

            This notebook constructs a shared spatial-cluster label space for
            the IL3 and NL3 mouse kidney sections. It follows the same
            AnnData-native structure as the MERFISH mouse-brain tutorial:

            1. load and validate the two samples;
            2. retain spatially filtered spots and prepare expression features;
            3. run joint BANKSY, PCA, Harmony and Leiden clustering;
            4. refine spatial boundaries independently within each section; and
            5. save a clustered AnnData for the alignment notebook.
            """
        ),
        markdown(
            """
            ## Installation and input

            From the repository root:

            ```bash
            cd /path/to/spAlignDE
            python -m pip install -e ".[clustering,tutorial]"
            export SPALIGNDE_KIDNEY_INPUT=/path/to/kidney_combined.h5ad
            ```

            Download the IL3 and NL3 Visium matrices, coordinates and region
            annotations from the [STcompare Zenodo record](https://zenodo.org/records/20647680).
            The input may also be a directory using the standard paired-CSV
            contract described in the main cross-sample tutorial.
            """
        ),
        code(
            """
            %matplotlib inline

            from pathlib import Path
            import os
            import warnings

            import anndata as ad
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import numpy as np
            import scipy.sparse as sp
            import spAlignDE

            WORKFLOW_SEED = 1000
            seed_controls = spAlignDE.set_random_seed(
                WORKFLOW_SEED,
                deterministic_torch=True,
            )

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings(
                "ignore", message="pkg_resources is deprecated as an API"
            )
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                "font.size": 8,
                "axes.spines.right": False,
                "axes.spines.top": False,
                "legend.frameon": False,
            })


            def find_project_root(start: Path) -> Path:
                start = start.resolve()
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists():
                        return candidate
                raise FileNotFoundError("Run this notebook from inside the spAlignDE repository")


            PROJECT_ROOT = find_project_root(Path.cwd())
            OUTPUT_DIR = PROJECT_ROOT / "tutorials" / "cross_sample" / "kidney" / "output"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            input_value = os.environ.get("SPALIGNDE_KIDNEY_INPUT")
            if not input_value:
                raise EnvironmentError(
                    "Set SPALIGNDE_KIDNEY_INPUT to a combined .h5ad file or paired-CSV directory"
                )
            INPUT_PATH = Path(input_value).expanduser().resolve()
            print("Input:", INPUT_PATH.name)
            """
        ),
        markdown(
            """
            ## Load and prepare the kidney data

            The combined source object contains IL3 and NL3 spots before and
            after spatial filtering. We retain the 6,180 spatially filtered
            spots, keep genes detected in at least 1% of spots in each sample,
            normalize every spot to a total of 250, add one, and apply the
            across-spot quantile normalization used in the original kidney
            analysis.
            """
        ),
        code(
            """
            if INPUT_PATH.is_file() and INPUT_PATH.suffix.lower() == ".h5ad":
                adata = ad.read_h5ad(INPUT_PATH)
            else:
                adata = spAlignDE.load_cross_sample_data(INPUT_PATH)
            if "is_spatial_filtered" in adata.obs:
                spatial_filter = adata.obs["is_spatial_filtered"].astype(bool).to_numpy()
                adata = adata[spatial_filter].copy()
            if "spatial" not in adata.obsm:
                adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=float)

            sample_values = adata.obs["sample_id"].astype(str)
            keep_per_sample = []
            for sample in sorted(sample_values.unique()):
                matrix = adata[sample_values.eq(sample).to_numpy()].X
                if sp.issparse(matrix):
                    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
                else:
                    detected = (np.asarray(matrix) > 0).sum(axis=0)
                threshold = max(1, int(np.ceil(0.01 * matrix.shape[0])))
                keep = detected >= threshold
                keep_per_sample.append(keep)
                print(f"{sample}: {keep.sum():,} genes detected in at least {threshold} spots")

            common_keep = np.logical_and.reduce(keep_per_sample)
            adata = adata[:, common_keep].copy()
            counts = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
            counts = counts.astype(np.float32, copy=False)
            totals = counts.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1
            normalized = counts / totals * 250.0 + 1.0

            target_distribution = np.sort(normalized, axis=1).mean(axis=0)
            order = np.argsort(normalized, axis=1)
            quantile_normalized = np.empty_like(normalized)
            np.put_along_axis(
                quantile_normalized,
                order,
                np.broadcast_to(target_distribution, normalized.shape),
                axis=1,
            )
            adata.X = quantile_normalized
            adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=float)
            spAlignDE.validate_cross_sample_anndata(adata, require_cluster=False)

            print(adata)
            print(adata.obs["sample_id"].value_counts().sort_index())
            print("Genes after the shared 1% filter:", adata.n_vars)
            """
        ),
        markdown(
            """
            ## Run joint BANKSY/Harmony/Leiden clustering

            The kidney-specific settings reproduce the four-domain structure
            used for this spot-level example: BANKSY lambda 0.2, 30 principal
            components, 100 SNN neighbors and Leiden resolution 0.2.
            """
        ),
        code(
            """
            config = spAlignDE.JointClusteringConfig(
                num_neighbors=30,
                banksy_lambda=0.2,
                pca_dim=30,
                resolution=0.2,
                snn_neighbors=100,
                harmony_theta=2.0,
                harmony_max_iter=20,
                refine_boundaries=True,
                compute_umap=False,
                random_state=WORKFLOW_SEED,
                leiden_flavor="leidenalg",
                leiden_n_iterations=-1,
            )

            adata_clustered, clustering_details = spAlignDE.cluster_joint(
                adata,
                config=config,
                return_details=True,
            )
            print("Raw clusters:", adata_clustered.obs["cluster_raw"].nunique())
            print("Refined clusters:", adata_clustered.obs["cluster_refined"].nunique())
            print(
                adata_clustered.obs.groupby(
                    ["sample_id", "cluster"], observed=False
                ).size().unstack(fill_value=0)
            )
            """
        ),
        markdown(
            """
            ## Inspect raw and refined kidney structures

            Columns correspond to NL3 and IL3. The top row shows raw joint
            Leiden clusters and the bottom row shows boundary-refined clusters.
            All panels share one cluster palette.
            """
        ),
        code(
            """
            fig, axes = spAlignDE.plot_joint_cluster_refinement(
                adata_clustered,
                samples=["NL3", "IL3"],
                point_size=1.5,
                alpha=0.9,
                figsize=(6, 8),
            )
            plt.show()
            clustering_details["refinement_stats"]
            """
        ),
        markdown(
            """
            ## Save the clustered kidney AnnData

            The alignment notebook uses the refined `cluster` labels while
            retaining `cluster_raw` and `cluster_refined` for inspection.
            """
        ),
        code(
            """
            clustered_path = OUTPUT_DIR / "kidney_IL3_NL3_joint_clustered.h5ad"
            adata_clustered.write_h5ad(clustered_path)
            print("Saved:", clustered_path.relative_to(PROJECT_ROOT))
            print("Output shape:", adata_clustered.shape)
            print("Cluster columns:", ["cluster_raw", "cluster_refined", "cluster"])
            """
        ),
    ]


def kidney_alignment_cells() -> list:
    return [
        markdown(
            """
            # Cross-sample alignment - two mouse kidney sections from Visium

            This notebook aligns the IL3 query section to the fixed NL3
            reference. It starts from the clustered AnnData and follows the
            same visual sequence as the MERFISH mouse-brain tutorial:
            pre-alignment, pseudo-image construction, S-LDDMM, and overlap
            comparisons without and with cluster colors.
            """
        ),
        markdown(
            """
            ## Installation and input

            ```bash
            cd /path/to/spAlignDE
            python -m pip install -e ".[clustering,tutorial]"
            ```

            By default the notebook reads the output of the kidney clustering
            notebook. Override it with:

            ```bash
            export SPALIGNDE_KIDNEY_CLUSTERED_H5AD=/path/to/clustered_kidney.h5ad
            ```
            """
        ),
        code(
            """
            %matplotlib inline

            from pathlib import Path
            import os
            import warnings

            import anndata as ad
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import spAlignDE

            WORKFLOW_SEED = 1000
            seed_controls = spAlignDE.set_random_seed(
                WORKFLOW_SEED,
                deterministic_torch=True,
            )

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings(
                "ignore", message="pkg_resources is deprecated as an API"
            )
            warnings.filterwarnings("ignore", message="Some cells have zero counts")
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                "font.size": 8,
                "axes.spines.right": False,
                "axes.spines.top": False,
                "legend.frameon": False,
            })


            def find_project_root(start: Path) -> Path:
                start = start.resolve()
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists():
                        return candidate
                raise FileNotFoundError("Run this notebook from inside the spAlignDE repository")


            PROJECT_ROOT = find_project_root(Path.cwd())
            OUTPUT_DIR = PROJECT_ROOT / "tutorials" / "cross_sample" / "kidney" / "output"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            default_path = OUTPUT_DIR / "kidney_IL3_NL3_joint_clustered.h5ad"
            clustered_path = Path(
                os.environ.get("SPALIGNDE_KIDNEY_CLUSTERED_H5AD", default_path)
            ).expanduser().resolve()

            adata = ad.read_h5ad(clustered_path)
            spAlignDE.validate_cross_sample_anndata(adata)
            QUERY = "IL3"
            REFERENCE = "NL3"
            COORDINATE_SCALE = 50.0
            raw_spatial = np.asarray(adata.obsm["spatial"], dtype=float).copy()
            adata_scaled = adata.copy()
            adata_scaled.obsm["spatial"] = raw_spatial * COORDINATE_SCALE

            print(adata)
            print(adata.obs["sample_id"].value_counts().sort_index())
            print("Internal coordinate scale:", COORDINATE_SCALE)
            """
        ),
        markdown(
            """
            ## Selected manual pre-alignment

            Automatic shared-cluster centroid fitting is disabled for this
            spot-level pair because only four shared structures are available
            and the automatic similarity fit is unstable. The fixed manual
            transform preserves scale and orientation and applies the selected
            center-to-center translation used by the validated kidney workflow.
            The four values remain explicit and may be overridden for another
            pair.
            """
        ),
        code(
            """
            manual_config = spAlignDE.ManualPrealignmentConfig(
                scale=float(os.environ.get("SPALIGNDE_MANUAL_SCALE", 1.0)),
                theta_deg=float(os.environ.get("SPALIGNDE_MANUAL_THETA_DEG", 0.0)),
                translation_x=float(
                    os.environ.get("SPALIGNDE_MANUAL_TX", -36.20040965)
                ),
                translation_y=float(
                    os.environ.get("SPALIGNDE_MANUAL_TY", -153.38356513)
                ),
            )
            prealignment = spAlignDE.prealign_cross_sample_manual(
                adata_scaled,
                query_sample=QUERY,
                reference_sample=REFERENCE,
                config=manual_config,
            )
            for key in ("scale", "theta_deg", "translation_x", "translation_y"):
                print(f"{key}: {prealignment.params[key]}")
            spAlignDE.plot_prealignment_result(
                prealignment,
                point_size=3.0,
                point_alpha=0.35,
                figsize=(8, 4.5),
            )
            """
        ),
        markdown(
            """
            ## Convert spots to continuous pseudo-images

            The four shared kidney domains define composition channels.
            Density is retained for visualization but given zero optimization
            weight, matching the original kidney analysis.
            """
        ),
        code(
            """
            fields = spAlignDE.rasterize_cross_sample(
                prealignment.adata,
                query_sample=QUERY,
                reference_sample=REFERENCE,
                config=spAlignDE.RasterizationConfig(
                    grid_spacing=30,
                    grid_expand=1.05,
                    blur_sigma=1,
                    cluster_weight=1,
                    density_weight=0,
                ),
            )
            print("Shared cluster channels:", len(fields.shared_clusters))
            print("Pseudo-image shape:", fields.query_image.shape)
            print("Shared grid:", len(fields.grid_y), "x", len(fields.grid_x))
            spAlignDE.plot_rasterized_fields(fields, figsize=(6, 5))
            """
        ),
        markdown(
            """
            ## Shooting-based LDDMM refinement

            This run uses the requested kidney parameters:
            `a=500`, `nt=5`, `grid_step=250`, `lrM=50`, and `niter=5000`.
            `minimum_momentum_lr` is also set to 50 so the requested momentum
            learning rate is not replaced by the package default lower bound.
            """
        ),
        code(
            """
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            slddmm_config = spAlignDE.SLDDMMConfig(
                kernel_scale=500,
                kernel_power=2,
                velocity_expand=2,
                time_steps=5,
                velocity_grid_spacing=250,
                iterations=5000,
                momentum_lr=50,
                minimum_momentum_lr=50,
                restore_best_checkpoint=False,
                dtype="float32",
            )
            result = spAlignDE.run_slddmm_alignment(
                prealignment.adata,
                fields,
                config=slddmm_config,
                device=device,
                prealignment=prealignment,
                verbose=True,
                print_every=500,
            )
            print("Alignment summary (scaled-coordinate fit):")
            for key, value in result.metrics.items():
                print(f"  {key}: {value}")
            """
        ),
        markdown(
            """
            ## Return to the original kidney coordinate scale

            S-LDDMM is fitted after multiplying the compact array coordinates
            by 50. The standardized output columns are divided by the same
            factor, and `adata.obsm["spatial"]` is restored before plotting and
            saving.
            """
        ),
        code(
            """
            output_columns = ["x_prealigned", "y_prealigned", "x_aligned", "y_aligned"]
            result.adata.obs.loc[:, output_columns] = (
                result.adata.obs.loc[:, output_columns].to_numpy(dtype=float)
                / COORDINATE_SCALE
            )
            result.adata.obsm["spatial"] = raw_spatial
            result.adata.uns["spAlignDE"]["cross_sample_alignment"][
                "internal_coordinate_scale"
            ] = COORDINATE_SCALE

            spAlignDE.plot_alignment_result(
                result,
                point_size=3.0,
                alpha=0.28,
                figsize=(7, 4.2),
            )
            spAlignDE.plot_cluster_alignment_result(
                result,
                point_size=2.2,
                target_alpha=0.55,
                source_alpha=0.75,
                figsize=(8, 4.4),
            )
            result.cluster_performance.sort_values("cluster")
            """
        ),
        markdown(
            """
            ## Save the aligned kidney AnnData

            The fixed NL3 coordinates are unchanged. The IL3 output coordinates
            are stored in `x_prealigned`, `y_prealigned`, `x_aligned`, and
            `y_aligned`, all in the original array-coordinate scale.
            """
        ),
        code(
            """
            aligned_path = OUTPUT_DIR / "kidney_IL3_to_NL3_aligned.h5ad"
            result.adata.write_h5ad(aligned_path)

            sample = result.adata.obs["sample_id"].astype(str)
            reference_mask = sample.eq(REFERENCE).to_numpy()
            reference_output = result.adata.obs.loc[
                reference_mask, output_columns
            ].to_numpy(dtype=float)
            assert np.allclose(reference_output[:, :2], raw_spatial[reference_mask])
            assert np.allclose(reference_output[:, 2:], raw_spatial[reference_mask])

            print("Saved:", aligned_path.relative_to(PROJECT_ROOT))
            print(result.adata.obs.loc[sample.eq(QUERY), output_columns].head())
            print("Reference coordinates verified unchanged.")
            """
        ),
    ]


def breast_clustering_cells() -> list:
    return [
        markdown(
            """
            # Joint clustering - two breast cancer sections from Xenium

            This notebook builds shared spatial structures for Xenium breast
            cancer Rep1 and Rep2. It begins with the original combined
            286,532-cell expression object, performs the selected sample-wise
            QC and batch-aware HVG300 preprocessing, and then runs the
            AnnData-native joint clustering workflow.
            """
        ),
        markdown(
            """
            ## Installation and input

            ```bash
            cd /path/to/spAlignDE
            python -m pip install -e ".[clustering,tutorial]"
            export SPALIGNDE_BREAST_CANCER_INPUT=/path/to/xenium_rep1_rep2_joint_input.h5ad
            ```

            Download In Situ Sample 1 Replicates 1 and 2 from the
            [10x Genomics Xenium human-breast dataset](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast).
            """
        ),
        code(
            """
            %matplotlib inline

            from pathlib import Path
            import os
            import warnings

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import scanpy as sc
            import spAlignDE

            WORKFLOW_SEED = 1000
            seed_controls = spAlignDE.set_random_seed(
                WORKFLOW_SEED,
                deterministic_torch=True,
            )

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings(
                "ignore", message="pkg_resources is deprecated as an API"
            )
            warnings.filterwarnings("ignore", message="Some cells have zero counts")
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                "font.size": 8,
                "axes.spines.right": False,
                "axes.spines.top": False,
                "legend.frameon": False,
            })


            def find_project_root(start: Path) -> Path:
                start = start.resolve()
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists():
                        return candidate
                raise FileNotFoundError("Run this notebook from inside the spAlignDE repository")


            PROJECT_ROOT = find_project_root(Path.cwd())
            OUTPUT_DIR = PROJECT_ROOT / "tutorials" / "cross_sample" / "breast_cancer" / "output"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            input_value = os.environ.get("SPALIGNDE_BREAST_CANCER_INPUT")
            if not input_value:
                raise EnvironmentError(
                    "Set SPALIGNDE_BREAST_CANCER_INPUT to the original combined Xenium .h5ad"
                )
            INPUT_PATH = Path(input_value).expanduser().resolve()
            print("Input:", INPUT_PATH.name)
            """
        ),
        markdown(
            """
            ## Sample-wise QC and batch-aware HVG selection

            Within each replicate, cells below the 10th percentile of
            transcript count or outside the 1st-99th percentile ranges of cell
            and nucleus area are removed. The top 300 highly variable genes are
            then selected after total-count normalization and log
            transformation, using `sample_id` as the batch key.
            """
        ),
        code(
            """
            adata = spAlignDE.load_cross_sample_data(INPUT_PATH)
            adata.obs["sample_id"] = adata.obs["sample_id"].astype(str)

            thresholds = {}
            keep = pd.Series(False, index=adata.obs_names)
            summary_rows = []
            for sample in sorted(adata.obs["sample_id"].unique()):
                sample_mask = adata.obs["sample_id"].eq(sample)
                obs = adata.obs.loc[sample_mask]
                values = {
                    "transcript_counts_min": float(obs["transcript_counts"].quantile(0.10)),
                    "cell_area_min": float(obs["cell_area"].quantile(0.01)),
                    "cell_area_max": float(obs["cell_area"].quantile(0.99)),
                    "nucleus_area_min": float(obs["nucleus_area"].quantile(0.01)),
                    "nucleus_area_max": float(obs["nucleus_area"].quantile(0.99)),
                }
                thresholds[sample] = values
                pass_qc = (
                    (obs["transcript_counts"] >= values["transcript_counts_min"])
                    & (obs["cell_area"] >= values["cell_area_min"])
                    & (obs["cell_area"] <= values["cell_area_max"])
                    & (obs["nucleus_area"] >= values["nucleus_area_min"])
                    & (obs["nucleus_area"] <= values["nucleus_area_max"])
                )
                keep.loc[obs.index] = pass_qc.to_numpy()
                summary_rows.append({
                    "sample_id": sample,
                    "n_cells_before": int(len(obs)),
                    "n_cells_after_qc": int(pass_qc.sum()),
                    **values,
                })

            qc = adata[keep.to_numpy()].copy()
            normalized = qc.copy()
            sc.pp.normalize_total(normalized, target_sum=1e4)
            sc.pp.log1p(normalized)
            sc.pp.highly_variable_genes(
                normalized,
                n_top_genes=300,
                flavor="seurat",
                batch_key="sample_id",
            )
            hvg_mask = normalized.var["highly_variable"].to_numpy()
            adata_filtered = qc[:, hvg_mask].copy()
            adata_filtered.uns["qc_thresholds"] = thresholds
            adata_filtered.uns["filtering"] = {
                "qc_transcript_min_quantile": 0.10,
                "qc_area_low_quantile": 0.01,
                "qc_area_high_quantile": 0.99,
                "hvg_n_top": int(hvg_mask.sum()),
                "hvg_batch_key": "sample_id",
            }
            print(pd.DataFrame(summary_rows))
            print(adata_filtered)
            """
        ),
        markdown(
            """
            ## Run the selected joint clustering model

            The selected breast-cancer setting uses BANKSY lambda 1.0,
            Harmony theta 4.0 and Leiden resolution 0.2.
            """
        ),
        code(
            """
            config = spAlignDE.JointClusteringConfig(
                num_neighbors=30,
                banksy_lambda=1.0,
                pca_dim=20,
                resolution=0.2,
                snn_neighbors=50,
                harmony_theta=4.0,
                harmony_max_iter=30,
                refine_boundaries=True,
                compute_umap=False,
                random_state=WORKFLOW_SEED,
                leiden_flavor="leidenalg",
                leiden_n_iterations=-1,
            )
            adata_clustered, clustering_details = spAlignDE.cluster_joint(
                adata_filtered,
                config=config,
                return_details=True,
            )
            print("Raw clusters:", adata_clustered.obs["cluster_raw"].nunique())
            print("Refined clusters:", adata_clustered.obs["cluster_refined"].nunique())
            print(
                adata_clustered.obs.groupby(
                    ["sample_id", "cluster"], observed=False
                ).size().unstack(fill_value=0)
            )
            """
        ),
        markdown(
            """
            ## Inspect raw and refined breast-cancer structures

            Rep1 and Rep2 are shown in columns. Raw and boundary-refined
            clusters are shown in rows with one shared color map.
            """
        ),
        code(
            """
            fig, axes = spAlignDE.plot_joint_cluster_refinement(
                adata_clustered,
                samples=["Rep1", "Rep2"],
                point_size=0.35,
                alpha=0.7,
            )
            plt.show()
            clustering_details["refinement_stats"]
            """
        ),
        markdown(
            """
            ## Save the clustered breast-cancer AnnData

            The refined `cluster` column is the semantic input to the Rep2 to
            Rep1 alignment notebook.
            """
        ),
        code(
            """
            clustered_path = OUTPUT_DIR / "breast_cancer_Rep1_Rep2_joint_clustered.h5ad"
            adata_clustered.write_h5ad(clustered_path)
            print("Saved:", clustered_path.relative_to(PROJECT_ROOT))
            print("Output shape:", adata_clustered.shape)
            print("Cluster columns:", ["cluster_raw", "cluster_refined", "cluster"])
            """
        ),
    ]


def breast_alignment_cells() -> list:
    return [
        markdown(
            """
            # Cross-sample alignment - two breast cancer sections from Xenium

            This notebook aligns the Rep2 query section to the fixed Rep1
            reference. It uses the selected manual pre-alignment and shows the
            same sequence of pre-alignment, pseudo-image and post-S-LDDMM
            comparisons as the MERFISH mouse-brain tutorial.
            """
        ),
        markdown(
            """
            ## Installation and input

            ```bash
            cd /path/to/spAlignDE
            python -m pip install -e ".[clustering,tutorial]"
            ```

            By default the notebook reads the output of the breast-cancer
            clustering notebook. Override it with:

            ```bash
            export SPALIGNDE_BREAST_CANCER_CLUSTERED_H5AD=/path/to/clustered_breast_cancer.h5ad
            ```
            """
        ),
        code(
            """
            %matplotlib inline

            from pathlib import Path
            import os
            import warnings

            import anndata as ad
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import spAlignDE

            WORKFLOW_SEED = 1000
            seed_controls = spAlignDE.set_random_seed(
                WORKFLOW_SEED,
                deterministic_torch=True,
            )

            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            mpl.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                "font.size": 8,
                "axes.spines.right": False,
                "axes.spines.top": False,
                "legend.frameon": False,
            })


            def find_project_root(start: Path) -> Path:
                start = start.resolve()
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists():
                        return candidate
                raise FileNotFoundError("Run this notebook from inside the spAlignDE repository")


            PROJECT_ROOT = find_project_root(Path.cwd())
            OUTPUT_DIR = PROJECT_ROOT / "tutorials" / "cross_sample" / "breast_cancer" / "output"
            default_path = OUTPUT_DIR / "breast_cancer_Rep1_Rep2_joint_clustered.h5ad"
            clustered_path = Path(
                os.environ.get("SPALIGNDE_BREAST_CANCER_CLUSTERED_H5AD", default_path)
            ).expanduser().resolve()

            adata = ad.read_h5ad(clustered_path)
            spAlignDE.validate_cross_sample_anndata(adata)
            QUERY = "Rep2"
            REFERENCE = "Rep1"
            print(adata)
            print(adata.obs["sample_id"].value_counts().sort_index())
            """
        ),
        markdown(
            """
            ## Selected manual pre-alignment

            Automatic shared-cluster centroid fitting is disabled.
            The selected Rep2-to-Rep1 transform preserves scale and
            orientation and applies the translation determined during the
            original manual QC. Environment variables may override the four
            values for another pair.
            """
        ),
        code(
            """
            manual_config = spAlignDE.ManualPrealignmentConfig(
                scale=float(os.environ.get("SPALIGNDE_MANUAL_SCALE", 1.0)),
                theta_deg=float(os.environ.get("SPALIGNDE_MANUAL_THETA_DEG", 0.0)),
                translation_x=float(
                    os.environ.get("SPALIGNDE_MANUAL_TX", -177.46841472056394)
                ),
                translation_y=float(
                    os.environ.get("SPALIGNDE_MANUAL_TY", 2215.1264424752544)
                ),
            )
            prealignment = spAlignDE.prealign_cross_sample_manual(
                adata,
                query_sample=QUERY,
                reference_sample=REFERENCE,
                config=manual_config,
            )
            for key in ("scale", "theta_deg", "translation_x", "translation_y"):
                print(f"{key}: {prealignment.params[key]}")
            spAlignDE.plot_prealignment_result(prealignment)
            """
        ),
        markdown(
            """
            ## Convert cells to continuous pseudo-images

            Shared refined clusters define composition channels, and the
            log-scaled pooled density channel receives equal optimization
            weight.
            """
        ),
        code(
            """
            fields = spAlignDE.rasterize_cross_sample(
                prealignment.adata,
                query_sample=QUERY,
                reference_sample=REFERENCE,
                config=spAlignDE.RasterizationConfig(
                    grid_spacing=30,
                    grid_expand=1.05,
                    blur_sigma=1,
                    cluster_weight=1,
                    density_weight=1,
                ),
            )
            print("Shared cluster channels:", len(fields.shared_clusters))
            print("Pseudo-image shape:", fields.query_image.shape)
            print("Shared grid:", len(fields.grid_y), "x", len(fields.grid_x))
            spAlignDE.plot_rasterized_fields(fields)
            """
        ),
        markdown(
            """
            ## Shooting-based LDDMM refinement

            The selected breast-cancer run retains the original S-LDDMM
            settings: `a=300`, `nt=3`, `grid_step=100`, `lrM=4000`, and
            `niter=500`.
            """
        ),
        code(
            """
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            slddmm_config = spAlignDE.SLDDMMConfig(
                kernel_scale=300,
                kernel_power=2,
                velocity_expand=2,
                time_steps=3,
                velocity_grid_spacing=100,
                iterations=500,
                momentum_lr=4000,
                minimum_momentum_lr=4000,
                momentum_gradient_clip=1000,
                restore_best_checkpoint=False,
                dtype="float32",
            )
            result = spAlignDE.run_slddmm_alignment(
                prealignment.adata,
                fields,
                config=slddmm_config,
                device=device,
                prealignment=prealignment,
                verbose=True,
                print_every=50,
            )
            print("Alignment summary:")
            for key, value in result.metrics.items():
                print(f"  {key}: {value}")
            """
        ),
        markdown(
            """
            ## Alignment visual comparisons

            The first figure emphasizes tissue geometry without cluster
            colors. The second uses the shared cluster palette to expose local
            structural correspondence before and after S-LDDMM.
            """
        ),
        code(
            """
            spAlignDE.plot_alignment_result(result)
            spAlignDE.plot_cluster_alignment_result(result)
            result.cluster_performance.sort_values("cluster")
            """
        ),
        markdown(
            """
            ## Save the aligned breast-cancer AnnData

            Rep1 remains fixed. Rep2 pre-aligned and final coordinates are
            written to the four standardized output columns.
            """
        ),
        code(
            """
            aligned_path = OUTPUT_DIR / "breast_cancer_Rep2_to_Rep1_aligned.h5ad"
            result.adata.write_h5ad(aligned_path)

            output_columns = ["x_prealigned", "y_prealigned", "x_aligned", "y_aligned"]
            sample = result.adata.obs["sample_id"].astype(str)
            reference_mask = sample.eq(REFERENCE).to_numpy()
            reference_output = result.adata.obs.loc[
                reference_mask, output_columns
            ].to_numpy(dtype=float)
            reference_raw = np.asarray(result.adata.obsm["spatial"])[reference_mask]
            assert np.allclose(reference_output[:, :2], reference_raw)
            assert np.allclose(reference_output[:, 2:], reference_raw)

            print("Saved:", aligned_path.relative_to(PROJECT_ROOT))
            print(result.adata.obs.loc[sample.eq(QUERY), output_columns].head())
            print("Reference coordinates verified unchanged.")
            """
        ),
    ]


def main() -> None:
    write_notebook(
        TUTORIAL_ROOT / "kidney" / "01_joint_clustering.ipynb",
        kidney_clustering_cells(),
    )
    write_notebook(
        TUTORIAL_ROOT / "kidney" / "02_alignment.ipynb",
        kidney_alignment_cells(),
    )
    write_notebook(
        TUTORIAL_ROOT / "breast_cancer" / "01_joint_clustering.ipynb",
        breast_clustering_cells(),
    )
    write_notebook(
        TUTORIAL_ROOT / "breast_cancer" / "02_alignment.ipynb",
        breast_alignment_cells(),
    )


if __name__ == "__main__":
    main()
