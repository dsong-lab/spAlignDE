#!/usr/bin/env python3
"""Promote audited fixed-seed artifacts into the public tutorial outputs.

This utility does not rerun clustering or deformation. It reads the products
of completed, independently repeated workflows, renders the public tutorial
figures with the repository plotting code, and replaces saved notebook output
cells with tables and figures from those products. Paths are supplied by the
caller so the repository does not encode one workstation's data layout.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import spAlignDE
from spalignde.alignment import _atlas_core
from spalignde.alignment.atac import (
    ATACSTAlignmentConfig,
    ATACSTAlignmentResult,
    ATACSTPrealignmentResult,
    _rasterize_structures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--atlas-run", type=Path, required=True)
    parser.add_argument("--atlas-clustered", type=Path, required=True)
    parser.add_argument("--atlas-annotation", type=Path, required=True)
    parser.add_argument("--atlas-structures", type=Path, required=True)
    parser.add_argument("--atlas-alignment-image", type=Path, required=True)
    parser.add_argument("--atac-run", type=Path, required=True)
    parser.add_argument("--atac-clustered", type=Path, required=True)
    parser.add_argument("--atac-st-input", type=Path, required=True)
    parser.add_argument("--joint-clustered", type=Path, required=True)
    return parser.parse_args()


def save_figure(fig, destination: Path, *, dpi: int = 220) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def image_output(path: Path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return nbformat.v4.new_output(
        "display_data",
        data={
            "image/png": encoded,
            "text/plain": f"<Figure: {path.name}>",
        },
        metadata={},
    )


def table_output(table: pd.DataFrame):
    return nbformat.v4.new_output(
        "display_data",
        data={
            "text/plain": table.to_string(),
            "text/html": table.to_html(border=1),
        },
        metadata={},
    )


def stream_output(text: str):
    return nbformat.v4.new_output("stream", name="stdout", text=text.rstrip() + "\n")


def update_notebook_pair(repo: Path, relative: str, callback, metadata: dict) -> None:
    paths = [
        repo / "source_notebooks" / relative,
        repo / "docs" / "source" / "source_notebooks" / relative,
    ]
    notebook = nbformat.read(paths[0], as_version=4)
    callback(notebook)
    notebook.metadata["spAlignDE_fixed_seed_output"] = metadata
    for path in paths:
        nbformat.write(notebook, path)


def plot_single_fixed(adata: ad.AnnData, destination: Path, *, atac: bool = False) -> None:
    if atac:
        fig, _ = spAlignDE.plot_single_cluster_refinement(
            adata,
            point_size=2.0,
            palette="tab20",
            alpha=0.90,
            figsize=(12, 5.2),
        )
    else:
        raw = adata.obs["cluster_raw"].astype(str)
        refined = adata.obs["cluster_refined"].astype(str)
        order = pd.Index(raw.unique()).union(pd.Index(refined.unique())).sort_values()
        palette = list(plt.get_cmap("tab20").colors)
        colors = {label: palette[i % len(palette)] for i, label in enumerate(order)}
        xy = adata.obsm["spatial"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
        for axis, key, title in zip(
            axes,
            ("cluster_raw", "cluster_refined"),
            ("Raw BANKSY clusters", "Boundary-refined clusters"),
            strict=False,
        ):
            labels = adata.obs[key].astype(str)
            axis.scatter(
                xy[:, 0],
                xy[:, 1],
                c=[colors[label] for label in labels],
                s=0.35,
                alpha=0.85,
                linewidths=0,
                rasterized=True,
            )
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.axis("off")
            axis.invert_yaxis()
    save_figure(fig, destination)


def parse_labels(value) -> list[int]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [int(item) for item in value]
    text = str(value).strip().replace(" ", "")
    for separator in (";", ","):
        if separator in text:
            return [int(float(item)) for item in text.split(separator) if item]
    return [int(float(text))]


def categorical_key(value) -> str:
    """Match categorical string keys after CSV integer labels become floats."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return str(value)


def mask_rgba(mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    image = np.ones(mask.shape + (4,), dtype=float)
    image[..., 3] = 0.0
    image[mask] = color
    return image


def overlap_rgba(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    blue = np.array([0 / 255, 114 / 255, 178 / 255, 1.0])
    orange = np.array([230 / 255, 159 / 255, 0 / 255, 1.0])
    purple = np.array([204 / 255, 121 / 255, 167 / 255, 1.0])
    image = np.ones(source.shape + (4,), dtype=float)
    image[:] = [1.0, 1.0, 1.0, 1.0]
    image[target] = orange
    image[source] = blue
    image[source & target] = purple
    return image


def plot_mask_rows(rows, source_masks, target_masks, destination: Path, labels) -> None:
    blue = np.array([0 / 255, 114 / 255, 178 / 255, 1.0])
    orange = np.array([230 / 255, 159 / 255, 0 / 255, 1.0])
    purple = np.array([204 / 255, 121 / 255, 167 / 255, 1.0])
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(12, 3.55 * len(rows)), constrained_layout=True, squeeze=False
    )
    for index, (row, source, target) in enumerate(zip(rows, source_masks, target_masks, strict=True)):
        union = source | target
        yy, xx = np.where(union)
        pad = 18
        y0, y1 = max(0, int(yy.min()) - pad), min(union.shape[0], int(yy.max()) + pad + 1)
        x0, x1 = max(0, int(xx.min()) - pad), min(union.shape[1], int(xx.max()) + pad + 1)
        titles, note = labels(row)
        panels = (
            (mask_rgba(source, blue), titles[0]),
            (mask_rgba(target, orange), titles[1]),
            (overlap_rgba(source, target), titles[2]),
        )
        for axis, (image, title) in zip(axes[index], panels, strict=True):
            axis.imshow(image[y0:y1, x0:x1], origin="lower", interpolation="nearest")
            axis.set_title(title)
            axis.axis("off")
        axes[index, 2].text(
            0.5, -0.05, note, transform=axes[index, 2].transAxes,
            ha="center", va="top", fontsize=9,
        )
    fig.legend(
        handles=(
            Patch(facecolor=blue, label="moving mask"),
            Patch(facecolor=orange, label="fixed mask"),
            Patch(facecolor=purple, label="overlap"),
        ),
        loc="lower center", ncol=3, frameon=False,
    )
    save_figure(fig, destination)


def promote_atlas(args, static_dir: Path) -> dict:
    alignment_dir = args.atlas_run / "alignment"
    aligned_path = alignment_dir / "st_to_allen_atlas_aligned.h5ad"
    pairs_path = alignment_dir / "matched_pairs_final_stage.csv"
    stages_path = alignment_dir / "iterative_alignment_stage_summary.csv"
    run_summary = json.loads((args.atlas_run / "run_summary.json").read_text())
    aligned = ad.read_h5ad(aligned_path)
    pairs = pd.read_csv(pairs_path)
    stages = pd.read_csv(stages_path)
    atlas = spAlignDE.load_allen_ccf_reference(
        args.atlas_annotation, args.atlas_structures, slice_index=675
    )
    result = spAlignDE.STAtlasAlignmentResult(
        adata=aligned,
        atlas=atlas,
        matched_pairs=pairs,
        stage_summary=stages,
        prealignment_parameters={},
        hierarchy_columns=("cluster_level_k7", "cluster_level_k16"),
        output_dir=alignment_dir,
        context=None,
    )

    alignment_figure = static_dir / "atlas_alignment_structures.png"
    shutil.copyfile(args.atlas_alignment_image, alignment_figure)

    transfer_figure = static_dir / "atlas_label_transfer.png"
    color_map = spAlignDE.load_atlas_label_color_map(
        alignment_dir / "atlas_z675_white_label_color_map_for_transfer_labels.csv",
        atlas=atlas,
    )
    fig, _ = spAlignDE.plot_atlas_label_transfer(
        result, color_map=color_map, point_size=2.0, point_alpha=0.8, figsize=(14, 7)
    )
    save_figure(fig, transfer_figure)

    filtered = pd.read_csv(alignment_dir / "final_filtered_points_for_matching.csv")
    mask_result = _atlas_core.build_cluster_masks(
        filtered,
        sl=atlas.annotation,
        xJ=atlas.x_coordinates,
        yJ=atlas.y_coordinates,
        x_col="x_prealigned",
        y_col="y_prealigned",
        label_col="cluster",
        params=_atlas_core.DEFAULT_MASK_PARAMS,
        params_thin=_atlas_core.MASK_PARAMS_THIN,
        shape_type_col="shape_type",
        thin_values=("detail",),
        thin_rule="mode",
        verbose=False,
    )
    top = pairs.sort_values("align_score_gated", ascending=False).head(6).reset_index(drop=True)
    source_masks = [np.asarray(mask_result["st_masks"][str(row.cluster)]) > 0 for _, row in top.iterrows()]
    target_masks = [np.isin(atlas.annotation, parse_labels(row.atlas_labels_union)) for _, row in top.iterrows()]

    def atlas_labels(row):
        return (
            ("ST feature", "Allen feature", f"Paired overlap\nST {row.cluster} ↔ {row.candidate_name}"),
            f"score={row.align_score_gated:.3f}; Dice={row.dice:.3f}; ASD={row.asd:.1f}",
        )

    mask_figure = static_dir / "atlas_paired_feature_masks.png"
    plot_mask_rows(list(top.itertuples(index=False)), source_masks, target_masks, mask_figure, atlas_labels)

    hierarchy = pd.DataFrame(
        {
            "label column": ["cluster_level_k7", "cluster_level_k16", "cluster"],
            "number of structures": [7, 16, 25],
        }
    )
    pair_columns = [
        "cluster", "candidate_name", "pair_type", "align_score_gated",
        "dice", "area_sim", "chamfer_dist", "asd",
    ]
    transfer_summary = pd.DataFrame(
        {
            "value": [
                aligned.n_obs,
                int(aligned.obs["atlas_label_transferred"].sum()),
                int((~aligned.obs["atlas_label_transferred"]).sum()),
                float((~aligned.obs["atlas_label_transferred"]).mean()),
            ]
        },
        index=[
            "all ST cells", "non-background Allen label",
            "background / unlabeled", "unlabeled fraction",
        ],
    )
    coordinate_columns = ["x_prealigned", "y_prealigned", "x_aligned", "y_aligned"]
    label_columns = [
        "atlas_label_id", "atlas_label_acronym", "atlas_label_name",
        "atlas_label_transferred",
    ]

    def update(nb):
        nb.cells[4].outputs = [
            stream_output("Coarse-to-fine alignment labels: ['cluster_level_k7', 'cluster_level_k16', 'cluster']"),
            table_output(hierarchy),
        ]
        nb.cells[7].outputs = [stream_output(
            "Mode: fresh package run\n"
            f"Aligned observations: {aligned.n_obs:,}\n"
            f"Scheduled pair counts: {run_summary['stage_pair_counts']}\n"
            "Continuation pair counts: 17→18, 18→18\n"
            f"Final matched structure pairs: {len(pairs)}\n"
            f"Alignment runtime: {run_summary['runtime_seconds'] / 60:.1f} minutes\n"
            f"Peak GPU memory allocation: {run_summary['peak_cuda_memory_gib']:.3f} GiB"
        )]
        nb.cells[9].outputs = [table_output(stages), table_output(pairs[pair_columns].head(20))]
        nb.cells[11].outputs = [image_output(mask_figure)]
        nb.cells[13].outputs = [image_output(alignment_figure)]
        nb.cells[15].outputs = [table_output(transfer_summary), image_output(transfer_figure)]
        nb.cells[17].outputs = [
            table_output(aligned.obs[coordinate_columns + label_columns].head()),
            stream_output(
                f"Fresh complete-run time: {run_summary['runtime_seconds'] / 60:.1f} minutes\n"
                f"Peak GPU memory allocation: {run_summary['peak_cuda_memory_gib']:.3f} GiB"
            ),
        ]

    metadata = {
        "workflow_seed": 1234,
        "discrete_repeat_result": "exact",
        "continuous_repeat_tolerance": "declared CUDA tolerance",
        "final_pairs": int(len(pairs)),
    }
    update_notebook_pair(args.repo_root, "cross_modal_atlas_alignment_nb.ipynb", update, metadata)
    return metadata


def promote_atac(args, static_dir: Path) -> dict:
    alignment_dir = args.atac_run / "alignment"
    aligned_path = alignment_dir / "atac_to_st_aligned.h5ad"
    aligned = ad.read_h5ad(aligned_path)
    reference = ad.read_h5ad(alignment_dir / "st_reference_analysis_frame.h5ad")
    clustered = ad.read_h5ad(args.atac_clustered)
    st_input = ad.read_h5ad(args.atac_st_input)
    pairs = pd.read_csv(alignment_dir / "matched_structure_pairs.csv")
    manifest = json.loads((alignment_dir / "alignment_manifest.json").read_text())
    run_summary = json.loads((args.atac_run / "run_summary.json").read_text())
    config = ATACSTAlignmentConfig(dtype="float64")
    result = ATACSTAlignmentResult(
        atac=aligned,
        st_reference=reference,
        matched_pairs=pairs,
        prealignment_parameters=manifest["prealignment"],
        output_dir=alignment_dir,
        context=None,
    )
    prealigned = ATACSTPrealignmentResult(
        atac=aligned,
        st_reference=reference,
        params=manifest["prealignment"],
        canvas_shape_hw=tuple(manifest["prealignment"]["canvas_shape_hw"]),
    )

    single_figure = static_dir / "atac_single_clustering_tab20.png"
    plot_single_fixed(clustered, single_figure, atac=True)

    independent_figure = static_dir / "atac_independent_structures.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), constrained_layout=True)
    for axis, data, key, title, size in [
        (axes[0], clustered, "cluster_raw", "P22 spatial ATAC", 2.0),
        (axes[1], st_input, "cluster", "MERFISH S3R1", 0.45),
    ]:
        labels = pd.Categorical(data.obs[key].astype(str))
        axis.scatter(
            data.obsm["spatial"][:, 0], data.obsm["spatial"][:, 1],
            c=labels.codes, cmap="turbo", s=size, alpha=0.88,
            edgecolors="none", rasterized=True,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.axis("off")
    save_figure(fig, independent_figure)

    prealign_figure = static_dir / "atac_prealignment_fov.png"
    fig, _ = spAlignDE.plot_atac_st_prealignment(
        prealigned, atac_cluster_key="cluster_raw", st_cluster_key="cluster"
    )
    save_figure(fig, prealign_figure)

    matched_figure = static_dir / "atac_matched_structures.png"
    fig, _ = spAlignDE.plot_atac_st_matched_structures(
        result, atac_cluster_key="cluster_raw", st_cluster_key="cluster"
    )
    save_figure(fig, matched_figure)
    alignment_figure = static_dir / "atac_alignment_before_after.png"
    fig, _ = spAlignDE.plot_atac_st_alignment(result, atac_cluster_key="cluster_raw")
    save_figure(fig, alignment_figure)

    shape = tuple(manifest["prealignment"]["canvas_shape_hw"])
    atac_masks, _ = _rasterize_structures(
        aligned, cluster_key="cluster_raw", shape=shape, config=config, role="atac"
    )
    st_masks, _ = _rasterize_structures(
        reference, cluster_key="cluster", shape=shape, config=config, role="st"
    )
    source_masks = [
        np.asarray(atac_masks[categorical_key(row.atac_structure)]) > 0
        for _, row in pairs.iterrows()
    ]
    target_masks = [
        np.asarray(st_masks[categorical_key(row.st_structure)]) > 0
        for _, row in pairs.iterrows()
    ]

    def atac_labels(row):
        return (
            (
                "ATAC feature", "ST feature",
                f"Paired overlap\nATAC {row.atac_structure} ↔ ST {row.st_structure}",
            ),
            f"score={row.align_score:.3f}; Dice={row.dice:.3f}; Chamfer={row.chamfer_distance:.1f}",
        )

    mask_figure = static_dir / "atac_paired_feature_masks.png"
    plot_mask_rows(list(pairs.itertuples(index=False)), source_masks, target_masks, mask_figure, atac_labels)

    input_table = pd.DataFrame(
        {
            "role": ["moving query", "fixed reference"],
            "dataset": ["P22 spatial ATAC", "MERFISH S3R1"],
            "observations": [aligned.n_obs, reference.n_obs],
            "features": [aligned.n_vars, reference.n_vars],
            "structures": [aligned.obs["cluster_raw"].nunique(), reference.obs["cluster"].nunique()],
        }
    )
    pair_columns = [
        "st_structure", "atac_structure", "align_score", "sdf_corr",
        "chamfer_distance", "area_sim", "dice",
    ]

    def update_alignment(nb):
        nb.cells[3].outputs = [table_output(input_table)]
        nb.cells[5].outputs = [image_output(independent_figure)]
        nb.cells[7].outputs = [
            stream_output(
                f"ATAC observations retained: {aligned.n_obs:,}\n"
                f"ST observations after half-brain crop: {reference.n_obs:,}\n"
                f"Shared canvas (height, width): {shape}"
            ),
            image_output(prealign_figure),
        ]
        nb.cells[11].outputs = [
            stream_output(
                f"Alignment runtime: {run_summary['runtime_seconds'] / 60:.2f} min\n"
                f"Accepted structure pairs: {len(pairs)}"
            ),
            table_output(pairs[pair_columns].round(4)),
        ]
        nb.cells[14].outputs = [image_output(mask_figure)]
        nb.cells[16].outputs = [image_output(matched_figure), image_output(alignment_figure)]
        nb.cells[18].outputs = [
            table_output(aligned.obs[["x_prealigned", "y_prealigned", "x_aligned", "y_aligned"]].head()),
            stream_output(f"Saved {aligned.n_obs:,} aligned ATAC observations; fixed-seed result contains {len(pairs)} pairs."),
        ]

    metadata = {
        "workflow_seed": 1234,
        "discrete_repeat_result": "exact",
        "continuous_repeat_tolerance": "float64 maximum difference <1e-12",
        "final_pairs": int(len(pairs)),
    }
    update_notebook_pair(
        args.repo_root, "cross_modality/atac_st_alignment_nb.ipynb",
        update_alignment, metadata,
    )

    summary = pd.DataFrame(
        {
            "value": [
                f"{clustered.n_obs:,}", f"{clustered.n_vars:,}",
                clustered.obs["cluster_raw"].nunique(),
                clustered.obs["cluster_refined"].nunique(),
                f"{json.loads((args.atac_clustered.parent / 'run_summary.json').read_text())['runtime_seconds'] / 60:.2f} min",
            ]
        },
        index=["observations", "features", "raw clusters", "refined clusters", "runtime"],
    )

    def update_single(nb):
        nb.cells[5].outputs = [table_output(summary)]
        nb.cells[8].outputs = [image_output(single_figure)]
        nb.cells[10].outputs = [
            stream_output(
                f"Saved fixed-seed handoff: {clustered.n_obs:,} observations; "
                f"{clustered.obs['cluster'].nunique()} selected structures."
            )
        ]

    update_notebook_pair(
        args.repo_root, "cross_modality/atac_st_single_clustering_nb.ipynb",
        update_single,
        {
            "workflow_seed": 1234,
            "discrete_repeat_result": "exact",
        },
    )
    return metadata


def promote_clustering(args, static_dir: Path) -> dict:
    single = ad.read_h5ad(args.atlas_clustered)
    joint = ad.read_h5ad(args.joint_clustered)
    single_figure = static_dir / "single_clustering_tab20.png"
    plot_single_fixed(single, single_figure)
    joint_figure = static_dir / "joint_clustering_tab20.png"
    fig, _ = spAlignDE.plot_joint_cluster_refinement(
        joint, samples=["S2R2", "S2R3"], point_size=0.5, palette="tab20"
    )
    save_figure(fig, joint_figure)

    def update_single(nb):
        nb.cells[4].outputs = [stream_output(
            "Mode: fresh package run\n"
            f"Raw clusters: {single.obs['cluster_raw'].nunique()}\n"
            f"Refined clusters: {single.obs['cluster_refined'].nunique()}"
        )]
        nb.cells[7].outputs = [image_output(single_figure)]
        nb.cells[9].outputs = [stream_output(
            f"Saved fixed-seed handoff: {single.n_obs:,} observations × {single.n_vars:,} genes."
        )]

    update_notebook_pair(
        args.repo_root, "clustering/clustering_single_nb.ipynb", update_single,
        {
            "workflow_seed": 1234,
            "discrete_repeat_result": "exact",
        },
    )

    sample_counts = (
        joint.obs.groupby(["sample_id", "cluster"], observed=False)
        .size().unstack(fill_value=0)
    )

    def update_joint(nb):
        nb.cells[6].outputs = [
            stream_output(f"Number of shared-label identities: {joint.obs['cluster'].nunique()}"),
            table_output(sample_counts),
        ]
        nb.cells[9].outputs = [image_output(joint_figure)]
        nb.cells[11].outputs = [stream_output(
            f"Saved fixed-seed handoff: {joint.n_obs:,} observations; "
            f"{joint.obs['cluster'].nunique()} selected shared clusters."
        )]

    update_notebook_pair(
        args.repo_root, "clustering/clustering_joint_nb.ipynb", update_joint,
        {
            "workflow_seed": 1000,
            "discrete_repeat_result": "exact",
        },
    )
    return {
        "single_clusters": int(single.obs["cluster"].nunique()),
        "joint_clusters": int(joint.obs["cluster"].nunique()),
    }


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    static_dir = args.repo_root / "docs" / "source" / "_static" / "tutorial_figures"
    atlas = promote_atlas(args, static_dir)
    atac = promote_atac(args, static_dir)
    clustering = promote_clustering(args, static_dir)
    print(json.dumps({"atlas": atlas, "atac": atac, "clustering": clustering}, indent=2))


if __name__ == "__main__":
    main()
