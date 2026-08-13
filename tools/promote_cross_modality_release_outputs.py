#!/usr/bin/env python3
"""Promote validated ST-to-H&E and ATAC-to-ST outputs into public docs.

The computational notebooks remain the canonical executable workflows. This
script only renders their validated artifacts into the stable static filenames
used by the prose tutorial and records cross-process reproducibility evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

import spAlignDE
from spalignde.alignment import _atlas_core
from spalignde.alignment.atac import (
    ATACSTAlignmentConfig,
    ATACSTAlignmentResult,
    ATACSTPrealignmentResult,
    _rasterize_structures,
)
from promote_fixed_seed_tutorial_outputs import (
    categorical_key,
    plot_mask_rows,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--atac-output", type=Path, required=True)
    parser.add_argument("--atac-repeat", type=Path, required=True)
    parser.add_argument("--he-figure-dir", type=Path, required=True)
    parser.add_argument("--atlas-output", type=Path)
    parser.add_argument("--atlas-repeat", type=Path)
    parser.add_argument("--atlas-annotation", type=Path)
    parser.add_argument("--atlas-structures", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse_atlas_labels(value) -> list[int]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [int(item) for item in value]
    text = str(value).strip().replace(" ", "")
    for separator in (";", ","):
        if separator in text:
            return [int(float(item)) for item in text.split(separator) if item]
    return [int(float(text))]


def notebook_result_summary(repo: Path, relative: str) -> dict:
    notebook = nbformat.read(repo / "source_notebooks" / relative, as_version=4)
    execution = notebook.metadata.get("spAlignDE_execution", {})
    return {
        "notebook": relative,
        "workflow_seed": execution.get("workflow_seed"),
        "fully_executed": execution.get("fully_executed"),
        "source_sha256": execution.get("source_sha256"),
        "saved_output_sha256": execution.get("saved_output_sha256"),
        "source_refresh_only": execution.get("source_refresh_only", False),
    }


def promote_atlas(args: argparse.Namespace, static_dir: Path) -> dict:
    required = {
        "atlas_output": args.atlas_output,
        "atlas_repeat": args.atlas_repeat,
        "atlas_annotation": args.atlas_annotation,
        "atlas_structures": args.atlas_structures,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Atlas promotion requires all Atlas arguments; missing: "
            + ", ".join(missing)
        )

    alignment_dir = args.atlas_output.resolve()
    repeat_dir = args.atlas_repeat.resolve()
    pairs = pd.read_csv(alignment_dir / "matched_pairs_final_stage.csv")
    repeat_pairs = pd.read_csv(repeat_dir / "matched_pairs_final_stage.csv")
    stages = pd.read_csv(alignment_dir / "iterative_alignment_stage_summary.csv")
    continuation = pd.read_csv(
        alignment_dir
        / "continue_alignment"
        / "continue_alignment_pair_counts.csv"
    )
    coordinates = pd.read_csv(
        alignment_dir / "final_aligned_all_points.csv",
        index_col=0,
    )
    repeat_coordinates = pd.read_csv(
        repeat_dir / "final_aligned_all_points.csv",
        index_col=0,
    )
    cell_order_exact = coordinates.index.equals(repeat_coordinates.index)
    repeat_coordinates = repeat_coordinates.reindex(coordinates.index)

    pair_key_columns = [
        "cluster",
        "atlas_labels_union",
        "pair_type",
        "candidate_name",
    ]
    pair_keys = sorted(
        map(tuple, pairs[pair_key_columns].astype(str).values.tolist())
    )
    repeat_pair_keys = sorted(
        map(tuple, repeat_pairs[pair_key_columns].astype(str).values.tolist())
    )
    pair_table_repeat_exact = pair_keys == repeat_pair_keys
    x_difference = float(
        np.max(
            np.abs(
                coordinates["x_aligned"].to_numpy(float)
                - repeat_coordinates["x_aligned"].to_numpy(float)
            )
        )
    )
    y_difference = float(
        np.max(
            np.abs(
                coordinates["y_aligned"].to_numpy(float)
                - repeat_coordinates["y_aligned"].to_numpy(float)
            )
        )
    )
    coordinate_tolerance = 0.1
    if not cell_order_exact or not pair_table_repeat_exact:
        raise AssertionError("Atlas discrete repeat contract failed")
    if max(x_difference, y_difference) > coordinate_tolerance:
        raise AssertionError(
            "Atlas coordinate repeat tolerance failed: "
            f"{x_difference}, {y_difference}"
        )

    atlas = spAlignDE.load_allen_ccf_reference(
        args.atlas_annotation,
        args.atlas_structures,
        slice_index=675,
    )
    filtered = pd.read_csv(
        alignment_dir / "final_filtered_points_for_matching.csv"
    )
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
    top = pairs.sort_values(
        "align_score_gated",
        ascending=False,
    ).head(6).reset_index(drop=True)
    source_masks = [
        np.asarray(mask_result["st_masks"][str(row.cluster)]) > 0
        for _, row in top.iterrows()
    ]
    target_masks = [
        np.isin(atlas.annotation, parse_atlas_labels(row.atlas_labels_union))
        for _, row in top.iterrows()
    ]

    def labels(row):
        return (
            (
                "ST feature",
                "Allen feature",
                f"Paired overlap\nST {row.cluster} ↔ {row.candidate_name}",
            ),
            (
                f"score={row.align_score_gated:.3f}; "
                f"Dice={row.dice:.3f}; ASD={row.asd:.1f}"
            ),
        )

    plot_mask_rows(
        list(top.itertuples(index=False)),
        source_masks,
        target_masks,
        static_dir / "atlas_paired_feature_masks.png",
        labels,
    )
    source_figure_dir = alignment_dir.parent / "figures"
    for source_name, destination_name in (
        ("st_to_allen_before_after.png", "atlas_alignment_structures.png"),
        ("st_to_allen_label_transfer.png", "atlas_label_transfer.png"),
    ):
        source = source_figure_dir / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        (static_dir / destination_name).write_bytes(source.read_bytes())

    return {
        "seed": 1234,
        "input_observations": int(len(coordinates)),
        "st_structure_levels": [7, 16, 25],
        "stage_iterations": [100, 500, 100],
        "stage_pair_counts": stages["n_pairs"].astype(int).tolist(),
        "continuation_iterations": 200,
        "continuation_pair_counts": continuation[
            ["n_pairs_before", "n_pairs_after"]
        ].astype(int).values.tolist(),
        "accepted_pairs": int(len(pairs)),
        "pairing_weights": [0.05, 0.05, 0.20, 0.50, 0.20],
        "continuation_kernel_scale": 200.0,
        "continuation_velocity_grid_spacing": 50.0,
        "restore_best_checkpoint": False,
        "continuation_restore_best_checkpoint": False,
        "pair_table_repeat_exact": pair_table_repeat_exact,
        "cell_order_repeat_exact": cell_order_exact,
        "coordinate_repeat_max_abs_difference": {
            "x": x_difference,
            "y": y_difference,
        },
        "coordinate_repeat_tolerance": coordinate_tolerance,
        "aligned_h5ad_sha256": sha256(
            alignment_dir / "st_to_allen_atlas_aligned.h5ad"
        ),
    }


def promote_atac(args: argparse.Namespace, static_dir: Path) -> dict:
    alignment_dir = args.atac_output / "alignment"
    aligned_path = alignment_dir / "atac_to_st_aligned.h5ad"
    reference_path = alignment_dir / "st_reference_analysis_frame.h5ad"
    aligned = ad.read_h5ad(aligned_path)
    reference = ad.read_h5ad(reference_path)
    pairs = read_pairs(alignment_dir / "matched_structure_pairs.csv")
    manifest = json.loads((alignment_dir / "alignment_manifest.json").read_text())

    pair_table = pd.DataFrame(pairs)
    numeric_columns = [
        "align_score", "base_score", "distance_gate", "sdf_corr",
        "chamfer_sim", "chamfer_distance", "area_sim", "dice",
        "st_area", "atac_area",
    ]
    for column in numeric_columns:
        pair_table[column] = pd.to_numeric(pair_table[column])
    result = ATACSTAlignmentResult(
        atac=aligned,
        st_reference=reference,
        matched_pairs=pair_table,
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

    fig, _ = spAlignDE.plot_atac_st_prealignment(
        prealigned, atac_cluster_key="cluster_raw", st_cluster_key="cluster"
    )
    save_figure(fig, static_dir / "atac_prealignment_fov.png")
    fig, _ = spAlignDE.plot_atac_st_matched_structures(
        result, atac_cluster_key="cluster_raw", st_cluster_key="cluster"
    )
    save_figure(fig, static_dir / "atac_matched_structures.png")
    fig, _ = spAlignDE.plot_atac_st_alignment(result, atac_cluster_key="cluster_raw")
    save_figure(fig, static_dir / "atac_alignment_before_after.png")

    shape = tuple(manifest["prealignment"]["canvas_shape_hw"])
    config = ATACSTAlignmentConfig(
        pair_score_threshold=0.21,
        velocity_grid_spacing=50.0,
        restore_best_checkpoint=False,
        dtype="float64",
    )
    atac_masks, _ = _rasterize_structures(
        aligned, cluster_key="cluster_raw", shape=shape, config=config, role="atac"
    )
    st_masks, _ = _rasterize_structures(
        reference, cluster_key="cluster", shape=shape, config=config, role="st"
    )
    source_masks = [
        np.asarray(atac_masks[categorical_key(row["atac_structure"])]) > 0
        for row in pairs
    ]
    target_masks = [
        np.asarray(st_masks[categorical_key(row["st_structure"])]) > 0
        for row in pairs
    ]

    def labels(row):
        return (
            (
                "ATAC feature",
                "ST feature",
                f"Paired overlap\nATAC {row['atac_structure']} ↔ ST {row['st_structure']}",
            ),
            "score={:.3f}; Dice={:.3f}; Chamfer={:.1f}".format(
                float(row["align_score"]),
                float(row["dice"]),
                float(row["chamfer_distance"]),
            ),
        )

    plot_mask_rows(
        pairs,
        source_masks,
        target_masks,
        static_dir / "atac_paired_feature_masks.png",
        labels,
    )

    repeat_pairs = read_pairs(args.atac_repeat / "matched_structure_pairs.csv")
    pair_keys = [(row["st_structure"], row["atac_structure"]) for row in pairs]
    repeat_keys = [
        (row["st_structure"], row["atac_structure"]) for row in repeat_pairs
    ]
    if pair_keys != repeat_keys:
        raise AssertionError("ATAC accepted pair table differs from fixed-seed repeat")
    score_diff = max(
        abs(float(left["align_score"]) - float(right["align_score"]))
        for left, right in zip(pairs, repeat_pairs)
    )
    if score_diff != 0.0:
        raise AssertionError(f"ATAC pair-score repeat difference: {score_diff}")

    repeat_csv = args.atac_repeat / "atac_to_st_alignment.csv"
    with repeat_csv.open(newline="", encoding="utf-8") as stream:
        repeat_rows = list(csv.DictReader(stream))
    x_repeat = np.asarray([float(row["x_aligned_to_st"]) for row in repeat_rows])
    y_repeat = np.asarray([float(row["y_aligned_to_st"]) for row in repeat_rows])
    x_diff = float(np.max(np.abs(aligned.obs["x_aligned"].to_numpy(float) - x_repeat)))
    y_diff = float(np.max(np.abs(aligned.obs["y_aligned"].to_numpy(float) - y_repeat)))
    if max(x_diff, y_diff) >= 1e-12:
        raise AssertionError(f"ATAC coordinate repeat tolerance failed: {x_diff}, {y_diff}")

    return {
        "seed": 1234,
        "input_st_observations": int(manifest["prealignment"]["reference_observations_before_crop"]),
        "analysis_st_observations": reference.n_obs,
        "atac_observations": aligned.n_obs,
        "accepted_pairs": len(pairs),
        "pair_score_threshold": 0.21,
        "kernel_scale": 100.0,
        "velocity_grid_spacing": 50.0,
        "restore_best_checkpoint": False,
        "dtype": "float64",
        "pair_table_repeat_exact": True,
        "pair_score_repeat_max_abs_difference": score_diff,
        "coordinate_repeat_max_abs_difference": {"x": x_diff, "y": y_diff},
        "coordinate_repeat_tolerance": 1e-12,
        "aligned_h5ad_sha256": sha256(aligned_path),
    }


def promote_he(args: argparse.Namespace, static_dir: Path) -> dict:
    # The public static figures are generated by the executable H&E notebook;
    # this audit verifies that the current files are present and records the
    # latest paper-figure sources without rasterizing PDFs in a second backend.
    for name in (
        "he_high_corr_marker_gene_spatial_canvas.pdf",
        "he_pair_overlap_after_lddmm.pdf",
    ):
        if not (args.he_figure_dir / name).is_file():
            raise FileNotFoundError(args.he_figure_dir / name)
    for name in (
        "histology_alignment_before_after.png",
        "histology_feature_clustering.png",
        "histology_paired_feature_overlap.png",
        "histology_paired_feature_overlap_after_lddmm.png",
    ):
        if not (static_dir / name).is_file():
            raise FileNotFoundError(static_dir / name)
    return {
        "seed": 0,
        "merged_cluster_target": 26,
        "final_cleaned_structures": 21,
        "accepted_pairs": 2,
        "kernel_scale": 60.0,
        "velocity_grid_spacing": 6.0,
        "restore_best_checkpoint": False,
        "dtype": "float64",
        "published_static_sha256": {
            name: sha256(static_dir / name)
            for name in (
                "histology_alignment_before_after.png",
                "histology_feature_clustering.png",
                "histology_paired_feature_overlap.png",
                "histology_paired_feature_overlap_after_lddmm.png",
            )
        },
    }


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    static_dir = args.repo_root / "docs" / "source" / "_static" / "tutorial_figures"
    report = {
        "schema_version": 1,
        "atac_to_st": promote_atac(args, static_dir),
        "st_to_he": promote_he(args, static_dir),
        "notebooks": [
            notebook_result_summary(
                args.repo_root, "cross_modality/atac_st_single_clustering_nb.ipynb"
            ),
            notebook_result_summary(
                args.repo_root, "cross_modality/atac_st_alignment_nb.ipynb"
            ),
            notebook_result_summary(
                args.repo_root, "cross_modality/st_he_feature_clustering_nb.ipynb"
            ),
            notebook_result_summary(
                args.repo_root, "cross_modality/st_he_alignment_nb.ipynb"
            ),
        ],
    }
    if args.atlas_output is not None:
        report["st_to_atlas"] = promote_atlas(args, static_dir)
        report["notebooks"].insert(
            0,
            notebook_result_summary(
                args.repo_root,
                "cross_modal_atlas_alignment_nb.ipynb",
            ),
        )
    destination = (
        args.repo_root / "docs" / "source" / "_static" /
        "cross_modality_reproducibility_manifest.json"
    )
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
