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
    destination = (
        args.repo_root / "docs" / "source" / "_static" /
        "cross_modality_reproducibility_manifest.json"
    )
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
