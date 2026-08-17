#!/usr/bin/env python3
"""Verify that a built spAlignDE wheel contains every public runtime asset."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


REQUIRED_FILES = {
    "spAlignDE.py",
    "spalignde/__init__.py",
    "spalignde/io.py",
    "spalignde/random.py",
    "spalignde/uncertainty.py",
    "spalignde/alignment/__init__.py",
    "spalignde/alignment/_atlas_core.py",
    "spalignde/alignment/_hipt_feature_extractor.py",
    "spalignde/alignment/_histology_alignment_core.py",
    "spalignde/alignment/_histology_clustering_core.py",
    "spalignde/alignment/_prealignment_core.py",
    "spalignde/alignment/_slddmm_core.py",
    "spalignde/alignment/cross_sample.py",
    "spalignde/alignment/atlas.py",
    "spalignde/alignment/atlas_ui.py",
    "spalignde/alignment/histology.py",
    "spalignde/alignment/atac.py",
    "spalignde/alignment/interactive.py",
    "spalignde/clustering/__init__.py",
    "spalignde/clustering/_joint_core.py",
    "spalignde/clustering/joint.py",
    "spalignde/clustering/single.py",
    "spalignde/datasets/__init__.py",
    "spalignde/datasets/examples.py",
    "spalignde/datasets/visium.py",
    "spalignde/inference/__init__.py",
    "spalignde/inference/_calibration.py",
    "spalignde/inference/_legacy_core.py",
    "spalignde/inference/_types.py",
    "spalignde/inference/prepare.py",
    "spalignde/inference/risk.py",
    "spalignde/inference/summaries.py",
    "spalignde/inference/testing.py",
    "spalignde/inference/plotting.py",
    "spalignde/datasets/toy_metadata.json",
    "spalignde/datasets/toy_post_alignment.csv.gz",
    "spalignde/datasets/toy_truth.csv.gz",
    "spalignde/datasets/kidney/metadata.json",
    "spalignde/datasets/kidney/aligned_coords_IL3.csv.gz",
    "spalignde/datasets/kidney/aligned_coords_NL3.csv.gz",
    "spalignde/datasets/aging_brain/metadata.json",
    "spalignde/datasets/aging_brain/genes.json",
}

AGING_AGES = ("4_3", "6_6", "15_8", "30_9", "34_5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Path to one built .whl file")
    return parser.parse_args()


def main() -> None:
    wheel = parse_args().wheel
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise SystemExit(f"Expected one wheel file, found: {wheel}")

    required = set(REQUIRED_FILES)
    for age in AGING_AGES:
        required.add(f"spalignde/datasets/aging_brain/age_{age}_counts.npz")
        required.add(
            f"spalignde/datasets/aging_brain/age_{age}_observations.csv.gz"
        )

    with ZipFile(wheel) as archive:
        members = set(archive.namelist())
        empty = {
            name
            for name in required.intersection(members)
            if archive.getinfo(name).file_size == 0
        }

    missing = sorted(required - members)
    if missing or empty:
        messages = []
        if missing:
            messages.append("missing=" + ", ".join(missing))
        if empty:
            messages.append("empty=" + ", ".join(sorted(empty)))
        raise SystemExit("Wheel content check failed: " + "; ".join(messages))

    print(
        f"PASS: {wheel.name} contains all {len(required)} required modules and "
        "packaged tutorial datasets."
    )
    print(
        "INFO: documentation, source notebooks, UI source and manuscript-scale "
        "raw inputs are repository resources, not wheel runtime assets."
    )


if __name__ == "__main__":
    main()
