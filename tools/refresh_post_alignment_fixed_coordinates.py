"""Refresh packaged post-alignment coordinates from validated fixed-seed runs.

This utility performs the exact coordinate handoff used by the public kidney
and aging-brain post-alignment inference notebooks.  It intentionally leaves
the packaged expression counts and annotations unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KIDNEY_OUTPUT = ROOT / "src/spalignde/datasets/kidney"
AGING_OUTPUT = ROOT / "src/spalignde/datasets/aging_brain"
KIDNEY_COUNTS = {"NL3": 3215, "IL3": 2965}
AGING_COUNTS = {
    "4.3": 79824,
    "6.6": 78862,
    "15.8": 72453,
    "30.9": 66416,
    "34.5": 73775,
}


def _write_deterministic_csv_gz(frame: pd.DataFrame, path: Path) -> str:
    """Write a stable gzip member and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_stream,
            mtime=0,
        ) as compressed_stream:
            with io.TextIOWrapper(
                compressed_stream,
                encoding="utf-8",
                newline="",
            ) as text_stream:
                frame.to_csv(
                    text_stream,
                    index=False,
                    float_format="%.17g",
                    lineterminator="\n",
                )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_coordinates(frame: pd.DataFrame, *, label: str, expected: int) -> None:
    if len(frame) != expected:
        raise RuntimeError(f"{label}: expected {expected} rows, found {len(frame)}")
    if frame["cell_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{label}: duplicate cell_id values")
    if not np.isfinite(frame[["x", "y"]].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{label}: non-finite aligned coordinates")


def refresh_kidney(aligned_h5ad: Path) -> dict[str, str]:
    """Package the fixed kidney alignment using the inference coordinate scale."""

    aligned = ad.read_h5ad(aligned_h5ad, backed="r")
    try:
        required = {"sample_id", "x_aligned", "y_aligned"}
        missing = required.difference(aligned.obs.columns)
        if missing:
            raise RuntimeError(f"Kidney H5AD is missing columns: {sorted(missing)}")
        observations = aligned.obs[
            ["sample_id", "x_aligned", "y_aligned"]
        ].copy()
    finally:
        aligned.file.close()

    observations["source_cell_id"] = observations.index.astype(str)
    observations["barcode"] = observations["source_cell_id"].str.split(
        "|", n=1, regex=False
    ).str[0]

    digests: dict[str, str] = {}
    for sample_id, expected in KIDNEY_COUNTS.items():
        selected = observations.loc[
            observations["sample_id"].astype(str).eq(sample_id)
        ]
        frame = pd.DataFrame(
            {
                "cell_id": sample_id + "__" + selected["barcode"].astype(str),
                "sample_id": sample_id,
                # The cross-sample tutorial stores coordinates after division by
                # 50; inference uses the original Visium plotting scale.
                "x": selected["x_aligned"].to_numpy(dtype=float) * 50.0,
                "y": selected["y_aligned"].to_numpy(dtype=float) * 50.0,
            }
        )
        _validate_coordinates(frame, label=f"kidney {sample_id}", expected=expected)
        target = KIDNEY_OUTPUT / f"aligned_coords_{sample_id}.csv.gz"
        digests[target.relative_to(ROOT).as_posix()] = _write_deterministic_csv_gz(
            frame, target
        )
    return digests


def _replacement_coordinates(
    *,
    age: str,
    aging_h5ad: Path,
    alignment_root: Path,
) -> pd.DataFrame:
    if age == "4.3":
        source = ad.read_h5ad(aging_h5ad, backed="r")
        try:
            age_values = source.obs["age"].to_numpy(dtype=float)
            selected = np.isclose(age_values, 4.3)
            spatial = np.asarray(source.obsm["spatial"], dtype=float)
            return pd.DataFrame(
                {
                    "cell_id": source.obs_names[selected].astype(str),
                    "x": spatial[selected, 0],
                    "y": spatial[selected, 1],
                }
            )
        finally:
            source.file.close()

    path = alignment_root / f"{age}_to_4.3" / "query_coordinates.csv.gz"
    coordinates = pd.read_csv(
        path,
        usecols=["cell_id", "x_aligned", "y_aligned"],
    )
    return coordinates.rename(columns={"x_aligned": "x", "y_aligned": "y"})


def refresh_aging_brain(
    aging_h5ad: Path,
    alignment_root: Path,
) -> dict[str, str]:
    """Replace only x_aligned/y_aligned in the packaged aging observations."""

    digests: dict[str, str] = {}
    for age, expected in AGING_COUNTS.items():
        stem = age.replace(".", "_")
        target = AGING_OUTPUT / f"age_{stem}_observations.csv.gz"
        # Preserve every non-alignment floating-point annotation exactly when
        # the already packaged table is refreshed more than once.
        observations = pd.read_csv(target, float_precision="round_trip")
        replacements = _replacement_coordinates(
            age=age,
            aging_h5ad=aging_h5ad,
            alignment_root=alignment_root,
        )
        replacements["cell_id"] = replacements["cell_id"].astype(str)
        if replacements["cell_id"].duplicated().any():
            raise RuntimeError(f"aging age_{age}: duplicate replacement cell_id")

        indexed = replacements.set_index("cell_id")
        matched = indexed.reindex(observations["cell_id"].astype(str))
        if matched[["x", "y"]].isna().any().any():
            missing = int(matched["x"].isna().sum())
            raise RuntimeError(f"aging age_{age}: {missing} cells lack coordinates")
        observations["x_aligned"] = matched["x"].to_numpy(dtype=float)
        observations["y_aligned"] = matched["y"].to_numpy(dtype=float)

        validation = observations.rename(
            columns={"x_aligned": "x", "y_aligned": "y"}
        )
        _validate_coordinates(
            validation,
            label=f"aging age_{age}",
            expected=expected,
        )
        digests[target.relative_to(ROOT).as_posix()] = _write_deterministic_csv_gz(
            observations, target
        )
    return digests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kidney-h5ad",
        type=Path,
        help="Fixed-seed kidney alignment H5AD.",
    )
    parser.add_argument(
        "--aging-h5ad",
        type=Path,
        help="Original aging-brain H5AD supplying the 4.3-month reference.",
    )
    parser.add_argument(
        "--aging-alignment-root",
        type=Path,
        help="Directory containing <age>_to_4.3/query_coordinates.csv.gz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kidney_h5ad is None and args.aging_h5ad is None:
        raise SystemExit("Provide --kidney-h5ad and/or the two aging-brain inputs.")
    if (args.aging_h5ad is None) != (args.aging_alignment_root is None):
        raise SystemExit(
            "--aging-h5ad and --aging-alignment-root must be supplied together."
        )

    digests: dict[str, str] = {}
    if args.kidney_h5ad is not None:
        digests.update(refresh_kidney(args.kidney_h5ad))
    if args.aging_h5ad is not None:
        digests.update(
            refresh_aging_brain(args.aging_h5ad, args.aging_alignment_root)
        )
    for relative_path, digest in sorted(digests.items()):
        print(f"{digest}  {relative_path}")


if __name__ == "__main__":
    main()
