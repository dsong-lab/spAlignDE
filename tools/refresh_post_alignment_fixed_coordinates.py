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
AGING_ALL_QUERY_AGES = (
    "3.4",
    "3.8",
    "5.4",
    "6.6",
    "9.8",
    "12.9",
    "15.5",
    "15.8",
    "18.8",
    "19.8",
    "21.4",
    "23.5",
    "24.6",
    "26.7",
    "28.5",
    "30.9",
    "32.6",
    "33.2",
    "34.5",
)


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


def _terminal_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.split("|", n=1, regex=False).str[0]


def refresh_kidney(
    query_coordinates: Path,
    cluster_labels: Path,
) -> dict[str, str]:
    """Package the formal run-1 IL3 query and unchanged NL3 reference."""

    query = pd.read_csv(
        query_coordinates,
        usecols=["cell_id", "sample_id", "x_aligned", "y_aligned"],
    )
    query = query.loc[query["sample_id"].astype(str).eq("IL3")].copy()
    reference = pd.read_csv(
        cluster_labels,
        usecols=["cell_id", "sample_id", "x", "y"],
    )
    reference = reference.loc[
        reference["sample_id"].astype(str).eq("NL3")
    ].copy()
    sources = {
        "NL3": reference.rename(columns={"x": "x_aligned", "y": "y_aligned"}),
        "IL3": query,
    }

    digests: dict[str, str] = {}
    for sample_id, expected in KIDNEY_COUNTS.items():
        selected = sources[sample_id]
        frame = pd.DataFrame(
            {
                "cell_id": sample_id + "__" + _terminal_id(selected["cell_id"]),
                "sample_id": sample_id,
                # The reproducibility audit stores coordinates after division
                # by 50; inference uses the original Visium plotting scale.
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
    reference_cluster_labels: Path,
    alignment_root: Path,
) -> pd.DataFrame:
    if age == "4.3":
        reference = pd.read_csv(
            reference_cluster_labels,
            usecols=["cell_id", "sample_id", "x", "y"],
        )
        return reference.loc[
            np.isclose(
                pd.to_numeric(reference["sample_id"], errors="coerce"),
                4.3,
            ),
            ["cell_id", "x", "y"],
        ].copy()

    path = alignment_root / f"{age}_to_4.3_query_coordinates.csv.gz"
    coordinates = pd.read_csv(
        path,
        usecols=["cell_id", "x_aligned", "y_aligned"],
    )
    return coordinates.rename(columns={"x_aligned": "x", "y_aligned": "y"})


def refresh_aging_brain(
    reference_cluster_labels: Path,
    alignment_root: Path,
) -> dict[str, str]:
    """Refresh the five-section example from the formal 19-query archive."""

    missing_queries = [
        age
        for age in AGING_ALL_QUERY_AGES
        if not (
            alignment_root / f"{age}_to_4.3_query_coordinates.csv.gz"
        ).is_file()
    ]
    if missing_queries:
        raise RuntimeError(
            "Aging alignment archive is incomplete; missing ages: "
            + ", ".join(missing_queries)
        )

    digests: dict[str, str] = {}
    for age, expected in AGING_COUNTS.items():
        stem = age.replace(".", "_")
        target = AGING_OUTPUT / f"age_{stem}_observations.csv.gz"
        # Preserve every non-alignment floating-point annotation exactly when
        # the already packaged table is refreshed more than once.
        observations = pd.read_csv(target, float_precision="round_trip")
        replacements = _replacement_coordinates(
            age=age,
            reference_cluster_labels=reference_cluster_labels,
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
        "--kidney-query-coordinates",
        type=Path,
        help="Formal run-1 IL3-to-NL3 query_coordinates.csv.gz.",
    )
    parser.add_argument(
        "--kidney-cluster-labels",
        type=Path,
        help="Formal kidney run-1 cluster_labels.csv.gz supplying NL3.",
    )
    parser.add_argument(
        "--aging-reference-cluster-labels",
        type=Path,
        help="Formal aging run-1 cluster_labels.csv.gz supplying age 4.3.",
    )
    parser.add_argument(
        "--aging-alignment-root",
        type=Path,
        help="Directory containing <age>_to_4.3_query_coordinates.csv.gz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kidney_values = (
        args.kidney_query_coordinates,
        args.kidney_cluster_labels,
    )
    aging_values = (
        args.aging_reference_cluster_labels,
        args.aging_alignment_root,
    )
    if not any(kidney_values) and not any(aging_values):
        raise SystemExit("Provide the two kidney and/or two aging-brain inputs.")
    if any(kidney_values) and not all(kidney_values):
        raise SystemExit(
            "--kidney-query-coordinates and --kidney-cluster-labels "
            "must be supplied together."
        )
    if any(aging_values) and not all(aging_values):
        raise SystemExit(
            "--aging-reference-cluster-labels and --aging-alignment-root "
            "must be supplied together."
        )

    digests: dict[str, str] = {}
    if args.kidney_query_coordinates is not None:
        digests.update(
            refresh_kidney(
                args.kidney_query_coordinates,
                args.kidney_cluster_labels,
            )
        )
    if args.aging_reference_cluster_labels is not None:
        digests.update(
            refresh_aging_brain(
                args.aging_reference_cluster_labels,
                args.aging_alignment_root,
            )
        )
    for relative_path, digest in sorted(digests.items()):
        print(f"{digest}  {relative_path}")


if __name__ == "__main__":
    main()
