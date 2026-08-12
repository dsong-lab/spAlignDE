#!/usr/bin/env python3
"""Remap a validated UI pairing export onto fixed-seed ST cluster labels.

The Allen selections and grouping are preserved.  Only ST-side cluster IDs
and their descriptive statistics are updated from cell-indexed overlap with
the original UI labels.  A dominant overlap of at least 90% maps to one new
cluster; otherwise the smallest set covering at least 98% is retained.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list of IDs, received {value!r}")
    return [str(item) for item in parsed]


def format_ids(values: list[str]) -> str:
    normalized = [int(value) if value.isdigit() else value for value in values]
    return repr(normalized)


def choose_fixed_clusters(
    old_labels: pd.Series,
    fixed_labels: pd.Series,
    old_ids: list[str],
    dominant_threshold: float = 0.90,
    cumulative_threshold: float = 0.98,
) -> tuple[list[str], float]:
    selected = old_labels.astype(str).isin(old_ids)
    counts = fixed_labels.loc[selected].astype(str).value_counts()
    if counts.empty:
        raise ValueError(f"No cells found for old UI cluster IDs {old_ids}")

    fractions = counts / counts.sum()
    if float(fractions.iloc[0]) >= dominant_threshold:
        chosen = [str(fractions.index[0])]
    else:
        cumulative = fractions.cumsum()
        stop = int(np.searchsorted(cumulative.to_numpy(), cumulative_threshold)) + 1
        chosen = [str(value) for value in fractions.index[:stop]]
    coverage = float(fractions.loc[chosen].sum())
    return sorted(chosen, key=lambda value: int(value) if value.isdigit() else value), coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-pairing", type=Path, required=True)
    parser.add_argument("--old-cluster-metadata", type=Path, required=True)
    parser.add_argument("--fixed-h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = pd.read_csv(args.old_pairing)
    pairs["left_selected_id"] = pairs["left_selected_id"].astype(object)
    old = pd.read_csv(args.old_cluster_metadata, dtype={"cell_id": str}).set_index("cell_id")
    fixed = ad.read_h5ad(args.fixed_h5ad)
    fixed.obs_names = fixed.obs_names.astype(str)

    missing = fixed.obs_names.difference(old.index)
    if len(missing):
        raise ValueError(f"Original UI metadata is missing {len(missing):,} fixed-run cells")

    old_labels = old.reindex(fixed.obs_names)["banksy_cluster_refined"].astype(str)
    fixed_labels = fixed.obs["cluster"].astype(str)
    coordinates = np.asarray(fixed.obsm["spatial"])

    summary: list[dict[str, object]] = []
    for group_id, row_indices in pairs.groupby("group_id", sort=False).groups.items():
        first = pairs.loc[next(iter(row_indices))]
        old_ids = parse_ids(first["left_raw_selected_ids"])
        new_ids, coverage = choose_fixed_clusters(old_labels, fixed_labels, old_ids)

        selected = fixed_labels.isin(new_ids).to_numpy()
        centroid = coordinates[selected].mean(axis=0)
        display_name = "+".join(f"cluster_{value}" for value in new_ids)
        encoded_ids = format_ids(new_ids)

        pairs.loc[row_indices, "left_selected_ids"] = encoded_ids
        pairs.loc[row_indices, "left_raw_selected_ids"] = encoded_ids
        pairs.loc[row_indices, "right_paired_to_ids"] = encoded_ids
        pairs.loc[row_indices, "left_selected_id"] = new_ids[0]
        pairs.loc[row_indices, "left_selected_name"] = display_name
        pairs.loc[row_indices, "left_point_count"] = int(selected.sum())
        pairs.loc[row_indices, "left_x"] = float(centroid[0])
        pairs.loc[row_indices, "left_y"] = float(centroid[1])

        summary.append(
            {
                "group_id": group_id,
                "old_ids": old_ids,
                "fixed_seed_ids": new_ids,
                "old_cell_coverage": coverage,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Wrote {len(pairs)} rows to {args.output}")


if __name__ == "__main__":
    main()
