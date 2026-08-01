"""Small Visium input helpers for post-alignment tutorials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(slots=True)
class VisiumInferenceInput:
    """Validated long-form input for post-alignment Visium inference."""

    data: pd.DataFrame
    coordinates: pd.DataFrame
    genes: tuple[str, ...]
    risk_genes: tuple[str, ...]
    sample_sizes: dict[str, int]
    n_common_genes: int


def canonical_visium_barcodes(
    values: Sequence[object] | pd.Series | pd.Index,
    *,
    source_name: str = "barcode values",
) -> pd.Series:
    """Extract terminal 10x barcodes from optionally prefixed identifiers."""

    original = pd.Series(values, dtype="string")
    canonical = original.str.strip().str.extract(
        r"([ACGT]+-[0-9]+)(?:$|\|)",
        expand=False,
    )
    missing = canonical.isna()
    if missing.any():
        examples = original.loc[missing].head(3).astype(str).tolist()
        raise ValueError(
            f"Could not extract Visium barcodes from {int(missing.sum())} "
            f"{source_name}; examples: {examples}"
        )
    return canonical.astype(str)


def build_visium_coordinate_table(
    tissue_positions: pd.DataFrame,
    aligned_coordinates: pd.DataFrame,
    *,
    sample_id: str,
    position_barcode_key: str = "barcode",
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    aligned_id_key: str = "cell_id",
    aligned_sample_key: str = "sample_id",
    aligned_coordinate_key: tuple[str, str] = ("x", "y"),
) -> pd.DataFrame:
    """Match raw Visium positions to one sample's aligned coordinates.

    The returned table has one row per spot and the standardized columns
    `barcode`, `cell_id`, `sample_id`, `x`, `y`, `x_aligned`, and `y_aligned`.
    Matching is one-to-one by the terminal 10x barcode, never by row order.
    """

    if not isinstance(tissue_positions, pd.DataFrame):
        raise TypeError("tissue_positions must be a pandas DataFrame.")
    if not isinstance(aligned_coordinates, pd.DataFrame):
        raise TypeError("aligned_coordinates must be a pandas DataFrame.")

    position_required = {position_barcode_key, array_row_key, array_col_key}
    aligned_required = {
        aligned_id_key,
        aligned_coordinate_key[0],
        aligned_coordinate_key[1],
    }
    missing_positions = position_required.difference(tissue_positions.columns)
    missing_aligned = aligned_required.difference(aligned_coordinates.columns)
    if missing_positions:
        raise ValueError(
            "tissue_positions is missing columns: "
            + ", ".join(sorted(missing_positions))
        )
    if missing_aligned:
        raise ValueError(
            "aligned_coordinates is missing columns: "
            + ", ".join(sorted(missing_aligned))
        )

    positions = pd.DataFrame(
        {
            "barcode": canonical_visium_barcodes(
                tissue_positions[position_barcode_key],
                source_name=f"{sample_id} tissue-position barcodes",
            ),
            "x": pd.to_numeric(tissue_positions[array_col_key], errors="coerce"),
            "y": pd.to_numeric(tissue_positions[array_row_key], errors="coerce"),
        }
    )
    aligned = pd.DataFrame(
        {
            "barcode": canonical_visium_barcodes(
                aligned_coordinates[aligned_id_key],
                source_name=f"{sample_id} aligned-coordinate identifiers",
            ),
            "cell_id": aligned_coordinates[aligned_id_key].astype(str),
            "x_aligned": pd.to_numeric(
                aligned_coordinates[aligned_coordinate_key[0]],
                errors="coerce",
            ),
            "y_aligned": pd.to_numeric(
                aligned_coordinates[aligned_coordinate_key[1]],
                errors="coerce",
            ),
        }
    )

    if aligned_sample_key in aligned_coordinates.columns:
        observed_samples = set(
            aligned_coordinates[aligned_sample_key].dropna().astype(str)
        )
        if observed_samples != {str(sample_id)}:
            raise ValueError(
                f"Expected aligned sample_id {sample_id!r}, found "
                f"{sorted(observed_samples)}."
            )

    for label, table in (("tissue positions", positions), ("aligned coordinates", aligned)):
        duplicated = table["barcode"].duplicated(keep=False)
        if duplicated.any():
            examples = table.loc[duplicated, "barcode"].head(3).tolist()
            raise ValueError(
                f"{sample_id} {label} contains duplicate barcodes; "
                f"examples: {examples}"
            )

    invalid_positions = ~np.isfinite(positions[["x", "y"]].to_numpy(float)).all(axis=1)
    invalid_aligned = ~np.isfinite(
        aligned[["x_aligned", "y_aligned"]].to_numpy(float)
    ).all(axis=1)
    if invalid_positions.any():
        raise ValueError(
            f"{sample_id} tissue positions contain "
            f"{int(invalid_positions.sum())} non-finite coordinate rows."
        )
    if invalid_aligned.any():
        raise ValueError(
            f"{sample_id} aligned coordinates contain "
            f"{int(invalid_aligned.sum())} non-finite coordinate rows."
        )

    merged = positions.merge(
        aligned,
        on="barcode",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        counts = merged.loc[unmatched, "_merge"].value_counts().to_dict()
        examples = merged.loc[unmatched, "barcode"].head(5).tolist()
        raise ValueError(
            f"{sample_id} spot identifiers do not match one-to-one between "
            f"tissue positions and aligned coordinates: {counts}; "
            f"examples: {examples}"
        )

    merged = merged.drop(columns="_merge")
    merged.insert(2, "sample_id", str(sample_id))
    return merged[
        [
            "barcode",
            "cell_id",
            "sample_id",
            "x",
            "y",
            "x_aligned",
            "y_aligned",
        ]
    ].copy()


def _read_visium_counts(
    source: ad.AnnData | str | Path,
    *,
    sample_id: str,
) -> ad.AnnData:
    if isinstance(source, ad.AnnData):
        counts = source.copy()
    else:
        try:
            import scanpy as sc
        except ImportError as error:
            raise ImportError(
                "Reading a 10x HDF5 count matrix requires Scanpy. Install "
                "spAlignDE with the tutorial extra: pip install -e '.[tutorial]'."
            ) from error
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Variable names are not unique.*",
                category=UserWarning,
            )
            counts = sc.read_10x_h5(Path(source), gex_only=True)

    counts.var_names_make_unique()
    counts.obs_names = canonical_visium_barcodes(
        counts.obs_names,
        source_name=f"{sample_id} count-matrix barcodes",
    ).to_numpy()
    if counts.obs_names.duplicated().any():
        raise ValueError(f"{sample_id} count matrix contains duplicate barcodes.")
    return counts


def build_visium_inference_table(
    aligned: ad.AnnData | str | Path,
    raw_counts: Mapping[str, ad.AnnData | str | Path],
    *,
    genes: Sequence[str],
    sample_key: str = "sample_id",
    coordinate_key: tuple[str, str] = ("x", "y"),
    aligned_coordinate_key: tuple[str, str] = ("x_aligned", "y_aligned"),
    min_detected_spots: int = 10,
    min_total_counts: float = 10,
    batch: str | Mapping[str, str] = "spatial_pair",
) -> VisiumInferenceInput:
    """Join aligned Visium coordinates to raw counts and select risk genes.

    Parameters
    ----------
    aligned
        Aligned AnnData object or H5AD path. Observation identifiers must
        contain extractable 10x barcodes. The function reads sample identity,
        original coordinates, and final aligned coordinates from ``.obs``;
        it never treats ``aligned.X`` as raw expression.
    raw_counts
        Mapping from sample identifier to a raw 10x HDF5 path or an in-memory
        AnnData count matrix. Samples must match ``aligned.obs[sample_key]``.
    genes
        Genes intended for downstream local testing. They are always retained
        in the risk-gene set after validation.
    min_detected_spots, min_total_counts
        Across-sample thresholds used to select the broad mismatch-risk gene
        pool from raw counts.
    batch
        One batch label shared by all samples, or a sample-to-batch mapping.

    Returns
    -------
    VisiumInferenceInput
        Standardized long-form table, aligned-coordinate table, tested genes,
        selected risk genes, sample sizes, and common-gene count.
    """

    if not isinstance(raw_counts, Mapping) or len(raw_counts) < 2:
        raise ValueError("raw_counts must map at least two sample IDs to count matrices.")
    if not genes:
        raise ValueError("genes must contain at least one gene to test.")
    if int(min_detected_spots) < 1:
        raise ValueError("min_detected_spots must be at least 1.")
    if not np.isfinite(min_total_counts) or float(min_total_counts) < 0:
        raise ValueError("min_total_counts must be a finite non-negative number.")

    sample_ids = [str(value) for value in raw_counts]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("raw_counts sample IDs must be unique after string conversion.")

    owns_aligned = not isinstance(aligned, ad.AnnData)
    aligned_adata = ad.read_h5ad(Path(aligned), backed="r") if owns_aligned else aligned
    required_obs = {
        sample_key,
        coordinate_key[0],
        coordinate_key[1],
        aligned_coordinate_key[0],
        aligned_coordinate_key[1],
    }
    missing_obs = required_obs.difference(aligned_adata.obs.columns)
    if missing_obs:
        if owns_aligned:
            aligned_adata.file.close()
        raise ValueError(
            "Aligned AnnData is missing .obs columns: "
            + ", ".join(sorted(missing_obs))
        )

    coordinates = aligned_adata.obs[
        [
            sample_key,
            coordinate_key[0],
            coordinate_key[1],
            aligned_coordinate_key[0],
            aligned_coordinate_key[1],
        ]
    ].copy()
    coordinates.insert(
        0,
        "barcode",
        canonical_visium_barcodes(
            aligned_adata.obs_names,
            source_name="aligned AnnData observation identifiers",
        ).to_numpy(),
    )
    coordinates.insert(1, "cell_id", aligned_adata.obs_names.astype(str))
    if owns_aligned:
        aligned_adata.file.close()

    coordinates = coordinates.rename(
        columns={
            sample_key: "sample_id",
            coordinate_key[0]: "x",
            coordinate_key[1]: "y",
            aligned_coordinate_key[0]: "x_aligned",
            aligned_coordinate_key[1]: "y_aligned",
        }
    )
    coordinates["sample_id"] = coordinates["sample_id"].astype(str)
    for column in ("x", "y", "x_aligned", "y_aligned"):
        coordinates[column] = pd.to_numeric(coordinates[column], errors="coerce")
    finite_coordinates = np.isfinite(
        coordinates[["x", "y", "x_aligned", "y_aligned"]].to_numpy(float)
    ).all(axis=1)
    if not finite_coordinates.all():
        raise ValueError(
            f"Aligned AnnData contains {int((~finite_coordinates).sum())} "
            "observations with non-finite coordinates."
        )

    observed_samples = set(coordinates["sample_id"])
    requested_samples = set(sample_ids)
    if observed_samples != requested_samples:
        raise ValueError(
            "Sample IDs differ between aligned AnnData and raw_counts: "
            f"aligned={sorted(observed_samples)}, raw_counts={sorted(requested_samples)}."
        )
    duplicated = coordinates.duplicated(["sample_id", "barcode"], keep=False)
    if duplicated.any():
        examples = coordinates.loc[duplicated, ["sample_id", "barcode"]].head(3)
        raise ValueError(
            "Aligned AnnData contains duplicate sample/barcode pairs; examples: "
            + examples.astype(str).agg(":".join, axis=1).str.cat(sep=", ")
        )

    counts_by_sample: dict[str, ad.AnnData] = {}
    for original_id, source in raw_counts.items():
        sample_id = str(original_id)
        counts = _read_visium_counts(source, sample_id=sample_id)
        sample_coordinates = coordinates.loc[
            coordinates["sample_id"].eq(sample_id)
        ]
        missing_expression = sorted(
            set(sample_coordinates["barcode"]) - set(counts.obs_names)
        )
        if missing_expression:
            raise ValueError(
                f"{sample_id} has {len(missing_expression)} aligned spots without "
                f"raw expression rows; examples: {missing_expression[:5]}"
            )
        counts_by_sample[sample_id] = counts

    common_genes = pd.Index(counts_by_sample[sample_ids[0]].var_names)
    for sample_id in sample_ids[1:]:
        common_genes = common_genes.intersection(counts_by_sample[sample_id].var_names)
    missing_genes = sorted(set(map(str, genes)) - set(common_genes))
    if missing_genes:
        raise ValueError(f"Genes intended for testing were not found: {missing_genes}")

    detected_total = np.zeros(len(common_genes), dtype=np.int64)
    count_total = np.zeros(len(common_genes), dtype=np.float64)
    for sample_id in sample_ids:
        matrix = counts_by_sample[sample_id][:, common_genes].X
        detected_total += np.asarray((matrix > 0).sum(axis=0)).reshape(-1)
        count_total += np.asarray(matrix.sum(axis=0)).reshape(-1)
    keep = (
        (detected_total >= int(min_detected_spots))
        & (count_total >= float(min_total_counts))
    )
    risk_genes = common_genes[keep].astype(str).tolist()
    tested_genes = list(dict.fromkeys(map(str, genes)))
    risk_genes = list(dict.fromkeys([*risk_genes, *tested_genes]))

    reserved = {
        "barcode",
        "cell_id",
        "sample_id",
        "x",
        "y",
        "x_aligned",
        "y_aligned",
        "batch",
        "celltype",
    }
    collisions = reserved.intersection(risk_genes)
    if collisions:
        raise ValueError(
            "Gene names collide with required metadata columns: "
            + ", ".join(sorted(collisions))
        )

    sample_sizes: dict[str, int] = {}
    tables = []
    for sample_id in sample_ids:
        sample_coordinates = coordinates.loc[
            coordinates["sample_id"].eq(sample_id)
        ].copy()
        ordered_barcodes = sample_coordinates["barcode"].tolist()
        counts = counts_by_sample[sample_id][ordered_barcodes, risk_genes]
        matrix = counts.X
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        expression = pd.DataFrame(
            np.asarray(matrix, dtype=np.float32),
            index=ordered_barcodes,
            columns=risk_genes,
        )
        table = sample_coordinates.set_index("barcode").loc[ordered_barcodes]
        table = table.join(expression, how="inner")
        table.index.name = "barcode"
        if isinstance(batch, Mapping):
            if sample_id not in batch:
                raise ValueError(f"batch mapping has no label for sample {sample_id!r}.")
            table["batch"] = str(batch[sample_id])
        else:
            table["batch"] = str(batch)
        tables.append(table.reset_index())
        sample_sizes[sample_id] = len(table)

    data = pd.concat(tables, ignore_index=True)
    return VisiumInferenceInput(
        data=data,
        coordinates=coordinates.reset_index(drop=True),
        genes=tuple(tested_genes),
        risk_genes=tuple(risk_genes),
        sample_sizes=sample_sizes,
        n_common_genes=len(common_genes),
    )
