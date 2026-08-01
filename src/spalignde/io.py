"""Input validation and coordinate conventions for spAlignDE."""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


REQUIRED_OUTPUT_COLUMNS = (
    "x_prealigned",
    "y_prealigned",
    "x_aligned",
    "y_aligned",
)


def validate_single_sample_anndata(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    cluster_key: str = "cluster",
    require_cluster: bool = False,
) -> None:
    """Validate the public AnnData contract for one spatial sample.

    The expression matrix must be cell/spot by gene, observation names must be
    unique, and ``adata.obsm[spatial_key]`` must contain finite x/y coordinates.
    A cluster column is only required by downstream structure-guided alignment.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if adata.n_obs == 0:
        raise ValueError("adata contains no observations")
    if adata.n_vars == 0:
        raise ValueError("adata contains no variables")
    if not adata.obs_names.is_unique:
        raise ValueError("adata.obs_names must be unique")
    if spatial_key not in adata.obsm:
        raise ValueError(f"Missing adata.obsm[{spatial_key!r}]")

    spatial = np.asarray(adata.obsm[spatial_key])
    if spatial.shape != (adata.n_obs, 2):
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must have shape "
            f"({adata.n_obs}, 2), got {spatial.shape}"
        )
    if not np.issubdtype(spatial.dtype, np.number):
        raise TypeError(f"adata.obsm[{spatial_key!r}] must be numeric")
    if not np.isfinite(spatial).all():
        raise ValueError(f"adata.obsm[{spatial_key!r}] contains non-finite values")

    if require_cluster:
        if cluster_key not in adata.obs:
            raise ValueError(
                f"Missing adata.obs[{cluster_key!r}]. Run single clustering first "
                "or copy the selected labels to this column."
            )
        if adata.obs[cluster_key].isna().any():
            raise ValueError(f"adata.obs[{cluster_key!r}] contains missing values")


def read_single_sample_csv(
    metadata_csv: str | PathLike[str],
    expression_csv: str | PathLike[str],
    *,
    spatial_key: str = "spatial",
    cell_id_key: str = "cell_id",
    x_key: str = "x",
    y_key: str = "y",
) -> ad.AnnData:
    """Read one metadata/expression CSV pair into AnnData.

    The metadata table requires ``cell_id``, ``x`` and ``y`` columns. The
    expression table requires ``cell_id`` followed by numeric gene columns.
    Both files must contain the same unique cell/spot identifiers; expression
    rows are reordered to the metadata table before constructing AnnData.
    """
    metadata_path = Path(metadata_csv).expanduser()
    expression_path = Path(expression_csv).expanduser()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not expression_path.is_file():
        raise FileNotFoundError(f"Expression CSV not found: {expression_path}")

    metadata = pd.read_csv(metadata_path)
    expression = pd.read_csv(expression_path)
    missing_metadata = {cell_id_key, x_key, y_key}.difference(metadata.columns)
    if missing_metadata:
        raise ValueError(
            f"Metadata CSV is missing columns: {sorted(missing_metadata)}"
        )
    if cell_id_key not in expression:
        raise ValueError(f"Expression CSV is missing {cell_id_key!r}")

    metadata = metadata.copy()
    expression = expression.copy()
    metadata[cell_id_key] = metadata[cell_id_key].astype(str)
    expression[cell_id_key] = expression[cell_id_key].astype(str)
    if metadata[cell_id_key].duplicated().any():
        raise ValueError("Metadata CSV contains duplicate cell IDs")
    if expression[cell_id_key].duplicated().any():
        raise ValueError("Expression CSV contains duplicate cell IDs")

    metadata_ids = pd.Index(metadata[cell_id_key])
    expression_ids = pd.Index(expression[cell_id_key])
    if set(metadata_ids) != set(expression_ids):
        only_metadata = len(set(metadata_ids).difference(expression_ids))
        only_expression = len(set(expression_ids).difference(metadata_ids))
        raise ValueError(
            "Metadata and expression cell IDs do not match "
            f"({only_metadata} metadata-only, {only_expression} expression-only)"
        )

    for coordinate_key in (x_key, y_key):
        if not pd.api.types.is_numeric_dtype(metadata[coordinate_key]):
            raise TypeError(f"Metadata column {coordinate_key!r} must be numeric")
        if not np.isfinite(metadata[coordinate_key].to_numpy()).all():
            raise ValueError(
                f"Metadata column {coordinate_key!r} contains non-finite values"
            )

    genes = [column for column in expression.columns if column != cell_id_key]
    if not genes:
        raise ValueError("Expression CSV has no gene columns")
    non_numeric = [
        gene for gene in genes if not pd.api.types.is_numeric_dtype(expression[gene])
    ]
    if non_numeric:
        raise TypeError(f"Non-numeric gene columns: {non_numeric[:10]}")
    values = expression[genes].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Expression CSV contains non-finite values")
    if (values < 0).any():
        raise ValueError("Expression CSV contains negative values")

    metadata = metadata.set_index(cell_id_key, drop=False)
    expression = expression.set_index(cell_id_key).loc[metadata.index, genes]
    metadata.index = pd.Index(metadata.index.astype(str), name="observation_id")
    expression.index = metadata.index
    result = ad.AnnData(
        X=sp.csr_matrix(expression.to_numpy(dtype=np.float64, copy=False)),
        obs=metadata,
        var=pd.DataFrame(index=pd.Index([str(gene) for gene in genes], name="gene")),
    )
    result.obsm[spatial_key] = metadata[[x_key, y_key]].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    _normalize_package_metadata(result)
    validate_single_sample_anndata(result, spatial_key=spatial_key)
    return result


def load_single_sample_data(
    data: ad.AnnData | str | PathLike[str],
    *,
    expression_csv: str | PathLike[str] | None = None,
    spatial_key: str = "spatial",
    copy: bool = True,
) -> ad.AnnData:
    """Load one spatial sample from AnnData/H5AD or a paired CSV input.

    For CSV input, pass the metadata path as ``data`` and the cell-by-gene path
    as ``expression_csv``. H5AD inputs may store coordinates either in
    ``obsm[spatial_key]`` or in numeric ``obs['x']``/``obs['y']`` columns.
    """
    if isinstance(data, ad.AnnData):
        if expression_csv is not None:
            raise ValueError("expression_csv cannot be used with an AnnData input")
        source = data.to_memory() if data.isbacked else data
        result = source.copy() if copy else source
    else:
        path = Path(data).expanduser()
        if path.suffix.lower() == ".h5ad":
            if expression_csv is not None:
                raise ValueError("expression_csv cannot be used with an H5AD input")
            if not path.is_file():
                raise FileNotFoundError(f"AnnData input not found: {path}")
            result = ad.read_h5ad(path)
        elif path.suffix.lower() == ".csv":
            if expression_csv is None:
                raise ValueError(
                    "CSV input requires expression_csv in addition to metadata CSV"
                )
            result = read_single_sample_csv(
                path,
                expression_csv,
                spatial_key=spatial_key,
            )
        else:
            raise ValueError(
                "Input must be an AnnData object, a .h5ad file, or a metadata "
                ".csv file accompanied by expression_csv"
            )

    if spatial_key not in result.obsm and {"x", "y"}.issubset(result.obs.columns):
        result.obsm[spatial_key] = result.obs[["x", "y"]].to_numpy(
            dtype=np.float64,
            copy=True,
        )
    _normalize_package_metadata(result)
    validate_single_sample_anndata(result, spatial_key=spatial_key)
    return result


def _normalize_package_metadata(adata: ad.AnnData) -> None:
    """Replace the historical lowercase AnnData key with ``spAlignDE``."""
    legacy = adata.uns.pop("spalignde", None)
    if legacy is None:
        return
    canonical = adata.uns.setdefault("spAlignDE", {})
    if isinstance(legacy, dict) and isinstance(canonical, dict):
        for key, value in legacy.items():
            canonical.setdefault(key, value)


def read_cross_sample_csv(
    data_dir: str | PathLike[str],
    *,
    sample_key: str = "sample_id",
    spatial_key: str = "spatial",
    cell_id_key: str = "cell_id",
    x_key: str = "x",
    y_key: str = "y",
) -> ad.AnnData:
    """Read paired per-sample CSV files into one combined AnnData.

    ``data_dir`` must contain one metadata and one expression file per sample,
    named ``cell_metadata_<sample_id>.csv`` and
    ``cell_by_gene_<sample_id>.csv``. Metadata files require ``cell_id``,
    ``x`` and ``y`` columns. Expression files require ``cell_id`` followed by
    numeric gene columns. Genes are aligned by name and missing genes in an
    individual sample are filled with zero.
    """
    directory = Path(data_dir).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"CSV input directory not found: {directory}")

    metadata_paths: dict[str, Path] = {}
    expression_paths: dict[str, Path] = {}
    for path in directory.glob("*.csv"):
        if path.name.startswith("cell_metadata_"):
            sample_id = path.stem.removeprefix("cell_metadata_")
            if sample_id:
                metadata_paths[sample_id] = path
        elif path.name.startswith("cell_by_gene_"):
            sample_id = path.stem.removeprefix("cell_by_gene_")
            if sample_id:
                expression_paths[sample_id] = path

    metadata_ids = set(metadata_paths)
    expression_ids = set(expression_paths)
    if metadata_ids != expression_ids or len(metadata_ids) < 2:
        missing_metadata = sorted(expression_ids.difference(metadata_ids))
        missing_expression = sorted(metadata_ids.difference(expression_ids))
        raise FileNotFoundError(
            "CSV input requires at least two matched file pairs named "
            "'cell_metadata_<sample_id>.csv' and "
            "'cell_by_gene_<sample_id>.csv'. "
            f"Missing metadata for: {missing_metadata}; "
            f"missing expression for: {missing_expression}."
        )

    sample_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_genes: set[str] = set()
    for sample_id in sorted(metadata_ids):
        metadata = pd.read_csv(metadata_paths[sample_id])
        expression = pd.read_csv(expression_paths[sample_id])

        required_metadata = {cell_id_key, x_key, y_key}
        missing = required_metadata.difference(metadata.columns)
        if missing:
            raise ValueError(
                f"[{sample_id}] metadata CSV is missing columns: {sorted(missing)}"
            )
        if cell_id_key not in expression:
            raise ValueError(
                f"[{sample_id}] expression CSV is missing {cell_id_key!r}"
            )

        metadata = metadata.copy()
        expression = expression.copy()
        metadata[cell_id_key] = metadata[cell_id_key].astype(str)
        expression[cell_id_key] = expression[cell_id_key].astype(str)
        if metadata[cell_id_key].duplicated().any():
            raise ValueError(f"[{sample_id}] metadata contains duplicate cell IDs")
        if expression[cell_id_key].duplicated().any():
            raise ValueError(f"[{sample_id}] expression contains duplicate cell IDs")

        metadata_ids_for_sample = pd.Index(metadata[cell_id_key])
        expression_ids_for_sample = pd.Index(expression[cell_id_key])
        if set(metadata_ids_for_sample) != set(expression_ids_for_sample):
            only_metadata = len(
                set(metadata_ids_for_sample).difference(expression_ids_for_sample)
            )
            only_expression = len(
                set(expression_ids_for_sample).difference(metadata_ids_for_sample)
            )
            raise ValueError(
                f"[{sample_id}] metadata and expression cell IDs do not match "
                f"({only_metadata} metadata-only, {only_expression} expression-only)"
            )

        for coordinate_key in (x_key, y_key):
            if not pd.api.types.is_numeric_dtype(metadata[coordinate_key]):
                raise TypeError(
                    f"[{sample_id}] metadata column {coordinate_key!r} must be numeric"
                )
            if not np.isfinite(metadata[coordinate_key].to_numpy()).all():
                raise ValueError(
                    f"[{sample_id}] metadata column {coordinate_key!r} "
                    "contains non-finite values"
                )

        genes = [column for column in expression.columns if column != cell_id_key]
        if not genes:
            raise ValueError(f"[{sample_id}] expression CSV has no gene columns")
        non_numeric = [
            gene
            for gene in genes
            if not pd.api.types.is_numeric_dtype(expression[gene])
        ]
        if non_numeric:
            raise TypeError(
                f"[{sample_id}] non-numeric gene columns: {non_numeric[:10]}"
            )
        expression_values = expression[genes].to_numpy()
        if not np.isfinite(expression_values).all():
            raise ValueError(f"[{sample_id}] expression contains non-finite values")
        if (expression_values < 0).any():
            raise ValueError(f"[{sample_id}] expression contains negative values")

        metadata = metadata.set_index(cell_id_key, drop=False)
        expression = expression.set_index(cell_id_key).loc[
            metadata.index,
            genes,
        ]
        metadata[sample_key] = str(sample_id)
        cell_ids = metadata[cell_id_key].astype(str)
        if cell_ids.str.startswith(f"{sample_id}_").all():
            observation_names = cell_ids
        else:
            observation_names = f"{sample_id}_" + cell_ids
        metadata.index = pd.Index(observation_names, name="observation_id")
        expression.index = metadata.index

        sample_tables[sample_id] = (metadata, expression)
        all_genes.update(str(gene) for gene in genes)

    gene_order = sorted(all_genes)
    observation_tables: list[pd.DataFrame] = []
    expression_matrices: list[sp.csr_matrix] = []
    for sample_id in sorted(sample_tables):
        metadata, expression = sample_tables[sample_id]
        aligned = expression.reindex(columns=gene_order, fill_value=0.0)
        observation_tables.append(metadata)
        expression_matrices.append(
            sp.csr_matrix(aligned.to_numpy(dtype=np.float64, copy=False))
        )

    observations = pd.concat(observation_tables, axis=0)
    combined = ad.AnnData(
        X=sp.vstack(expression_matrices, format="csr"),
        obs=observations,
        var=pd.DataFrame(index=pd.Index(gene_order, name="gene")),
    )
    combined.obsm[spatial_key] = observations[[x_key, y_key]].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    _normalize_package_metadata(combined)
    validate_cross_sample_anndata(
        combined,
        sample_key=sample_key,
        spatial_key=spatial_key,
        require_cluster=False,
    )
    return combined


def load_cross_sample_data(
    data: ad.AnnData | str | PathLike[str],
    *,
    sample_key: str = "sample_id",
    spatial_key: str = "spatial",
    copy: bool = True,
) -> ad.AnnData:
    """Load an AnnData object, ``.h5ad`` file or paired-CSV directory.

    All supported inputs are normalized to the public combined-AnnData
    contract before clustering or alignment.
    """
    if isinstance(data, ad.AnnData):
        source = data.to_memory() if data.isbacked else data
        combined = source.copy() if copy else source
    else:
        path = Path(data).expanduser()
        if path.is_dir():
            combined = read_cross_sample_csv(
                path,
                sample_key=sample_key,
                spatial_key=spatial_key,
            )
        elif path.suffix.lower() == ".h5ad":
            if not path.is_file():
                raise FileNotFoundError(f"AnnData input not found: {path}")
            combined = ad.read_h5ad(path)
            if spatial_key not in combined.obsm and {"x", "y"}.issubset(
                combined.obs.columns
            ):
                combined.obsm[spatial_key] = combined.obs[["x", "y"]].to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
        else:
            raise ValueError(
                "Input must be an AnnData object, a .h5ad file, or a directory "
                "containing paired CSV files"
            )

    _normalize_package_metadata(combined)
    validate_cross_sample_anndata(
        combined,
        sample_key=sample_key,
        spatial_key=spatial_key,
        require_cluster=False,
    )
    return combined


def validate_cross_sample_anndata(
    adata: ad.AnnData,
    *,
    sample_key: str = "sample_id",
    spatial_key: str = "spatial",
    cluster_key: str = "cluster",
    require_cluster: bool = True,
) -> None:
    """Validate the public AnnData contract for cross-sample workflows.

    The input must contain a cell/spot-by-gene matrix, globally unique
    observation names, sample labels in ``adata.obs`` and finite two-dimensional
    coordinates in ``adata.obsm``. Alignment additionally requires a shared
    cluster label column.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("adata must be an anndata.AnnData object")
    if adata.n_obs == 0:
        raise ValueError("adata contains no observations")
    if adata.n_vars == 0:
        raise ValueError("adata contains no variables")
    if not adata.obs_names.is_unique:
        raise ValueError("adata.obs_names must be globally unique")
    if sample_key not in adata.obs:
        raise ValueError(f"Missing adata.obs[{sample_key!r}]")

    sample = adata.obs[sample_key]
    if sample.isna().any():
        raise ValueError(f"adata.obs[{sample_key!r}] contains missing values")
    if sample.astype(str).nunique() < 2:
        raise ValueError("Cross-sample analysis requires at least two samples")

    if spatial_key not in adata.obsm:
        raise ValueError(f"Missing adata.obsm[{spatial_key!r}]")
    spatial = np.asarray(adata.obsm[spatial_key])
    if spatial.shape != (adata.n_obs, 2):
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must have shape "
            f"({adata.n_obs}, 2), got {spatial.shape}"
        )
    if not np.issubdtype(spatial.dtype, np.number):
        raise TypeError(f"adata.obsm[{spatial_key!r}] must be numeric")
    if not np.isfinite(spatial).all():
        raise ValueError(f"adata.obsm[{spatial_key!r}] contains non-finite values")

    if require_cluster:
        if cluster_key not in adata.obs:
            raise ValueError(
                f"Missing adata.obs[{cluster_key!r}]. Run joint clustering first "
                "or copy the selected shared labels to this column."
            )
        cluster = adata.obs[cluster_key]
        if cluster.isna().any():
            raise ValueError(f"adata.obs[{cluster_key!r}] contains missing values")


def validate_sample_selection(
    adata: ad.AnnData,
    *,
    query_sample: str,
    reference_sample: str,
    sample_key: str = "sample_id",
) -> None:
    """Validate an explicit query-to-reference sample selection."""
    query_sample = str(query_sample)
    reference_sample = str(reference_sample)
    if query_sample == reference_sample:
        raise ValueError("query_sample and reference_sample must be different")
    available = set(adata.obs[sample_key].astype(str))
    missing = {query_sample, reference_sample}.difference(available)
    if missing:
        raise ValueError(
            f"Samples not found in adata.obs[{sample_key!r}]: {sorted(missing)}. "
            f"Available samples: {sorted(available)}"
        )


def spatial_coordinates(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
) -> np.ndarray:
    """Return an independent float64 copy of the canonical input coordinates."""
    return np.asarray(adata.obsm[spatial_key], dtype=np.float64).copy()


def initialize_output_coordinates(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    columns: Iterable[str] = REQUIRED_OUTPUT_COLUMNS,
) -> None:
    """Initialize output columns from the immutable input coordinates."""
    spatial = spatial_coordinates(adata, spatial_key=spatial_key)
    values = {
        "x_prealigned": spatial[:, 0],
        "y_prealigned": spatial[:, 1],
        "x_aligned": spatial[:, 0],
        "y_aligned": spatial[:, 1],
    }
    for column in columns:
        adata.obs[column] = values[column]
