#!/usr/bin/env python3
"""Bind the public inference notebooks to the validated fixed-seed coordinates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source_notebooks"
DOCS_DIR = ROOT / "docs/source/source_notebooks"


def _source_hash(notebook) -> str:
    payload = "\n\n".join(
        f"{cell.cell_type}\n{cell.source}" for cell in notebook.cells
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_hash(notebook) -> str:
    payload = "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _output_hash(notebook) -> str:
    payload = [
        cell.get("outputs", [])
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _base_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _record_execution(notebook, *, aging_brain: bool) -> None:
    """Record hashes after a successful execution of the generated source."""
    notebook.metadata["spAlignDE_execution"] = {
        "fully_executed": True,
        "workflow_seed": 1,
        "source_sha256": _source_hash(notebook),
        "executed_code_sha256": _code_hash(notebook),
        "saved_output_sha256": _output_hash(notebook),
        "repository_commit": _base_revision(),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_root": "repository root with validated fixed-seed coordinate handoff",
        "package_source": "current integrated checkout (public version 0.1.0)",
        "input_manifest": "docs/source/_static/tutorial_execution_manifest.json",
        "coordinate_provenance": (
            "current PyPI five-section aging-brain precomputed coordinates"
            if aging_brain
            else (
                "fixed-seed kidney manual-prealignment coordinates, "
                "resolution 0.2, 5000 iterations"
            )
        ),
        "aging_brain_included": aging_brain,
    }


KIDNEY_INTRO = r"""# Mouse kidney local inference from fixed-seed manual-alignment coordinates

This real-data notebook continues from the validated fixed-seed NL3/IL3 kidney workflow and runs post-alignment local inference with the current public `spalignde` API. Upstream clustering and alignment use seed `1000`, Leiden resolution `0.2`, and a selected manual similarity pre-alignment (`scale=1`, `theta=0`, `translation_x=-36.20040965`, `translation_y=-153.38356513`) before 5,000 S-LDDMM iterations with `restore_best_checkpoint=False`. The raw count matrices and tissue-position files come from Zenodo record `17676992`. IL3 is the 2,965-spot query and NL3 is the unchanged 3,215-spot reference.

The analysis proceeds through one continuous handoff:

1. load public expression data and the packaged fixed-seed manual-alignment coordinates;
2. match spots by terminal Visium barcode;
3. construct the shared grid and local neighborhoods;
4. estimate stable-gene and density-based mismatch risk;
5. fit gene-specific mismatch-aware local tests with the public `fit_local_de` function; and
6. report grid-level statistics, FDR-adjusted regions, and representative gene maps.

NL3 defines the reference coordinate system and IL3 is the query. Users may replace the packaged coordinates with their own standardized spAlignDE output, but a custom alignment is not expected to reproduce the recorded numerical results unless its evaluated spot set and coordinates are identical.

**Fixed-seed reproducibility.** Upstream clustering and alignment use seed `1000`; this inference workflow uses seed `1` and requests `n_jobs=1`. During `prepare_inference`, the two seeded per-sample auto-geometry subsampling and parameter-estimation passes always use one worker so thread scheduling cannot change which sample consumes each RNG draw. All later preparation and fitting stages use the caller-requested `n_jobs` value, so this safeguard does not make the complete analysis single-threaded. The preparation metadata records `n_jobs`, `auto_geometry_n_jobs=1`, and `random_state`. For the closest numerical reproduction, set `PYTHONHASHSEED=1` before kernel startup and keep the documented input order, package version, and configuration fixed. Small floating-point differences in the last displayed digits can remain across numerical-library builds."""


KIDNEY_HANDOFF = r"""## Alignment-to-inference handoff

No path configuration is required for the recorded example because the package includes a compact, hash-tracked copy of the fixed-seed manual-alignment coordinates as `aligned_coords_NL3.csv.gz` and `aligned_coords_IL3.csv.gz`. They come from `kidney_IL3_to_NL3_aligned.h5ad`, the output of the public kidney manual-alignment notebook. The inference stage reloads the public NL3/IL3 10x matrices and tissue-position tables from Zenodo record `17676992` and joins them to the packaged coordinates one-to-one by terminal Visium barcode, never by row order.

For a custom alignment, point `SPALIGNDE_ALIGNMENT_DIR` to a directory containing `aligned_coords_NL3.csv` and `aligned_coords_IL3.csv`. Each file must contain one row per retained spot and the standardized columns `cell_id`, `x`, and `y`; `barcode` may be used instead of `cell_id`. `SPALIGNDE_KIDNEY_DATA_DIR` and `SPALIGNDE_TUTORIAL_WORK_DIR` can redirect the raw-data and working directories.

```python
import os
os.environ["SPALIGNDE_ALIGNMENT_DIR"] = "/path/to/custom_alignment_output"
```

Raw 10x matrices are always reloaded because an alignment output may contain only the gene-filtered matrix used for clustering and registration."""


KIDNEY_INTERPRETATION = r"""## 7. Interpretation and extensions

This example demonstrates the complete alignment-output $\rightarrow$ shared-grid inference $\rightarrow$ local DE maps $\rightarrow$ gene-level ACAT P value handoff. It is a single matched-section NL3-versus-IL3 comparison and should not be interpreted as replicate-level population inference. The packaged coordinates reproduce the public fixed-seed manual-alignment handoff by default, and `SPALIGNDE_ALIGNMENT_DIR` remains available for standardized coordinate CSV files. Users with reliable cell-type annotations can add a `celltype` column and enable `cell_type_adjustment=True`. Multiple query sections can be analyzed against one reference and summarized with gene-level ACAT or trajectory clustering."""


KIDNEY_IMPORTS = r"""%matplotlib inline

from pathlib import Path
import os
import random
import warnings

WORK_DIR = Path(
    os.environ.get(
        "SPALIGNDE_TUTORIAL_WORK_DIR",
        Path.cwd() / "spalignde_kidney_tutorial",
    )
).expanduser().resolve()
DATA_DIR = Path(
    os.environ.get("SPALIGNDE_KIDNEY_DATA_DIR", WORK_DIR / "raw")
).expanduser().resolve()
USER_ALIGNMENT_DIR_VALUE = os.environ.get("SPALIGNDE_ALIGNMENT_DIR")
USER_ALIGNMENT_DIR = (
    Path(USER_ALIGNMENT_DIR_VALUE).expanduser().resolve()
    if USER_ALIGNMENT_DIR_VALUE
    else None
)
os.environ.setdefault("MPLCONFIGDIR", str(WORK_DIR / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

import urllib.request

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from IPython.display import display

import spalignde
from spalignde.inference import (
    fit_local_de,
    gene_level_acat_pvalue,
    plot_local_result,
    prepare_inference,
)
from spalignde.datasets import (
    build_visium_coordinate_table,
    canonical_visium_barcodes,
    kidney_alignment_metadata,
    load_kidney_aligned_coordinates,
)

WORKFLOW_SEED = 1
random.seed(WORKFLOW_SEED)
np.random.seed(WORKFLOW_SEED)

alignment_metadata = kidney_alignment_metadata()
alignment_source = (
    "user-provided standardized coordinate CSV files"
    if USER_ALIGNMENT_DIR is not None
    else "packaged fixed-seed manual-alignment coordinates"
)
print("spalignde version:", spalignde.__version__)
print("Raw-data source: Zenodo record 17676992 (local cache configured)")
print("Aligned-coordinate source:", alignment_source)
print("Risk-map radius rule: 1.5 x shared-grid spacing")"""


KIDNEY_RAW_INPUTS = r"""DOWNLOAD_ZENODO = False
ZENODO_BASE = "https://zenodo.org/records/17676992/files"
RAW_FILES = [
    "NL3_filtered_feature_bc_matrix.h5",
    "IL3_filtered_feature_bc_matrix.h5",
    "NL3_tissue_positions.csv",
    "IL3_tissue_positions.csv",
]

if DOWNLOAD_ZENODO:
    for filename in RAW_FILES:
        destination = DATA_DIR / filename
        if not destination.exists():
            url = f"{ZENODO_BASE}/{filename}?download=1"
            print("Downloading", url)
            urllib.request.urlretrieve(url, destination)

missing_raw = [name for name in RAW_FILES if not (DATA_DIR / name).exists()]
if missing_raw:
    raise FileNotFoundError(
        "Missing raw Visium files in DATA_DIR: "
        + ", ".join(missing_raw)
        + ". Download them from Zenodo record 17676992 or set "
        + "DOWNLOAD_ZENODO=True."
    )

if USER_ALIGNMENT_DIR is not None:
    missing_aligned = [
        name
        for name in ("aligned_coords_NL3.csv", "aligned_coords_IL3.csv")
        if not (USER_ALIGNMENT_DIR / name).exists()
    ]
    if missing_aligned:
        raise FileNotFoundError(
            "SPALIGNDE_ALIGNMENT_DIR is set but is missing: "
            + ", ".join(missing_aligned)
        )

print("Raw Visium inputs are present.")
print("Packaged coordinate version:", alignment_metadata["coordinate_version"])"""


KIDNEY_COORDINATES = r"""coordinate_tables = []
for sample_id in ("NL3", "IL3"):
    if USER_ALIGNMENT_DIR is None:
        aligned = load_kidney_aligned_coordinates(sample_id)
    else:
        aligned = pd.read_csv(
            USER_ALIGNMENT_DIR / f"aligned_coords_{sample_id}.csv"
        )
    if "cell_id" not in aligned.columns and "barcode" in aligned.columns:
        aligned = aligned.rename(columns={"barcode": "cell_id"})
    positions = pd.read_csv(DATA_DIR / f"{sample_id}_tissue_positions.csv")
    coordinates = build_visium_coordinate_table(
        positions,
        aligned,
        sample_id=sample_id,
    )
    coordinate_tables.append(coordinates)
    coordinate_range = coordinates[["x_aligned", "y_aligned"]].agg(["min", "max"])
    print(sample_id, "matched spots:", len(coordinates))
    display(coordinate_range)

coordinate_data = pd.concat(coordinate_tables, ignore_index=True)"""


AGING_INTRO = r"""# Aging mouse brain: inference from the current PyPI Figure 5A coordinates

This tutorial applies the current public `spalignde` inference API to the packaged five-section Figure 5A aging-brain example. It contains the 6.6-, 15.8-, 30.9-, and 34.5-month queries plus the 4.3-month reference, uses the packaged precomputed `x_aligned` and `y_aligned` coordinates, and does not rerun alignment.

`Gamt` and `Vip` are fitted so the notebook can demonstrate both the reported global linear age-trend test and automatic trajectory-cluster selection; the local-statistic map remains focused on `Gamt`. The original 300-gene MERFISH data were published by Sun et al. and are available from [Zenodo record 13883177](https://doi.org/10.5281/zenodo.13883177). Raw counts are normalized to a library size of 250. The density channel receives 25% of standardized mismatch-feature energy; the gene-specific condition offset and local technical covariates are included. Mismatch-aware variance adjustment and connected-region cleanup are enabled, cell-type adjustment is disabled, and grid-level significance uses $q \leq 0.05$.

**Fixed-seed reproducibility.** Preparation, fitting, and trajectory clustering use seed `1`, and this tutorial requests `n_jobs=1`. During `prepare_inference`, the two seeded per-sample auto-geometry subsampling and parameter-estimation passes always use one worker so thread scheduling cannot change which sample consumes each RNG draw. All later preparation and fitting stages use the caller-requested `n_jobs` value, so this safeguard does not make the complete analysis single-threaded. The preparation metadata records `n_jobs`, `auto_geometry_n_jobs=1`, and `random_state`. Set `PYTHONHASHSEED=1` before kernel startup and retain the documented input order and numerical-library environment. Repeated runs reproduce the reported summaries; floating-point calibration values can differ in the last displayed digits."""


AGING_IMPORTS = r"""from pathlib import Path
import json
import os
import random

import numpy as np
import pandas as pd
from IPython.display import display

import spalignde
from spalignde.inference import (
    cluster_trajectories,
    fit_local_de,
    gene_level_acat_pvalue,
    gene_level_age_trend_acat,
    plot_local_result,
    prepare_inference,
)
from spalignde.datasets import (
    AGING_BRAIN_FIGURE5A_REFERENCE,
    AGING_BRAIN_FIGURE5A_SAMPLES,
    aging_brain_figure5a_genes,
    load_aging_brain_figure5a,
)

WORKFLOW_SEED = 1
random.seed(WORKFLOW_SEED)
np.random.seed(WORKFLOW_SEED)

print("spalignde version:", spalignde.__version__)"""


AGING_AUTO_K_MARKDOWN = r"""## Global age-trend test and automatic trajectory clustering

`gene_level_age_trend_acat` is the reported multi-age global test. At each retained grid location it regresses the unsmoothed adjusted local expression (`muA_adj_by_time`) on the four query ages with an intercept, using mismatch-aware `Wv_by_time` as local precision. It performs a two-sided slope test and combines local slope P values across space with ACAT. The 4.3-month reference is not added as a fifth regression observation. This test precedes trajectory clustering and does not use smoothed trajectories, cluster labels, or the selected `K`. `gene_level_acat_pvalue` is retained below only as the distinct any-spatial-change diagnostic.

`cluster_trajectories(..., n_clusters="auto")` evaluates candidate `K` values using held-out cluster-specific trajectory gain relative to a shared time trend. If the best dynamic gain is not positive after subtracting its one-SE uncertainty, it selects the smallest candidate `K`. Otherwise, it retains candidates within one SE of the best dynamic gain and examines them from fine to coarse using the fraction of grid locations in connected components smaller than one `R_map` footprint. If the next coarser candidate fails to reduce fragmentation, the current finer-side local minimum is retained. If fragmentation decreases throughout the scan and therefore supplies no elbow, the rule takes one conservative coarsening step from the finest retained candidate. Fragmentation is not an objective that is globally minimized, and the no-elbow fallback does not select the coarsest candidate. The notebook calls the public API directly and displays the complete stable selection diagnostics."""


AGING_AUTO_K_CODE = r"""trajectory_time_ids = list(
    result.fits[GENES[0]]["terrain_data"]["time_ids"]
)
time_values = [
    float(str(time_id).removeprefix("age_"))
    for time_id in trajectory_time_ids
]

trend_results = {}
trajectory_results = {}
gene_summary_rows = []
contrast_summary_rows = []
selection_fields = (
    "mode",
    "recommended_k",
    "best_dynamic_gain",
    "best_dynamic_gain_lower_1SE",
    "dynamic_candidates",
    "fine_to_coarse_order",
    "fine_to_coarse_scan",
    "fragmentation_stop_at_k",
    "rejected_coarser_k",
    "no_elbow_fallback_k",
    "reason",
    "rule",
)

for gene in GENES:
    assert list(result.fits[gene]["terrain_data"]["time_ids"]) == trajectory_time_ids
    trend = gene_level_age_trend_acat(
        result,
        gene,
        time_values=time_values,
        alpha=0.05,
    )
    trajectory = cluster_trajectories(
        result,
        gene,
        n_clusters="auto",
        time_values=time_values,
        random_state=WORKFLOW_SEED,
    )
    trend_results[gene] = trend
    trajectory_results[gene] = trajectory
    selection = trajectory.metadata["selection"]
    gene_summary_rows.append(
        {
            "gene": gene,
            "age-trend spatial ACAT P value": trend["summary"][
                "gene_level_trend_acat_p"
            ],
            "any-spatial-change diagnostic P value": gene_level_acat_pvalue(
                result, gene
            ),
            "selected K": trajectory.result["K_TRAJ"],
        }
    )
    display(
        pd.Series(
            {field: selection.get(field) for field in selection_fields},
            name=f"{gene} auto-K diagnostics",
        ).to_frame("value")
    )
    display(trajectory.result["auto_k"]["table"])

    terrain = result.fits[gene]["terrain_data"]
    for query_id in terrain["time_ids"]:
        statistic = np.asarray(terrain["stat_by_time"][query_id], dtype=float)
        q_value = np.asarray(terrain["q_by_time"][query_id], dtype=float)
        reported_mask = np.asarray(
            terrain["sig_mask_by_time"][query_id], dtype=bool
        )
        raw_q_mask = np.isfinite(q_value) & (q_value <= 0.05)
        contrast_summary_rows.append(
            {
                "gene": gene,
                "contrast": f"{query_id} - age_4.3",
                "q <= 0.05 grid locations": int(raw_q_mask.sum()),
                "reported grids after cleanup": int(reported_mask.sum()),
                "minimum q-value": float(np.nanmin(q_value)),
                "median |t|": float(np.nanmedian(np.abs(statistic))),
            }
        )

gene_summary = pd.DataFrame(gene_summary_rows)
contrast_summary = pd.DataFrame(contrast_summary_rows)
display(gene_summary)
display(contrast_summary)"""


AGING_LOAD_MARKDOWN = r"""## Load counts and the packaged precomputed coordinates

All 300 expression columns are retained because mismatch-risk estimation screens a broad stable-gene candidate panel even though only `Gamt` and `Vip` are fitted below. The public expression loader supplies raw counts, annotations and the current PyPI package's precomputed `x_aligned` and `y_aligned` coordinates. The validation below matches those coordinates explicitly by cell ID; row order is never used.

Set `SPALIGNDE_AGING_COORDINATE_DIR` only when validating an equivalent packaged coordinate directory with the same five observation tables."""


AGING_LOAD_CODE = r"""def find_repository_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src/spalignde/datasets/aging_brain").is_dir():
            return candidate
    raise FileNotFoundError(
        "Run this notebook from a spAlignDE checkout containing "
        "src/spalignde/datasets/aging_brain."
    )


REPOSITORY_ROOT = find_repository_root()
FORMAL_COORDINATE_DIR = Path(
    os.environ.get(
        "SPALIGNDE_AGING_COORDINATE_DIR",
        REPOSITORY_ROOT / "src/spalignde/datasets/aging_brain",
    )
).expanduser().resolve()

risk_genes = list(aging_brain_figure5a_genes())
data = load_aging_brain_figure5a()
metadata = json.loads(
    (FORMAL_COORDINATE_DIR / "metadata.json").read_text(encoding="utf-8")
)

for sample_id in AGING_BRAIN_FIGURE5A_SAMPLES:
    stem = sample_id.replace(".", "_")
    coordinate_path = FORMAL_COORDINATE_DIR / f"{stem}_observations.csv.gz"
    coordinates = pd.read_csv(
        coordinate_path,
        usecols=["cell_id", "x_aligned", "y_aligned"],
    )
    if coordinates["cell_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate coordinate cell IDs in {coordinate_path.name}.")
    selected = data["sample_id"].astype(str).eq(sample_id)
    ordered_ids = data.loc[selected, "cell_id"].astype(str)
    matched = coordinates.set_index(coordinates["cell_id"].astype(str)).reindex(
        ordered_ids
    )
    if matched[["x_aligned", "y_aligned"]].isna().any().any():
        raise ValueError(f"{sample_id} contains cells without formal coordinates.")
    data.loc[selected, ["x_aligned", "y_aligned"]] = matched[
        ["x_aligned", "y_aligned"]
    ].to_numpy(dtype=float)

print(f"Samples: {AGING_BRAIN_FIGURE5A_SAMPLES}")
print(f"Cells: {len(data):,}; risk-gene candidates: {len(risk_genes)}")
print("Packaged query subset:", metadata["queries"])
print("Aligned-coordinate source:", metadata["coordinate_system"])
print("Coordinate directory:", FORMAL_COORDINATE_DIR)
data.groupby("sample_id", sort=False).size().rename("n_cells").to_frame()"""


def _read(name: str):
    return nbformat.read(SOURCE_DIR / name, as_version=4)


def _write(name: str, notebook) -> None:
    for directory in (SOURCE_DIR, DOCS_DIR):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(copy.deepcopy(notebook), target)
        print(target.relative_to(ROOT))


def build_kidney(*, record_execution: bool) -> None:
    name = "post_alignment_inference_nb.ipynb"
    notebook = _read(name)
    notebook.cells[0].source = KIDNEY_INTRO
    notebook.cells[1].source = KIDNEY_HANDOFF
    cells_by_id = {cell.get("id"): cell for cell in notebook.cells}
    cells_by_id["87f4a039"].source = KIDNEY_IMPORTS
    cells_by_id["b28c1a97"].source = KIDNEY_RAW_INPUTS
    cells_by_id["16100433"].source = KIDNEY_COORDINATES
    cells_by_id["a9ba1f99"].source = KIDNEY_INTERPRETATION
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if (
                "import matplotlib.pyplot as plt" in cell.source
                and not cell.source.lstrip().startswith("%matplotlib inline")
            ):
                cell.source = "%matplotlib inline\n\n" + cell.source
            cell.source = cell.source.replace(
                'print("spalignde import:", Path(spalignde.__file__).resolve())',
                'print("spalignde version:", spalignde.__version__)',
            )
            cell.source = cell.source.replace(
                'print("Raw-data directory:", DATA_DIR)',
                'print("Raw-data source: Zenodo record 17676992 (local cache configured)")',
            )
            cell.source = cell.source.replace(
                "from spalignde import (",
                "from spalignde.inference import (",
            )
    if record_execution:
        _record_execution(notebook, aging_brain=False)
    _write(name, notebook)


def build_aging(*, record_execution: bool) -> None:
    name = "post_alignment_inference_aging_brain_nb.ipynb"
    notebook = _read(name)
    notebook.cells[0].source = AGING_INTRO
    notebook.cells[1].source = AGING_IMPORTS
    notebook.cells[2].source = AGING_LOAD_MARKDOWN
    notebook.cells[3].source = AGING_LOAD_CODE
    notebook.cells[3].source = notebook.cells[3].source.replace(
        'print("Coordinate directory:", FORMAL_COORDINATE_DIR)\n',
        '',
    )
    cells_by_id = {cell.get("id"): cell for cell in notebook.cells}
    cells_by_id["aging-fixed-summary-heading-0814"].source = AGING_AUTO_K_MARKDOWN
    cells_by_id["aging-fixed-summary-0814"].source = AGING_AUTO_K_CODE
    if record_execution:
        _record_execution(notebook, aging_brain=True)
    _write(name, notebook)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-execution",
        action="store_true",
        help=(
            "Refresh execution hashes after both generated notebooks have "
            "completed successfully. Do not use this option for a source-only build."
        ),
    )
    args = parser.parse_args()
    build_kidney(record_execution=args.record_execution)
    build_aging(record_execution=args.record_execution)


if __name__ == "__main__":
    main()
