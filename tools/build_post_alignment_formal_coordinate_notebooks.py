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
            "formal 19-query aging-brain archive, resolution 0.8, 800 iterations"
            if aging_brain
            else (
                "fixed-seed kidney manual prealignment, resolution 0.2, "
                "5000 iterations"
            )
        ),
        "aging_brain_included": aging_brain,
    }


KIDNEY_INTRO = r"""# Mouse kidney local inference from fixed-seed manual-alignment coordinates

This real-data notebook continues from the reported fixed-seed NL3/IL3 kidney alignment and runs post-alignment local inference with the current public `spalignde` API. The upstream Kidney workflow uses Leiden resolution `0.2`, seed `1000`, and a selected manual similarity pre-alignment (`scale=1`, `theta=0`, `translation_x=-36.20040965`, `translation_y=-153.38356513`) before 5,000 S-LDDMM iterations with `restore_best_checkpoint=False`. IL3 is the 2,965-spot query and NL3 is the unchanged 3,215-spot reference.

The raw count matrices and tissue-position files come from Zenodo record `17676992`. The checked-out repository supplies compact coordinate tables generated directly from `kidney_IL3_to_NL3_aligned.h5ad`, the output of the public manual-alignment notebook. The source H5AD and packaged coordinate SHA-256 values are recorded in the metadata.

The analysis proceeds through one continuous handoff:

1. load public expression data and the fixed-seed manual-alignment coordinates;
2. match spots by terminal Visium barcode;
3. construct the shared grid and local neighborhoods;
4. estimate stable-gene and density-based mismatch risk;
5. fit gene-specific mismatch-aware local tests with the public `fit_local_de` function; and
6. report grid-level statistics, FDR-adjusted regions, and representative gene maps.

NL3 defines the reference coordinate system and IL3 is the query. Users may replace the checked-in coordinates with their own spAlignDE output, but a custom alignment is not expected to reproduce the recorded numerical results unless its evaluated spot set and coordinates are identical.

**Fixed-seed reproducibility.** Upstream clustering and alignment use seed `1000`; this inference workflow uses seed `1` and `n_jobs=1`. For the closest numerical reproduction, set `PYTHONHASHSEED=1` before kernel startup and keep the documented input order, package version, and configuration fixed. Small floating-point differences in the last displayed digits can remain across numerical-library builds."""


KIDNEY_HANDOFF = r"""## Alignment-to-inference handoff

The recorded website example starts from the output of the public fixed-seed Kidney alignment notebook rather than rerunning alignment inside this inference notebook:

- upstream notebook: `cross_sample_alignment_mouse_kidney_alignment_nb.ipynb`;
- handoff artifact: `tutorials/cross_sample/kidney/output/kidney_IL3_to_NL3_aligned.h5ad`;
- evaluated coordinate columns: `x_aligned` and `y_aligned` for IL3 and the unchanged NL3 reference;
- coordinate scale: the saved compact array coordinates are multiplied by the recorded factor of 50 for inference.

The repository packages a compact, hash-tracked copy of this exact manual-alignment handoff as `src/spalignde/datasets/kidney/aligned_coords_IL3.csv.gz` and `aligned_coords_NL3.csv.gz`. The alignment input and region annotations come from STcompare record `20647680`; this inference stage reloads the NL3/IL3 10x matrices and tissue-position tables from source record `17676992` and joins them to the saved coordinates by terminal barcode. The two records therefore serve different handoff roles.

For a custom alignment, either point `SPALIGNDE_KIDNEY_ALIGNED_H5AD` to an H5AD containing `sample_id`, `x_aligned`, and `y_aligned`, or point `SPALIGNDE_ALIGNMENT_DIR` to a directory containing `aligned_coords_NL3.csv` and `aligned_coords_IL3.csv`. Set only one input variable. `SPALIGNDE_KIDNEY_DATA_DIR` and `SPALIGNDE_TUTORIAL_WORK_DIR` can redirect the raw-data and working directories.

```python
import os
os.environ["SPALIGNDE_KIDNEY_ALIGNED_H5AD"] = "/path/to/custom_alignment.h5ad"
```

Raw 10x matrices are always reloaded because an alignment object may contain only the gene-filtered matrix used for clustering and registration."""


AGING_INTRO = r"""# Aging mouse brain: inference from the formal 19-age fixed-seed alignment

This tutorial applies the current public `spalignde` inference API to the five-section Figure 5A subset of the formal aging-brain alignment. The complete upstream archive contains 19 fixed-seed query-age alignments to the unchanged 4.3-month reference, produced with Leiden resolution `0.8`, alignment seed `1000`, and 800 S-LDDMM iterations. The website example consumes four of those formal query files—6.6, 15.8, 30.9, and 34.5 months—plus the unchanged 4.3-month reference. It uses their saved `x_aligned` and `y_aligned` coordinates and does not rerun alignment.

`Gamt` and `Vip` are fitted so the notebook can demonstrate both the reported global linear age-trend test and automatic trajectory-cluster selection; the local-statistic map remains focused on `Gamt`. The original 300-gene MERFISH data were published by Sun et al. and are available from [Zenodo record 13883177](https://doi.org/10.5281/zenodo.13883177). Raw counts are normalized to a library size of 250. The density channel receives 25% of standardized mismatch-feature energy; the gene-specific condition offset and local technical covariates are included. Mismatch-aware variance adjustment and connected-region cleanup are enabled, cell-type adjustment is disabled, and grid-level significance uses $q \leq 0.05$.

**Fixed-seed reproducibility.** Upstream alignment uses seed `1000`; preparation, fitting, and trajectory clustering use seed `1` and `n_jobs=1`. Set `PYTHONHASHSEED=1` before kernel startup and retain the documented input order and numerical-library environment. Repeated runs reproduce the reported summaries; floating-point calibration values can differ in the last displayed digits."""


AGING_IMPORTS = r"""from pathlib import Path
import json
import os
import random

import numpy as np
import pandas as pd
from IPython.display import display

import spalignde
from spalignde import (
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


AGING_LOAD_MARKDOWN = r"""## Load counts and the formal fixed-seed coordinates

All 300 expression columns are retained because mismatch-risk estimation screens a broad stable-gene candidate panel even though only `Gamt` and `Vip` are fitted below. The expression loader supplies raw counts and annotations. The code then replaces only `x_aligned` and `y_aligned` from the repository's hash-tracked coordinate handoff generated from the formal 19-query archive. Cell IDs are matched explicitly; row order is never used.

The website subset uses four of the 19 formal query outputs. The 4.3-month reference retains its coordinates from the corresponding fixed-seed `run_1/cluster_labels.csv.gz`. Set `SPALIGNDE_AGING_COORDINATE_DIR` only when validating an equivalent packaged coordinate directory."""


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
print("Formal upstream query alignments:", metadata["alignment"]["formal_query_count"])
print("Website query subset:", metadata["queries"])
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
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if (
                "import matplotlib.pyplot as plt" in cell.source
                and not cell.source.lstrip().startswith("%matplotlib inline")
            ):
                cell.source = "%matplotlib inline\n\n" + cell.source
            cell.source = cell.source.replace(
                'else "checked-in formal run_1 coordinates"',
                'else "checked-in fixed-seed manual alignment"',
            )
            cell.source = cell.source.replace(
                'print("spalignde import:", Path(spalignde.__file__).resolve())',
                'print("spalignde version:", spalignde.__version__)',
            )
            cell.source = cell.source.replace(
                'print("Raw-data directory:", DATA_DIR)',
                'print("Raw-data source: Zenodo record 17676992 (local cache configured)")',
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
