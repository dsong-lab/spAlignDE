#!/usr/bin/env python3
"""Execute every public computational tutorial with its canonical seed.

The executor starts each notebook in a fresh kernel from the repository root,
forces the kernel to import the current checkout, and writes a notebook only
after every cell succeeds.  Large datasets remain external and are selected
through the environment variables documented by the notebooks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from sanitize_notebook_outputs import sanitize_notebook


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source_notebooks"
MIRROR_ROOT = ROOT / "docs" / "source" / "source_notebooks"

# Dependency order is intentional.  Clustering/feature notebooks create the
# hand-off files consumed by the alignment notebooks that follow them.
WORKFLOWS = (
    ("clustering/clustering_single_nb.ipynb", 1234),
    ("cross_modal_atlas_alignment_nb.ipynb", 1234),
    ("cross_modality/ui_paired_atlas_alignment_nb.ipynb", 1234),
    ("cross_modality/atac_st_single_clustering_nb.ipynb", 1234),
    ("cross_modality/atac_st_alignment_nb.ipynb", 1234),
    ("cross_modality/st_he_feature_extraction_nb.ipynb", 0),
    ("cross_modality/st_he_feature_clustering_nb.ipynb", 0),
    ("cross_modality/st_he_alignment_nb.ipynb", 0),
    ("clustering/clustering_joint_nb.ipynb", 1000),
    ("cross_sample_alignment_nb.ipynb", 1000),
    ("cross_sample_alignment_mouse_kidney_clustering_nb.ipynb", 1000),
    ("cross_sample_alignment_mouse_kidney_alignment_nb.ipynb", 1000),
    ("cross_sample_alignment_breast_cancer_clustering_nb.ipynb", 1000),
    ("cross_sample_alignment_breast_cancer_alignment_nb.ipynb", 1000),
    ("cross_sample_uncertainty_report.ipynb", 1000),
    ("post_alignment_inference_aging_brain_nb.ipynb", 1),
    ("post_alignment_inference_nb.ipynb", 1),
)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hash(notebook) -> str:
    payload = "\n\n".join(
        f"{cell.cell_type}\n{cell.source}" for cell in notebook.cells
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_hash(notebook) -> str:
    """Hash executable code separately from explanatory Markdown."""
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


@contextmanager
def _kernel_environment(seed: int):
    previous = os.environ.copy()
    source_path = str(ROOT / "src")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ.update(
        {
            "PYTHONPATH": (
                source_path
                if not existing_pythonpath
                else os.pathsep.join((source_path, existing_pythonpath))
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def execute(relative: str, seed: int, timeout: int) -> None:
    path = SOURCE_ROOT / relative
    notebook = nbformat.read(path, as_version=4)
    source_hash = _source_hash(notebook)
    executor = ExecutePreprocessor(
        timeout=timeout,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        allow_errors=False,
        record_timing=True,
    )
    print(f"EXECUTE {relative} (seed={seed})", flush=True)
    with _kernel_environment(seed):
        executed, _ = executor.preprocess(
            notebook,
            resources={"metadata": {"path": str(ROOT)}},
        )

    sanitize_notebook(executed)
    executed.metadata.pop("spAlignDE_fixed_seed_output", None)
    executed.metadata["spAlignDE_execution"] = {
        "fully_executed": True,
        "workflow_seed": seed,
        "source_sha256": source_hash,
        "executed_code_sha256": _code_hash(executed),
        "saved_output_sha256": _output_hash(executed),
        "repository_commit": _git_revision(),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_root": "repository root",
        "package_source": "current checkout src",
        "input_manifest": "docs/source/_static/tutorial_execution_manifest.json",
        "aging_brain_included": (
            relative == "post_alignment_inference_aging_brain_nb.ipynb"
        ),
    }
    temporary = path.with_suffix(".executed.tmp.ipynb")
    nbformat.write(executed, temporary)
    temporary.replace(path)

    mirror = MIRROR_ROOT / relative
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_bytes(path.read_bytes())
    print(f"PASS    {relative}", flush=True)


def refresh_source_hash(relative: str, seed: int) -> None:
    """Refresh generated source while preserving validated saved outputs.

    This is intended only when a notebook's executable source is regenerated
    from its canonical builder and the corresponding computation has already
    been repeated outside the public checkout. It prevents regenerated source
    from silently discarding the published result while clearly distinguishing
    a validated imported output from an in-checkout execution.
    """
    path = SOURCE_ROOT / relative
    mirror = MIRROR_ROOT / relative
    validated_path_env = os.environ.get("SPALIGNDE_VALIDATED_NOTEBOOK")
    validated_path = (
        Path(validated_path_env).expanduser().resolve()
        if validated_path_env
        else mirror
    )
    generated = nbformat.read(path, as_version=4)
    validated = nbformat.read(validated_path, as_version=4)
    generated_code = [cell for cell in generated.cells if cell.cell_type == "code"]
    validated_code = [cell for cell in validated.cells if cell.cell_type == "code"]
    if len(generated_code) != len(validated_code):
        raise ValueError(
            f"Cannot restore outputs for {relative}: generated and validated "
            "notebooks have different code-cell counts."
        )
    for destination, source in zip(generated_code, validated_code):
        destination["execution_count"] = source.get("execution_count")
        destination["outputs"] = source.get("outputs", [])

    notebook = generated
    source_hash = _source_hash(notebook)
    execution = notebook.metadata.setdefault("spAlignDE_execution", {})
    execution.update(
        {
            "fully_executed": False,
            "workflow_seed": seed,
            "source_sha256": source_hash,
            "executed_code_sha256": _code_hash(notebook),
            "saved_output_sha256": _output_hash(notebook),
            "repository_commit": _git_revision(),
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_root": "validated external run",
            "package_source": "current checkout src",
            "input_manifest": "docs/source/_static/tutorial_execution_manifest.json",
            "output_provenance": "validated fixed-seed run imported after source refresh",
            "source_refresh_only": True,
            "aging_brain_included": False,
        }
    )
    nbformat.write(notebook, path)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_bytes(path.read_bytes())
    print(f"REFRESH {relative} (seed={seed})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Execute only a listed repository-relative notebook; repeat as needed.",
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--refresh-source-hash-only",
        action="store_true",
        help="Preserve validated outputs and refresh source/provenance hashes only.",
    )
    args = parser.parse_args()

    selected = set(args.only)
    if args.list:
        for relative, seed in WORKFLOWS:
            print(f"{seed:4d}  {relative}")
        return

    unknown = selected.difference(relative for relative, _ in WORKFLOWS)
    if unknown:
        raise SystemExit("Unknown notebook(s): " + ", ".join(sorted(unknown)))

    executed = 0
    for relative, seed in WORKFLOWS:
        if selected and relative not in selected:
            continue
        if args.refresh_source_hash_only:
            refresh_source_hash(relative, seed)
        else:
            execute(relative, seed, args.timeout)
        executed += 1
    print(f"Executed {executed} computational tutorial notebook(s).")


if __name__ == "__main__":
    main()
