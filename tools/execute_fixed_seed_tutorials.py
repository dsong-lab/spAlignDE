#!/usr/bin/env python3
"""Execute every public computational tutorial with its specified seed.

The executor starts each notebook in a fresh kernel from the repository root,
forces the kernel to import the current checkout, and writes a notebook only
after every cell succeeds.  Large datasets remain external and are selected
through the environment variables documented by the notebooks.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
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


@contextmanager
def _kernel_environment():
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
    executor = ExecutePreprocessor(
        timeout=timeout,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        allow_errors=False,
        record_timing=True,
    )
    print(f"EXECUTE {relative} (seed={seed})", flush=True)
    with _kernel_environment():
        executed, _ = executor.preprocess(
            notebook,
            resources={"metadata": {"path": str(ROOT)}},
        )

    sanitize_notebook(executed)
    executed.metadata.pop("spAlignDE_fixed_seed_output", None)
    executed.metadata["spAlignDE_execution"] = {
        "fully_executed": True,
        "workflow_seed": seed,
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


def refresh_source(relative: str, seed: int) -> None:
    """Refresh generated source while preserving repeated-run outputs.

    This is intended only when a notebook's executable source is regenerated
    from its notebook builder and the corresponding computation has already
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
    execution = notebook.metadata.setdefault("spAlignDE_execution", {})
    validated_execution = validated.metadata.get("spAlignDE_execution", {})
    if isinstance(validated_execution, dict):
        execution.update(validated_execution)
    execution.update(
        {
            "workflow_seed": seed,
            "source_refresh_only": True,
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
        "--refresh-source-only",
        action="store_true",
        help="Preserve repeated-run outputs while refreshing generated notebook text.",
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
        if args.refresh_source_only:
            refresh_source(relative, seed)
        else:
            execute(relative, seed, args.timeout)
        executed += 1
    print(f"Executed {executed} computational tutorial notebook(s).")


if __name__ == "__main__":
    main()
