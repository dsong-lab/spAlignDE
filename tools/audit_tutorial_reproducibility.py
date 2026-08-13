#!/usr/bin/env python3
"""Fail when a computational tutorial loses its fixed-seed contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_notebooks"
MIRROR = ROOT / "docs" / "source" / "source_notebooks"

SEEDS = {
    "clustering/clustering_joint_nb.ipynb": 1000,
    "clustering/clustering_single_nb.ipynb": 1234,
    "cross_modal_atlas_alignment_nb.ipynb": 1234,
    "cross_modality/atac_st_alignment_nb.ipynb": 1234,
    "cross_modality/atac_st_single_clustering_nb.ipynb": 1234,
    "cross_modality/st_he_alignment_nb.ipynb": 0,
    "cross_modality/st_he_feature_clustering_nb.ipynb": 0,
    "cross_modality/st_he_feature_extraction_nb.ipynb": 0,
    "cross_modality/ui_paired_atlas_alignment_nb.ipynb": 1234,
    "cross_sample_alignment_breast_cancer_alignment_nb.ipynb": 1000,
    "cross_sample_alignment_breast_cancer_clustering_nb.ipynb": 1000,
    "cross_sample_alignment_mouse_kidney_alignment_nb.ipynb": 1000,
    "cross_sample_alignment_mouse_kidney_clustering_nb.ipynb": 1000,
    "cross_sample_alignment_nb.ipynb": 1000,
    "cross_sample_uncertainty_report.ipynb": 1000,
    "post_alignment_inference_aging_brain_nb.ipynb": 1,
    "post_alignment_inference_nb.ipynb": 1,
}

STOCHASTIC_MARKERS = (
    "SingleClusteringConfig(",
    "JointClusteringConfig(",
    "ImageFeatureClusteringConfig(",
    "KMeans(",
    "np.random.default_rng(",
    "subprocess.run(command",
)


def source_hash(notebook) -> str:
    """Hash exactly the executable source captured by the notebook executor."""
    payload = "\n\n".join(
        f"{cell.cell_type}\n{cell.source}" for cell in notebook.cells
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_notebook(relative: str, seed: int) -> list[str]:
    problems: list[str] = []
    source_path = SOURCE / relative
    mirror_path = MIRROR / relative
    if not source_path.is_file():
        return [f"{relative}: missing source notebook"]
    if not mirror_path.is_file():
        return [f"{relative}: missing documentation mirror"]
    if source_path.read_bytes() != mirror_path.read_bytes():
        problems.append(f"{relative}: source and documentation mirror differ")

    notebook = nbformat.read(source_path, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    seed_call = "seed_controls = spAlignDE.set_random_seed("
    if f"WORKFLOW_SEED = {seed}" not in code:
        problems.append(f"{relative}: WORKFLOW_SEED is not {seed}")
    if seed_call not in code:
        problems.append(f"{relative}: missing spAlignDE.set_random_seed call")
    else:
        seed_position = code.index(seed_call)
        for marker in STOCHASTIC_MARKERS:
            marker_position = code.find(marker)
            if marker_position >= 0 and marker_position < seed_position:
                problems.append(
                    f"{relative}: {marker} appears before the workflow seed reset"
                )

    if "SingleClusteringConfig(" in code and "random_state=WORKFLOW_SEED" not in code:
        problems.append(f"{relative}: single clustering lacks explicit random_state")
    if "JointClusteringConfig(" in code:
        for required in (
            "random_state=WORKFLOW_SEED",
            'leiden_flavor="leidenalg"',
            "leiden_n_iterations=-1",
        ):
            if required not in code:
                problems.append(f"{relative}: missing joint backend control {required}")
    if "ImageFeatureClusteringConfig(" in code and "random_state=WORKFLOW_SEED" not in code:
        problems.append(f"{relative}: histology clustering lacks explicit random_state")
    if "subprocess.run(command" in code:
        if code.count('"--seed", str(WORKFLOW_SEED)') != 1:
            problems.append(f"{relative}: external runner must receive exactly one seed")
    if "np.random.default_rng(" in code and "np.random.default_rng(WORKFLOW_SEED)" not in code:
        problems.append(f"{relative}: display subsampling uses a different seed")
    if relative == "cross_modal_atlas_alignment_nb.ipynb":
        for required in (
            "stage_iterations=(100, 500, 100)",
            "restore_best_checkpoint=False",
            "continuation_iterations=200",
            "continuation_restore_best_checkpoint=False",
        ):
            if required not in code:
                problems.append(
                    f"{relative}: missing automatic Atlas optimizer control {required}"
                )
    if relative in {
        "post_alignment_inference_aging_brain_nb.ipynb",
        "post_alignment_inference_nb.ipynb",
    }:
        if code.count("n_jobs=1") < 2:
            problems.append(
                f"{relative}: preparation and fitting must both use n_jobs=1"
            )
        if code.count("random_state=WORKFLOW_SEED") < 2:
            problems.append(
                f"{relative}: inference calls must use WORKFLOW_SEED explicitly"
            )

    metadata = notebook.metadata.get("spAlignDE_reproducibility", {})
    if metadata.get("workflow_seed") != seed:
        problems.append(f"{relative}: notebook seed metadata is missing or inconsistent")
    if metadata.get("discrete_repeat_contract") != "exact":
        problems.append(f"{relative}: exact discrete-output contract is missing")
    if "tolerance" not in metadata.get("cuda_coordinate_contract", ""):
        problems.append(f"{relative}: CUDA coordinate tolerance contract is missing")

    execution = notebook.metadata.get("spAlignDE_execution", {})
    if execution.get("fully_executed") is not True:
        problems.append(f"{relative}: full fixed-seed execution receipt is missing")
    if execution.get("workflow_seed") != seed:
        problems.append(f"{relative}: execution receipt seed is missing or inconsistent")
    recorded_source_hash = execution.get("source_sha256")
    if not recorded_source_hash:
        problems.append(f"{relative}: executed source hash is missing")
    elif recorded_source_hash != source_hash(notebook):
        problems.append(
            f"{relative}: source changed after the saved outputs were executed"
        )
    if not execution.get("repository_commit"):
        problems.append(f"{relative}: execution source revision is missing")

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        if cell.execution_count is None:
            problems.append(f"{relative}: code cell {index} was not executed")
        if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
            problems.append(f"{relative}: code cell {index} contains a saved error")

    output_payload = [
        cell.get("outputs", [])
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]
    rendered = json.dumps(output_payload, sort_keys=True, separators=(",", ":"))
    output_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    if execution.get("saved_output_sha256") != output_hash:
        problems.append(f"{relative}: saved output hash does not match notebook outputs")
    return problems


def main() -> None:
    problems: list[str] = []
    for relative, seed in SEEDS.items():
        problems.extend(audit_notebook(relative, seed))

    discovered = {
        path.relative_to(SOURCE).as_posix() for path in SOURCE.rglob("*.ipynb")
    }
    expected = set(SEEDS) | {"cross_modality/interactive_region_pairing_nb.ipynb"}
    if discovered != expected:
        problems.append(
            "Notebook inventory changed; classify new notebooks explicitly: "
            f"unexpected={sorted(discovered - expected)}, missing={sorted(expected - discovered)}"
        )

    if problems:
        raise SystemExit("\n".join(problems))
    print(
        f"PASS: {len(SEEDS)} computational notebooks have fixed seeds, explicit "
        "backend controls, full execution receipts, verified output hashes, and "
        "identical documentation mirrors."
    )
    print(
        "INFO: interactive_region_pairing_nb.ipynb is manual input capture and is "
        "not classified as a stochastic computational workflow."
    )


if __name__ == "__main__":
    main()
