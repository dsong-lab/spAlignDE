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


def code_hash(notebook) -> str:
    """Hash executable code independently of explanatory Markdown."""
    payload = "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def saved_text(notebook) -> str:
    """Flatten saved textual outputs for result-level contract checks."""
    chunks: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append(str(output.get("text", "")))
            data = output.get("data", {})
            plain = data.get("text/plain", "")
            chunks.append("".join(plain) if isinstance(plain, list) else str(plain))
    return "\n".join(chunks)


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
    inference_notebooks = {
        "post_alignment_inference_aging_brain_nb.ipynb",
        "post_alignment_inference_nb.ipynb",
    }
    seed_call = "seed_controls = spAlignDE.set_random_seed("
    if f"WORKFLOW_SEED = {seed}" not in code:
        problems.append(f"{relative}: WORKFLOW_SEED is not {seed}")
    if relative in inference_notebooks:
        for required in (
            "random.seed(WORKFLOW_SEED)",
            "np.random.seed(WORKFLOW_SEED)",
        ):
            if required not in code:
                problems.append(
                    f"{relative}: missing explicit inference seed control {required}"
                )
    elif seed_call not in code:
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
            "pairing_weight_sdf=0.05",
            "pairing_weight_chamfer=0.05",
            "pairing_weight_dice=0.20",
            "pairing_weight_area=0.50",
            "pairing_weight_thickness=0.20",
        ):
            if required not in code:
                problems.append(
                    f"{relative}: missing automatic Atlas optimizer control {required}"
                )
    if relative == "cross_modality/st_he_alignment_nb.ipynb":
        for required in (
            "kernel_scale=60.0",
            "velocity_grid_spacing=6.0",
            "restore_best_checkpoint=False",
            'dtype="float64"',
        ):
            if required not in code:
                problems.append(f"{relative}: missing validated H&E control {required}")
    if relative == "cross_modality/atac_st_alignment_nb.ipynb":
        for required in (
            "pair_score_threshold=0.21",
            "kernel_scale=100.0",
            "velocity_grid_spacing=50.0",
            "restore_best_checkpoint=False",
            'dtype="float64"',
        ):
            if required not in code:
                problems.append(f"{relative}: missing validated ATAC control {required}")
    if relative == "cross_sample_alignment_mouse_kidney_alignment_nb.ipynb":
        for required in (
            "spAlignDE.ManualPrealignmentConfig(",
            "spAlignDE.prealign_cross_sample_manual(",
            'SPALIGNDE_MANUAL_SCALE", 1.0',
            'SPALIGNDE_MANUAL_THETA_DEG", 0.0',
            'SPALIGNDE_MANUAL_TX", -36.20040965',
            'SPALIGNDE_MANUAL_TY", -153.38356513',
        ):
            if required not in code:
                problems.append(
                    f"{relative}: missing validated manual kidney control {required}"
                )
        if "prealignment = spAlignDE.prealign_cross_sample(" in code:
            problems.append(
                f"{relative}: still uses the unstable automatic kidney initialization"
            )
    if relative == "cross_sample_uncertainty_report.ipynb":
        for required in (
            "WORKFLOW_SEED = 1000",
            '"niter": 500',
            '"lrM": 2e3',
            '"restore_best": False',
        ):
            if required not in code:
                problems.append(f"{relative}: missing validated uncertainty control {required}")
    if relative in inference_notebooks:
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
        if not execution.get("documentation_only_refresh"):
            problems.append(
                f"{relative}: source changed after the saved outputs were executed"
            )
        elif execution.get("documentation_source_sha256") != source_hash(notebook):
            problems.append(
                f"{relative}: documentation-only source hash is inconsistent"
            )
        elif execution.get("executed_code_sha256") != code_hash(notebook):
            problems.append(
                f"{relative}: executable code changed during a documentation-only refresh"
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

    outputs = saved_text(notebook)
    expected_output_markers = {
        "cross_modal_atlas_alignment_nb.ipynb": (
            "Final matched structure pairs: 18",
        ),
        "cross_modality/atac_st_alignment_nb.ipynb": (
            "Accepted structure pairs: 8",
        ),
        "cross_modality/st_he_feature_clustering_nb.ipynb": (
            "final cleaned regions",
            "21",
        ),
        "cross_modality/st_he_alignment_nb.ipynb": (
            "Reference: 21 image-derived structures",
        ),
        "cross_sample_alignment_mouse_kidney_alignment_nb.ipynb": (
            "nearest_label_agreement_prealigned: 0.663743676222597",
            "nearest_label_agreement_aligned: 0.736593591905565",
        ),
        "cross_sample_uncertainty_report.ipynb": (
            "429.54",
            "2352.74",
            "Checkpoint policy: final optimizer iterate (restore_best=False)",
        ),
    }
    for marker in expected_output_markers.get(relative, ()):
        if marker not in outputs:
            problems.append(f"{relative}: saved outputs lack current result marker {marker!r}")
    if relative == "cross_sample_uncertainty_report.ipynb":
        if "median  68.43" in outputs or "300.01" in outputs:
            problems.append(
                f"{relative}: saved outputs still contain the retired restore_best=True result"
            )
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

    cross_modality_manifest = json.loads(
        (ROOT / "docs/source/_static/cross_modality_reproducibility_manifest.json")
        .read_text(encoding="utf-8")
    )
    observed_knn = cross_modality_manifest.get("atac_to_st", {}).get(
        "local_neighborhood_preservation"
    )
    expected_knn = {
        "10": 0.8940965816603367,
        "20": 0.9756158437330441,
        "30": 0.9167046482184844,
        "50": 0.9287726532826914,
    }
    if observed_knn != expected_knn:
        problems.append(
            "cross-modality manifest lacks the exact fixed ATAC neighborhood "
            f"preservation values: observed={observed_knn!r}"
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
