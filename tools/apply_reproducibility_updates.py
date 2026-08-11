#!/usr/bin/env python3
"""Apply fixed-seed controls to canonical notebooks and their Sphinx mirrors.

The transformation preserves saved outputs. Data-heavy notebooks whose
numeric configuration changes must be re-executed before release.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "source_notebooks"
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
    "post_alignment_inference_nb.ipynb": 1,
}


def _add_page_note(notebook, seed: int) -> None:
    marker = "**Fixed-seed reproducibility.**"
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        if marker in cell.source:
            return
        cell.source = (
            cell.source.rstrip()
            + "\n\n"
            + marker
            + f" This workflow uses seed `{seed}`. Launch Python with "
            + "`PYTHONHASHSEED` set before kernel startup, then keep the "
            + "documented input order, package versions and configuration fixed. "
            + "Discrete labels/pairs are expected to match exactly; CUDA "
            + "coordinates are compared with the documented numerical tolerance."
        )
        return


def _add_seed_call(notebook, seed: int) -> None:
    marker = "seed_controls = spAlignDE.set_random_seed("
    if any(marker in cell.source for cell in notebook.cells if cell.cell_type == "code"):
        return

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if "import spAlignDE" not in source:
            continue
        import_line = "import spAlignDE"
        insertion = (
            f"{import_line}\n\n"
            f"WORKFLOW_SEED = {seed}\n"
            "seed_controls = spAlignDE.set_random_seed(\n"
            "    WORKFLOW_SEED,\n"
            "    deterministic_torch=True,\n"
            ")"
        )
        cell.source = source.replace(import_line, insertion, 1)
        return

    for cell in notebook.cells:
        if cell.cell_type != "code" or "from spAlignDE import" not in cell.source:
            continue
        cell.source = cell.source.replace(
            "from spAlignDE import",
            "import spAlignDE\n\n"
            f"WORKFLOW_SEED = {seed}\n"
            "seed_controls = spAlignDE.set_random_seed(\n"
            "    WORKFLOW_SEED,\n"
            "    deterministic_torch=True,\n"
            ")\n\n"
            "from spAlignDE import",
            1,
        )
        return

    # The breast-cancer clustering notebook delegates work to a subprocess and
    # previously did not import the package in its setup cell.
    setup_updated = False
    for cell in notebook.cells:
        if cell.cell_type == "code" and "from IPython.display import Image, display" in cell.source:
            cell.source = cell.source.replace(
                "from IPython.display import Image, display",
                "from IPython.display import Image, display\n\n"
                "import spAlignDE\n\n"
                f"WORKFLOW_SEED = {seed}\n"
                "seed_controls = spAlignDE.set_random_seed(\n"
                "    WORKFLOW_SEED,\n"
                "    deterministic_torch=True,\n"
                ")",
                1,
            )
            setup_updated = True
            break
    if setup_updated:
        for cell in notebook.cells:
            if cell.cell_type != "code" or "subprocess.run(command" not in cell.source:
                continue
            duplicate = (
                '        "--seed", str(WORKFLOW_SEED),\n'
                '        "--seed", str(WORKFLOW_SEED),'
            )
            cell.source = cell.source.replace(
                duplicate,
                '        "--seed", str(WORKFLOW_SEED),',
            )
            if '"--seed", str(WORKFLOW_SEED)' not in cell.source:
                cell.source = cell.source.replace(
                    '        "--lambdas", "0", "0.2", "0.5", "0.8", "1",',
                    '        "--lambdas", "0", "0.2", "0.5", "0.8", "1",\n'
                    '        "--seed", str(WORKFLOW_SEED),',
                    1,
                )
            break
        return

    raise RuntimeError("No spAlignDE setup cell found")


def _replace_once(notebook, old: str, new: str, *, required: bool = True) -> None:
    for cell in notebook.cells:
        if old in cell.source:
            cell.source = cell.source.replace(old, new, 1)
            return
    if any(new in cell.source for cell in notebook.cells):
        return
    if required:
        raise RuntimeError(f"Notebook replacement target not found: {old!r}")


def _update_config(relative: str, notebook) -> None:
    if relative == "clustering/clustering_joint_nb.ipynb":
        _replace_once(
            notebook,
            "    compute_umap=False,\n)",
            "    compute_umap=False,\n"
            "    random_state=WORKFLOW_SEED,\n"
            '    leiden_flavor="leidenalg",\n'
            "    leiden_n_iterations=-1,\n)",
        )
    elif relative == "clustering/clustering_single_nb.ipynb":
        _replace_once(
            notebook,
            '    decay="scaled_gaussian",\n    refine_boundaries=True,',
            '    decay="scaled_gaussian",\n'
            "    random_state=WORKFLOW_SEED,\n"
            "    refine_boundaries=True,",
        )
    elif relative == "cross_modal_atlas_alignment_nb.ipynb":
        replacements = {
            "    n_levels=4,": "    n_levels=3,",
            "    pairing_weight_sdf=0.08,": "    pairing_weight_sdf=0.05,",
            "    pairing_weight_chamfer=0.06,": "    pairing_weight_chamfer=0.05,",
            "    pairing_weight_dice=0.18,": "    pairing_weight_dice=0.20,",
            "    pairing_weight_area=0.47,": "    pairing_weight_area=0.50,",
            "    pairing_weight_thickness=0.21,": (
                "    pairing_weight_thickness=0.20,\n"
                "    continuation_kernel_scale=200,\n"
                "    continuation_velocity_grid_spacing=50,\n"
                "    continuation_restore_best_checkpoint=False,"
            ),
            "Three coarser partitions precede the final\nBANKSY partition. Four levels make the coarse-to-fine deformation more gradual\nand reduce sensitivity to small changes in the finest BANKSY partition.": (
                "Two coarser partitions precede the final BANKSY partition. "
                "Three levels provide the validated coarse-to-fine path."
            ),
            "These levels are ST-only: Allen labels and coordinates are not used to create\nthem.": (
                "These levels are ST-only. Allen candidates from hierarchy depths 2–10 "
                "remain eligible at every stage; only the ST partition becomes finer."
            ),
            "`0.08 × SDF + 0.06 × Chamfer + 0.18 × Dice + 0.47 × area + 0.21 × thickness`.": (
                "`0.05 × SDF + 0.05 × Chamfer + 0.20 × Dice + 0.50 × area + 0.20 × thickness`."
            ),
            "The four-level package workflow": "The three-level package workflow",
        }
        for old, new in replacements.items():
            _replace_once(notebook, old, new)
        _replace_once(
            notebook,
            "    pairing_weight_asd=0.00,\n",
            "",
            required=False,
        )
    elif relative == "cross_modality/atac_st_single_clustering_nb.ipynb":
        _replace_once(notebook, "    random_state=1234,", "    random_state=WORKFLOW_SEED,")
    elif relative == "cross_modality/atac_st_alignment_nb.ipynb":
        _replace_once(
            notebook,
            "    matching_scale=0.5,\n)",
            '    matching_scale=0.5,\n    dtype="float64",\n)',
        )
    elif relative == "cross_modality/st_he_feature_clustering_nb.ipynb":
        _replace_once(notebook, "    random_state=0,", "    random_state=WORKFLOW_SEED,")
    elif relative == "cross_modality/st_he_alignment_nb.ipynb":
        _replace_once(
            notebook,
            "        device=None,\n    ),",
            '        device=None,\n        dtype="float64",\n    ),',
        )
    elif relative in {
        "cross_sample_alignment_nb.ipynb",
        "cross_sample_alignment_mouse_kidney_alignment_nb.ipynb",
    }:
        _replace_once(
            notebook,
            '    dtype="float32",\n)',
            '    restore_best_checkpoint=False,\n    dtype="float32",\n)',
        )
    elif relative == "cross_sample_alignment_mouse_kidney_clustering_nb.ipynb":
        _replace_once(notebook, "    random_state=1000,", "    random_state=WORKFLOW_SEED,")
        _replace_once(
            notebook,
            "    random_state=WORKFLOW_SEED,\n)",
            "    random_state=WORKFLOW_SEED,\n"
            '    leiden_flavor="leidenalg",\n'
            "    leiden_n_iterations=-1,\n)",
        )
    elif relative == "cross_sample_alignment_breast_cancer_clustering_nb.ipynb":
        # This notebook calls the dedicated runner; the seed argument is added
        # together with the setup cell by _add_seed_call. Reuse the same seed
        # for display-only subsampling.
        _replace_once(
            notebook,
            "rng = np.random.default_rng(1000)",
            "rng = np.random.default_rng(WORKFLOW_SEED)",
        )


def update_notebook(relative: str, seed: int) -> None:
    path = CANONICAL / relative
    notebook = nbformat.read(path, as_version=4)
    _add_page_note(notebook, seed)
    _add_seed_call(notebook, seed)
    _update_config(relative, notebook)
    notebook.metadata["spAlignDE_reproducibility"] = {
        "workflow_seed": seed,
        "seed_scope": "Python, NumPy, Torch and configured stochastic methods",
        "discrete_repeat_contract": "exact",
        "cuda_coordinate_contract": "workflow-specific numerical tolerance",
    }
    nbformat.write(notebook, path)
    mirror = MIRROR / relative
    mirror.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, mirror)


def main() -> None:
    missing = [relative for relative in SEEDS if not (CANONICAL / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing canonical notebooks: " + ", ".join(missing))
    for relative, seed in SEEDS.items():
        update_notebook(relative, seed)
        print(f"updated {relative}: seed={seed}")


if __name__ == "__main__":
    main()
