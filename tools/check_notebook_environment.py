#!/usr/bin/env python3
"""Report the environment used to execute spAlignDE source notebooks."""

from __future__ import annotations

import argparse
from importlib import import_module, metadata
import json
import os
import platform
import sys


PACKAGES = {
    "spAlignDE": ("spAlignDE",),
    "anndata": ("anndata",),
    "einops": ("einops",),
    "h5py": ("h5py",),
    "harmonypy": ("harmonypy",),
    "imagecodecs": ("imagecodecs",),
    "igraph": ("igraph",),
    "ipykernel": ("ipykernel",),
    "ipywidgets": ("ipywidgets",),
    "jupyterlab": ("jupyterlab",),
    "leidenalg": ("leidenalg",),
    "louvain": ("louvain",),
    "matplotlib": ("matplotlib",),
    "myst-nb": ("myst-nb",),
    "nbconvert": ("nbconvert",),
    "nbformat": ("nbformat",),
    "numpy": ("numpy",),
    # Either wheel exposes the same cv2 import. The public environment uses
    # headless; a developer workstation may already contain the desktop wheel.
    "opencv": ("opencv-python-headless", "opencv-python"),
    "pandas": ("pandas",),
    "pillow": ("pillow",),
    "plotly": ("plotly",),
    "pybanksy": ("pybanksy",),
    "pynrrd": ("pynrrd",),
    "pyyaml": ("pyyaml",),
    "pytest": ("pytest",),
    "scanpy": ("scanpy",),
    "scikit-image": ("scikit-image",),
    "scikit-learn": ("scikit-learn",),
    "scipy": ("scipy",),
    "seaborn": ("seaborn",),
    "shapely": ("shapely",),
    "sphinx": ("sphinx",),
    "sphinx-rtd-theme": ("sphinx-rtd-theme",),
    "streamlit": ("streamlit",),
    "tifffile": ("tifffile",),
    "torch": ("torch",),
    "torchvision": ("torchvision",),
    "umap-learn": ("umap-learn",),
    "webdataset": ("webdataset",),
}

EXPECTED_VERSIONS = {
    "anndata": "0.10.9",
    "einops": "0.8.1",
    "h5py": "3.15.1",
    "harmonypy": "0.2.0",
    "imagecodecs": "2025.3.30",
    "igraph": "0.11.9",
    "ipykernel": "7.1.0",
    "ipywidgets": "8.1.8",
    "jupyterlab": "4.5.0",
    "leidenalg": "0.10.2",
    "louvain": "0.8.2",
    "matplotlib": "3.10.7",
    "myst-nb": "1.1.2",
    "nbconvert": "7.16.6",
    "nbformat": "5.10.4",
    "numpy": "1.26.4",
    "opencv": "4.10.0.84",
    "pandas": "2.3.3",
    "pillow": "12.1.0",
    "plotly": "5.14.1",
    "pybanksy": "1.3.4",
    "pynrrd": "1.0.0",
    "pyyaml": "6.0",
    "pytest": "9.1.1",
    "scanpy": "1.10.3",
    "scikit-image": "0.24.0",
    "scikit-learn": "1.7.2",
    "scipy": "1.10.1",
    "seaborn": "0.13.2",
    "shapely": "2.1.2",
    "sphinx": "7.4.7",
    "sphinx-rtd-theme": "2.0.0",
    "streamlit": "1.60.0",
    "tifffile": "2025.5.10",
    "torch": "2.10.0+cu128",
    "torchvision": "0.25.0+cu128",
    "umap-learn": "0.5.11",
    "webdataset": "1.0.2",
}
EXPECTED_PYTHON = "3.10.14"
EXPECTED_CUDA_RUNTIME = "12.8"

IMPORT_CHECKS = {
    "spAlignDE": "spAlignDE",
    "anndata": "anndata",
    "einops": "einops",
    "h5py": "h5py",
    "harmonypy": "harmonypy",
    "imagecodecs": "imagecodecs",
    "igraph": "igraph",
    "ipykernel": "ipykernel",
    "ipywidgets": "ipywidgets",
    "jupyterlab": "jupyterlab",
    "leidenalg": "leidenalg",
    "louvain": "louvain",
    "matplotlib": "matplotlib",
    "myst-nb": "myst_nb",
    "nbconvert": "nbconvert",
    "nbformat": "nbformat",
    "numpy": "numpy",
    "opencv": "cv2",
    "pandas": "pandas",
    "pillow": "PIL",
    "plotly": "plotly",
    "pybanksy": "banksy",
    "pynrrd": "nrrd",
    "pyyaml": "yaml",
    "pytest": "pytest",
    "scanpy": "scanpy",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "shapely": "shapely",
    "sphinx": "sphinx",
    "sphinx-rtd-theme": "sphinx_rtd_theme",
    "streamlit": "streamlit",
    "tifffile": "tifffile",
    "torch": "torch",
    "torchvision": "torchvision",
    "umap-learn": "umap",
    "webdataset": "webdataset",
}

IMPORT_FALLBACKS = {
    "numpy": "numpy",
}


def distribution_versions() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for label, candidates in PACKAGES.items():
        for distribution in candidates:
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                continue
            if not version and label in IMPORT_FALLBACKS:
                module = import_module(IMPORT_FALLBACKS[label])
                version = getattr(module, "__version__", "unknown")
            versions[label] = f"{version} ({distribution})"
            break
        else:
            missing.append(label)
    return versions, missing


def import_checks() -> tuple[dict[str, str], list[str]]:
    """Import notebook-critical modules, including binary extension stacks."""
    versions: dict[str, str] = {}
    errors: list[str] = []
    for label, module_name in IMPORT_CHECKS.items():
        try:
            module = import_module(module_name)
        except Exception as error:
            errors.append(f"{label} ({module_name}): {type(error).__name__}: {error}")
            continue
        version = getattr(module, "__version__", None)
        if version is not None:
            versions[label] = str(version)
    try:
        import torch
        from torchvision.ops import nms

        boxes = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [0.5, 0.5, 2.5, 2.5]],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        kept = nms(boxes, scores, 0.3)
        if kept.numel() != 1:
            errors.append(
                "torchvision binary smoke test returned an unexpected NMS result."
            )
    except Exception as error:
        errors.append(
            "torchvision binary smoke test: "
            f"{type(error).__name__}: {error}"
        )
    return versions, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless a visible GPU completes the CUDA kernel smoke test.",
    )
    args = parser.parse_args()
    import torch

    versions, missing = distribution_versions()
    imported_versions, import_errors = import_checks()
    warnings: list[str] = []
    if os.environ.get("PYTHONPATH"):
        warnings.append(
            "PYTHONPATH is set and can inject packages from outside the Conda "
            "environment. Run `unset PYTHONPATH` before creation, validation "
            "and notebook execution."
        )
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        warnings.append(
            "PYTHONNOUSERSITE is not 1; user-site packages may shadow the "
            "validated environment. Set `export PYTHONNOUSERSITE=1`."
        )
    python_version = sys.version.split()[0]
    if python_version != EXPECTED_PYTHON:
        warnings.append(
            f"Python version mismatch: expected={EXPECTED_PYTHON}, "
            f"actual={python_version}."
        )
    for package, expected in EXPECTED_VERSIONS.items():
        actual = versions.get(package, "").split(" (", maxsplit=1)[0]
        if actual and actual != expected and not actual.startswith(expected + "+"):
            warnings.append(
                f"Version mismatch for {package}: expected={expected}, actual={actual}."
            )
    installed_torch = versions.get("torch", "").split(" (", maxsplit=1)[0]
    if installed_torch and installed_torch != torch.__version__:
        warnings.append(
            "The imported torch version differs from distribution metadata: "
            f"import={torch.__version__}, metadata={installed_torch}. "
            "Create a fresh environment before executing release notebooks."
        )
    if torch.version.cuda and torch.version.cuda != EXPECTED_CUDA_RUNTIME:
        warnings.append(
            "CUDA runtime mismatch: "
            f"expected={EXPECTED_CUDA_RUNTIME}, actual={torch.version.cuda}."
        )
    cuda_smoke: dict[str, object] = {"ran": False}
    if torch.cuda.is_available():
        try:
            import torch.nn.functional as functional

            device = torch.device("cuda:0")
            capability = torch.cuda.get_device_capability(device)
            expected_arch = f"sm_{capability[0]}{capability[1]}"
            compiled_arches = torch.cuda.get_arch_list()
            if expected_arch not in compiled_arches:
                warnings.append(
                    "The imported Torch build does not list the visible GPU "
                    f"architecture: expected={expected_arch}, compiled={compiled_arches}."
                )
            image = torch.arange(
                16,
                dtype=torch.float64,
                device=device,
            ).reshape(1, 1, 4, 4)
            grid = torch.zeros((1, 2, 2, 2), dtype=torch.float64, device=device)
            sampled = functional.grid_sample(
                image,
                grid,
                mode="bilinear",
                align_corners=True,
            )
            value = float((sampled.square().sum()).cpu())
            torch.cuda.synchronize(device)
            cuda_smoke = {
                "ran": True,
                "device": torch.cuda.get_device_name(device),
                "capability": list(capability),
                "expected_arch": expected_arch,
                "compiled_arches": compiled_arches,
                "float64_grid_sample_sum": value,
            }
        except Exception as error:
            import_errors.append(
                "CUDA kernel smoke test: "
                f"{type(error).__name__}: {error}"
            )
    elif args.require_cuda:
        import_errors.append(
            "CUDA is required for this validation run, but Torch reports no "
            "visible CUDA device."
        )
    report = {
        "python": python_version,
        "platform": platform.platform(),
        "isolation": {
            "pythonpath": os.environ.get("PYTHONPATH"),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        },
        "packages": versions,
        "imported_versions": imported_versions,
        "import_errors": import_errors,
        "torch": {
            "imported_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "cuda_kernel_smoke": cuda_smoke,
        "warnings": warnings,
        "missing_distributions": missing,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing or import_errors or warnings:
        problems = []
        if missing:
            problems.append("missing distributions: " + ", ".join(missing))
        if warnings:
            problems.append(f"{len(warnings)} version or runtime warning(s)")
        if import_errors:
            problems.append(f"{len(import_errors)} import or CUDA smoke failure(s)")
        raise SystemExit("Environment validation failed; " + "; ".join(problems))


if __name__ == "__main__":
    main()
