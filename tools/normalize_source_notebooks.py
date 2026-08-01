"""Normalize public source notebooks before execution and Sphinx syncing."""

from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "source_notebooks"


def canonicalize_text(value: str) -> str:
    return value.replace("spalignde", "spAlignDE")


for notebook_path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        cell.source = canonicalize_text(cell.source)

    nbformat.write(notebook, notebook_path)
    print(notebook_path.relative_to(PROJECT_ROOT))
