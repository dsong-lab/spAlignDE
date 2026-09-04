#!/usr/bin/env python3
"""Stage generated documentation inputs from their canonical repository files."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCE = ROOT / "source_notebooks"
NOTEBOOK_TARGET = ROOT / "docs" / "source" / "source_notebooks"
DOWNLOADS = {
    ROOT / "environment.yml": (
        ROOT / "docs" / "source" / "_static" / "environment" / "environment.yml"
    ),
    ROOT / "ENVIRONMENT.md": (
        ROOT / "docs" / "source" / "_static" / "environment" / "ENVIRONMENT.md"
    ),
    ROOT / "tools" / "check_notebook_environment.py": (
        ROOT
        / "docs"
        / "source"
        / "_static"
        / "environment"
        / "check_notebook_environment.py"
    ),
}


def main() -> None:
    notebooks = sorted(NOTEBOOK_SOURCE.rglob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No notebooks found under {NOTEBOOK_SOURCE}")

    for stale in NOTEBOOK_TARGET.rglob("*.ipynb"):
        stale.unlink()
    for source in notebooks:
        target = NOTEBOOK_TARGET / source.relative_to(NOTEBOOK_SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for source, target in DOWNLOADS.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    print(
        f"Staged {len(notebooks)} notebooks and {len(DOWNLOADS)} downloads for Sphinx."
    )


if __name__ == "__main__":
    main()
