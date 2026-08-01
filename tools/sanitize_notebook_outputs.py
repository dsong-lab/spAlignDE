"""Remove known non-result warnings from executed public notebooks."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat


SUPPRESSED = (
    "pkg_resources is deprecated as an API",
    "Some cells have zero counts",
    "No data for colormapping provided via 'c'",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH_REPLACEMENTS = {
    str(PROJECT_ROOT): "/path/to/spAlignDE",
}


def public_paths(text: str) -> str:
    """Replace validated-machine paths without changing numeric results."""
    for source, destination in PUBLIC_PATH_REPLACEMENTS.items():
        text = text.replace(source, destination)
    return text


def sanitize(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    changed = False

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        outputs = []
        for output in cell.get("outputs", []):
            if output.output_type != "stream":
                outputs.append(output)
                continue

            text = output.get("text", "")
            lines = []
            for line in text.splitlines(keepends=True):
                if any(message in line for message in SUPPRESSED):
                    continue
                if "/miniconda3/" in line or "/anaconda3/" in line:
                    continue
                lines.append(public_paths(line))
            cleaned = "".join(lines)
            if cleaned:
                output["text"] = cleaned
                outputs.append(output)
            changed |= cleaned != text

        cell.outputs = outputs

    if changed:
        nbformat.write(notebook, path)


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        sanitize(Path(filename))
