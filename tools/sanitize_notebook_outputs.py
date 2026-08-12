"""Remove known non-result warnings from executed public notebooks."""

from __future__ import annotations

import sys
import re
import hashlib
import json
from pathlib import Path

import nbformat


SUPPRESSED = (
    "pkg_resources is deprecated as an API",
    "Some cells have zero counts",
    "No data for colormapping provided via 'c'",
    "return Variable._execution_engine.run_backward",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH_REPLACEMENTS = {
    str(PROJECT_ROOT): "/path/to/spAlignDE",
}
DEVELOPER_HOME = re.compile(r"/home/[^/\s<]+/")


def public_paths(text: str) -> str:
    """Replace validated-machine paths without changing numeric results."""
    for source, destination in PUBLIC_PATH_REPLACEMENTS.items():
        text = text.replace(source, destination)
    return DEVELOPER_HOME.sub("/path/to/", text)


def sanitize_notebook(notebook) -> bool:
    """Sanitize an in-memory notebook and report whether it changed."""
    changed = False

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        outputs = []
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
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
                continue

            data = output.get("data", {})
            for mime in ("text/plain", "text/html", "text/markdown"):
                value = data.get(mime)
                if isinstance(value, str):
                    cleaned = public_paths(value)
                    data[mime] = cleaned
                    changed |= cleaned != value
            outputs.append(output)

        cell.outputs = outputs
    return changed


def sanitize(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    changed = sanitize_notebook(notebook)
    execution = notebook.metadata.get("spAlignDE_execution")
    if isinstance(execution, dict):
        payload = [
            cell.get("outputs", [])
            for cell in notebook.cells
            if cell.cell_type == "code"
        ]
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        output_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if execution.get("saved_output_sha256") != output_hash:
            execution["saved_output_sha256"] = output_hash
            changed = True
    if changed:
        nbformat.write(notebook, path)


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        sanitize(Path(filename))
