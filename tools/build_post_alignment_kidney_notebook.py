"""Publish the executed mouse-kidney post-alignment inference notebook.

The real-data notebook under ``Post_alignment_inference/tutorials`` is the
canonical analysis record.  This tool preserves its saved outputs and the
lowercase public ``spalignde`` imports while writing the two notebook mirrors
used by the documentation website.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
WEBSITE_SOURCE = (
    ROOT / "docs/source"
    if (ROOT / "docs/source").is_dir()
    else ROOT.parents[2] / "readdoc/docs/source"
)
TARGETS = (
    ROOT / "source_notebooks/post_alignment_inference_nb.ipynb",
    WEBSITE_SOURCE / "source_notebooks/post_alignment_inference_nb.ipynb",
)
KIDNEY_RECORD_HANDOFF = (
    "The alignment input and region annotations come from STcompare record "
    "`20647680`; this inference stage reloads the NL3/IL3 10x matrices and "
    "tissue-position tables from source record `17676992` and joins them to "
    "the aligned coordinates by terminal barcode. The two records therefore "
    "serve different handoff roles."
)


def _source_notebook() -> Path:
    configured = os.environ.get("SPALIGNDE_POST_ALIGNMENT_NOTEBOOK")
    candidates = [
        Path(configured).expanduser() if configured else None,
        ROOT
        / "Post_alignment_inference/tutorials/mouse_kidney_from_aligned_coordinates.ipynb",
        ROOT / "source_notebooks/post_alignment_inference_nb.ipynb",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the executed mouse-kidney inference notebook. Set "
        "SPALIGNDE_POST_ALIGNMENT_NOTEBOOK to its path."
    )


def _scrub_saved_output(value):
    if isinstance(value, str):
        return value.replace(
            "/home/ywang/projects/multi_model/PyPI/tutorial_data/kidney/raw",
            "spalignde_kidney_tutorial/raw",
        )
    if isinstance(value, list):
        return [_scrub_saved_output(item) for item in value]
    if isinstance(value, dict):
        return {key: _scrub_saved_output(item) for key, item in value.items()}
    return value


def build_notebook(source: Path | None = None):
    """Return the executed notebook in its portable public form."""

    notebook = nbformat.read(source or _source_notebook(), as_version=4)
    notebook = copy.deepcopy(notebook)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace(
                "### Optional: use coordinates from your own alignment",
                "## Optional: use coordinates from your own alignment",
            )
            if (
                cell.source.startswith("## Alignment-to-inference handoff")
                and KIDNEY_RECORD_HANDOFF not in cell.source
            ):
                marker = "\n\nThe H5AD route extracts"
                cell.source = cell.source.replace(
                    marker,
                    f"\n\n{KIDNEY_RECORD_HANDOFF}{marker}",
                    1,
                )
        if "outputs" in cell:
            cell.outputs = nbformat.from_dict(_scrub_saved_output(cell.outputs))
    return notebook


def main() -> None:
    source = _source_notebook()
    notebook = build_notebook(source)
    for target in TARGETS:
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, target)
        print(target)
    print("source:", source)


if __name__ == "__main__":
    main()
