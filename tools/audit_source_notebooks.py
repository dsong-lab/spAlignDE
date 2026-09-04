#!/usr/bin/env python3
"""Check public source notebooks for portability and saved execution state."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager


LEGACY_PUBLIC_PATTERNS = (
    re.compile(r"uns\s*\[\s*['\"]spalignde['\"]\s*\]"),
)
DEVELOPER_PATH_PATTERN = re.compile(r"(?:/home/[^/]+/|[A-Za-z]:\\Users\\[^\\]+\\)")
WORKFLOW_SEED_PATTERN = re.compile(r"(?m)^\s*WORKFLOW_SEED\s*=\s*(\d+)\s*$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook_dir", type=Path)
    args = parser.parse_args()

    notebook_paths = sorted(args.notebook_dir.rglob("*.ipynb"))
    transformer = TransformerManager()
    failures: list[str] = []
    code_cells = 0
    output_errors = 0

    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        relative = path.relative_to(args.notebook_dir)
        raw_text = path.read_text(encoding="utf-8")
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        if DEVELOPER_PATH_PATTERN.search(raw_text):
            failures.append(f"{relative}: contains a developer-only absolute path")
        for pattern in LEGACY_PUBLIC_PATTERNS:
            if pattern.search(raw_text):
                failures.append(f"{relative}: contains legacy public name/key {pattern.pattern!r}")

        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            code_cells += 1
            try:
                ast.parse(transformer.transform_cell(cell.source))
            except SyntaxError as exc:
                failures.append(f"{relative}: code cell {index} is invalid Python: {exc}")
            if cell.execution_count is None:
                failures.append(f"{relative}: code cell {index} has no saved execution count")
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    output_errors += 1
                    failures.append(
                        f"{relative}: code cell {index} saved "
                        f"{output.get('ename', 'an error')} output"
                    )

        reproducibility = notebook.metadata.get("spAlignDE_reproducibility", {})
        if reproducibility:
            seed = reproducibility.get("workflow_seed")
            seed_match = WORKFLOW_SEED_PATTERN.search(code)
            if not isinstance(seed, int):
                failures.append(
                    f"{relative}: reproducibility metadata has no integer workflow_seed"
                )
            elif seed_match is None or int(seed_match.group(1)) != seed:
                failures.append(
                    f"{relative}: WORKFLOW_SEED does not match notebook metadata"
                )

            execution = notebook.metadata.get("spAlignDE_execution", {})
            if execution.get("fully_executed") is not True:
                failures.append(
                    f"{relative}: execution metadata does not mark the notebook complete"
                )
            if execution.get("workflow_seed") != seed:
                failures.append(
                    f"{relative}: execution seed does not match reproducibility metadata"
                )

    print(f"Notebooks: {len(notebook_paths)}")
    print(f"Non-empty code cells: {code_cells}")
    print(f"Saved error outputs: {output_errors}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Failures: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
