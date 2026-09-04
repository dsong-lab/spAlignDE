#!/usr/bin/env python3
"""Check public notebook references and shared release settings."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_COMMAND = re.compile(r"jupyter\s+lab\s+([^\s`]+\.ipynb)")
SOURCE_NOTEBOOK = re.compile(r"source_notebooks/[A-Za-z0-9_./-]+\.ipynb")
BANKSY_PIN = re.compile(r"Banksy_py\.git@([0-9a-f]{40})", re.IGNORECASE)
PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
PACKAGE_VERSION = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)
CITATION_VERSION = re.compile(r"^version:\s*([^\s]+)", re.MULTILINE)


def main() -> int:
    failures: list[str] = []
    text_paths = [ROOT / "README.md"]
    text_paths.extend(sorted((ROOT / "docs/source").rglob("*.rst")))
    text_paths.extend(sorted((ROOT / "source_notebooks").rglob("*.ipynb")))

    checked_references: set[str] = set()
    for text_path in text_paths:
        text = text_path.read_text(encoding="utf-8")
        references = set(NOTEBOOK_COMMAND.findall(text))
        references.update(SOURCE_NOTEBOOK.findall(text))
        for reference in sorted(references):
            if reference.startswith("/path/to/"):
                continue
            checked_references.add(reference)
            if not (ROOT / reference).is_file():
                relative_source = text_path.relative_to(ROOT)
                failures.append(
                    f"{relative_source}: notebook reference does not exist: {reference}"
                )

    pin_paths = (
        ROOT / "pyproject.toml",
        ROOT / "environment.yml",
    )
    pins: dict[Path, str] = {}
    for path in pin_paths:
        match = BANKSY_PIN.search(path.read_text(encoding="utf-8"))
        if match is None:
            failures.append(f"{path.relative_to(ROOT)}: missing full BANKSY commit pin")
        else:
            pins[path] = match.group(1).lower()
    if pins and len(set(pins.values())) != 1:
        rendered = ", ".join(
            f"{path.relative_to(ROOT)}={pin}" for path, pin in pins.items()
        )
        failures.append(f"BANKSY commit pins disagree: {rendered}")

    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")

    version_sources = {
        "pyproject.toml": PYPROJECT_VERSION.search(pyproject_text),
        "src/spalignde/__init__.py": PACKAGE_VERSION.search(
            (ROOT / "src/spalignde/__init__.py").read_text(encoding="utf-8")
        ),
        "CITATION.cff": CITATION_VERSION.search(citation_text),
    }
    versions = {
        source: match.group(1) for source, match in version_sources.items() if match
    }
    if len(versions) != len(version_sources):
        missing = sorted(set(version_sources).difference(versions))
        failures.append(f"missing release version in: {', '.join(missing)}")
    elif len(set(versions.values())) != 1:
        rendered = ", ".join(f"{source}={version}" for source, version in versions.items())
        failures.append(f"release versions disagree: {rendered}")

    print(f"Public notebook references: {len(checked_references)}")
    print(f"BANKSY pin files: {len(pins)}")
    print(f"Release version files: {len(versions)}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Failures: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
