#!/usr/bin/env python3
"""Audit public notebook references, mirrors, and reproducibility pins."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_COMMAND = re.compile(r"jupyter\s+lab\s+([^\s`]+\.ipynb)")
CANONICAL_NOTEBOOK = re.compile(r"source_notebooks/[A-Za-z0-9_./-]+\.ipynb")
BANKSY_PIN = re.compile(r"Banksy_py\.git@([0-9a-f]{40})", re.IGNORECASE)
PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
PACKAGE_VERSION = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)
CITATION_VERSION = re.compile(r"^version:\s*([^\s]+)", re.MULTILINE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    failures: list[str] = []
    text_paths = [ROOT / "README.md"]
    text_paths.extend(sorted((ROOT / "docs/source").rglob("*.rst")))
    text_paths.extend(sorted((ROOT / "source_notebooks").rglob("*.ipynb")))

    checked_references: set[str] = set()
    for text_path in text_paths:
        text = text_path.read_text(encoding="utf-8")
        references = set(NOTEBOOK_COMMAND.findall(text))
        references.update(CANONICAL_NOTEBOOK.findall(text))
        for reference in sorted(references):
            if reference.startswith("/path/to/"):
                continue
            checked_references.add(reference)
            if not (ROOT / reference).is_file():
                relative_source = text_path.relative_to(ROOT)
                failures.append(
                    f"{relative_source}: notebook reference does not exist: {reference}"
                )

    canonical_root = ROOT / "source_notebooks"
    mirror_root = ROOT / "docs/source/source_notebooks"
    notebook_paths = sorted(canonical_root.rglob("*.ipynb"))
    for canonical in notebook_paths:
        relative = canonical.relative_to(canonical_root)
        mirror = mirror_root / relative
        if not mirror.is_file():
            failures.append(f"missing documentation notebook mirror: {relative}")
        elif _sha256(canonical) != _sha256(mirror):
            failures.append(f"documentation notebook mirror is stale: {relative}")

    pin_paths = (
        ROOT / "pyproject.toml",
        ROOT / "environment.yml",
        ROOT / "docs/source/_static/environment/environment.yml",
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

    license_path = ROOT / "LICENSE"
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if not license_path.is_file() or not license_path.read_text(
        encoding="utf-8"
    ).startswith("MIT License\n"):
        failures.append("LICENSE: missing the standard MIT License heading")
    elif "Copyright (c) 2026 Dongyuan Song Lab\n" not in license_path.read_text(
        encoding="utf-8"
    ):
        failures.append("LICENSE: copyright holder is not Dongyuan Song Lab")
    if 'license = {file = "LICENSE"}' not in pyproject_text:
        failures.append("pyproject.toml: project license does not reference LICENSE")
    if "License :: OSI Approved :: MIT License" not in pyproject_text:
        failures.append("pyproject.toml: missing MIT license classifier")
    if not re.search(r"^license:\s*MIT\s*$", citation_text, re.MULTILINE):
        failures.append("CITATION.cff: missing license: MIT")

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
    print(f"Canonical notebook mirrors: {len(notebook_paths)}")
    print(f"BANKSY pin files: {len(pins)}")
    print(f"Release version files: {len(versions)}")
    print("License metadata: MIT")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Failures: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
