#!/usr/bin/env python3
"""Synchronize the authoritative local ``spalignde`` tree into docs copies.

The source path is supplied explicitly at synchronization time. Published
documentation imports only the vendored copies and never depends on the source
checkout at runtime.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import shutil


DOCS_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORY_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    "build",
    "dist",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
INTEGRATED_COMPATIBILITY_FILES = {
    Path("__init__.py"),
    Path("datasets/__init__.py"),
    Path("datasets/aging_brain/__init__.py"),
    Path("datasets/visium.py"),
}


def default_targets() -> tuple[Path, ...]:
    """Return only vendored package locations present in this checkout."""

    targets = [DOCS_ROOT / "src" / "spalignde"]
    standalone_parent = DOCS_ROOT / "Post_alignment_inference"
    if standalone_parent.is_dir():
        targets.append(standalone_parent / "spalignde")
    return tuple(targets)


def include_file(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(
        part in EXCLUDED_DIRECTORY_NAMES or part.endswith(".egg-info")
        for part in relative.parts
    ):
        return False
    return path.is_file() and path.suffix not in EXCLUDED_SUFFIXES


def source_files(source: Path) -> dict[Path, Path]:
    return {
        path.relative_to(source): path
        for path in sorted(source.rglob("*"))
        if include_file(path, source)
    }


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(
    source: Path,
    target: Path,
    *,
    check: bool,
    preserve_integrated_namespace: bool,
) -> list[str]:
    managed = source_files(source)
    if preserve_integrated_namespace:
        managed = {
            relative: path
            for relative, path in managed.items()
            if relative not in INTEGRATED_COMPATIBILITY_FILES
        }
    differences = []
    for relative, source_path in managed.items():
        target_path = target / relative
        if not target_path.is_file() or file_digest(source_path) != file_digest(target_path):
            differences.append(relative.as_posix())
            if not check:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)

    if not check:
        for cache_name in EXCLUDED_DIRECTORY_NAMES:
            for cache_path in target.rglob(cache_name):
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
        for suffix in EXCLUDED_SUFFIXES:
            for generated_path in target.rglob(f"*{suffix}"):
                generated_path.unlink()
    return differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Authoritative spalignde package directory",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=Path,
        help="Destination package directory; repeat for multiple targets",
    )
    parser.add_argument("--check", action="store_true", help="Report differences without writing")
    parser.add_argument(
        "--preserve-integrated-namespace",
        action="store_true",
        help=(
            "Keep the integrated repository's combined alignment/inference "
            "namespace and compatibility dataset adapters while synchronizing "
            "all inference implementation files and packaged data"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir() or not (source / "inference" / "__init__.py").is_file():
        raise SystemExit(f"not a spalignde package tree: {source}")
    targets = tuple(
        path.expanduser().resolve() for path in (args.target or default_targets())
    )
    any_differences = False
    for target in targets:
        differences = synchronize(
            source,
            target,
            check=args.check,
            preserve_integrated_namespace=args.preserve_integrated_namespace,
        )
        any_differences |= bool(differences)
        action = "DIFF" if args.check else "SYNC"
        print(f"{action} {target}: {len(differences)} managed files changed")
        for relative in differences:
            print(f"  {relative}")
    if args.check and any_differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
