#!/usr/bin/env python3
"""Check local links, fragments, and media in a built Sphinx HTML tree."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


SKIPPED_SCHEMES = {"data", "ftp", "http", "https", "javascript", "mailto", "tel"}
GENERATED_PAGES = {Path("genindex.html"), Path("py-modindex.html"), Path("search.html")}


class ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.references: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for key in ("id", "name"):
            if values.get(key):
                self.ids.add(values[key] or "")
        for key in ("href", "src"):
            if values.get(key):
                self.references.append((key, values[key] or ""))


def parse_html(path: Path) -> ReferenceParser:
    parser = ReferenceParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser


def resolve_target(build_root: Path, source: Path, raw_url: str) -> tuple[Path, str] | None:
    if raw_url.startswith("//"):
        return None
    parsed = urlsplit(raw_url)
    if parsed.scheme.lower() in SKIPPED_SCHEMES:
        return None

    decoded_path = unquote(parsed.path)
    if decoded_path.startswith("/"):
        target = build_root / decoded_path.lstrip("/")
    elif decoded_path:
        target = source.parent / decoded_path
    else:
        target = source

    if target.is_dir():
        target = target / "index.html"
    return target.resolve(), unquote(parsed.fragment)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, help="Sphinx HTML output directory")
    args = parser.parse_args()

    build_root = args.build_dir.resolve()
    html_files = sorted(build_root.rglob("*.html"))
    parsed_pages = {path.resolve(): parse_html(path) for path in html_files}
    failures: list[str] = []
    checked_references = 0
    checked_fragments = 0

    source_root = Path(__file__).resolve().parents[1] / "docs" / "source"
    expected_pages = {
        source.relative_to(source_root).with_suffix(".html")
        for pattern in ("*.rst", "*.ipynb")
        for source in source_root.rglob(pattern)
    }
    for html_path in html_files:
        relative = html_path.relative_to(build_root)
        if relative not in expected_pages and relative not in GENERATED_PAGES:
            failures.append(
                f"stale or unexpected HTML page without a documentation source: {relative}"
            )

    for source, document in parsed_pages.items():
        for attribute, raw_url in document.references:
            resolved = resolve_target(build_root, source, raw_url)
            if resolved is None:
                continue
            target, fragment = resolved
            checked_references += 1
            if not target.exists():
                failures.append(
                    f"{source.relative_to(build_root)}: missing {attribute} target {raw_url!r}"
                )
                continue
            if fragment and target.suffix.lower() in {".htm", ".html"}:
                checked_fragments += 1
                target_document = parsed_pages.get(target)
                if target_document is None:
                    target_document = parse_html(target)
                    parsed_pages[target] = target_document
                if fragment not in target_document.ids:
                    failures.append(
                        f"{source.relative_to(build_root)}: missing fragment "
                        f"{fragment!r} in {raw_url!r}"
                    )

    print(f"HTML pages: {len(html_files)}")
    print(f"Expected source pages: {len(expected_pages)}")
    print(f"Local references checked: {checked_references}")
    print(f"HTML fragments checked: {checked_fragments}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Failures: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
