#!/usr/bin/env python3
"""Fail when a public spAlignDE symbol is absent from the API guide."""

from __future__ import annotations

import re
from pathlib import Path

import spAlignDE


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    api_text = (ROOT / "docs/source/api.rst").read_text(encoding="utf-8")
    missing = [
        name
        for name in spAlignDE.__all__
        if not re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
            api_text,
        )
    ]
    if missing:
        raise SystemExit(
            "Public API symbols missing from docs/source/api.rst: "
            + ", ".join(sorted(missing))
        )
    print(
        f"PASS: all {len(spAlignDE.__all__)} public API symbols are named in "
        "the API guide."
    )


if __name__ == "__main__":
    main()
