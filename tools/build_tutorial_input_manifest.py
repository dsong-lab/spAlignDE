#!/usr/bin/env python3
"""Build a public checksum manifest for fixed-seed tutorial inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_entry(value: str) -> tuple[str, str, Path]:
    try:
        notebook, role, raw_path = value.split("=", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "entries must use NOTEBOOK=ROLE=PATH"
        ) from error
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input file not found: {path}")
    return notebook, role, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="append", required=True, type=parse_entry)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = []
    for notebook, role, path in args.entry:
        records.append(
            {
                "notebook": notebook,
                "role": role,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        print(f"HASH {path.name}", flush=True)

    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "aging_brain_alignment_included": False,
        "aging_brain_post_alignment_inference_included": True,
        "inputs": sorted(records, key=lambda item: (item["notebook"], item["role"], item["filename"])),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
