#!/usr/bin/env python3
"""Create a content-addressed manifest for a frozen intraday research dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict:
    frame = pd.read_parquet(path)
    index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": list(frame.columns),
        "start": str(index.min()) if len(index) else None,
        "end": str(index.max()) if len(index) else None,
        "invalid_timestamps": int(index.isna().sum()),
        "duplicate_timestamps": int(index.duplicated().sum()),
        "null_cells": int(frame.isna().sum().sum()),
    }


def build_manifest(data_dir: Path, result_paths: list[Path] | None = None) -> dict:
    files = {
        path.name: describe(path)
        for path in sorted(data_dir.glob("*_15min.parquet"))
    }
    if not files:
        raise FileNotFoundError(f"no *_15min.parquet files in {data_dir}")
    dataset_id = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    results = {
        str(path.resolve()): {
            "sha256": sha256(path.resolve()),
            "bytes": path.resolve().stat().st_size,
        }
        for path in sorted(result_paths or [])
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "dataset_id": dataset_id,
        "files": files,
        "baseline_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="research_data/intraday_15m")
    parser.add_argument("--output", default="research_manifests/intraday_15m_frozen.json")
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Existing baseline result file to fingerprint; may be repeated.",
    )
    args = parser.parse_args()
    manifest = build_manifest(
        Path(args.data_dir).resolve(), [Path(value) for value in args.result]
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[FREEZE] dataset_id={manifest['dataset_id']}")
    print(
        f"[FREEZE] files={len(manifest['files'])} "
        f"baseline_results={len(manifest['baseline_results'])} manifest={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
