#!/usr/bin/env python3
"""Assemble locally-qualified generated operators without touching submission/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import driver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--qualified", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    package = args.output_dir / "cann_bench"
    provenance = args.output_dir / "provenance"
    package.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)

    sources: dict[str, dict[str, str]] = {}
    base_package = args.base / "cann_bench"
    base_provenance = args.base / "provenance"
    for op in driver.ALL_OPS:
        candidate = base_package / f"{op}.py"
        record = base_provenance / f"{op}.json"
        if candidate.exists():
            shutil.copy2(candidate, package / candidate.name)
            shutil.copy2(record, provenance / record.name)
            sources[op] = {
                "candidate": str(candidate.resolve()),
                "provenance": str(record.resolve()),
            }

    for qualified in args.qualified:
        records = list(qualified.glob("*.py"))
        if len(records) != 1:
            raise RuntimeError(f"expected one operator module in {qualified}")
        candidate = records[0]
        op = candidate.stem
        if op not in driver.ALL_OPS:
            raise RuntimeError(f"unknown operator {op}")
        record = qualified / "provenance.json"
        shutil.copy2(candidate, package / candidate.name)
        shutil.copy2(record, provenance / f"{op}.json")
        sources[op] = {
            "candidate": str(candidate.resolve()),
            "provenance": str(record.resolve()),
        }

    missing = sorted(set(driver.ALL_OPS) - set(sources))
    if missing:
        raise RuntimeError(f"bundle is missing operators: {missing}")
    shutil.copy2(driver.SUBMISSION_PKG / "_pyasc_runtime.py", package)
    (package / "__init__.py").write_text(
        driver.render_init(driver.ALL_OPS), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "status": "assembled-awaiting-full-matrix",
        "evidence": "verified-local-compile",
        "operators": driver.ALL_OPS,
        "cases_per_operator": 20,
        "pyasc_commit": "ac1222a48c8914d3f81297c7570d1a84f0f26778",
        "sources": sources,
        "limitations": [
            "no numerical execution",
            "no NPU performance measurement",
            "not promoted to the canonical submission package",
        ],
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
