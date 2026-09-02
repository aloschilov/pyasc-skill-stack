#!/usr/bin/env python3
"""Vendor one authoritative CANNBench task through the official site API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SECRETS = ROOT / ".secrets" / "cannbench.env"


def load_env() -> None:
    if SECRETS.exists():
        for raw in SECRETS.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.removeprefix("export ").split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    for site in sorted(
        (ROOT / ".tools/benchsite-mcp/lib").glob("python*/site-packages")
    ):
        sys.path.insert(0, str(site))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operator", help="catalog operator name, e.g. RmsNorm")
    parser.add_argument("--benchmark", default="official-tasks")
    args = parser.parse_args()

    load_env()
    from benchsite_mcp.client import BenchSiteClient  # type: ignore

    task = BenchSiteClient().get_benchmark_task(args.benchmark, args.operator)
    function_name = task["function_name"]
    destination = Path(__file__).resolve().parent / "tasks" / function_name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "desc.md").write_text(task["desc_md"], encoding="utf-8")
    (destination / "golden.py").write_text(
        task["golden_source"], encoding="utf-8"
    )
    (destination / "proto.yaml").write_text(
        yaml.safe_dump(
            {"operator": task["proto"]}, sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    (destination / "cases.yaml").write_text(
        yaml.safe_dump(
            {"cases": task["cases_yaml"]}, sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    print(f"vendored {args.benchmark}/{args.operator} -> {destination}")


if __name__ == "__main__":
    main()
