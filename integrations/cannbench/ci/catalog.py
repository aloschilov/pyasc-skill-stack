#!/usr/bin/env python3
"""Snapshot the complete official CANNBench catalog and task contracts.

Run on the CANN VM with BENCHSITE_API_TOKEN. No uploads or evaluation requests
are made. A changed catalog/version during capture invalidates the snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


class SiteClient:
    def __init__(self) -> None:
        self.base_url = os.environ.get("BENCHSITE_URL", "https://cannbench.com").rstrip("/")
        self.token = os.environ.get("BENCHSITE_API_TOKEN")
        if not self.token:
            raise ValueError("BENCHSITE_API_TOKEN must be supplied by the VM environment")

    def get(self, route: str) -> dict:
        request = Request(self.base_url + route, headers={"Authorization": "Bearer " + self.token})
        with urlopen(request, timeout=120) as response:
            return json.load(response)


def select_catalog(payload: dict, slug: str) -> dict:
    matches = [b for b in payload["benchmarks"] if b["slug"] == slug]
    if len(matches) != 1:
        raise ValueError("Expected exactly one benchmark catalog")
    benchmark = matches[0]
    operators = benchmark["multi_operators"]
    names = [o["operator"] for o in operators]
    paths = [o["rel_path"] for o in operators]
    if len(names) != len(set(names)) or len(paths) != len(set(paths)):
        raise ValueError("Duplicate operator or task path in official catalog")
    if len(operators) != benchmark["operator_count"]:
        raise ValueError("Catalog operator count does not match its task inventory")
    if sum(o["case_count"] for o in operators) != benchmark["total_case_count"]:
        raise ValueError("Catalog case count does not match its task inventory")
    return benchmark


def snapshot(client, slug: str, destination: Path) -> dict:
    before = select_catalog(client.get("/api/benchmarks"), slug)
    version = before["submission_version"]
    if version not in before["accepted_submission_versions"]:
        raise ValueError("Catalog submission version is not accepted")
    if destination.exists():
        raise ValueError("Snapshot destination must be new; historical contracts are immutable")
    destination.mkdir(parents=True)
    (destination / "catalog.raw.json").write_text(json.dumps(before, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "source": "cannbench-api", "benchmark_slug": slug,
        "benchmark_version": version, "fetched_at": datetime.now(timezone.utc).isoformat(),
        "required_operators": before["operator_count"],
        "required_cases": before["total_case_count"], "operators": [],
    }
    for operator in before["multi_operators"]:
        name = operator["operator"]
        function = operator["function_name"]
        rel_path = operator["rel_path"]
        if not re.fullmatch(r"[A-Za-z_]\w*", function) or not re.fullmatch(r"level\d+/[a-z0-9_]+", rel_path):
            raise ValueError("Invalid function name or task path")
        task = client.get("/api/benchmarks/" + quote(slug, safe="") + "/operators/" + quote(name, safe=""))
        if task["benchmark_slug"] != slug or task["operator"] != name or task["function_name"] != function:
            raise ValueError("Task identity does not match the catalog")
        cases = task["cases_yaml"]
        case_ids = [rel_path + "_" + str(c["case_id"]) for c in cases]
        if len(cases) != operator["case_count"] or len(case_ids) != len(set(case_ids)):
            raise ValueError("Task case inventory does not match the catalog")
        directory = destination / "tasks" / function
        directory.mkdir(parents=True)
        # JSON is valid YAML and preserves the exact API values without PyYAML.
        files = {
            "proto.yaml": json.dumps({"operator": task["proto"]}, ensure_ascii=False, indent=2) + "\n",
            "cases.yaml": json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n",
            "golden.py": task["golden_source"], "desc.md": task["desc_md"],
            "task.raw.json": json.dumps(task, ensure_ascii=False, indent=2) + "\n",
        }
        hashes = {}
        for filename, content in files.items():
            path = directory / filename
            path.write_text(content, encoding="utf-8")
            hashes[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["operators"].append({**operator, "case_ids": case_ids, "task_files": hashes})
        print(f"Captured {name}: {len(cases)} cases", flush=True)
    after = select_catalog(client.get("/api/benchmarks"), slug)
    if before != after:
        raise ValueError("Official catalog changed during capture; snapshot is not qualified")
    manifest["catalog_sha256"] = hashlib.sha256((destination / "catalog.raw.json").read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="official-tasks")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = snapshot(SiteClient(), args.benchmark, args.output)
    except Exception as exc:
        # No headers, tokens, or response bodies are printed on failure.
        print(f"Catalog capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Captured {manifest['required_operators']} operators / {manifest['required_cases']} cases")


if __name__ == "__main__":
    main()
