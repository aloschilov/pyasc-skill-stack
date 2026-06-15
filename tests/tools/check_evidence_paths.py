#!/usr/bin/env python3
"""Fail CI when committed perf evidence contains machine-specific absolute paths."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_HOME_RE = re.compile(r"/(?:home|Users)/[^/]+/")
_SKIP_KEYS = frozenset({"container_view"})

SCAN_GLOBS = (
    "evidence/perf/**/*.json",
    "evidence/perf-vs-ascendc/*.json",
    "evidence/vf-fusion/*.json",
)

EXCLUDE_PARTS = frozenset({"history", "legacy-cann-mirror-wip", "_build_cache"})


def _scan_strings(obj, *, path: str = "", parent_key: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _SKIP_KEYS:
                continue
            child = f"{path}.{key}" if path else key
            hits.extend(_scan_strings(value, path=child, parent_key=key))
        return hits
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(_scan_strings(value, path=f"{path}[{idx}]", parent_key=parent_key))
        return hits
    if isinstance(obj, str) and _HOME_RE.search(obj):
        if obj.startswith("/workspace/"):
            return hits
        hits.append(f"{path}: {obj}")
    return hits


def check_file(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return [f"{path}: unreadable ({exc})"]
    return [f"{path}: {hit}" for hit in _scan_strings(data)]


def iter_evidence_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in SCAN_GLOBS:
        for path in sorted(repo_root.glob(pattern)):
            if not path.is_file():
                continue
            if EXCLUDE_PARTS & set(path.relative_to(repo_root).parts):
                continue
            files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reject home-dir paths in perf evidence JSON")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = ap.parse_args(argv)

    violations: list[str] = []
    for path in iter_evidence_files(args.repo_root):
        violations.extend(check_file(path))

    if violations:
        print("EVIDENCE PATH CHECK: FAIL", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 1

    count = len(iter_evidence_files(args.repo_root))
    print(f"EVIDENCE PATH CHECK: PASS ({count} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
