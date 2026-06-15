#!/usr/bin/env python3
"""Rewrite perf evidence JSON to use repo-relative paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from check_evidence_paths import iter_evidence_files
from repo_paths import REPO_ROOT, relativize_evidence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sanitize absolute paths in perf evidence JSON")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--write", action="store_true", help="rewrite files in place")
    args = ap.parse_args(argv)

    changed = 0
    for path in iter_evidence_files(args.repo_root):
        try:
            original = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"SKIP {path}: {exc}", file=sys.stderr)
            continue
        cleaned = relativize_evidence(original, root=args.repo_root)
        if cleaned == original:
            continue
        changed += 1
        if args.write:
            try:
                path.write_text(json.dumps(cleaned, indent=2) + "\n")
            except OSError as exc:
                print(f"SKIP {path}: {exc}", file=sys.stderr)
                continue
            print(f"updated {path.relative_to(args.repo_root)}")
        else:
            print(f"would update {path.relative_to(args.repo_root)}")

    if not args.write:
        print(f"{changed} file(s) would change (pass --write to apply)")
    else:
        print(f"{changed} file(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
