#!/usr/bin/env python3
"""Run the exact-v2 QEMU compile gate for every comparison candidate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
GATE = REPO_ROOT / "integrations/cannbench/workers/run_local_compile_gate.sh"
VARIANTS = ("handwritten", "no_skills", "with_skills")
OPS = ("gelu", "foreach_addcdiv_scalar")


def main() -> int:
    summary = []
    for variant in VARIANTS:
        output_dir = ROOT / variant / "local_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        for op in OPS:
            candidate = ROOT / variant / "candidates" / f"{op}.py"
            completed = subprocess.run(
                [str(GATE), "--candidate", str(candidate.relative_to(REPO_ROOT)),
                 "--op", op],
                cwd=REPO_ROOT, capture_output=True, text=True, check=False,
                timeout=900)
            (output_dir / f"{op}.stderr.txt").write_text(
                completed.stderr, encoding="utf-8")
            try:
                report = json.loads(completed.stdout)
            except json.JSONDecodeError:
                report = {
                    "status": "gate_process_failed",
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                }
            (output_dir / f"{op}.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            row = {
                "variant": variant,
                "operator": op,
                "gate_returncode": completed.returncode,
                "status": report.get("status"),
                "dispatch_passed": report.get("dispatch_passed"),
                "compile_passed": report.get("compile_passed"),
                "total_cases": report.get("cases"),
                "unique_specializations": report.get("unique_specializations"),
                "unique_specializations_passed": report.get(
                    "unique_specializations_passed"),
            }
            summary.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "schema_version": 1,
        "runtime_commit": "ac1222a48c8914d3f81297c7570d1a84f0f26778",
        "results": summary,
    }
    (ROOT / "local_validation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
