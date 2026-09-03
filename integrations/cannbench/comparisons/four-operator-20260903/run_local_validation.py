#!/usr/bin/env python3
"""Run the pinned-current-v2 QEMU compile gate for all 12 candidates."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
IMAGE = "pyasc-cannbench-local-current:030e9b2"
VARIANTS = ("handwritten", "no_skills", "with_skills")
OPS = ("gelu", "rms_norm", "softmax", "transpose")
RUNTIME_COMMIT = "030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d"


def write_json_atomic(path: Path, value: dict) -> None:
    """Replace stale container-owned evidence through the writable directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    def validate(variant: str, op: str) -> dict:
        output_dir = ROOT / variant / "local_validation"
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate = ROOT / variant / "candidates" / f"{op}.py"
        relative = candidate.relative_to(REPO_ROOT)
        command = [
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "-v", f"{REPO_ROOT}:/workspace:ro", "-w", "/workspace",
            IMAGE, "--candidate", str(relative), "--op", op,
        ]
        completed = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True,
            check=False, timeout=1800,
        )
        (output_dir / f"{op}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8")
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError:
            report = {
                "status": "gate_process_failed",
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
            }
        write_json_atomic(output_dir / f"{op}.json", report)
        return {
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
            "fatal_error": report.get("fatal_error"),
        }

    summary = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(validate, variant, op)
                   for variant in VARIANTS for op in OPS]
        for future in as_completed(futures):
            row = future.result()
            summary.append(row)
            print(json.dumps(row), flush=True)
    summary.sort(key=lambda row: (VARIANTS.index(row["variant"]),
                                  OPS.index(row["operator"])))
    result = {
        "schema_version": 1,
        "runtime_commit": RUNTIME_COMMIT,
        "image": IMAGE,
        "results": summary,
    }
    write_json_atomic(ROOT / "local_validation_summary.json", result)
    return 0 if all(row["status"] == "passed" for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
