#!/usr/bin/env python3
"""Run the credit-free pyasc v2 compile gate over the full CANNBench matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


WORKERS_DIR = Path(__file__).resolve().parent
CANNBENCH_DIR = WORKERS_DIR.parent
REPO_ROOT = CANNBENCH_DIR.parent.parent
DEFAULT_CANDIDATES = CANNBENCH_DIR / "submission" / "cann_bench"
ALL_OPS = (
    "sigmoid", "exp", "mish", "gelu", "masked_scale", "swi_glu",
    "foreach_addcdiv_scalar", "foreach_norm", "rms_norm",
)


def evaluate(op: str, candidate_root: Path) -> dict:
    candidate = (candidate_root / f"{op}.py").resolve()
    try:
        container_candidate = candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"candidate must be inside {REPO_ROOT}: {candidate}") from exc
    command = [
        str(WORKERS_DIR / "run_local_compile_gate.sh"),
        "--candidate", str(container_candidate),
        "--op", op,
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = {
            "operator": op,
            "candidate": str(candidate),
            "status": "failed",
            "fatal_error": "local gate emitted non-JSON output",
            "stdout_tail": proc.stdout[-2000:],
        }
    if proc.stderr:
        report["stderr_tail"] = proc.stderr[-4000:]
    report["returncode"] = proc.returncode
    return report


def summarize(reports: list[dict]) -> dict:
    signatures: Counter[str] = Counter()
    for report in reports:
        if report.get("fatal_error"):
            signatures[report["fatal_error"]] += 1
        for case in report.get("case_results", []):
            for error in case.get("compile_errors", []):
                signatures[error] += 1
            if case.get("error"):
                signatures[case["error"]] += 1
    total_cases = sum(report.get("cases", 0) for report in reports)
    dispatch_passed = sum(report.get("dispatch_passed", 0) for report in reports)
    compile_passed = sum(report.get("compile_passed", 0) for report in reports)
    return {
        "schema_version": 1,
        "operators": len(reports),
        "operators_passed": sum(report.get("status") == "passed" for report in reports),
        "cases": total_cases,
        "dispatch_passed": dispatch_passed,
        "compile_passed": compile_passed,
        "status": "passed" if reports and all(
            report.get("status") == "passed" for report in reports
        ) else "failed",
        "failure_signatures": dict(signatures),
        "results": [{
            "operator": report.get("operator"),
            "status": report.get("status"),
            "cases": report.get("cases", 0),
            "dispatch_passed": report.get("dispatch_passed", 0),
            "compile_passed": report.get("compile_passed", 0),
            "unique_specializations": report.get("unique_specializations", 0),
            "unique_specializations_passed": report.get(
                "unique_specializations_passed", 0
            ),
            "fatal_error": report.get("fatal_error"),
        } for report in reports],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--ops", default=",".join(ALL_OPS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    ops = tuple(op.strip() for op in args.ops.split(",") if op.strip())
    unknown = sorted(set(ops) - set(ALL_OPS))
    if unknown:
        parser.error(f"unknown operators: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, op, args.candidate_root): op for op in ops}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            op = report.get("operator", futures[future])
            print(
                f"{op}: {report.get('status')} "
                f"{report.get('compile_passed', 0)}/{report.get('cases', 0)}",
                flush=True,
            )
            (args.output_dir / f"{op}.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    order = {op: index for index, op in enumerate(ops)}
    reports.sort(key=lambda report: order[report["operator"]])
    summary = summarize(reports)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
