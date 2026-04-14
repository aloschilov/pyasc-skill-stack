#!/usr/bin/env python3
"""Collect verification evidence for a pyasc kernel and write an evidence JSON file.

Runs score_kernel.py and verify_kernel.py (static checks), and optionally
run_and_verify.py (runtime), then writes evidence/<op>-<dtype>-<kind>.json.

Usage:
    python collect_evidence.py <kernel.py> --op abs --dtype float16 [--kind golden] [--shapes '[[1,128]]'] [--runtime]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EVIDENCE_DIR = REPO_ROOT / "evidence"
VERIFY_SCRIPT = SCRIPT_DIR / "verify_kernel.py"
SCORE_SCRIPT = SCRIPT_DIR / "score_kernel.py"
RUN_VERIFY_SCRIPT = SCRIPT_DIR / "run_and_verify.py"
PYTHON = "python3.10"


def _run_tool(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return 1, "timeout"
    except FileNotFoundError:
        return 2, f"not found: {cmd[0]}"
    except Exception as exc:
        return 1, str(exc)


def collect_score(kernel_path: str) -> dict | None:
    code, out = _run_tool([PYTHON, str(SCORE_SCRIPT), kernel_path, "--json"])
    if code == 0:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            pass
    return None


def collect_static_verify(kernel_path: str) -> str:
    code, out = _run_tool([PYTHON, str(VERIFY_SCRIPT), kernel_path, "--json"])
    if code == 0:
        try:
            data = json.loads(out)
            return "pass" if data.get("passed", False) else "fail"
        except json.JSONDecodeError:
            pass
    return "fail"


def collect_runtime(kernel_path: str, mode: str, backend: str, platform: str) -> dict:
    cmd = [
        PYTHON, str(RUN_VERIFY_SCRIPT), kernel_path,
        "--json", "--mode", mode, "--backend", backend, "--platform", platform,
    ]
    code, out = _run_tool(cmd, timeout=180)
    result = {"mode": mode, "backend": backend, "platform": platform}
    if code == 0:
        result["status"] = "pass"
    elif code == 2:
        result["status"] = "skip"
    else:
        result["status"] = "fail"

    try:
        parsed = json.loads(out)
        result["detail"] = parsed.get("detail", "")
    except (json.JSONDecodeError, TypeError):
        result["detail"] = out[:300] if out else ""

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect evidence for a pyasc kernel and write to evidence/.",
    )
    parser.add_argument("kernel_py", help="Path to kernel .py file")
    parser.add_argument("--op", required=True, help="Operation name (e.g. abs, add)")
    parser.add_argument("--dtype", required=True, help="Data type (e.g. float16, float32)")
    parser.add_argument("--kind", default="golden", choices=("golden", "generative"),
                        help="Evidence kind (default: golden)")
    parser.add_argument(
        "--shapes", default="[]",
        help='JSON array of tested shapes, e.g. \'[[1,128],[4,2048]]\'',
    )
    parser.add_argument("--runtime", action="store_true", help="Also run runtime verification")
    parser.add_argument("--mode", default="auto", choices=("jit", "simulator", "auto"),
                        help="Runtime verification mode (default: auto)")
    parser.add_argument("--backend", default="Model", help="Backend for runtime (default: Model)")
    parser.add_argument("--platform", default="Ascend910B1", help="Platform for runtime")
    parser.add_argument("--notes", default="", help="Optional notes to include")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON to stdout instead of writing file")
    args = parser.parse_args()

    kernel_path = Path(args.kernel_py)
    if not kernel_path.exists():
        print(f"ERROR: kernel not found: {kernel_path}", file=sys.stderr)
        sys.exit(1)

    try:
        kernel_rel = str(kernel_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        kernel_rel = str(kernel_path)

    try:
        shapes = json.loads(args.shapes)
    except json.JSONDecodeError:
        shapes = []

    print(f"  Collecting {args.kind} evidence for {args.op}/{args.dtype}...")
    print(f"  Kernel: {kernel_rel}")

    score_data = collect_score(str(kernel_path))
    static_result = collect_static_verify(str(kernel_path))

    print(f"  Static verify: {static_result}")
    if score_data:
        print(f"  Score: {score_data.get('score', '?')}/10 (accepted: {score_data.get('accepted', '?')})")
    else:
        print("  Score: could not be determined")

    verification_section: dict = {
        "mode": "static_only",
        "status": static_result,
        "shapes_verified": shapes,
    }

    if args.runtime:
        print(f"  Running runtime verification (mode={args.mode})...")
        rt = collect_runtime(str(kernel_path), args.mode, args.backend, args.platform)
        verification_section = {
            "mode": rt["mode"],
            "backend": rt["backend"],
            "platform": rt["platform"],
            "status": rt["status"],
            "shapes_verified": shapes,
        }
        print(f"  Runtime: {rt['status']}")

    evidence: dict = {
        "schema_version": "2",
        "kind": args.kind,
        "operation": args.op,
        "dtype": args.dtype,
        "kernel_path": kernel_rel,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verification": verification_section,
        "score": {
            "value": score_data.get("score", 0.0) if score_data else 0.0,
            "threshold": 8.5,
            "accepted": score_data.get("accepted", False) if score_data else False,
            "checks": score_data.get("checks", {}) if score_data else {},
        },
        "static_verify": static_result,
        "notes": args.notes,
    }

    dtype_short = args.dtype.replace("float", "f")
    out_name = f"{args.op}-{dtype_short}-{args.kind}.json"
    out_path = EVIDENCE_DIR / out_name

    if args.dry_run:
        print()
        print(json.dumps(evidence, indent=2))
    else:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(evidence, f, indent=2)
            f.write("\n")
        print(f"  Written: {out_path.relative_to(REPO_ROOT)}")

    overall = "pass" if (static_result == "pass" and evidence["score"]["accepted"]) else "fail"
    print(f"  Overall: {overall}")
    sys.exit(0 if overall == "pass" else 1)


if __name__ == "__main__":
    main()
