#!/usr/bin/env python3
"""Run a fork target test (representative case) on the simulator.

The fork tests have ~40-60 cases each (production shapes); running ALL on the
simulator is infeasible for the merge-gate. This parses the declarative
# PYASC_TESTS_BEGIN block, selects the case with the smallest block_num
(fastest on the single-threaded simulator), and runs it via pytest.

Usage:
    python3.11 tests/tools/run_target_test.py golden/target/test_vadd.py
    python3.11 tests/tools/run_target_test.py golden/target/test_vadd.py --full  # all cases
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

IMAGE_PYTEST = ["python3.11", "-m", "pytest"]


def parse_cases(filepath: str) -> list[tuple[str, int]]:
    """Parse the # PYASC_TESTS_BEGIN block; return [(test_name, block_num), ...]."""
    source = Path(filepath).read_text()
    m = re.search(r"# PYASC_TESTS_BEGIN\n(.+?)# PYASC_TESTS_END", source, re.DOTALL)
    if not m:
        return []
    # Extract ("test_name", block_num, ...) tuples
    return [(name, int(bn)) for name, bn in re.findall(r'\("([^"]+)",\s*(\d+)', m.group(1))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("test_file", help="Path to the fork target test file")
    ap.add_argument("--backend", default="Model")
    ap.add_argument("--platform", default="Ascend950PR_9599")
    ap.add_argument("--timeout", type=int, default=300, help="Per-case timeout (s)")
    ap.add_argument("--compile-only", action="store_true", default=True, help="Compile-only check (default; fast)")
    ap.add_argument("--run", action="store_true", help="Run the kernel (slow on simulator)")
    ap.add_argument("--full", help="Run ALL cases (for nightly)")
    args = ap.parse_args()

    if not args.full:
        args.compile_only = not args.run
        cases = parse_cases(args.test_file)
        if not cases:
            print(f"[SKIP] {args.test_file}: no PYASC_TESTS_BEGIN block")
            sys.exit(0)
        rep = min(cases, key=lambda c: c[1])
        k_pattern = f"{rep[0]}-{rep[1]}-"
        label = f"{rep[0]} (block_num={rep[1]})"
        print(f"[INFO] {args.test_file}: representative case {label}")
    else:
        k_pattern = None
        label = "ALL cases"
        print(f"[INFO] {args.test_file}: {label}")

    cmd = IMAGE_PYTEST + [args.test_file]
    if k_pattern:
        cmd += ["-k", k_pattern]
    cmd += [
        "--backend", args.backend,
        "--platform", args.platform,
        "-x", "-q", "-p", "no:cacheprovider",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout + 60)
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {args.test_file}: timeout after {args.timeout+60}s ({label})")
        sys.exit(1)
    if r.returncode == 0:
        n = r.stdout.count(" passed")
        print(f"[PASS] {args.test_file} ({label})")
        sys.exit(0)
    else:
        print(f"[FAIL] {args.test_file} ({label})")
        print(r.stdout[-600:])
        print(r.stderr[-400:])
        sys.exit(1)


if __name__ == "__main__":
    main()
