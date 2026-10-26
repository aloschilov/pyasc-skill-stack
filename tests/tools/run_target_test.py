#!/usr/bin/env python3
"""Run a fork target test (representative case) on the simulator.

The fork tests have ~40-60 cases each (production shapes); running ALL on the
simulator is infeasible for the merge-gate. This parses the declarative
# PYASC_TESTS_BEGIN block, selects the case with the smallest block_num
(fastest on the single-threaded simulator), and runs it via pytest.

Note what the default mode proves. `--compile-only` is on by default, and in
that mode upstream mocks kernel launch and the numerical assertions: a pass
means the kernel built against the pinned runtime, not that it computed
anything correctly. Pass `--run` for numerical execution. Never report a
compile-only sweep as numerical qualification.

Usage:
    python3.11 tests/tools/run_target_test.py reference/asctile/target/test_vadd.py
    python3.11 tests/tools/run_target_test.py reference/asctile/target/test_vadd.py --full
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# The interpreter must be the one the pinned image was built around --
# reference/images/ENV.json records it as .image.python_bin. Overridable so
# that re-pinning onto another Python cell does not need a code change here.
IMAGE_PYTEST = [os.environ.get("PYASC_PYTHON_BIN", "python3.11"), "-m", "pytest"]


def parse_cases(filepath: str) -> list[tuple[str, int]]:
    """Parse the # PYASC_TESTS_BEGIN block; return [(test_name, block_num), ...]."""
    source = Path(filepath).read_text()
    m = re.search(r"# PYASC_TESTS_BEGIN\n(.+?)# PYASC_TESTS_END", source, re.DOTALL)
    if not m:
        return []
    # Extract ("test_name", block_num, ...) tuples
    return [(name, int(bn)) for name, bn in re.findall(r'\("([^"]+)",\s*(\d+)', m.group(1))]


def select_pattern(args, candidates: list[str]) -> str | None:
    """Return the first -k pattern that actually selects a case, else None.

    Probing with --collect-only costs well under a second and turns a silent
    "0 selected, exit 5" into a deliberate choice.
    """
    base = IMAGE_PYTEST + [
        args.test_file, "--collect-only", "-q", "-p", "no:cacheprovider",
        "--backend", args.backend, "--platform", args.platform,
    ]
    for pat in candidates:
        try:
            r = subprocess.run(base + ["-k", pat], capture_output=True,
                               text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return None
        # With -k, pytest reports "N/M tests collected (K deselected)"; with no
        # filter it reports plain "N tests collected". Match both, and require
        # N > 0 -- "0/28 tests collected" is exactly the case to reject.
        m = re.search(r"^(\d+)(?:/\d+)? tests? collected", r.stdout, re.MULTILINE)
        if r.returncode == 0 and m and int(m.group(1)) > 0:
            return pat
    return None


def run_pytest(extra: list[str], timeout: int):
    """Run pytest; return (returncode, stdout, stderr). -1 marks a timeout."""
    try:
        r = subprocess.run(IMAGE_PYTEST + extra, capture_output=True,
                           text=True, timeout=timeout + 60)
    except subprocess.TimeoutExpired:
        return None, "", ""
    return r.returncode, r.stdout, r.stderr


def collect_ids(args) -> list[str]:
    """Every selectable node id in the module, in declaration order."""
    r = subprocess.run(
        IMAGE_PYTEST + [args.test_file, "--collect-only", "-q",
                        "-p", "no:cacheprovider",
                        "--backend", args.backend, "--platform", args.platform],
        capture_output=True, text=True, timeout=300)
    return [ln.strip() for ln in r.stdout.splitlines()
            if "::" in ln and ln.strip().startswith(args.test_file)]


# Exit codes this script returns to the gate.
EXIT_OK = 0
EXIT_FAILED = 1   # the kernel genuinely did not compile
EXIT_BLOCKED = 2  # the check could not be carried out on this machine


def report_failure(test_file: str, label: str, rc, out: str, err: str) -> int:
    """Name the failure mode and return the exit code the gate should see.

    `failed` and `blocked` are different results, and the repo's gate
    vocabulary already separates them. A kernel that does not compile is a
    defect; a box that ran out of memory holding a 7.4 GiB input tensor is not
    evidence about the kernel at all. Reporting both as "FAIL" sends the next
    person hunting a compiler bug that is not there.
    """
    if rc is None:
        print(f"[BLOCKED] {test_file}: no result in time ({label})")
        verdict = EXIT_BLOCKED
    elif rc in (-9, 137):
        # SIGKILL, cause unknown. Memory pressure is the usual explanation and
        # the largest cases do allocate multi-GiB host tensors, but exit 137
        # does not prove it: a timeout, a container limit or an operator with
        # a kill switch produce the same code. Report what was observed and
        # leave the diagnosis to whoever reads the logs.
        print(f"[BLOCKED] {test_file}: killed with SIGKILL ({label}); "
              "cause not established from the exit code alone -- check host "
              "memory, container limits and the kernel log")
        verdict = EXIT_BLOCKED
    elif rc == 5:
        print(f"[FAIL] {test_file}: pytest selected no cases ({label})")
        verdict = EXIT_FAILED
    elif rc < 0:
        print(f"[BLOCKED] {test_file}: killed by signal {-rc} ({label})")
        verdict = EXIT_BLOCKED
    else:
        print(f"[FAIL] {test_file} ({label})")
        verdict = EXIT_FAILED
    print(out[-600:])
    print(err[-400:])
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("test_file", help="Path to the fork target test file")
    ap.add_argument("--backend", default="Model")
    ap.add_argument("--platform", default="Ascend950PR_9599")
    # Per-module, not per-case. Modules without a PYASC_TESTS_BEGIN block run
    # every case: matmul_v3/test_matmul_v3_cube_only.py has 184 and needs ~275s
    # compile-only, which the old 300s default clipped into a spurious failure.
    ap.add_argument("--timeout", type=int,
                    default=int(os.environ.get("PYASC_TARGET_TIMEOUT", "900")),
                    help="Per-module timeout (s)")
    ap.add_argument("--compile-only", action="store_true", default=True,
                    help="Build only; launch and numerical asserts are mocked "
                         "(default). NOT numerical qualification.")
    ap.add_argument("--run", action="store_true",
                    help="Execute the kernel and check numbers (slow on simulator)")
    ap.add_argument("--full", action="store_true", help="Run ALL cases (for nightly)")
    ap.add_argument("--batch", type=int,
                    default=int(os.environ.get("PYASC_TARGET_BATCH", "24")),
                    help="Cases per pytest process on whole-module runs "
                         "(0 disables batching)")
    args = ap.parse_args()

    args.compile_only = not args.run
    cases = parse_cases(args.test_file)
    if cases and not args.full:
        rep = min(cases, key=lambda c: c[1])
        # Two id conventions exist upstream. Most modules build the id from the
        # parameters, so it starts "<name>-<block_num>-"; others pass an
        # explicit id= to pytest.param, giving just "<name>". Picking the wrong
        # one deselects everything, pytest exits 5, and the module looks broken
        # when nothing ran at all -- which is how test_gather.py first failed.
        k_pattern = select_pattern(args, [f"{rep[0]}-{rep[1]}-", rep[0]])
        if k_pattern is None:
            print(f"[WARN] {args.test_file}: no id matches {rep[0]}; running ALL cases")
            label = "ALL cases"
        else:
            label = f"{rep[0]} (block_num={rep[1]})"
            print(f"[INFO] {args.test_file}: representative case {label}")
    else:
        # No PYASC_TESTS_BEGIN block (direct @pytest.mark.parametrize) OR --full:
        # run the entire test. With --compile-only this is fast (~2s per test).
        k_pattern = None
        label = "ALL cases"
        print(f"[INFO] {args.test_file}: {label}")

    common = [
        "--backend", args.backend,
        "--platform", args.platform,
        "-x", "-q", "-p", "no:cacheprovider",
    ]
    if args.compile_only:
        common.append("--compile-only")

    # Whole-module runs go in batches, each in its own pytest process.
    #
    # Compiling many kernels inside one interpreter accumulates memory, and
    # long single processes are the ones that get killed: both
    # matmul_v3/test_matmul_v3_cube_only.py (184 cases) and test_kl_div.py
    # died at exit 137 on a box with 27 GB free. Exit 137 alone does not
    # identify the killer, but batching fixed it, which is the evidence that
    # matters here: capping the peak costs no coverage, because the allocator
    # starts clean on every batch.
    if k_pattern is None and args.batch > 0:
        ids = collect_ids(args)
        if len(ids) > args.batch:
            batches = [ids[i:i + args.batch] for i in range(0, len(ids), args.batch)]
            print(f"[INFO] {args.test_file}: {len(ids)} cases in "
                  f"{len(batches)} batch(es) of up to {args.batch}")
            for n, batch in enumerate(batches, 1):
                rc, out, err = run_pytest(batch + common, args.timeout)
                if rc != 0:
                    sys.exit(report_failure(
                        args.test_file, f"batch {n}/{len(batches)}", rc, out, err))
            print(f"[PASS] {args.test_file} ({len(ids)} cases, {len(batches)} batches)")
            sys.exit(0)

    cmd = [args.test_file]
    if k_pattern:
        cmd += ["-k", k_pattern]
    rc, out, err = run_pytest(cmd + common, args.timeout)
    if rc == 0:
        print(f"[PASS] {args.test_file} ({label})")
        sys.exit(EXIT_OK)
    sys.exit(report_failure(args.test_file, label, rc, out, err))



if __name__ == "__main__":
    main()
