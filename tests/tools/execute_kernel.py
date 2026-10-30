#!/usr/bin/env python3.10
"""End-to-end execution of a pyasc kernel under a harness-owned platform contract.

Unlike :mod:`pytest_verify_kernel`, which mocks ``Launcher.run`` and therefore
proves only that codegen ran, this script performs a **real** launch and
requires evidence that one happened.

Two design points are load-bearing:

**The harness owns platform setup, the kernel owns nothing.** Generated kernels
do not call ``config.set_platform()``; without it they ask for a real NPU device
and die with ``SetDeviceWrapper returned 107001`` even on a simulator-capable
host. This script sets the platform *before* the kernel is imported, which is
what makes an otherwise-unmodified generated kernel runnable.

**The kernel is run via ``runpy`` under ``run_name="__main__"``, not by calling a
discovered entry point.** Generated kernels have no common entry point: of the
three kernels in ``docs/evidence/generation-gate-20261028/`` one does its work in
``verify()``, one in ``main()``, and one inline under ``if __name__ ==
"__main__":`` with no callable entry point at all. Name-based discovery reaches
at most one of the three and cannot reach the inline one by any means. Running
the module as ``__main__`` reaches all of them, needs no signature inference,
and never invokes a ``test_*`` function outside its pytest fixture context.

A clean return does not by itself prove anything ran, so ``Launcher.run`` is
wrapped in a call-through spy and at least one launch is required. This is the
same evidence ``pytest_verify_kernel`` relies on (``call_count >= 1``), except
the original is invoked rather than suppressed.

``--mock-launch`` suppresses the real launch instead, leaving the output buffer
with its pre-launch contents. It is the control half of the numerics
falsification test driven by ``run_and_verify.py --mode execute``: a kernel that
genuinely checks its results must FAIL when launches are suppressed.

Usage::

    python3.10 execute_kernel.py /path/to/kernel.py \\
        [--backend Model] [--platform Ascend950PR_9599] \\
        [--mock-launch] [--json]

Exit codes:
    0 — PASS (ran to completion, and at least one launch was observed)
    1 — FAIL (raised, exited non-zero, or launched nothing)
    2 — SKIP (pyasc / a runtime dependency is not usable here)
"""

from __future__ import annotations

import ast
import argparse
import json
import os
import runpy
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the environment probes rather than duplicating them: these encode the
# "partial environment -> SKIP, not FAIL" contract shared with the JIT mode.
from numerical_checks import NumericalChecks

from pytest_verify_kernel import (  # noqa: E402
    EXIT_FAIL,
    EXIT_PASS,
    EXIT_SKIP,
    _SKIPPABLE_MISSING_MODULES,
    _asc_package_dir,
    _is_provisioning_error,
    _try_import_asc,
)

# Verification levels, weakest first. Reported verbatim in the JSON payload so a
# consumer never has to infer how much was actually demonstrated.
LEVEL_COMPILED = "compiled"
LEVEL_EXECUTED = "executed"


def _emit(payload: dict[str, Any], use_json: bool, json_out: Optional[Path] = None) -> int:
    code = int(payload["exit_code"])
    # The kernel and the pyasc runtime both write to stdout (the simulator
    # prints "Model RUN TIME: ..." there), so stdout is not a reliable channel
    # for a machine-readable result. --json-out gives the caller an unpolluted
    # one; without it a parse failure would silently look like a zero-launch run.
    if json_out is not None:
        try:
            json_out.write_text(json.dumps(payload, indent=2))
        except OSError as exc:
            print(f"warning: could not write {json_out}: {exc}", file=sys.stderr)
    if use_json:
        print(json.dumps(payload, indent=2))
    else:
        status = str(payload["status"]).upper()
        detail = payload.get("error") or payload.get("reason") or payload.get("detail") or ""
        stream = sys.stdout if code == EXIT_PASS else sys.stderr
        print(f"{status}: {detail}".rstrip(": "), file=stream)
    return code


def _install_launch_spy(asc_mod: Any, suppress: bool) -> tuple[Any, Any, dict[str, int]]:
    """Wrap ``Launcher.run``; return (cls, original, counter).

    With ``suppress=False`` the original runs (a real launch). With
    ``suppress=True`` it does not, so results never reach the output buffer.
    """
    import asc.runtime.launcher as launcher_mod

    cls = launcher_mod.Launcher
    original = cls.run
    counter = {"calls": 0}

    def spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        counter["calls"] += 1
        if suppress:
            return None
        return original(self, *args, **kwargs)

    cls.run = spy  # type: ignore[method-assign]
    return cls, original, counter


def _missing_skippable_module(exc: BaseException) -> Optional[str]:
    """Name of the missing runtime dependency, if this is a skippable import."""
    if not isinstance(exc, ModuleNotFoundError):
        return None
    name = (exc.name or "").split(".")[0]
    return name if name in _SKIPPABLE_MISSING_MODULES else None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a pyasc kernel end-to-end with a harness-set platform.",
    )
    parser.add_argument("kernel_py", type=Path, help="Path to kernel.py")
    parser.add_argument("--backend", default="Model", help="Backend (default: Model)")
    parser.add_argument(
        "--platform",
        default="Ascend950PR_9599",
        help="Platform / SoC version (default: Ascend950PR_9599)",
    )
    parser.add_argument("--device-id", type=int, default=0, help="Device id (default: 0)")
    parser.add_argument(
        "--mock-launch",
        action="store_true",
        help="Suppress the real launch (control run for the numerics check)",
    )
    parser.add_argument("--json", action="store_true", help="JSON report on stdout")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help=(
            "Write the JSON report to this path. Preferred over --json for "
            "programmatic use: the kernel and the pyasc runtime also write to "
            "stdout, so stdout cannot be parsed reliably."
        ),
    )
    args = parser.parse_args(argv)
    json_out: Optional[Path] = args.json_out

    base: dict[str, Any] = {
        "kernel_path": str(args.kernel_py),
        "backend": args.backend,
        "platform": args.platform,
        "mock_launch": bool(args.mock_launch),
        "launcher_calls": 0,
        "verification_level": None,
    }

    kernel_path = args.kernel_py.resolve()
    if not kernel_path.is_file():
        return _emit(
            {**base, "status": "fail", "exit_code": EXIT_FAIL,
             "error": f"Not a file: {kernel_path}"},
            args.json,
            json_out,
        )
    base["kernel_path"] = str(kernel_path)

    asc_mod, import_err = _try_import_asc()
    if asc_mod is None:
        return _emit(
            {**base, "status": "skip", "exit_code": EXIT_SKIP,
             "reason": "pyasc (asc) not importable", "import_error": import_err},
            args.json,
            json_out,
        )
    asc_dir = _asc_package_dir(asc_mod)

    from asc.runtime import config

    # Platform first, before the kernel module is imported: this is the whole
    # point of the harness-owned contract.
    try:
        config.set_platform(
            config.Backend(args.backend),
            config.Platform(args.platform),
            args.device_id,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - classified below
        bad_value = isinstance(exc, (ValueError, KeyError))
        return _emit(
            {**base,
             "status": "fail" if bad_value else "skip",
             "exit_code": EXIT_FAIL if bad_value else EXIT_SKIP,
             "error" if bad_value else "reason":
                 f"set_platform({args.backend}, {args.platform}) failed: "
                 f"{type(exc).__name__}: {exc}"},
            args.json,
            json_out,
        )

    cls, original, counter = _install_launch_spy(asc_mod, suppress=args.mock_launch)

    checks = NumericalChecks(kernel_path, counter)
    checks.install()
    numerical_failure = None

    saved_argv = list(sys.argv)
    saved_cwd = os.getcwd()
    saved_path = list(sys.path)
    kernel_dir = str(kernel_path.parent)

    # Kernels that *do* parse arguments (the skill template) see the same
    # backend/platform the harness just configured; kernels that ignore argv
    # are unaffected.
    sys.argv = [str(kernel_path), "-r", args.backend, "-v", args.platform]
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    os.chdir(kernel_dir)

    status = "pass"
    exit_code = EXIT_PASS
    error: Optional[str] = None
    tb: Optional[str] = None

    try:
        runpy.run_path(str(kernel_path), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code not in (None, 0):
            status, exit_code = "fail", EXIT_FAIL
            error = f"kernel exited with status {code}"
    except BaseException as exc:  # noqa: BLE001 - classified below
        numerical_failure = checks.failure(exc)
        missing = _missing_skippable_module(exc)
        if missing is not None:
            status, exit_code = "skip", EXIT_SKIP
            error = f"runtime dependency not installed: {missing}"
        elif _is_provisioning_error(exc, asc_dir):
            # A broken pyasc install, not a kernel defect. Same contract as the
            # JIT mode. Compiler errors are deliberately excluded (see helper).
            status, exit_code = "skip", EXIT_SKIP
            error = (
                "pyasc native extension not usable (partial environment): "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            status, exit_code = "fail", EXIT_FAIL
            error = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
    finally:
        checks.restore()
        cls.run = original  # type: ignore[method-assign]
        sys.argv = saved_argv
        sys.path = saved_path
        try:
            os.chdir(saved_cwd)
        except OSError:
            pass

    calls = counter["calls"]
    base["launcher_calls"] = calls
    base["numerical_checks"] = checks.events
    base["numerical_failure"] = numerical_failure

    # A clean return proves nothing on its own. An empty ``__main__``, a module
    # with no ``__main__`` block, and a script that only prints "PASS" all reach
    # this point without a launch, and all must FAIL.
    if status == "pass" and calls == 0:
        status, exit_code = "fail", EXIT_FAIL
        has_main = any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)
                       for n in ast.walk(checks.tree))
        error = (
            ("no kernel launch: " if has_main else "no __main__ entry point; no kernel launch: ") +
            " the module ran to completion but Launcher.run "
            "was never called"
        )

    payload: dict[str, Any] = {**base, "status": status, "exit_code": exit_code}
    if status == "pass":
        payload["verification_level"] = (
            LEVEL_COMPILED if args.mock_launch else LEVEL_EXECUTED
        )
        payload["detail"] = (
            f"Kernel ran on {args.backend}/{args.platform} with "
            f"{calls} launch{'' if calls == 1 else 'es'}"
            + (" (launch suppressed)" if args.mock_launch else "")
        )
    elif status == "skip":
        payload["reason"] = error
    else:
        payload["error"] = error
        if tb:
            payload["traceback"] = tb

    return _emit(payload, args.json, json_out)


if __name__ == "__main__":
    sys.exit(main())
