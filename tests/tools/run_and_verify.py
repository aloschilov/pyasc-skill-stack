#!/usr/bin/env python3
"""Runtime verification for pyasc kernels.

Supports four strategies:

* **jit** — Runs ``pytest_verify_kernel.py``: mocks ``Launcher.run``, no simulator.
  Proves codegen ran; proves nothing about results.
* **execute** — Runs ``execute_kernel.py``: the *harness* sets the platform, then
  runs the kernel as ``__main__`` and requires a counted real launch.
* **simulator** — Compatibility alias for execute; same counted-launch contract.
* **auto** — Tries execute first; on SKIP (exit 2), falls back to JIT. A failure
  is never downgraded into a fallback success.

Every report carries an explicit ``verification_level`` so that "exited cleanly"
is never mistaken for "numerically verified":

* ``compiled`` — codegen ran, the launch was mocked.
* ``executed`` — at least one real launch occurred and the script exited cleanly.
* ``numerically_verified`` — ``executed``, *and* the kernel's own check was shown
  to fire when results are wrong. Established by falsification: the kernel is run
  a second time with launches suppressed, and must fail that control run. A
  kernel that passes both runs is not checking its results, and is reported
  ``executed``.

Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP.

Usage::

    python run_and_verify.py <kernel.py> \\
        [--mode jit|execute|simulator|auto] \\
        [--backend Model|NPU] \\
        [--platform Ascend950PR_9599] \\
        [--no-prove-numerics] \\
        [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from numerical_checks import proves_numerics
from pathlib import Path

PYTHON = sys.executable or "python3"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
JIT_VERIFY_SCRIPT = SCRIPT_DIR / "pytest_verify_kernel.py"
EXECUTE_SCRIPT = SCRIPT_DIR / "execute_kernel.py"

# Interpreter inside the pinned simulator image.
CONTAINER_PYTHON = "python3.11"

# Verification levels, weakest first. See the module docstring.
LEVEL_COMPILED = "compiled"
LEVEL_EXECUTED = "executed"
LEVEL_NUMERIC = "numerically_verified"

# Shared budgets for the nested CLI and its callers. Image provisioning is
# deliberately separate: docker run must never spend a kernel budget pulling.
DOCKER_PROBE_TIMEOUT = 30
DOCKER_CLEANUP_TIMEOUT = 30


def docker_run_timeout(execution_timeout: int) -> int:
    return 2 * execution_timeout + 120  # primary, control, JIT/startup allowance


def verification_command_timeout(execution_timeout: int) -> int:
    return (docker_run_timeout(execution_timeout) + DOCKER_PROBE_TIMEOUT
            + DOCKER_CLEANUP_TIMEOUT + 30)  # outer CLI startup/report headroom


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _simulator_env(platform: str, base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env.setdefault("PYASC_DUMP_PATH", "/tmp/pyasc-eval-dump")
    ascend = env.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_HOME_PATH")
    if ascend:
        sim_lib = os.path.join(ascend, "tools", "simulator", platform, "lib")
        if os.path.isdir(sim_lib):
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{sim_lib}:{prev}" if prev else sim_lib
    return env


def _extract_shapes(detail: str) -> list[list[int]]:
    """Extract shape info from PASS messages like 'size 128' or 'shape (32, 4096)'."""
    import re
    shapes: list[list[int]] = []
    for m in re.finditer(r"size\s+(\d+)", detail):
        shapes.append([int(m.group(1))])
    for m in re.finditer(r"shape\s*\(([^)]+)\)", detail):
        dims = [int(d.strip()) for d in m.group(1).split(",") if d.strip().isdigit()]
        if dims:
            shapes.append(dims)
    return shapes


def status_of_code(code: int) -> str:
    """Map an exit code to a status.

    Anything other than the three documented codes is a failure rather than
    an unknown state to crash on: a process killed by a signal arrives here
    as a negative number, and an OOM-killed one as 137.
    """
    return {0: "PASS", 1: "FAIL", 2: "SKIP"}.get(code, "FAIL")


def _report(
    use_json: bool,
    path: str,
    status: str,
    detail: str,
    mode: str | None = None,
    level: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    if use_json:
        payload: dict[str, object] = {
            "file": path,
            "status": status,
            "detail": detail,
        }
        if mode is not None:
            payload["mode"] = mode
        if extra:
            payload.update(extra)
        # Always present, even on failure, so no consumer has to infer how much
        # was actually demonstrated. null means "nothing was established".
        payload["verification_level"] = level if status == "PASS" else None
        payload["shapes_verified"] = _extract_shapes(detail) if status == "PASS" else []
        print(json.dumps(payload, indent=2))
    else:
        mode_s = f" [{mode}]" if mode else ""
        level_s = f" <{level}>" if level and status == "PASS" else ""
        print(f"  [{status}]{mode_s}{level_s} {Path(path).name}: {detail}")


def _run_jit(kernel_abs: str, use_json: bool, platform: str = "Ascend950PR_9599", scalar_args: str = "{}", evidence: dict | None = None) -> tuple[int, str]:
    cmd = [PYTHON, str(JIT_VERIFY_SCRIPT), kernel_abs, "--platform", platform, "--scalar-args", scalar_args]
    if use_json:
        cmd.append("--json")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(kernel_abs).parent),
        )
    except subprocess.TimeoutExpired:
        return 1, "JIT verify timeout after 120s"
    except FileNotFoundError:
        return 2, f"{PYTHON} or pytest_verify_kernel.py not found"
    except Exception as exc:
        return 1, f"JIT verify error: {exc}"

    combined = (result.stdout or "") + (result.stderr or "")
    if evidence is not None:
        evidence["jit"] = {"exit_code": result.returncode, "log": combined, "argv": cmd}
    if result.returncode == 0:
        return 0, "JIT compilation verified (Launcher.run mocked)"
    if result.returncode == 2:
        return 2, combined.strip() or "pyasc (asc) not importable"
    return 1, combined.strip() or f"JIT verify exit {result.returncode}"


def pinned_sim_image() -> str:
    """The simulator image this repo is pinned to, by digest.

    Read from ``reference/images/ENV.json`` rather than hardcoded, because the
    upstream image matrix moves between versions. Prefers the dev overlay: the
    minimal image carries pyasc alone, while the overlay adds CPU torch, and
    generated kernels verify themselves with ``torch.allclose``.
    """
    env_override = os.environ.get("PYASC_SIM_IMAGE")
    if env_override:
        return env_override
    pin = REPO_ROOT / "reference" / "images" / "ENV.json"
    try:
        data = json.loads(pin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    for key in ("dev_image", "image"):
        entry = data.get(key)
        if isinstance(entry, str) and entry:
            return entry
        if isinstance(entry, dict):
            # ``ref`` is the digest-pinned reference; ``tag`` is the mutable
            # fallback. Prefer the digest so a gate cannot drift under us.
            for field in ("ref", "tag"):
                value = entry.get(field)
                if isinstance(value, str) and value:
                    return value
    return ""


def _docker_available() -> bool:
    try:
        return subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=DOCKER_PROBE_TIMEOUT,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _run_in_docker(
    kernel_abs: str,
    argv_rest: list[str],
    image: str,
    timeout: int,
    project_dir: Path | None = None,
) -> tuple[int, str, str]:
    """Re-invoke this script inside the pinned image. Returns (code, stdout, raw).

    The repo is mounted read-only at ``/repo`` for the tool scripts and the
    project directory read-write at ``/workspace`` (compilation writes dumps
    next to the kernel). ``project_dir`` defaults to the kernel's own parent;
    callers with a wider project tree pass it explicitly so the kernel's
    relative location inside that tree is preserved. The inner invocation is
    pinned to ``--runtime-backend host`` so it cannot recurse.
    """
    if not image:
        return 2, "", (
            "no simulator image pinned: set PYASC_SIM_IMAGE or provide "
            "reference/images/ENV.json"
        )

    kernel = Path(kernel_abs)
    if project_dir is None:
        project_dir = kernel.parent
        rel_kernel = kernel.name
    else:
        project_dir = Path(project_dir).resolve()
        try:
            rel_kernel = str(kernel.relative_to(project_dir))
        except ValueError:
            return 1, "", (
                f"kernel {kernel} is not inside project dir {project_dir}"
            )
    container_name = "pyasc-verify-" + uuid.uuid4().hex
    cmd = [
        "docker", "run", "--rm", "--name", container_name,
        "--pull", "never", "--network", "none",
        # Heavy CANN simulator runs exhaust the default open-file limit.
        "--ulimit", "nofile=65536:65536",
        "-v", f"{REPO_ROOT}:/repo:ro",
        "-v", f"{project_dir}:/workspace",
        "-w", "/workspace",
        image,
        CONTAINER_PYTHON, "/repo/tests/tools/run_and_verify.py",
        f"/workspace/{rel_kernel}",
        "--runtime-backend", "host",
        *argv_rest,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 1, "", f"docker run timed out after {timeout}s"
    except FileNotFoundError:
        return 2, "", "docker not found on PATH"
    except Exception as exc:  # noqa: BLE001
        return 1, "", f"docker run error: {exc}"
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=DOCKER_CLEANUP_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            pass  # preserve the original failure/report if the daemon vanished
    return result.returncode, result.stdout or "", (result.stdout or "") + (result.stderr or "")


def _run_execute_once(
    kernel_abs: str,
    backend: str,
    platform: str,
    timeout: int,
    mock_launch: bool,
) -> tuple[int, dict[str, object], str]:
    """One execute_kernel.py subprocess. Returns (exit_code, payload, raw)."""
    # The report comes back through a file, not stdout: the kernel logs and the
    # simulator ("Model RUN TIME: ...") both write to stdout, so parsing it
    # would intermittently fail and make a real run look like a zero-launch one.
    with tempfile.TemporaryDirectory(prefix="pyasc-exec-") as tmpdir:
        report = Path(tmpdir) / "report.json"
        cmd = [
            PYTHON,
            str(EXECUTE_SCRIPT),
            kernel_abs,
            "--backend",
            backend,
            "--platform",
            platform,
            "--json-out",
            str(report),
        ]
        if mock_launch:
            cmd.append("--mock-launch")
        env = _simulator_env(platform, os.environ.copy())
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(Path(kernel_abs).parent),
            )
        except subprocess.TimeoutExpired:
            return 1, {}, f"Timeout after {timeout}s"
        except FileNotFoundError:
            return 2, {}, f"{PYTHON} or execute_kernel.py not found"
        except Exception as exc:
            return 1, {}, f"Execution error: {exc}"

        raw = (result.stdout or "") + (result.stderr or "")
        payload: dict[str, object] = {}
        if report.is_file():
            try:
                payload = json.loads(report.read_text())
            except (ValueError, OSError):
                pass
        return result.returncode, payload, raw


def _run_execute(
    kernel_abs: str,
    backend: str,
    platform: str,
    timeout: int = 300,
    prove_numerics: bool = True,
    evidence: dict | None = None,
) -> tuple[int, str, str | None, int]:
    """Execute the kernel for real; optionally falsify its numerics check.

    Returns ``(exit_code, detail, verification_level, launcher_calls)``.
    """
    code, payload, raw = _run_execute_once(
        kernel_abs, backend, platform, timeout, mock_launch=False
    )

    if evidence is not None:
        evidence["execution"] = {"exit_code": code, "report": payload, "log": raw}
    if code != 0:
        detail = str(
            payload.get("error") or payload.get("reason") or raw.strip()[:500]
        ) or f"execute exit {code}"
        if code not in (1, 2):
            # Killed by a signal (negative code) or an unexpected status: a
            # codegen failure can take the CANN runtime down at interpreter
            # shutdown, and an OOM-killed run returns 137. Either way it is a
            # failure, not an unknown state to crash on.
            detail = f"{detail} [process terminated with exit code {code}]"
            code = 1
        # Report the launches that did happen before the failure: a kernel that
        # launched and then raised is a different diagnosis from one that never
        # launched at all.
        return code, detail, None, int(payload.get("launcher_calls") or 0)

    # A missing report means we cannot say what happened. Do not fall back to a
    # default of "0 launches" and report PASS anyway — that is precisely the
    # conflation this mode exists to prevent.
    if not payload or payload.get("status") != "pass" or payload.get("exit_code") != 0:
        return 1, "execute reported success but produced no report (cannot verify)", None, 0

    calls = int(payload.get("launcher_calls") or 0)
    if calls < 1:
        return 1, f"execute exited 0 but recorded {calls} launches", None, calls

    detail = str(payload.get("detail") or f"Kernel ran with {calls} launch(es)")

    if not prove_numerics:
        return 0, detail, LEVEL_EXECUTED, calls

    # Control run: suppress the launches, so the output buffer keeps its
    # pre-launch contents. A kernel that genuinely verifies its results must
    # FAIL here. One that passes is not checking anything, and does not get to
    # claim the numeric level.
    ctl_code, ctl_payload, ctl_raw = _run_execute_once(
        kernel_abs, backend, platform, timeout, mock_launch=True
    )

    if evidence is not None:
        evidence["control"] = {"exit_code": ctl_code, "report": ctl_payload, "log": ctl_raw}
    if proves_numerics(payload, ctl_payload, ctl_code):
        reason = str(ctl_payload.get("error") or "control run failed")
        return 0, f"{detail}; numerics proven (suppressed launch -> {reason})", LEVEL_NUMERIC, calls
    if ctl_code == 0:
        return (
            0,
            f"{detail}; NOT numerically verified: the kernel passes even with "
            f"launches suppressed; this control did not establish a rejecting check",
            LEVEL_EXECUTED,
            calls,
        )
    # Control SKIPped (or could not run): the real execution still stands, but
    # the numeric claim is not established.
    return 0, f"{detail}; numerics check inconclusive (control exit {ctl_code}, no attributable numerical rejection)", LEVEL_EXECUTED, calls



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify pyasc kernel via simulator and/or JIT mock.",
    )
    parser.add_argument("kernel_py", help="Path to kernel.py")
    parser.add_argument(
        "--mode",
        choices=("jit", "execute", "simulator", "auto"),
        default="auto",
        help=(
            "jit=mocked launch only; execute=harness sets platform and runs the "
            "kernel as __main__; simulator=execute compatibility alias; "
            "auto=execute then JIT on SKIP"
        ),
    )
    parser.add_argument(
        "--backend",
        default="Model",
        help="Backend passed as -r (default: Model)",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Platform: detected on SSH; Ascend950PR_9599 for host/docker unless specified",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON report on stdout",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for kernel execution (default: 300)",
    )
    parser.add_argument(
        "--runtime-backend",
        choices=("host", "docker", "ssh", "auto"),
        default="host",
        help=(
            "Where to run: host=this interpreter; docker=inside the pinned "
            "simulator image; auto=docker only if the host cannot run pyasc. "
            "Standing gates should pass 'docker' explicitly — 'auto' can "
            "silently resolve to a host that SKIPs."
        ),
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Override the image for --runtime-backend docker (default: pinned)",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help=(
            "Directory to mount at /workspace for --runtime-backend docker "
            "(default: the kernel's own parent)"
        ),
    )
    parser.add_argument(
        "--no-prove-numerics",
        action="store_true",
        help=(
            "Skip the suppressed-launch control run. The result can then reach "
            "at most verification_level=executed."
        ),
    )
    parser.add_argument("--jit-scalar-args", default="{}", help="Explicit scalar arguments for JIT examples, as JSON")
    parser.add_argument('--ssh-target', default=os.environ.get('PYASC_SSH_TARGET', 'cannlab'))
    parser.add_argument('--ssh-remote-dir', default=os.environ.get('PYASC_SSH_REMOTE_DIR', '/tmp/pyasc-hardware-runs'))
    parser.add_argument('--ssh-python', default=os.environ.get('PYASC_SSH_PYTHON', 'python3'))
    parser.add_argument('--ssh-env', default=os.environ.get('PYASC_SSH_ENV'), help='Remote CANN environment script to source')
    args = parser.parse_args()

    kernel_path = args.kernel_py
    use_json = args.json

    if not Path(kernel_path).exists():
        _report(use_json, kernel_path, "FAIL", "File not found", args.mode)
        sys.exit(1)

    if args.runtime_backend == 'ssh':
        from ssh_runtime import run
        code, payload = run(args)
        if use_json:
            print(json.dumps(payload, indent=2))
        else:
            _report(False, kernel_path, payload['status'], payload['detail'],
                    payload.get('mode'), payload.get('verification_level'))
        sys.exit(code)
    args.platform = args.platform or 'Ascend950PR_9599'

    # Container delegation happens before the host asc probe: the whole point of
    # the docker backend is to run where pyasc *is* installed.
    backend_choice = args.runtime_backend
    if backend_choice == "auto":
        backend_choice = "host" if _can_import("asc") else "docker"
    if backend_choice == "docker":
        image = args.docker_image or pinned_sim_image()
        if not _docker_available():
            _report(
                use_json, kernel_path, "SKIP",
                "docker requested but the daemon is not available",
                args.mode, None,
            )
            sys.exit(2)
        passthrough = [
            "--mode", args.mode,
            "--backend", args.backend,
            "--platform", args.platform,
            "--timeout", str(args.timeout),
            "--jit-scalar-args", args.jit_scalar_args,
        ]
        if args.json:
            passthrough.append("--json")
        if args.no_prove_numerics:
            passthrough.append("--no-prove-numerics")
        code, stdout, raw = _run_in_docker(
            kernel_abs=str(Path(kernel_path).resolve()),
            argv_rest=passthrough,
            image=image,
            timeout=docker_run_timeout(args.timeout),
            project_dir=Path(args.project_dir) if args.project_dir else None,
        )
        # Docker startup errors are unavailable infrastructure, not kernel
        # verdicts. Other unexpected exits (signals/OOM) remain failures. Keep
        # the raw exit and log while honoring the public 0/1/2 exit contract.
        raw_code = code
        if code not in (0, 1, 2):
            code = 2 if code in (125, 126, 127) else 1
            detail = (f"Docker runtime unavailable (exit {raw_code}); pre-pull the "
                      f"pinned image with docker pull {image}. " if code == 2 else
                      f"Docker process terminated with exit {raw_code}. ")
            _report(use_json, kernel_path, status_of_code(code),
                    detail + raw.strip()[-500:], f"{args.mode} (docker)", None,
                    {"evidence": {"docker": {"exit_code": raw_code, "log": raw}}})
            sys.exit(code)
        # The inner invocation already emitted a complete report; forward it
        # verbatim so the two backends are indistinguishable to a consumer.
        if stdout.strip():
            sys.stdout.write(stdout if stdout.endswith("\n") else stdout + "\n")
        else:
            _report(
                use_json, kernel_path,
                status_of_code(code),
                (raw.strip()[-500:] or f"docker run exit {code}"),
                f"{args.mode} (docker)", None,
            )
        sys.exit(code)

    if not _can_import("asc"):
        _report(
            use_json,
            kernel_path,
            "SKIP",
            "pyasc (asc) not importable",
            args.mode,
        )
        sys.exit(2)

    kernel_abs = str(Path(kernel_path).resolve())
    evidence = {}

    if args.mode == "jit":
        code, detail = _run_jit(kernel_abs, use_json, args.platform, args.jit_scalar_args, evidence)
        _report(use_json, kernel_path, status_of_code(code), detail, "jit", LEVEL_COMPILED, {"evidence": evidence})
        sys.exit(code)

    if args.mode in ("execute", "simulator"):
        code, detail, level, calls = _run_execute(
            kernel_abs,
            args.backend,
            args.platform,
            timeout=args.timeout,
            prove_numerics=not args.no_prove_numerics, evidence=evidence,
        )
        _report(use_json, kernel_path, status_of_code(code), detail, args.mode, level,
                {"launcher_calls": calls, "evidence": evidence})
        sys.exit(code)

    # auto: execute first, JIT fallback on SKIP only.
    #
    # The fallback exists for hosts that cannot run kernels at all, not as a way
    # to turn a failure into a pass: a genuine execution failure (exit 1) stays
    # FAIL, and the fallback can only ever report the weaker ``compiled`` level.
    exe_code, exe_detail, exe_level, exe_calls = _run_execute(
        kernel_abs,
        args.backend,
        args.platform,
        timeout=args.timeout,
        prove_numerics=not args.no_prove_numerics, evidence=evidence,
    )
    if exe_code == 0:
        _report(use_json, kernel_path, "PASS", exe_detail, "auto (execute)", exe_level,
                {"launcher_calls": exe_calls, "evidence": evidence})
        sys.exit(0)
    if exe_code != 2:
        _report(use_json, kernel_path, "FAIL", exe_detail, "auto (execute)", None,
                {"launcher_calls": exe_calls, "evidence": evidence})
        sys.exit(1)

    jit_code, jit_detail = _run_jit(kernel_abs, use_json, args.platform, args.jit_scalar_args, evidence)
    if jit_code == 0:
        combined = (
            f"execution unavailable ({exe_detail}); fell back to compile-only: "
            f"{jit_detail}"
        )
        _report(
            use_json, kernel_path, "PASS", combined, "auto (jit fallback)", LEVEL_COMPILED,
            extra={"evidence": evidence},
        )
        sys.exit(0)
    if jit_code == 2:
        _report(
            use_json,
            kernel_path,
            "SKIP",
            f"Execute: {exe_detail}; JIT: {jit_detail}",
            "auto",
            None,
            extra={"evidence": evidence},
        )
        sys.exit(2)

    _report(
        use_json,
        kernel_path,
        "FAIL",
        f"Execution skipped ({exe_detail}); JIT: {jit_detail}",
        "auto (jit fallback failed)",
        None,
        extra={"evidence": evidence},
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
