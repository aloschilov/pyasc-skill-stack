#!/usr/bin/env python3
"""Drive an opencode agent run for a given prompt, verify the generated kernel,
and write a generative evidence JSON file.

Flow:
  1. Read prompt from --prompt or from capabilities.yaml for the given --op/--dtype
  2. Create a clean test project directory
  3. Run: opencode run "<prompt>" --dir <project>
  4. Search the project for the generated kernel.py
  5. Run score_kernel.py and verify_kernel.py (static, on host)
  6. Optionally run simulator verification inside Docker (--runtime)
  7. Write evidence/<op>-<dtype>-generative.json
  8. Clean up

Exit codes: 0 = pass, 1 = fail, 2 = skip (opencode unavailable)

Usage:
    python collect_generative_evidence.py --op abs --dtype float16 [--prompt "..."] [--runtime] [--timeout 300]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EVIDENCE_DIR = REPO_ROOT / "evidence"
VERIFY_SCRIPT = SCRIPT_DIR / "verify_kernel.py"
SCORE_SCRIPT = SCRIPT_DIR / "score_kernel.py"
RUN_VERIFY_SCRIPT = SCRIPT_DIR / "run_and_verify.py"
CAPABILITIES_FILE = REPO_ROOT / "capabilities.yaml"
PYTHON = "python3.10"

DOCKER_IMAGE = os.environ.get(
    "PYASC_SIM_IMAGE", "ghcr.io/aloschilov/pyasc-sim:py3.11"
)


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"
    except FileNotFoundError:
        return 2, "", f"not found: {cmd[0]}"
    except Exception as exc:
        return 1, "", str(exc)


def load_prompt_from_capabilities(op: str, dtype: str) -> str | None:
    """Extract the prompt for a given op/dtype from capabilities.yaml."""
    if not CAPABILITIES_FILE.exists():
        return None

    if yaml is not None:
        with open(CAPABILITIES_FILE) as f:
            data = yaml.safe_load(f)
    else:
        code, out, _ = _run(
            [PYTHON, "-c",
             f"import yaml,json; print(json.dumps(yaml.safe_load(open('{CAPABILITIES_FILE}'))))"],
            timeout=10,
        )
        if code != 0:
            return None
        data = json.loads(out)

    for operation in data.get("operations", []):
        if operation.get("name") != op:
            continue
        for cell in operation.get("cells", []):
            if cell.get("dtype") == dtype:
                return cell.get("prompt")
    return None


def create_test_project(prefix: str) -> Path:
    """Create a temporary directory with skills/teams/golden symlinked in."""
    tmp = Path(tempfile.mkdtemp(prefix=f"{prefix}."))
    for subdir in ("skills", "teams", "golden"):
        src = REPO_ROOT / subdir
        if src.exists():
            (tmp / subdir).symlink_to(src)

    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=str(tmp), capture_output=True, timeout=10,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@pyasc.test"],
        cwd=str(tmp), capture_output=True, timeout=10,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=str(tmp), capture_output=True, timeout=10,
    )
    return tmp


def find_kernel(project_dir: Path, op: str) -> Path | None:
    """Search for the generated kernel.py, trying likely paths first."""
    candidates = [
        project_dir / "kernels" / f"{op}_f16" / "kernel.py",
        project_dir / "kernels" / f"{op}_f32" / "kernel.py",
        project_dir / "kernel.py",
        project_dir / f"{op}_f16" / "kernel.py",
        project_dir / f"{op}_f32" / "kernel.py",
    ]
    for c in candidates:
        if c.is_file():
            return c

    matches = glob.glob(str(project_dir / "**" / "kernel.py"), recursive=True)
    for m in matches:
        if op in m:
            return Path(m)
    if matches:
        return Path(matches[0])

    py_files = glob.glob(str(project_dir / "**" / "*.py"), recursive=True)
    py_files = [f for f in py_files if not Path(f).name.startswith("__")]
    if py_files:
        return Path(py_files[0])

    return None


def find_artifacts(project_dir: Path) -> list[str]:
    """List workflow artifacts found in the project."""
    found = []
    for name in ("kernel.py", "design.md", "self_review.md",
                  "acceptance_review.md", "verification.md"):
        matches = glob.glob(str(project_dir / "**" / name), recursive=True)
        if matches:
            found.append(name)
    return found


def run_score(kernel_path: Path) -> dict | None:
    code, out, _ = _run([PYTHON, str(SCORE_SCRIPT), str(kernel_path), "--json"])
    if code == 0:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            pass
    return None


def run_static_verify(kernel_path: Path) -> str:
    code, out, _ = _run([PYTHON, str(VERIFY_SCRIPT), str(kernel_path), "--json"])
    if code == 0:
        try:
            data = json.loads(out)
            return "pass" if data.get("passed", False) else "fail"
        except json.JSONDecodeError:
            pass
    return "fail"


def run_docker_verify(kernel_path: Path, project_dir: Path) -> dict:
    """Run simulator verification inside the Docker container.

    Mounts the repo at /repo (for tool scripts) and the project at /workspace
    (for the generated kernel). Runs run_and_verify.py from /repo.
    """
    rel_kernel = kernel_path.relative_to(project_dir)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{REPO_ROOT}:/repo:ro",
        "-v", f"{project_dir}:/workspace",
        "-w", "/workspace",
        DOCKER_IMAGE,
        "python3.11", "/repo/tests/tools/run_and_verify.py",
        str(rel_kernel), "--mode", "simulator", "--json",
    ]
    code, out, err = _run(cmd, timeout=300)
    result = {
        "mode": "simulator", "backend": "Model", "platform": "Ascend910B1",
    }
    if code == 0:
        result["status"] = "pass"
    elif code == 2:
        result["status"] = "skip"
    else:
        result["status"] = "fail"

    try:
        parsed = json.loads(out)
        result["detail"] = parsed.get("detail", "")
        result["shapes_verified"] = parsed.get("shapes_verified", [])
    except (json.JSONDecodeError, TypeError):
        result["detail"] = (err or out)[:300]
        result["shapes_verified"] = []

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect generative evidence via opencode run.",
    )
    parser.add_argument("--op", required=True, help="Operation name (e.g. abs)")
    parser.add_argument("--dtype", required=True, help="Data type (e.g. float16)")
    parser.add_argument("--prompt", default=None,
                        help="Prompt to use (default: read from capabilities.yaml)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Agent timeout in seconds (default: 300)")
    parser.add_argument("--runtime", action="store_true",
                        help="Run simulator verification in Docker after generation")
    parser.add_argument("--notes", default="", help="Optional notes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print JSON to stdout, don't write file")
    parser.add_argument("--keep-project", action="store_true",
                        help="Don't delete the test project after run")
    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        prompt = load_prompt_from_capabilities(args.op, args.dtype)
    if not prompt:
        print(f"ERROR: No prompt provided and none found in capabilities.yaml "
              f"for {args.op}/{args.dtype}", file=sys.stderr)
        sys.exit(1)

    if shutil.which("opencode") is None:
        print("SKIP: opencode CLI not found on PATH", file=sys.stderr)
        sys.exit(2)

    print(f"  Generative evidence for {args.op}/{args.dtype}")
    print(f"  Prompt: {prompt[:80]}...")
    print()

    project = create_test_project(f"gen-{args.op}-{args.dtype}")
    print(f"  Project: {project}")

    output_file = project / "agent-output.txt"
    agent_completed = False
    try:
        env = os.environ.copy()
        env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"

        opencode_cmd = f'opencode run "{prompt}" --dir "{project}"'
        cmd = ["script", "-qc", opencode_cmd, "/dev/null"]

        print(f"  Running opencode (timeout={args.timeout}s)...")
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.timeout, env=env,
        )
        with open(output_file, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        agent_completed = True
        print("  Agent completed.")
    except subprocess.TimeoutExpired:
        print(f"  Agent timed out after {args.timeout}s")
    except Exception as exc:
        print(f"  Agent error: {exc}")

    kernel = find_kernel(project, args.op)
    artifacts = find_artifacts(project)
    print(f"  Kernel: {kernel}")
    print(f"  Artifacts: {artifacts}")

    score_data = None
    static_result = "fail"
    verification: dict = {
        "mode": "static_only", "status": "fail", "shapes_verified": [],
    }

    if kernel and kernel.is_file():
        score_data = run_score(kernel)
        static_result = run_static_verify(kernel)
        print(f"  Static verify: {static_result}")
        if score_data:
            print(f"  Score: {score_data.get('score', '?')}/10")
        verification = {
            "mode": "static_only", "status": static_result, "shapes_verified": [],
        }

        if args.runtime:
            print("  Running simulator in Docker...")
            rt = run_docker_verify(kernel, project)
            verification = rt
            print(f"  Runtime: {rt['status']}")
    else:
        print("  No kernel found — generation failed")

    try:
        kernel_rel = str(kernel.relative_to(project)) if kernel else ""
    except ValueError:
        kernel_rel = str(kernel) if kernel else ""

    evidence: dict = {
        "schema_version": "2",
        "kind": "generative",
        "operation": args.op,
        "dtype": args.dtype,
        "prompt": prompt,
        "kernel_path": kernel_rel,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent": {
            "platform": "opencode",
            "timeout_s": args.timeout,
            "completed": agent_completed,
            "artifacts_found": artifacts,
        },
        "verification": verification,
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
    out_name = f"{args.op}-{dtype_short}-generative.json"
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

    if not args.keep_project:
        shutil.rmtree(project, ignore_errors=True)
        print("  Cleaned up project directory")
    else:
        print(f"  Project kept at: {project}")

    overall_pass = (
        static_result == "pass"
        and evidence["score"]["accepted"]
        and agent_completed
        and kernel is not None
    )
    print(f"  Overall: {'pass' if overall_pass else 'fail'}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
