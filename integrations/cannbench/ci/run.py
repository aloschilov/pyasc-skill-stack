#!/usr/bin/env python3
"""Full catalog generation, installed evaluator-wheel qualification and hardware gate.

Schedules issue only GETs. --submit authorizes one full-catalog POST after
complete skill/compiler evidence and installed-wheel qualification.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from catalog import SiteClient, snapshot
from generation_gate import sha256
from hardware_gate import evaluate

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_vm_credentials(path: Path):
    if os.environ.get("BENCHSITE_API_TOKEN"):
        return
    for raw in path.read_text().splitlines():
        line = raw.strip().removeprefix("export ")
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def retain_job(payload):
    job = payload.get("job", payload)
    fields = ("id", "status", "benchmark_slug", "benchmark_version", "benchmark", "target_hardware",
              "runner_hardware", "selected_operators", "queued_at", "finished_at", "results", "submission_id", "job_tag")
    return {k: job[k] for k in fields if k in job}


def verify_binding(catalog, state, job_ids, repo_root):
    if not state or job_ids != [state.get("job_id")] or state.get("phase") not in ("submitted", "finished"):
        return False
    bundle = state["bundle"]
    if (bundle["catalog_sha256"] != catalog["catalog_sha256"] or bundle["benchmark_version"] != catalog["benchmark_version"]
            or set(bundle["operators"]) != {s["function_name"] for s in catalog["operators"]}):
        return False
    if not state.get("archive_path") or sha256(state["archive_path"]) != state["archive_sha256"]:
        return False
    if not bundle.get("generation_sources"):
        return False
    for relative, digest in bundle["generation_sources"].items():
        file = (repo_root / relative).resolve()
        if not file.is_relative_to(repo_root.resolve()) or not file.is_file() or sha256(file) != digest:
            return False
    import zipfile
    with zipfile.ZipFile(state["archive_path"]) as archive:
        if json.loads(archive.read("BUNDLE.json")) != bundle:
            return False
    return True


def qualify_bundle(bundle, catalog_file, output):
    from vm_locks import shared_locks
    runtime = json.loads((REPO_ROOT / "integrations/cannbench/ci/runtime.json").read_text())
    with shared_locks("execution"):
        actual = subprocess.run(["docker", "image", "inspect", runtime["compile_image"], "--format", "{{.Id}}"],
                                capture_output=True, text=True, check=True).stdout.strip()
        if actual != runtime["compile_image_id"]:
            raise ValueError("Compiler image changed; requalify its runtime before use")
        command = ["docker", "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "python",
                   "--volume", str(REPO_ROOT) + ":/workspace:ro", "--volume", str(output.resolve()) + ":/qualification",
                   "--workdir", "/workspace", runtime["compile_image"], "integrations/cannbench/ci/qualify_bundle.py",
                   "--bundle", str(bundle.relative_to(REPO_ROOT)), "--catalog", str(catalog_file.relative_to(REPO_ROOT)),
                   "--output", "/qualification/qualification.json"]
        with (output / "qualification.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError("Installed evaluator wheel qualification failed")
    import shutil
    shutil.copy2(output / "qualification.json", bundle / "qualification.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path, default=Path("/home/aloschilov/workspace/pyasc-skill-stack/.secrets/cannbench.env"))
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--state-file", type=Path, help="Resume an exact durable submission using GETs only")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--submit", action="store_true", help="One full-catalog job after all qualification gates")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    if args.submit and not args.generate:
        parser.error("--submit requires a fresh full-catalog --generate campaign")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(REPO_ROOT):
        parser.error("Campaign output must be inside the scoped VM checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    ids = args.job_id or [x.strip() for x in os.environ.get("CANNBENCH_JOB_IDS", "").split(",") if x.strip()]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"job_[a-z0-9]+", x) for x in ids):
        parser.error("Invalid or duplicate explicit CANNBench job IDs")
    catalog = None; payloads = []; generation_exit = None; state = None; failure = None
    try:
        load_vm_credentials(args.credentials_file)
        client = SiteClient()
        catalog_root = args.output / "catalog"
        catalog = snapshot(client, "official-tasks", catalog_root)
        generation = args.output / "generation"
        if args.generate:
            command = [sys.executable, "integrations/cannbench/workers/driver.py", "--catalog", str(catalog_root / "manifest.json"),
                       "--output-dir", str(generation), "--items", "all", "--evaluation", "local",
                       "--models", "gateway/glm-5.3,gateway/qwen3.8-max", "--workers", str(args.workers), "--iterations", str(args.iterations)]
            with (args.output / "generation.log").open("w") as log:
                generation_exit = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if args.submit:
            if generation_exit != 0:
                raise ValueError("Generation/compile failed; no submission was created")
            from package_bundle import package_bundle
            from submission import submit
            bundle = args.output / "bundle"
            package_bundle(catalog, generation, REPO_ROOT, bundle)
            qualify_bundle(bundle, catalog_root / "manifest.json", args.output)
            payload, state = submit(bundle, catalog, REPO_ROOT)
            ids = [state["job_id"]]; payloads = [retain_job(payload)]
        else:
            from submission import STATE_ROOT, resume
            selected = args.state_file
            if selected is None:
                selected = next((p for p in (STATE_ROOT / "pending.json", STATE_ROOT / "latest.json") if p.is_file()), None)
            if selected and not ids:
                payload, state = resume(selected, REPO_ROOT)
                ids = [state["job_id"]]; payloads = [retain_job(payload)]
            else:
                if selected:
                    state = json.loads(selected.read_text())
                payloads = [retain_job(client.get("/api/jobs/" + job_id)) for job_id in ids]
    except Exception as exc:
        # No response bodies, download URLs or credentials enter public reports.
        failure = type(exc).__name__
    if catalog is None:
        catalog = json.loads((REPO_ROOT / "integrations/cannbench/catalogs/official-tasks-1.1.2-20261001/manifest.json").read_text())
    jobs = args.output / "jobs"; jobs.mkdir()
    sources = {}
    for payload in payloads:
        path = jobs / (payload["id"] + ".json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        sources[path.name] = sha256(path)
    report = evaluate(catalog, payloads)
    try:
        bound = verify_binding(catalog, state, ids, REPO_ROOT)
    except (KeyError, ValueError, OSError):
        bound = False
    report["job_file_sha256"] = sources
    report["skill_stack_gate_passed"] = report["gate_passed"] and bound and not failure and generation_exit in (None, 0)
    report["generation_provenance"] = "Bound to exact generated sources, skills, installed evaluator wheel and durable submission" if bound else "No complete skill-generated submission identity binding"
    report["generation_requested"] = args.generate
    report["generation_exit"] = generation_exit
    if failure:
        report["gate_passed"] = False; report["errors"].append("Campaign stage failed: " + failure)
    (args.output / "hardware-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output / "run.json").write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
        "job_ids": ids, "generation_requested": args.generate, "generation_exit": generation_exit,
        "hardware_gate_passed": report["gate_passed"], "skill_stack_gate_passed": report["skill_stack_gate_passed"], "failure": failure}, indent=2) + "\n")
    if report["skill_stack_gate_passed"] and state:
        from submission import STATE_ROOT, save
        save(STATE_ROOT / "latest.json", state)
    print(f"Hardware {report['correct_cases']}/{report['required_cases']}; skill stack accepted={report['skill_stack_gate_passed']}")
    return 0 if report["skill_stack_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
