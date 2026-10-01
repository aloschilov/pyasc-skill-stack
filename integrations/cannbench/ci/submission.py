"""Durable submission primitives; ambiguous POSTs are reconciled by GET."""
import json
import os
from pathlib import Path
import sys
import time
import tempfile
import shutil
import zipfile

from generation_gate import sha256
from vm_locks import shared_locks, VM_ROOT

STATE_ROOT = Path("/home/aloschilov/ci-campaign-state/pyasc-cannbench")
TERMINAL = {"succeeded", "failed", "correctness_failed", "compile_failed", "cancelled", "canceled"}


def site_queue(repo_root):
    # The official adapter stays in its existing VM environment. A fresh
    # Actions checkout must not copy its credentials or entire .tools tree.
    for directory in (VM_ROOT / ".tools/benchsite-mcp/lib").glob("python*/site-packages"):
        sys.path.append(str(directory))
    sys.path.insert(0, str(repo_root / "integrations/cannbench/workers"))
    from evalqueue import EvalQueue
    return EvalQueue()


def save(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as output:
        output.write(json.dumps(data, indent=2) + "\n")
        output.flush(); os.fsync(output.fileno())
        temporary = Path(output.name)
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reconcile(client, state):
    """No empty GET result ever authorizes another POST."""
    payload = client.list_jobs(limit=100, benchmark_slug=state["benchmark_slug"])
    jobs = payload.get("jobs", [])
    matches = [job for job in jobs if job.get("job_tag") == state["job_tag"]]
    if len(matches) != 1:
        raise RuntimeError("Ambiguous submission requires GET reconciliation; POST will not be repeated")
    return matches[0]["id"]


def submit(bundle_root, catalog, repo_root, timeout=43200):
    queue = site_queue(repo_root)
    manifest = json.loads((bundle_root / "BUNDLE.json").read_text())
    qualification = json.loads((bundle_root / "qualification.json").read_text())
    if (qualification.get("status") != "passed" or qualification.get("wheel_sha256") != manifest["wheel_sha256"]
            or qualification.get("catalog_sha256") != catalog["catalog_sha256"]):
        raise ValueError("Exact evaluator wheel is not qualified")
    archive = bundle_root.parent / "submission.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
            for path in sorted(bundle_root.rglob("*")):
                if path.is_file():
                    target.write(path, str(path.relative_to(bundle_root)))
    digest = sha256(archive)
    with zipfile.ZipFile(archive) as uploaded:
        expected = {str(p.relative_to(bundle_root)): p for p in bundle_root.rglob("*") if p.is_file()}
        if set(uploaded.namelist()) != set(expected) or any(uploaded.read(name) != file.read_bytes() for name, file in expected.items()):
            raise ValueError("Existing archive differs from the qualified bundle")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    path = STATE_ROOT / (digest + ".json")
    with shared_locks("upload"):
        if path.exists():
            state = json.loads(path.read_text())
            if state["archive_sha256"] != digest:
                raise ValueError("Durable submission identity mismatch")
            job_id = state.get("job_id") or reconcile(queue._client, state)
        else:
            credits = queue._client.get_credits().get("credits") or {}
            if credits.get("unlimited") is not True and int(credits.get("remaining") or 0) < len(catalog["operators"]):
                raise RuntimeError("Insufficient credits for the full catalog; use quota-bounded campaign batches")
            recent = queue._client.list_jobs(limit=100).get("jobs", [])
            if any(j.get("status") in ("queued", "running", "compiling", "evaluating") for j in recent):
                raise RuntimeError("Another CANNBench job is active; no submission was created")
            archives = STATE_ROOT / "archives"; archives.mkdir(exist_ok=True)
            durable_archive = archives / (digest + ".zip")
            if not durable_archive.exists():
                with durable_archive.open("xb") as target, archive.open("rb") as source:
                    shutil.copyfileobj(source, target)
                    target.flush(); os.fsync(target.fileno())
                descriptor = os.open(archives, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            if sha256(durable_archive) != digest:
                raise ValueError("Durable archive hash mismatch")
            state = {"phase": "upload-intent", "archive_sha256": digest, "bundle": manifest,
                     "archive_path": str(durable_archive.resolve()),
                     "benchmark_slug": catalog["benchmark_slug"], "benchmark_version": catalog["benchmark_version"],
                     "job_tag": "pyasc-ci-" + digest[:24], "job_id": None}
            save(path, state)  # Durable intent precedes the only possible POST.
            save(STATE_ROOT / "pending.json", state)
            try:
                posted = queue._submit_streaming(str(durable_archive), [op["function_name"] for op in catalog["operators"]], state["job_tag"])
                job_id = posted.get("job_id") or (posted.get("job") or {}).get("id")
                if not job_id:
                    raise RuntimeError("Submission response has no hardware job identity")
                state["submission_id"] = posted.get("submission_id") or (posted.get("submission") or {}).get("id")
            except Exception:
                state["phase"] = "ambiguous"; save(path, state)
                save(STATE_ROOT / "pending.json", state)
                job_id = reconcile(queue._client, state)
        state.update(phase="submitted", job_id=job_id)
        save(path, state)
        save(STATE_ROOT / "pending.json", state)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = queue._client.get_job(job_id)
        job = payload.get("job", payload)
        if job.get("status") in TERMINAL:
            state["phase"] = "finished"; state["status"] = job["status"]
            save(path, state)
            save(STATE_ROOT / "pending.json", state)
            return payload, state
        time.sleep(30)
    raise RuntimeError("Hardware job remains active; resume by GET using durable job identity")


def resume(state_file, repo_root):
    """Refresh a durable intent or job using GETs only, even after cancellation."""
    state = json.loads(state_file.read_text())
    queue = site_queue(repo_root)
    if not state.get("job_id"):
        state["job_id"] = reconcile(queue._client, state)
    payload = queue._client.get_job(state["job_id"])
    status = payload.get("job", payload).get("status")
    state.update(phase="finished" if status in TERMINAL else "submitted", status=status)
    with shared_locks("upload"):
        save(STATE_ROOT / (state["archive_sha256"] + ".json"), state)
        save(STATE_ROOT / "pending.json", state)
    return payload, state
