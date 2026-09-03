#!/usr/bin/env python3
"""Submit or resume the current single-op GeLU iteration idempotently."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO / "integrations/cannbench/workers"))

from evalqueue import EvalQueue, TERMINAL_STATUSES, _unwrap_job  # noqa: E402


ITERATION = "iteration-03-lowlevel-tanh-safe-tile13824"
COMMIT = "0a631f70968c3cb7c33ce45330a85768dd5a6f06"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sanitized(value):
    if isinstance(value, dict):
        return {
            key: sanitized(item) for key, item in value.items()
            if key not in {"aggregation_token", "api_key", "token"}
        }
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    return value


def require_local_evidence(manifest: dict) -> None:
    if manifest.get("pyasc_commit") != COMMIT:
        raise RuntimeError("manifest is not pinned to the current v2 commit")
    report = json.loads(
        (ROOT / "local_validation/gelu-current-v2.json").read_text(encoding="utf-8")
    )
    if not (
        report.get("pyasc_commit") == COMMIT
        and report.get("status") == "passed"
        and report.get("dispatch_passed") == 20
        and report.get("compile_passed") == 20
        and all(
            item.get("status") == "passed"
            and item.get("has_ffts_arg") is False
            for item in report.get("specializations", [])
        )
    ):
        raise RuntimeError(
            "current-v2 20/20 compile/ABI evidence is incomplete"
        )


def main() -> int:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    require_local_evidence(manifest)
    archive = REPO / manifest["submission_zip"]["path"]
    if sha256(archive) != manifest["submission_zip"]["sha256"]:
        raise RuntimeError("submission archive differs from MANIFEST.json")

    evidence = ROOT / "remote_runs" / ITERATION
    submission_path = evidence / "submission.json"
    job_path = evidence / "job.json"
    queue = EvalQueue()
    if job_path.is_file():
        print(json.dumps({"status": "already_terminal", "job": str(job_path)}))
        return 0
    if submission_path.is_file():
        record = json.loads(submission_path.read_text(encoding="utf-8"))
        job_id = record["job_id"]
    else:
        credits = queue._client.get_credits()
        write_json(evidence / "credits-before.json", credits)
        info = credits.get("credits") or {}
        remaining = 999999 if info.get("unlimited") else int(info.get("remaining") or 0)
        if remaining < 1:
            print(json.dumps({
                "status": "waiting_for_credits",
                "remaining": remaining,
                "resets_at": info.get("resets_at"),
            }))
            return 3
        response = queue._submit_streaming(
            str(archive), ["gelu"],
            "pyasc-v2-current-gelu-lowlevel-safe-i03-20260903",
        )
        job_id = response.get("job_id") or (response.get("job") or {}).get("id")
        if not job_id:
            raise RuntimeError(f"submission returned no job id: {response}")
        write_json(submission_path, {
            "iteration": ITERATION,
            "job_id": job_id,
            "operator": "gelu",
            "runtime_commit": COMMIT,
            "archive_sha256": manifest["submission_zip"]["sha256"],
            "response": sanitized(response),
        })
        print(json.dumps({"submitted": ITERATION, "job_id": job_id}), flush=True)

    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        payload = queue._client.get_job(job_id)
        job = _unwrap_job(payload)
        if job.get("status") in TERMINAL_STATUSES or job.get("job_completed"):
            write_json(job_path, sanitized(payload))
            try:
                write_json(evidence / "logs.json", sanitized(queue._client.get_job_logs(job_id)))
            except Exception as exc:
                (evidence / "logs.error.txt").write_text(str(exc), encoding="utf-8")
            if job.get("results"):
                write_json(evidence / "results.json", sanitized(job["results"]))
            write_json(evidence / "credits-after.json", queue._client.get_credits())
            print(json.dumps({
                "completed": ITERATION,
                "job_id": job_id,
                "status": job.get("status"),
            }), flush=True)
            return 0
        time.sleep(queue.poll_interval)
    raise RuntimeError(f"job timed out: {job_id}")


if __name__ == "__main__":
    raise SystemExit(main())
