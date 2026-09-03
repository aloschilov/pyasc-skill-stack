#!/usr/bin/env python3
"""Submit and monitor the three private target-intersection runs in order."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "integrations/cannbench/workers"))

from evalqueue import EvalQueue, TERMINAL_STATUSES, _unwrap_job  # noqa: E402


ORDER = ("handwritten", "no_skills", "with_skills")
OPS = ["gelu", "foreach_addcdiv_scalar"]
ACTIVE = {"queued", "compiling", "correctness", "performance", "archiving", "running"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def current_jobs(queue: EvalQueue) -> list[dict]:
    payload = queue._client.list_jobs(limit=200, benchmark_slug=queue.benchmark_slug)
    return payload.get("jobs") or []


def wait_for_preexisting(queue: EvalQueue, evidence: Path) -> list[str]:
    initial = current_jobs(queue)
    ids = [job["id"] for job in initial if job.get("status") in ACTIVE]
    write_json(evidence / "preexisting_jobs.json", {"active_job_ids": ids, "jobs": initial})
    deadline = time.monotonic() + 7200
    pending = set(ids)
    while pending and time.monotonic() < deadline:
        for job_id in tuple(pending):
            job = _unwrap_job(queue._client.get_job(job_id))
            if job.get("status") in TERMINAL_STATUSES or job.get("job_completed"):
                pending.remove(job_id)
        if pending:
            time.sleep(queue.poll_interval)
    if pending:
        raise RuntimeError(f"pre-existing jobs did not finish: {sorted(pending)}")
    return ids


def main() -> int:
    evidence = ROOT / "remote_runs"
    evidence.mkdir(exist_ok=True)
    queue = EvalQueue()
    submissions = []
    missing = []
    for index, variant in enumerate(ORDER, start=1):
        path = evidence / f"{index:02d}-{variant}-submission.json"
        if path.is_file():
            submissions.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append((index, variant))
    if not missing and all(
        (evidence / f"{record['order']:02d}-{record['variant']}-job.json").is_file()
        for record in submissions
    ):
        print("all three comparison runs are already complete")
        return 0

    credits_payload = queue._client.get_credits()
    credits = credits_payload.get("credits") or {}
    write_json(evidence / "credits_before.json", credits_payload)
    remaining = 999999 if credits.get("unlimited") else int(credits.get("remaining") or 0)
    if remaining < len(missing):
        print(f"need {len(missing)} credits, currently {remaining}", file=sys.stderr)
        return 3

    preexisting = wait_for_preexisting(queue, evidence) if not submissions else []
    for index, variant in missing:
        archive = ROOT / "submissions" / f"{variant}.zip"
        tag = f"pyasc-v2-target-intersection-{index:02d}-{variant}-20260902"
        response = queue._submit_streaming(str(archive), OPS, tag)
        record = {
            "order": index,
            "variant": variant,
            "tag": tag,
            "selected_operators": OPS,
            "preexisting_job_ids": preexisting,
            "response": response,
        }
        write_json(evidence / f"{index:02d}-{variant}-submission.json", record)
        submissions.append(record)
        print(json.dumps({"variant": variant, "response": response}), flush=True)

    pending = {}
    for record in submissions:
        response = record["response"]
        job_id = response.get("job_id") or (response.get("job") or {}).get("id")
        if not job_id:
            raise RuntimeError(f"submission returned no job id: {record}")
        pending[job_id] = record
    deadline = time.monotonic() + 7200
    while pending and time.monotonic() < deadline:
        for job_id, record in tuple(pending.items()):
            payload = queue._client.get_job(job_id)
            job = _unwrap_job(payload)
            if job.get("status") in TERMINAL_STATUSES or job.get("job_completed"):
                variant = record["variant"]
                write_json(evidence / f"{record['order']:02d}-{variant}-job.json", payload)
                try:
                    logs = queue._client.get_job_logs(job_id)
                    write_json(evidence / f"{record['order']:02d}-{variant}-logs.json", logs)
                except Exception as exc:
                    (evidence / f"{record['order']:02d}-{variant}-logs.error.txt").write_text(
                        str(exc), encoding="utf-8")
                del pending[job_id]
        if pending:
            time.sleep(queue.poll_interval)
    if pending:
        raise RuntimeError(f"comparison jobs timed out: {sorted(pending)}")
    write_json(evidence / "credits_after.json", queue._client.get_credits())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
