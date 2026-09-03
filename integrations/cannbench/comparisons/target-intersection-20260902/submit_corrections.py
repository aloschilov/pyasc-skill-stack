#!/usr/bin/env python3
"""Idempotently submit and monitor the two corrective private runs."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
sys.path.insert(0, str(REPO_ROOT / "integrations/cannbench/workers"))

from evalqueue import EvalQueue, TERMINAL_STATUSES, _unwrap_job  # noqa: E402


ORDER = ("target_corrected", "with_skills_corrected")
OPS = ["gelu", "foreach_addcdiv_scalar"]
ACTIVE = {
    "queued", "compiling", "correctness", "performance", "archiving", "running"
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    evidence = ROOT / "correction/remote_runs"
    evidence.mkdir(parents=True, exist_ok=True)
    queue = EvalQueue()

    payload = queue._client.list_jobs(
        limit=200, benchmark_slug=queue.benchmark_slug
    )
    active = [
        job["id"] for job in payload.get("jobs", [])
        if job.get("status") in ACTIVE and not job.get("job_completed")
    ]
    write_json(evidence / "preexisting_jobs.json", {"active_job_ids": active})
    if active:
        print(f"pre-existing active jobs: {active}", file=sys.stderr)
        return 4

    missing = []
    records = []
    for index, variant in enumerate(ORDER, start=1):
        path = evidence / f"{index:02d}-{variant}-submission.json"
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append((index, variant))

    credits_payload = queue._client.get_credits()
    write_json(evidence / "credits_before.json", credits_payload)
    credits = credits_payload.get("credits") or {}
    remaining = (
        999999 if credits.get("unlimited")
        else int(credits.get("remaining") or 0)
    )
    required = len(missing) * len(OPS)
    if remaining < required:
        write_json(
            evidence / "deferred.json",
            {
                "reason": "insufficient_credits",
                "required": required,
                "remaining": remaining,
                "resets_at": credits.get("resets_at"),
                "missing_variants": [variant for _, variant in missing],
            },
        )
        print(
            f"need {required} credits, have {remaining}; reset "
            f"{credits.get('resets_at')}",
            file=sys.stderr,
        )
        return 3

    for index, variant in missing:
        archive = ROOT / "correction/submissions" / f"{variant}.zip"
        tag = f"pyasc-v2-target-intersection-correction-{index:02d}-{variant}-20260902"
        response = queue._submit_streaming(str(archive), OPS, tag)
        record = {
            "order": index,
            "variant": variant,
            "tag": tag,
            "selected_operators": OPS,
            "response": response,
        }
        write_json(
            evidence / f"{index:02d}-{variant}-submission.json", record
        )
        records.append(record)
        print(json.dumps({"variant": variant, "response": response}), flush=True)

    pending = {}
    for record in records:
        response = record["response"]
        job_id = response.get("job_id") or (response.get("job") or {}).get("id")
        if not job_id:
            raise RuntimeError(f"submission returned no job id: {record}")
        job_path = evidence / f"{record['order']:02d}-{record['variant']}-job.json"
        if not job_path.is_file():
            pending[job_id] = record

    deadline = time.monotonic() + 10800
    while pending and time.monotonic() < deadline:
        for job_id, record in tuple(pending.items()):
            payload = queue._client.get_job(job_id)
            job = _unwrap_job(payload)
            print(
                json.dumps({"job_id": job_id, "status": job.get("status")}),
                flush=True,
            )
            if job.get("status") in TERMINAL_STATUSES or job.get("job_completed"):
                stem = f"{record['order']:02d}-{record['variant']}"
                write_json(evidence / f"{stem}-job.json", payload)
                try:
                    write_json(
                        evidence / f"{stem}-logs.json",
                        queue._client.get_job_logs(job_id),
                    )
                except Exception as exc:
                    (evidence / f"{stem}-logs.error.txt").write_text(
                        str(exc), encoding="utf-8"
                    )
                del pending[job_id]
        if pending:
            time.sleep(max(queue.poll_interval, 20))

    if pending:
        raise RuntimeError(f"corrective jobs timed out: {sorted(pending)}")
    write_json(evidence / "credits_after.json", queue._client.get_credits())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
