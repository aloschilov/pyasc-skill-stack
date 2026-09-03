#!/usr/bin/env python3
"""Submit prepared arms in fixed order as credits become available."""

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
OPS = ["gelu", "rms_norm", "softmax", "transpose"]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    evidence = ROOT / "remote_runs"
    evidence.mkdir(exist_ok=True)
    queue = EvalQueue()
    submitted_now = 0
    for index, arm in enumerate(ORDER, start=1):
        submission_file = evidence / f"{index:02d}-{arm}-submission.json"
        job_file = evidence / f"{index:02d}-{arm}-job.json"
        if job_file.is_file():
            continue
        if submission_file.is_file():
            record = json.loads(submission_file.read_text(encoding="utf-8"))
            job_id = record["job_id"]
        else:
            credits = queue._client.get_credits()
            write_json(evidence / "credits-latest.json", credits)
            info = credits.get("credits") or {}
            remaining = 999999 if info.get("unlimited") else int(info.get("remaining") or 0)
            if remaining < len(OPS):
                print(json.dumps({
                    "status": "waiting_for_credits",
                    "next_arm": arm,
                    "required": len(OPS),
                    "remaining": remaining,
                    "resets_at": info.get("resets_at"),
                    "submitted_now": submitted_now,
                }))
                return 3
            archive = ROOT / "submissions" / f"{arm}.zip"
            response = queue._submit_streaming(
                str(archive), OPS,
                f"pyasc-v2-current-four-op-{index:02d}-{arm}-20260903",
            )
            job_id = response.get("job_id") or (response.get("job") or {}).get("id")
            if not job_id:
                raise RuntimeError(f"submission returned no job id: {response}")
            record = {
                "order": index,
                "arm": arm,
                "job_id": job_id,
                "operators": OPS,
                "runtime_commit": "030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d",
                "response": response,
            }
            write_json(submission_file, record)
            submitted_now += 1
            print(json.dumps({"submitted": arm, "job_id": job_id}), flush=True)
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            payload = queue._client.get_job(job_id)
            job = _unwrap_job(payload)
            if job.get("status") in TERMINAL_STATUSES or job.get("job_completed"):
                write_json(job_file, payload)
                try:
                    write_json(evidence / f"{index:02d}-{arm}-logs.json", queue._client.get_job_logs(job_id))
                except Exception as exc:
                    (evidence / f"{index:02d}-{arm}-logs.error.txt").write_text(str(exc), encoding="utf-8")
                if job.get("results"):
                    write_json(evidence / f"{index:02d}-{arm}-results.json", job["results"])
                print(json.dumps({"completed": arm, "job_id": job_id, "status": job.get("status")}), flush=True)
                break
            time.sleep(queue.poll_interval)
        else:
            raise RuntimeError(f"job timed out: {job_id}")
    write_json(evidence / "credits-after.json", queue._client.get_credits())
    print(json.dumps({"status": "all_complete", "submitted_now": submitted_now}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
