"""Atomic state writes and GET-only reconciliation for the smoke ledger."""
import json
import os
from pathlib import Path
import tempfile

TERMINAL = {"succeeded", "failed", "correctness_failed", "compile_failed", "cancelled", "canceled"}


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
