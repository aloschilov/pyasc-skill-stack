"""Serialized private CANNBench evaluation queue for worker candidates.

Candidates are staged into a self-contained submission directory and sent to
the official CANNBench control-plane API. This replaces the retired GitCode
pull-request-comment transport completely.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


WORKERS_DIR = Path(__file__).resolve().parent
CANNBENCH_DIR = WORKERS_DIR.parent
REPO_ROOT = CANNBENCH_DIR.parent.parent
SUBMISSION_ROOT = CANNBENCH_DIR / "submission"
SECRETS_FILE = REPO_ROOT / ".secrets" / "cannbench.env"

TERMINAL_STATUSES = {
    "succeeded",
    "correctness_failed",
    "compile_failed",
    "failed",
    "cancelled",
    "canceled",
}


class RemoteError(RuntimeError):
    """The site rejected or failed a private evaluation."""


def _load_secret_env() -> None:
    """Load the local gitignored BenchSite environment without echoing it."""
    if os.environ.get("BENCHSITE_API_TOKEN") or not SECRETS_FILE.exists():
        return
    for raw in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _import_site_client():
    """Import the official adapter from the project-local virtualenv."""
    candidates = sorted(
        (REPO_ROOT / ".tools" / "benchsite-mcp" / "lib").glob(
            "python*/site-packages"
        )
    )
    for candidate in candidates:
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from benchsite_mcp.client import BenchSiteClient  # type: ignore
        from benchsite_mcp.zip import prepare_submission_zip  # type: ignore
    except ImportError as exc:
        raise RemoteError(
            "BenchSite adapter is unavailable; run "
            "integrations/cannbench/setup-site-mcp.sh"
        ) from exc
    return BenchSiteClient, prepare_submission_zip


def _unwrap_job(payload: dict) -> dict:
    job = payload.get("job", payload)
    return job if isinstance(job, dict) else {}


def parse_site_job(payload: dict) -> dict:
    """Convert a BenchSite job payload to the worker feedback contract."""
    job = _unwrap_job(payload)
    results = job.get("results") or {}
    operators = results.get("operators") or []
    if not operators:
        return {
            "hard_failure": True,
            "log_tail": (
                f"site job {job.get('id', '?')} ended as "
                f"{job.get('status', 'unknown')}: "
                f"{job.get('error_message') or job.get('error_code') or 'no report'}"
            ),
            "job_id": job.get("id"),
            "submission_id": job.get("submission_id"),
        }

    operator = operators[0]
    digest = {
        "hard_failure": False,
        "score": round(operator.get("score") or 0.0, 2),
        "compile": round(operator.get("compile_runtime_score") or 0.0, 2),
        "accuracy": round(operator.get("function_score") or 0.0, 2),
        "perf": round(operator.get("performance_score") or 0.0, 2),
        "avg_speedup": round(operator.get("avg_speedup") or 0.0, 3),
        "passed": operator.get("passed_cases") or 0,
        "total": operator.get("total_cases") or 0,
        "anti_cheat_failed": (results.get("summary") or {}).get(
            "anti_cheat_failed_cases", 0
        ),
        "failed_cases": [],
        "timings": [],
        "job_id": job.get("id"),
        "submission_id": job.get("submission_id")
        or (job.get("submission") or {}).get("id"),
    }
    for case in operator.get("cases") or []:
        if case.get("status") != "success":
            error = (case.get("error_msg") or "unknown").splitlines()
            digest["failed_cases"].append(
                {
                    "case_id": case.get("case_id"),
                    "error": " | ".join(error[:3])[:400],
                }
            )
        else:
            digest["timings"].append(
                {
                    "case_id": str(case.get("case_id", "")).split("/")[-1],
                    "elapsed_us": case.get("elapsed_us"),
                    "baseline_us": case.get("baseline_perf_us"),
                    "speedup": round(case.get("speedup") or 0.0, 3),
                }
            )
    return digest


class EvalQueue:
    """Run candidate evaluations as private, credit-aware BenchSite jobs."""

    def __init__(self) -> None:
        _load_secret_env()
        if not os.environ.get("BENCHSITE_API_TOKEN"):
            raise RemoteError(
                f"BENCHSITE_API_TOKEN is missing from the environment or {SECRETS_FILE}"
            )
        client_cls, self._prepare_zip = _import_site_client()
        self._client = client_cls()
        self.lock = threading.Lock()
        self.target_hardware = os.environ.get(
            "CANNBENCH_TARGET_HARDWARE", "950pr"
        )
        self.benchmark_slug = os.environ.get(
            "CANNBENCH_BENCHMARK_SLUG", "official-tasks"
        )
        self.poll_interval = float(os.environ.get("CANNBENCH_POLL_INTERVAL", "15"))
        self.timeout = int(os.environ.get("CANNBENCH_JOB_TIMEOUT", "1800"))
        self.upload_timeout = int(
            os.environ.get("CANNBENCH_UPLOAD_TIMEOUT", "3600")
        )

    def _check_credit(self) -> None:
        credits = self._client.get_credits().get("credits") or {}
        if not credits.get("unlimited") and int(credits.get("remaining") or 0) < 1:
            raise RemoteError("CANNBench has no submission credits remaining")

    def _submit_streaming(
        self, zip_path: str, ops: str | list[str], tag: str
    ) -> dict:
        """Upload without the adapter's fixed 120-second in-memory timeout.

        The official adapter constructs the whole multipart body in memory and
        restarts it after 120 seconds.  Self-contained pyasc submissions can
        legitimately take longer on a constrained uplink, so curl streams the
        same documented form fields while keeping the bearer token out of the
        process argument list.
        """
        token = self._client.token
        selected_ops = [ops] if isinstance(ops, str) else list(ops)
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="cannbench-headers-", encoding="utf-8"
        ) as headers:
            headers.write(f"Authorization: Bearer {token}\n")
            headers.write("X-BenchSite-Client: pyasc-skill-stack/1\n")
            headers.write("X-BenchSite-MCP-Tool: submit_kernel\n")
            headers.flush()
            command = [
                "curl",
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--connect-timeout",
                "30",
                "--max-time",
                str(self.upload_timeout),
                # A submission POST is not safe to retry after an ambiguous
                # proxy disconnect: the server may have accepted the first
                # body even when the response never reached this client.
                "--retry",
                "0",
                "--header",
                f"@{headers.name}",
                "--form-string",
                "source=mcp",
                "--form-string",
                f"target_hardware={self.target_hardware}",
                "--form-string",
                f"selected_operators={json.dumps(selected_ops)}",
                "--form-string",
                f"benchmark_slug={self.benchmark_slug}",
                "--form-string",
                f"job_tag={tag}",
                "--form-string",
                "is_private=true",
                "--form",
                f"file=@{zip_path};type=application/octet-stream",
                f"{self._client.base_url}/api/submissions",
            ]
            resolve_ip = os.environ.get("CANNBENCH_RESOLVE_IP", "").strip()
            if resolve_ip:
                command[1:1] = [
                    "--resolve",
                    f"cannbench.com:443:{resolve_ip}",
                ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.upload_timeout + 30,
            )
        if completed.returncode:
            detail = (completed.stdout or completed.stderr).strip()
            raise RemoteError(
                f"CANNBench streaming upload failed (curl rc={completed.returncode}): "
                f"{detail[:800]}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RemoteError("CANNBench upload returned invalid JSON") from exc

    @staticmethod
    def _stage_submission(candidate: Path, init_source: str, op: str) -> Path:
        stage = Path(tempfile.mkdtemp(prefix=f"cannbench-{op}-"))
        shutil.copytree(
            SUBMISSION_ROOT,
            stage,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "dist", "build", ".base-dist", "*.egg-info", "__pycache__"
            ),
        )
        shutil.copy2(candidate, stage / "cann_bench" / f"{op}.py")
        (stage / "cann_bench" / "__init__.py").write_text(
            init_source, encoding="utf-8"
        )
        return stage

    def evaluate(
        self,
        op: str,
        candidate: Path,
        init_source: str,
        workdir: Path,
    ) -> dict:
        """Submit one candidate privately and return its score digest."""
        self._check_credit()
        stage = self._stage_submission(candidate, init_source, op)
        tag = f"pyasc-v2-worker-{op}-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            with self._prepare_zip(str(stage)) as zip_path:
                submitted = self._submit_streaming(zip_path, op, tag)
            (workdir / "site_submission.json").write_text(
                json.dumps(submitted, indent=2), encoding="utf-8"
            )
            job_id = submitted.get("job_id") or (submitted.get("job") or {}).get("id")
            if not job_id:
                raise RemoteError(f"submission returned no job id: {submitted}")

            deadline = time.monotonic() + self.timeout
            payload = {}
            while time.monotonic() < deadline:
                payload = self._client.get_job(job_id)
                job = _unwrap_job(payload)
                if job.get("status") in TERMINAL_STATUSES or job.get("job_completed"):
                    break
                time.sleep(self.poll_interval)
            else:
                raise RemoteError(f"site job {job_id} timed out after {self.timeout}s")

            (workdir / "site_job.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            try:
                logs = self._client.get_job_logs(job_id)
                (workdir / "site_logs.json").write_text(
                    json.dumps(logs, indent=2), encoding="utf-8"
                )
            except Exception as exc:  # logs are evidence, not job truth
                (workdir / "site_logs.error.txt").write_text(
                    str(exc), encoding="utf-8"
                )

            digest = parse_site_job(payload)
            job = _unwrap_job(payload)
            results = job.get("results")
            if results:
                (workdir / "report.json").write_text(
                    json.dumps(results, indent=2), encoding="utf-8"
                )
            return digest
        finally:
            shutil.rmtree(stage, ignore_errors=True)
