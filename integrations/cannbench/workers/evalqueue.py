"""Private streaming upload adapter; smoke.py owns the durable submission ledger.

This helper never stages candidates, polls jobs or retries uploads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


WORKERS_DIR = Path(__file__).resolve().parent
CANNBENCH_DIR = WORKERS_DIR.parent
REPO_ROOT = CANNBENCH_DIR.parent.parent
SECRETS_FILE = REPO_ROOT / ".secrets" / "cannbench.env"

class RemoteError(RuntimeError):
    """The site rejected or failed a private evaluation."""


class SubmissionRejected(RemoteError):
    """An explicit HTTP 4xx rejected this upload before a job was accepted."""


def load_credentials(path: Path) -> None:
    """Load the local gitignored BenchSite environment without echoing it."""
    if os.environ.get("BENCHSITE_API_TOKEN"):
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
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
    except ImportError as exc:
        raise RemoteError(
            "BenchSite adapter is unavailable; run "
            "integrations/cannbench/setup-site-mcp.sh"
        ) from exc
    return BenchSiteClient


class EvalQueue:
    """Run candidate evaluations as private, credit-aware BenchSite jobs."""

    def __init__(self) -> None:
        if SECRETS_FILE.exists():
            load_credentials(SECRETS_FILE)
        if not os.environ.get("BENCHSITE_API_TOKEN"):
            raise RemoteError(
                f"BENCHSITE_API_TOKEN is missing from the environment or {SECRETS_FILE}"
            )
        client_cls = _import_site_client()
        self._client = client_cls()
        self.target_hardware = os.environ.get(
            "CANNBENCH_TARGET_HARDWARE", "950pr"
        )
        self.benchmark_slug = os.environ.get(
            "CANNBENCH_BENCHMARK_SLUG", "official-tasks"
        )
        self.upload_timeout = int(
            os.environ.get("CANNBENCH_UPLOAD_TIMEOUT", "3600")
        )

    def _check_credit(self) -> None:
        credits = self._client.get_credits().get("credits") or {}
        if not credits.get("unlimited") and int(credits.get("remaining") or 0) < 1:
            raise RemoteError("CANNBench has no submission credits remaining")

    def _submit_streaming(
        self, zip_path: str, ops: str | list[str], tag: str, aggregation_token: str | None = None
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
                "--write-out",
                "\n%{http_code}",
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
            if aggregation_token:
                command[1:1] = ["--form-string", f"aggregation_token={aggregation_token}"]
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
        body, _, status = completed.stdout.rpartition("\n")
        if completed.returncode:
            detail = (body or completed.stderr).strip()
            error = SubmissionRejected if completed.returncode == 22 and status in ("400", "401", "403", "413", "422") else RemoteError
            raise error(f"CANNBench upload failed (HTTP {status}, curl rc={completed.returncode}): {detail[:800]}")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RemoteError("CANNBench upload returned invalid JSON") from exc
