"""Private streaming upload adapter; smoke.py owns the durable submission ledger.

This helper never stages candidates, polls jobs or retries uploads.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote, quote_plus


WORKERS_DIR = Path(__file__).resolve().parent
CANNBENCH_DIR = WORKERS_DIR.parent
REPO_ROOT = CANNBENCH_DIR.parent.parent
SECRETS_FILE = REPO_ROOT / ".secrets" / "cannbench.env"

class RemoteError(RuntimeError):
    """The site rejected or failed a private evaluation."""


class SubmissionRejected(RemoteError):
    """An explicit HTTP 4xx rejected this upload before a job was accepted."""


def redact_response(text: str, *secrets: str | None) -> str:
    """Redact decoded JSON strings, including alternate escapes and nested keys."""
    variants = set()
    for secret in secrets:
        if secret:
            variants.update((secret, json.dumps(secret)[1:-1], quote(secret, safe=""), quote_plus(secret)))
    sensitive_key = r"token|secret|password|authorization|api_key|credential"

    def redact_text(value):
        for secret in sorted(variants, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r'(?i)Bearer\s+[^\s"<>]+', 'Bearer [REDACTED]', value)
        return re.sub(r'https?://[^\s"<>]+', '[URL]', value)

    def redact_json(value):
        if isinstance(value, str):
            return redact_text(value)
        if isinstance(value, list):
            return [redact_json(item) for item in value]
        if isinstance(value, dict):
            return {redact_text(key): "[REDACTED]" if re.search(sensitive_key, key, re.I) else redact_json(item)
                    for key, item in value.items()}
        return value

    try:
        decoded = json.loads(text)
    except (ValueError, RecursionError):
        # Partial JSON and plain error messages still need escaped credentials
        # removed. Decode valid escape units even in an unterminated string.
        text = re.sub(r'\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt])',
                      lambda match: json.loads('"' + match[0] + '"'), text)
        text = redact_text(text)
        return re.sub(r'(?i)("[^"\n]*(?:' + sensitive_key + r')[^"\n]*"\s*:\s*)"(?:\\.|[^"\\])*"',
                      r'\1"[REDACTED]"', text)
    # Serialize only the scrubbed structure, never the original escaped text.
    return json.dumps(redact_json(decoded), ensure_ascii=True)


def _open_response_receipt(path: Path | None):
    if path is None:
        return nullcontext(None)
    path = Path(path).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents or any((p / '.git').exists() for p in path.parents):
        raise ValueError("Keep HTTP response receipts outside Git checkouts")
    # Reserve before POST: an existing attempt receipt must never be replaced.
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8")


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

    def _check_credit(self, required: int = 1) -> None:
        credits = self._client.get_credits().get("credits") or {}
        if not credits.get("unlimited") and int(credits.get("remaining") or 0) < required:
            raise RemoteError(f"CANNBench requires {required} submission credit(s)")

    def _submit_streaming(
        self, zip_path: str, ops: str | list[str], tag: str, aggregation_token: str | None = None,
        *, benchmark_version: str | None = None, public: bool = False,
        receipt_path: Path | None = None
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
        with _open_response_receipt(receipt_path) as receipt, tempfile.NamedTemporaryFile(
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
                "\n%{http_code} %{size_upload} %{time_total}",
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
                "is_private=false" if public else "is_private=true",
                "--form",
                f"file=@{zip_path};type=application/octet-stream",
                f"{self._client.base_url}/api/submissions",
            ]
            if benchmark_version:
                command[1:1] = ["--form-string", f"benchmark_version={benchmark_version}"]
            if aggregation_token:
                command[1:1] = ["--form-string", f"aggregation_token={aggregation_token}"]
            resolve_ip = os.environ.get("CANNBENCH_RESOLVE_IP", "").strip()
            if resolve_ip:
                command[1:1] = [
                    "--resolve",
                    f"cannbench.com:443:{resolve_ip}",
                ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    timeout=self.upload_timeout + 30,
                )
            except subprocess.TimeoutExpired:
                # TimeoutExpired embeds argv, including the aggregation credential.
                raise RemoteError("CANNBench upload process timed out; delivery is unknown") from None
            raw_body, separator, transfer = completed.stdout.rpartition(b"\n")
            trailer = re.fullmatch(rb"([0-9]{3}) ([0-9]+) ([0-9]+(?:\.[0-9]+)?)", transfer) if separator else None
            if trailer:
                status, uploaded, seconds = (value.decode("ascii") for value in trailer.groups())
            else:
                # Without a valid curl trailer, no stdout bytes can safely be
                # attributed to transport metadata rather than the response.
                raw_body = completed.stdout
                status = uploaded = seconds = None
            body = raw_body.decode("utf-8", errors="replace")
            safe_body = redact_response(body, token, aggregation_token)
            safe_stderr = redact_response(completed.stderr.decode("utf-8", errors="replace"), token, aggregation_token)
            if receipt is not None:
                json.dump({"received_at": datetime.now(timezone.utc).isoformat(),
                           "http_status": status, "curl_returncode": completed.returncode,
                           "transfer_metadata": "valid" if trailer else "missing-or-malformed",
                           "body_source": "http_body" if trailer else "curl_stdout",
                           "multipart_bytes": uploaded, "duration_seconds": seconds,
                           "body_bytes": len(raw_body), "body_sha256": hashlib.sha256(raw_body).hexdigest(),
                           "body_redacted": safe_body, "stderr_redacted": safe_stderr}, receipt, indent=2)
                receipt.write("\n")
                receipt.flush()
                os.fsync(receipt.fileno())
                directory = os.open(Path(receipt_path).expanduser().resolve().parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        if not trailer:
            raise RemoteError("CANNBench upload returned missing or malformed transfer metadata; delivery is unknown")
        if completed.returncode:
            detail = (safe_body or safe_stderr).strip()
            error = SubmissionRejected if completed.returncode == 22 and status in ("400", "401", "403", "413", "422") else RemoteError
            stats = f", {uploaded} multipart bytes in {seconds}s" if seconds else ""
            raise error(f"CANNBench upload failed (HTTP {status}, curl rc={completed.returncode}{stats}): {detail[:800]}")
        try:
            return json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Decode errors can include credential bytes; retain only redacted
            # evidence in the receipt, not a raw exception chain on stdout.
            raise RemoteError("CANNBench upload returned invalid JSON") from None
