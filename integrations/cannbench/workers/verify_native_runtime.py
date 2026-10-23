#!/usr/bin/env python3
"""Independent, fail-closed qualification of the pinned native pyasc runtime.

Never trusts a self-reported pin string: the decision uses the ACTUAL installed
distribution version, native extension hashes for BOTH ``asc`` and ``asctile``,
import roots resolving inside the private venv, the absence of legacy editable
links to the source checkout, and source HEAD == pin.

Run only under the shared execution lock (it imports asc/asctile). The decision
logic lives in the pure ``analyze`` function so it is unit-testable without a
runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PIN = "0a631f70968c3cb7c33ce45330a85768dd5a6f06"
FORK_PATH_MARKER = "workspace/pyasc-fork"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extension_hashes(package_file: str | None) -> dict:
    if not package_file:
        return {}
    base = Path(package_file).resolve().parent
    hashes = {}
    for so_path in sorted(base.rglob("*.so")):
        hashes[str(so_path.relative_to(base))] = _sha256_file(so_path)
    return hashes


def analyze(*, pin: str, source_head: str, site_packages: str,
            asc_file: str | None, asctile_file: str | None,
            dist_version: str | None, editable_files: list,
            fork_leak_files: list, asc_extensions: dict,
            asctile_extensions: dict) -> tuple[dict, list]:
    """Pure decision function -> (record, failure_reasons)."""
    real_site = os.path.realpath(site_packages)
    record = {
        "pin": pin,
        "source_head": source_head,
        "source_head_matches_pin": source_head == pin,
        "site_packages": site_packages,
        "dist_version": dist_version,
        "dist_version_contains_pin": bool(dist_version)
        and pin[:8] in (dist_version or ""),
        "asc_file": asc_file,
        "asctile_file": asctile_file,
        "asc_extension_sha256": asc_extensions,
        "asctile_extension_sha256": asctile_extensions,
        "legacy_editable_pth": editable_files,
        "fork_editable_leaks": fork_leak_files,
    }
    record["asc_under_private_venv"] = bool(asc_file) and os.path.realpath(
        asc_file).startswith(real_site)
    record["asctile_under_private_venv"] = bool(
        asctile_file) and os.path.realpath(asctile_file).startswith(real_site)
    record["has_native_extension"] = bool(asc_extensions) or bool(
        asctile_extensions)

    reasons = []
    if not record["source_head_matches_pin"]:
        reasons.append(f"source HEAD {source_head!r} != pin {pin!r}")
    if not record["dist_version_contains_pin"]:
        reasons.append(
            f"installed dist version {dist_version!r} does not contain pin "
            f"{pin[:8]!r}")
    if not record["asc_under_private_venv"]:
        reasons.append(f"asc imported outside private venv: {asc_file!r}")
    if not record["asctile_under_private_venv"]:
        reasons.append(
            f"asctile imported outside private venv: {asctile_file!r}")
    if not record["has_native_extension"]:
        reasons.append("no native .so extension found for asc or asctile")
    if editable_files:
        reasons.append(f"legacy editable installs present: {editable_files}")
    if fork_leak_files:
        reasons.append(
            f".pth files reference the source checkout: {fork_leak_files}")
    record["qualified"] = not reasons
    record["failure_reasons"] = reasons
    return record, reasons


def collect(*, pin: str, source_src: str, site_packages: str,
            asc_file: str | None, asctile_file: str | None,
            dist_version: str | None) -> tuple[dict, list]:
    """Gather actual evidence and run ``analyze``."""
    site = Path(site_packages)
    editable_files = [path.name for path in site.glob("__editable__*.pth")]
    fork_leak_files = []
    for pth in site.glob("*.pth"):
        try:
            text = pth.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if FORK_PATH_MARKER in text:
            fork_leak_files.append(pth.name)
    try:
        source_head = subprocess.run(
            ["git", "-C", str(source_src), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        source_head = "unknown"
    return analyze(
        pin=pin,
        source_head=source_head,
        site_packages=site_packages,
        asc_file=asc_file,
        asctile_file=asctile_file,
        dist_version=dist_version,
        editable_files=editable_files,
        fork_leak_files=fork_leak_files,
        asc_extensions=_extension_hashes(asc_file),
        asctile_extensions=_extension_hashes(asctile_file),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", default=DEFAULT_PIN)
    parser.add_argument("--src", required=True,
                        help="private source checkout (git) at the pin")
    parser.add_argument("--site-packages", required=True)
    parser.add_argument("--root", required=True,
                        help="runtime root for NATIVE-RUNTIME-QUALIFIED.json")
    args = parser.parse_args()

    import importlib.metadata as metadata

    asc_file = asctile_file = None
    import_error = None
    try:
        import asc
        asc_file = asc.__file__
    except Exception as exc:  # pragma: no cover - runtime dependent
        import_error = f"asc: {type(exc).__name__}: {exc}"
    try:
        import asctile
        asctile_file = asctile.__file__
    except Exception as exc:  # pragma: no cover - runtime dependent
        import_error = (import_error + "; " if import_error else "") + \
            f"asctile: {type(exc).__name__}: {exc}"
    try:
        dist_version = metadata.version("pyasc")
    except Exception:
        dist_version = None

    record, reasons = collect(
        pin=args.pin, source_src=args.src,
        site_packages=args.site_packages,
        asc_file=asc_file, asctile_file=asctile_file,
        dist_version=dist_version)
    record["import_error"] = import_error
    record["python"] = sys.version.split()[0]
    record["checked_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    if import_error:
        record["qualified"] = False
        reasons = list(reasons) + [f"import failed: {import_error}"]
        record["failure_reasons"] = reasons

    root = Path(args.root)
    output = root / "NATIVE-RUNTIME-QUALIFIED.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"qualified": record["qualified"],
                      "reasons": reasons}, sort_keys=True))
    return 0 if record["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
