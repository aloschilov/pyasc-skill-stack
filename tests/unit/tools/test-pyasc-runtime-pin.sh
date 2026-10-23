#!/usr/bin/env bash
# Keep every executable runner/dashboard pin aligned with Dockerfile's source pin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

python3 - "$REPO_ROOT" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])

def match(path: str, pattern: str) -> str:
    text = (root / path).read_text()
    found = re.search(pattern, text)
    if not found:
        raise SystemExit(f"FAIL: runtime pin not found in {path}")
    return found.group(1)

full = match("docker/Dockerfile", r"ARG PYASC_GIT_REV=([0-9a-f]{40})")
short = full[:8]
bundle = root / "docker" / f"pyasc-{short}.bundle"
if not bundle.is_file():
    raise SystemExit(f"FAIL: bundled pyasc source missing: {bundle}")
heads = subprocess.check_output(
    ["git", "bundle", "list-heads", str(bundle)], text=True
).splitlines()
bundle_revs = {line.split()[0] for line in heads if line.strip()}
if bundle_revs != {full}:
    raise SystemExit(
        f"FAIL: {bundle.name} contains {sorted(bundle_revs)}, expected only {full}"
    )

full_pins = {
    "docker/build-sim-image.sh": r'PYASC_GIT_REV="\$\{PYASC_GIT_REV:-([0-9a-f]{40})\}"',
    "docker/build-perf-image.sh": r'PYASC_GIT_REV="\$\{PYASC_GIT_REV:-([0-9a-f]{40})\}"',
    "tests/tools/verify_pyasc_image.sh": r'EXPECTED_REV="\$\{PYASC_GIT_REV:-([0-9a-f]{40})\}"',
    "tests/tools/generate_dashboard.py": r'PYASC_REVISION = "([0-9a-f]{40})"',
    ".github/workflows/build-sim-image.yml": r"default: ([0-9a-f]{40})",
    ".github/workflows/build-perf-image.yml": r"default: ([0-9a-f]{40})",
}
for path, pattern in full_pins.items():
    actual = match(path, pattern)
    if actual != full:
        raise SystemExit(f"FAIL: {path} pins {actual}, expected {full}")

short_pin_files = [
    ".github/workflows/ci.yml",
    "docker/Dockerfile.perf",
    "tests/tools/collect_generative_evidence.py",
]
for path in short_pin_files:
    text = (root / path).read_text()
    values = set(re.findall(r"pyasc-([0-9a-f]{8})", text))
    if values != {short}:
        raise SystemExit(
            f"FAIL: {path} short pins {sorted(values)}, expected only {short}"
        )

print(f"PASS: all runtime and dashboard pins agree on pyasc {full}")
PY
