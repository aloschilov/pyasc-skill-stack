#!/usr/bin/env python3
"""Validate capabilities.yaml (v3, tier-based) against golden kernels and evidence.

For each operation cell in capabilities.yaml, checks both golden_status and
generative_status independently:

  golden_status:
    confirmed   - golden file must exist + pass static verify + golden_evidence JSON valid
    golden_only - golden file must exist + pass static verify
    pending     - golden file should exist (warn if missing)
    claimed     - warn (no artifacts)
    untested    - info only
    blocked     - info only

  generative_status:
    confirmed   - generative_evidence JSON must exist, be valid, have kind=generative + agent section
    pending     - warn (prompt defined but no evidence yet)
    untested    - info only
    blocked     - info only

Structural validation (v3):
  - schema_version must be "3"
  - tiers block must exist with level + description
  - every operation must have a valid tier reference
  - representative_of entries are informational (no validation against asc2 API)

Exit 0 = all consistency checks pass, exit 1 = at least one confirmed cell is broken.

Usage:
    python check_capabilities.py [--json] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CAPABILITIES_FILE = REPO_ROOT / "capabilities.yaml"
EVIDENCE_DIR = REPO_ROOT / "evidence"
VERIFY_SCRIPT = SCRIPT_DIR / "verify_kernel.py"
PYTHON = "python3.10"


def _load_yaml(path: Path) -> dict:
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    try:
        result = subprocess.run(
            [PYTHON, "-c", f"import yaml, json; print(json.dumps(yaml.safe_load(open('{path}'))))"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    sys.stderr.write("ERROR: PyYAML is required. Install with: pip install pyyaml\n")
    sys.exit(2)


def _run_static_verify(kernel_path: Path) -> bool:
    try:
        result = subprocess.run(
            [PYTHON, str(VERIFY_SCRIPT), str(kernel_path), "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("passed", False)
    except Exception:
        pass
    return False


def _validate_evidence(evidence_path: Path, expected_kind: str | None = None) -> tuple[bool, str]:
    if not evidence_path.exists():
        return False, f"evidence file not found: {evidence_path}"
    try:
        with open(evidence_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"evidence file invalid JSON: {exc}"

    required_top = {"schema_version", "operation", "dtype", "kernel_path", "date"}
    missing = required_top - set(data.keys())
    if missing:
        return False, f"evidence missing top-level fields: {missing}"

    if "score" not in data:
        return False, "evidence missing 'score' section"
    score_section = data["score"]
    if not isinstance(score_section, dict) or "value" not in score_section:
        return False, "evidence 'score' section missing 'value'"

    if "verification" not in data:
        return False, "evidence missing 'verification' section"

    if expected_kind:
        actual_kind = data.get("kind", "")
        if actual_kind != expected_kind:
            return False, f"evidence kind mismatch: expected '{expected_kind}', got '{actual_kind}'"

    if expected_kind == "generative" and "agent" not in data:
        return False, "generative evidence missing 'agent' section"

    return True, "OK"


class CellResult:
    def __init__(self, op: str, dtype: str, tier: str):
        self.op = op
        self.dtype = dtype
        self.tier = tier
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.passed = True

    def fail(self, msg: str) -> None:
        self.issues.append(msg)
        self.passed = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)


def _check_golden(cell: dict, result: CellResult) -> None:
    status = cell.get("golden_status", "untested")

    if status == "confirmed":
        golden = cell.get("golden")
        if not golden:
            result.fail("golden confirmed but no 'golden' path")
        else:
            golden_path = REPO_ROOT / golden
            if not golden_path.exists():
                result.fail(f"golden kernel not found: {golden}")
            elif not _run_static_verify(golden_path):
                result.fail(f"golden kernel fails static verification: {golden}")
            else:
                result.note(f"golden passes: {golden}")

        evidence_ref = cell.get("golden_evidence")
        if not evidence_ref:
            result.fail("golden confirmed but no 'golden_evidence' path")
        else:
            evidence_path = REPO_ROOT / evidence_ref
            ok, detail = _validate_evidence(evidence_path, expected_kind="golden")
            if not ok:
                result.fail(f"golden evidence: {detail}")
            else:
                result.note(f"golden evidence valid: {evidence_ref}")

    elif status == "golden_only":
        golden = cell.get("golden")
        if not golden:
            result.fail("golden_only but no 'golden' path")
        else:
            golden_path = REPO_ROOT / golden
            if not golden_path.exists():
                result.fail(f"golden kernel not found: {golden}")
            elif not _run_static_verify(golden_path):
                result.fail(f"golden kernel fails static verification: {golden}")
            else:
                result.note(f"golden passes: {golden}")

    elif status == "pending":
        golden = cell.get("golden")
        if golden:
            golden_path = REPO_ROOT / golden
            if not golden_path.exists():
                result.warn(f"golden pending but kernel not found: {golden}")
            else:
                result.note(f"golden pending, kernel exists: {golden}")
        else:
            result.warn("golden pending — no golden kernel path")

    elif status == "claimed":
        result.warn("golden claimed — no golden kernel")

    elif status == "untested":
        result.note("golden untested")

    elif status == "blocked":
        notes = cell.get("notes", "no notes")
        result.warn(f"golden blocked: {notes}")


def _check_generative(cell: dict, result: CellResult) -> None:
    status = cell.get("generative_status", "untested")

    if status == "confirmed":
        evidence_ref = cell.get("generative_evidence")
        if not evidence_ref:
            result.fail("generative confirmed but no 'generative_evidence' path")
        else:
            evidence_path = REPO_ROOT / evidence_ref
            ok, detail = _validate_evidence(evidence_path, expected_kind="generative")
            if not ok:
                result.fail(f"generative evidence: {detail}")
            else:
                result.note(f"generative evidence valid: {evidence_ref}")

    elif status == "pending":
        prompt = cell.get("prompt")
        if prompt:
            result.warn("generative pending — prompt defined, no evidence yet")
        else:
            result.warn("generative pending — no prompt defined")

    elif status == "untested":
        result.note("generative untested")

    elif status == "blocked":
        notes = cell.get("notes", "no notes")
        result.warn(f"generative blocked: {notes}")


def check_structure(data: dict) -> list[str]:
    """Validate v3 structural requirements. Returns list of fatal errors."""
    errors: list[str] = []

    tiers = data.get("tiers")
    if not tiers or not isinstance(tiers, dict):
        errors.append("missing or invalid 'tiers' block")
        return errors

    for t_name, t_info in tiers.items():
        if "level" not in t_info:
            errors.append(f"tier '{t_name}' missing 'level'")
        if "description" not in t_info:
            errors.append(f"tier '{t_name}' missing 'description'")

    valid_tiers = set(tiers.keys())
    for op in data.get("operations", []):
        op_name = op.get("name", "?")
        op_tier = op.get("tier")
        if not op_tier:
            errors.append(f"operation '{op_name}' missing 'tier' field")
        elif op_tier not in valid_tiers:
            errors.append(f"operation '{op_name}' references unknown tier '{op_tier}'")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate capabilities.yaml (v3) consistency.")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show info-level notes")
    args = parser.parse_args()

    if not CAPABILITIES_FILE.exists():
        if args.json:
            print(json.dumps({"status": "fail", "error": "capabilities.yaml not found"}))
        else:
            print(f"FAIL: {CAPABILITIES_FILE} not found")
        sys.exit(1)

    data = _load_yaml(CAPABILITIES_FILE)

    schema = data.get("schema_version", "1")
    if schema != "3":
        msg = f"capabilities.yaml schema_version is '{schema}', expected '3'"
        if args.json:
            print(json.dumps({"status": "fail", "error": msg}))
        else:
            print(f"FAIL: {msg}")
        sys.exit(1)

    struct_errors = check_structure(data)
    if struct_errors:
        if args.json:
            print(json.dumps({"status": "fail", "structural_errors": struct_errors}))
        else:
            print("FAIL: Structural validation errors:")
            for e in struct_errors:
                print(f"  - {e}")
        sys.exit(1)

    operations = data.get("operations", [])
    tiers = data.get("tiers", {})

    results: list[CellResult] = []
    for op in operations:
        op_name = op.get("name", "unknown")
        op_tier = op.get("tier", "unknown")
        for cell in op.get("cells", []):
            dtype = cell.get("dtype", "unknown")
            result = CellResult(op_name, dtype, op_tier)
            _check_golden(cell, result)
            _check_generative(cell, result)
            results.append(result)

    failures = [r for r in results if not r.passed]
    warnings = [r for r in results if r.warnings]

    tier_counts: dict[str, dict] = {}
    for r in results:
        if r.tier not in tier_counts:
            tier_counts[r.tier] = {"total": 0, "golden_confirmed": 0, "gen_confirmed": 0}
        tc = tier_counts[r.tier]
        tc["total"] += 1
        for op in operations:
            if op.get("name") != r.op:
                continue
            for cell in op.get("cells", []):
                if cell.get("dtype") == r.dtype:
                    if cell.get("golden_status") == "confirmed":
                        tc["golden_confirmed"] += 1
                    if cell.get("generative_status") == "confirmed":
                        tc["gen_confirmed"] += 1
                    break
            break

    if args.json:
        out = {
            "status": "pass" if not failures else "fail",
            "schema_version": "3",
            "total_cells": len(results),
            "tier_counts": tier_counts,
            "failures": [
                {"op": r.op, "dtype": r.dtype, "tier": r.tier, "issues": r.issues}
                for r in failures
            ],
            "warnings": [
                {"op": r.op, "dtype": r.dtype, "tier": r.tier, "warnings": r.warnings}
                for r in warnings
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print("=" * 60)
        print("  Capabilities Matrix Validation (v3 — tier-based)")
        print("=" * 60)
        print()

        for r in results:
            tag = "PASS" if r.passed else "FAIL"
            line = f"  [{tag}] {r.op}/{r.dtype} (tier: {r.tier})"
            if r.issues:
                line += f" — {'; '.join(r.issues)}"
            if r.warnings:
                line += f" — {'; '.join(r.warnings)}"
            if args.verbose and r.info:
                line += f" — {'; '.join(r.info)}"
            print(line)

        print()
        print(f"  Cells: {len(results)} total")
        for t_name in sorted(tier_counts, key=lambda k: tiers.get(k, {}).get("level", 99)):
            tc = tier_counts[t_name]
            lvl = tiers.get(t_name, {}).get("level", "?")
            print(f"  Tier {lvl} ({t_name}): golden {tc['golden_confirmed']}/{tc['total']}, "
                  f"gen {tc['gen_confirmed']}/{tc['total']}")
        print()

        if failures:
            print(f"  FAIL: {len(failures)} cell(s) have broken artifacts")
        else:
            print("  PASS: All confirmed cells are consistent")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
