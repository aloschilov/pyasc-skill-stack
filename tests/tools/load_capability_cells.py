#!/usr/bin/env python3
"""Load perf and generative cell lists from capabilities.yaml."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES_FILE = REPO_ROOT / "capabilities.yaml"

DTYPE_ABBREV = {
    "float16": "f16",
    "float32": "f32",
    "bfloat16": "bf16",
}

CAP_DTYPE_TO_REF = {
    "float16": "f16",
    "float32": "f32",
    "bfloat16": "bf16",
}


class PerfCellError(RuntimeError):
    pass


def load_capabilities_yaml(path: Path | None = None) -> dict[str, Any]:
    path = path or CAPABILITIES_FILE
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        result = subprocess.run(
            ["python3", "-c",
             f"import yaml,json; print(json.dumps(yaml.safe_load(open('{path}'))))"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            sys.stderr.write("ERROR: PyYAML is required. pip install pyyaml\n")
            sys.exit(2)
        return json.loads(result.stdout)


def resolve_perf_kernel(op_name: str, cell: dict[str, Any], repo_root: Path) -> Path:
    """Pick the pyasc kernel measured on the gen side for a perf cell."""
    golden_rel = cell.get("golden")
    golden = repo_root / golden_rel if golden_rel else None
    prd = cell.get("perf_ratio_demo") or {}
    kernel_source = prd.get("kernel_source", "")
    dtype = cell.get("dtype", "?")
    abbrev = DTYPE_ABBREV.get(dtype)
    teams = (
        repo_root / f"teams/pyasc-kernel-dev-team/kernels/{op_name}_{abbrev}/kernel.py"
        if abbrev else None
    )

    if golden and ("vetted golden" in kernel_source
                    or cell.get("generative_status") != "confirmed"):
        if golden.exists():
            return golden
    if teams and teams.exists():
        return teams
    if golden and golden.exists():
        return golden
    raise PerfCellError(
        f"no perf kernel for {op_name}/{dtype}: golden={golden_rel!r}, "
        f"teams={teams}"
    )


def load_perf_cells(cap: dict[str, Any], repo_root: Path | None = None) -> dict[str, dict]:
    """Build demo CELLS dict from capabilities ``perf_ratio_demo`` blocks."""
    repo_root = repo_root or REPO_ROOT
    cells: dict[str, dict] = {}
    for op in cap.get("operations", []):
        op_name = op.get("name", "?")
        for cell in op.get("cells", []):
            prd = cell.get("perf_ratio_demo")
            if not prd:
                continue
            dtype = cell.get("dtype", "?")
            key = f"{op_name}/{dtype}"
            ref_dtype = CAP_DTYPE_TO_REF.get(dtype)
            if ref_dtype is None:
                raise PerfCellError(f"unsupported perf dtype {dtype!r} for {key}")
            cells[key] = {
                "ref_op": op_name,
                "ref_dtype": ref_dtype,
                "kernel": resolve_perf_kernel(op_name, cell, repo_root),
                "gen_dtype": dtype,
                "shape": list(prd.get("shape") or []),
            }
    return cells


def load_cell_to_op_dtype(cap: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Build ascendc_ref_runner CELL_TO_OP_DTYPE from perf cells."""
    return {
        cell: (spec["ref_op"], spec["ref_dtype"])
        for cell, spec in load_perf_cells(cap).items()
    }


def list_generative_cells(
    cap: dict[str, Any],
    *,
    platform: str = "Ascend950PR_9599",
) -> list[tuple[str, str, str, int]]:
    """Return (op, dtype, tier, tier_level) for cells with prompts on ``platform``."""
    tiers = cap.get("tiers", {})
    out: list[tuple[str, str, str, int]] = []
    for op in cap.get("operations", []):
        op_name = op.get("name", "?")
        tier = op.get("tier", "")
        tier_level = tiers.get(tier, {}).get("level", 99)
        for cell in op.get("cells", []):
            if cell.get("platform") != platform:
                continue
            prompt = (cell.get("prompt") or "").strip()
            if not prompt:
                continue
            dtype = cell.get("dtype", "?")
            out.append((op_name, dtype, tier, tier_level))
    out.sort(key=lambda row: (row[3], row[0], row[1]))
    return out
