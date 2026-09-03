#!/usr/bin/env python3
"""Assemble immutable single-op GeLU iteration 03."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
BASE = REPO / "integrations/cannbench/submission"
COMMIT = "0a631f70968c3cb7c33ce45330a85768dd5a6f06"
ITERATION = "iteration-03-lowlevel-tanh-safe-tile13824"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    runtime_wheels = list((ROOT / "runtime-dist").glob(
        "pyasc-*-cp312-cp312-linux_x86_64.whl"
    ))
    if len(runtime_wheels) != 1:
        raise RuntimeError(f"expected one current-v2 runtime wheel, got {runtime_wheels}")
    runtime = runtime_wheels[0]
    pybind = BASE / "vendor/runtime-wheels/pybind11-2.13.6-py3-none-any.whl"
    stage = ROOT / "submissions" / ITERATION
    if stage.exists():
        shutil.rmtree(stage)
    package = stage / "cann_bench"
    vendor = stage / "vendor/runtime-wheels"
    package.mkdir(parents=True)
    vendor.mkdir(parents=True)
    for source, target in (
        (BASE / "setup.py", stage / "setup.py"),
        (ROOT / "merge_wheels.py", stage / "merge_wheels.py"),
        (ROOT / "candidates/gelu.py", package / "gelu.py"),
        (ROOT / "candidates/_pyasc_runtime.py", package / "_pyasc_runtime.py"),
        (runtime, vendor / runtime.name),
        (pybind, vendor / pybind.name),
        (BASE / "vendor/PYASC-LICENSE", stage / "vendor/PYASC-LICENSE"),
    ):
        shutil.copy2(source, target)
    (package / "__init__.py").write_text(
        "from .gelu import gelu\n\n__all__ = ['gelu']\n", encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "iteration": ITERATION,
        "operator": "gelu",
        "source_class": "upstream-handwritten-derived-contract-corrected-and-tuned",
        "pyasc_ref": "v2",
        "pyasc_commit": COMMIT,
        "candidate_sha256": sha256(ROOT / "candidates/gelu.py"),
        "runtime_wheel_sha256": sha256(runtime),
        "integration_overlays": {
            "c310_lowlevel_compiler": (
                "base asc pipeline with set_ffts_addr disabled on C310"
            ),
            "asctile_jit_options": (
                "concrete compiler option discovery and extraction"
            ),
        },
        "launch": {
            "exact_fp16_tile": 13824,
            "lowlevel_tile": 13824,
            "tanh_tile": 13824,
            "max_ai_cores": 72,
            "tail_strategy": "real-shape-asctile; overlap-final-lowlevel",
            "exact_fp32_bf16": "asc.adv.erfc",
            "exact_fp16": "asctile.erf-vf-fused-reuse-alloc",
            "tanh": "single-exp-cancellation-free-lowlevel",
        },
    }
    (stage / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    build = f'''#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
rm -rf .base-dist build dist cann_bench.egg-info
mkdir -p .base-dist dist
python setup.py bdist_wheel --dist-dir .base-dist
python merge_wheels.py \\
  --base-wheel "$(find .base-dist -name 'cann_bench-*.whl' -print -quit)" \\
  --runtime-wheel "$(find vendor/runtime-wheels -name 'pyasc-*.whl' -print -quit)" \\
  --pybind11-wheel vendor/runtime-wheels/pybind11-2.13.6-py3-none-any.whl \\
  --license vendor/PYASC-LICENSE --output-dir dist
python - <<'PY'
import zipfile
from pathlib import Path
wheel = next(Path('dist').glob('cann_bench-*.whl'))
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    assert 'asctile/__init__.py' in names
    assert 'asc2/__init__.py' not in names
    assert 'cann_bench/gelu.py' in names
    metadata = archive.read(next(n for n in names if n.endswith('.dist-info/METADATA'))).decode()
    assert 'X-PyAsc-Source-Commit: {COMMIT}' in metadata
PY
'''
    build_path = stage / "build.sh"
    build_path.write_text(build, encoding="utf-8")
    build_path.chmod(0o755)
    archive_path = ROOT / "submissions" / f"{ITERATION}.zip"
    archive_path.unlink(missing_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(stage))
    provenance["submission_zip"] = {
        "path": str(archive_path.relative_to(REPO)),
        "sha256": sha256(archive_path),
        "size_bytes": archive_path.stat().st_size,
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
