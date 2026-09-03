#!/usr/bin/env python3
"""Assemble immutable source submissions for all three four-operator arms."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
OPS = ("gelu", "rms_norm", "softmax", "transpose")
ARMS = ("handwritten", "no_skills", "with_skills")
COMMIT = "030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d"
BASE = REPO_ROOT / "integrations/cannbench/submission"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zip_tree(source: Path, output: Path) -> None:
    output.unlink(missing_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(source))


def main() -> int:
    runtime = next((ROOT / "runtime-dist").glob("pyasc-*-cp312-cp312-linux_x86_64.whl"))
    pybind = BASE / "vendor/runtime-wheels/pybind11-2.13.6-py3-none-any.whl"
    manifest = {
        "schema_version": 1,
        "benchmark": "official-tasks",
        "benchmark_version": "1.1.1",
        "operators": list(OPS),
        "pyasc_ref": "v2",
        "pyasc_commit": COMMIT,
        "arms": {},
    }
    submissions = ROOT / "submissions"
    submissions.mkdir(exist_ok=True)
    for arm in ARMS:
        stage = submissions / arm
        if stage.exists():
            shutil.rmtree(stage)
        package = stage / "cann_bench"
        vendor = stage / "vendor/runtime-wheels"
        package.mkdir(parents=True)
        vendor.mkdir(parents=True)
        shutil.copy2(BASE / "setup.py", stage / "setup.py")
        shutil.copy2(ROOT / "merge_wheels.py", stage / "merge_wheels.py")
        shutil.copy2(BASE / "cann_bench/_pyasc_runtime.py", package / "_pyasc_runtime.py")
        shutil.copy2(runtime, vendor / runtime.name)
        shutil.copy2(pybind, vendor / pybind.name)
        shutil.copy2(BASE / "vendor/PYASC-LICENSE", stage / "vendor/PYASC-LICENSE")
        candidates = {}
        for op in OPS:
            source = ROOT / arm / "candidates" / f"{op}.py"
            shutil.copy2(source, package / f"{op}.py")
            candidates[op] = {"sha256": sha256(source), "path": str(source.relative_to(REPO_ROOT))}
        init = "".join(f"from .{op} import {op}\n" for op in OPS) + "\n__all__ = " + repr(list(OPS)) + "\n"
        (package / "__init__.py").write_text(init, encoding="utf-8")
        provenance = {
            "arm": arm,
            "operators": list(OPS),
            "runtime_commit": COMMIT,
            "runtime_wheel_sha256": sha256(runtime),
            "candidates": candidates,
        }
        (stage / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
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
    assert all(f'cann_bench/{{op}}.py' in names for op in {list(OPS)!r})
    metadata = archive.read(next(n for n in names if n.endswith('.dist-info/METADATA'))).decode()
    assert 'X-PyAsc-Source-Commit: {COMMIT}' in metadata
PY
'''
        (stage / "build.sh").write_text(build, encoding="utf-8")
        (stage / "build.sh").chmod(0o755)
        archive = submissions / f"{arm}.zip"
        zip_tree(stage, archive)
        provenance["submission_zip"] = {
            "path": str(archive.relative_to(REPO_ROOT)),
            "sha256": sha256(archive),
            "size_bytes": archive.stat().st_size,
        }
        manifest["arms"][arm] = provenance
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({arm: data["submission_zip"] for arm, data in manifest["arms"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
