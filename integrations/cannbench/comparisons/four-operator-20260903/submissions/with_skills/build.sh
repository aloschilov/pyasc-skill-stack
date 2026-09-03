#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
rm -rf .base-dist build dist cann_bench.egg-info
mkdir -p .base-dist dist
python setup.py bdist_wheel --dist-dir .base-dist
python merge_wheels.py \
  --base-wheel "$(find .base-dist -name 'cann_bench-*.whl' -print -quit)" \
  --runtime-wheel "$(find vendor/runtime-wheels -name 'pyasc-*.whl' -print -quit)" \
  --pybind11-wheel vendor/runtime-wheels/pybind11-2.13.6-py3-none-any.whl \
  --license vendor/PYASC-LICENSE --output-dir dist
python - <<'PY'
import zipfile
from pathlib import Path
wheel = next(Path('dist').glob('cann_bench-*.whl'))
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    assert 'asctile/__init__.py' in names
    assert 'asc2/__init__.py' not in names
    assert all(f'cann_bench/{op}.py' in names for op in ['gelu', 'rms_norm', 'softmax', 'transpose'])
    metadata = archive.read(next(n for n in names if n.endswith('.dist-info/METADATA'))).decode()
    assert 'X-PyAsc-Source-Commit: 030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d' in metadata
PY
