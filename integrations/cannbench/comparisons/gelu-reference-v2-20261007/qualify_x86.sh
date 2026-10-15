#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd "$(dirname "$0")" && pwd)
artifact_dir=$task_dir
if [[ ${1:-} == --repair ]]; then artifact_dir=$task_dir/repair; fi
task_sdk=/home/aloschilov/workspace/cannbench-competitors-20260910/sdk-x86-9.0.0.v4ebk5/builder-sdk-v2
mkdir "$artifact_dir/x86-qualification"
cp -a "$artifact_dir/bundle" "$artifact_dir/x86-qualification/work"
mkdir "$artifact_dir/x86-qualification/sdk-alias"
for name in include asc lib64; do
  ln -s "/sdk/$name" "$artifact_dir/x86-qualification/sdk-alias/$name"
done
common=(--rm --platform linux/amd64 --network none --user "$(id -u):$(id -g)"
  -v "$task_sdk:/sdk:ro" -v "$task_dir:/task:ro"
  -v "$artifact_dir/x86-qualification:/out"
  -v "$artifact_dir/x86-qualification/sdk-alias:/sdk-alias:ro"
  -e ASCEND_HOME_PATH=/sdk-alias -e LD_LIBRARY_PATH=/sdk-alias/lib64
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/out/installed
  --entrypoint /usr/bin/env)
docker run "${common[@]}" sha256:9fdec1126a3d95148c3c5c2cbe643dac232690be4230cc06c7bc3867eba0c40d bash /out/work/build.sh
docker run "${common[@]}" sha256:9fdec1126a3d95148c3c5c2cbe643dac232690be4230cc06c7bc3867eba0c40d \
  python -m pip install --no-index --no-deps --no-compile --target /out/installed \
  /out/work/dist/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl
docker run "${common[@]}" sha256:e5eeb7ec348639cd2911e895b94d3297e721b388c15019973d76e7b74f1906ac \
  PYASC_COMPILER=/bin/false PYASC_LINKER=/bin/false PYASC_CACHE_DIR=/out/cache \
  python /task/qualify_x86.py
