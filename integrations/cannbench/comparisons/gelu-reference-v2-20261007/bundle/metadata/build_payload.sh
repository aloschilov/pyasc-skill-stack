#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha256sum --check source.sha256.txt
: "${ASCEND_HOME_PATH:?Evaluator must select its installed CANN SDK}"
python3 -B metadata/build_metadata_v3.py --sdk "$ASCEND_HOME_PATH" --output metadata-build
mkdir dist
python3 -B metadata/augment_wheel.py \
  --source wheel/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl \
  --built metadata-build \
  --destination dist/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl
