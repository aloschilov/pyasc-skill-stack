#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_NAME="${1:?Usage: $0 <kernel_name>}"

TEAM_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/teams/pyasc-kernel-dev-team"
KERNELS_DIR="$TEAM_DIR/kernels"
KERNEL_DIR="$KERNELS_DIR/$KERNEL_NAME"

if [ -d "$KERNEL_DIR" ]; then
    echo "[WARN] Directory already exists: $KERNEL_DIR"
    echo "[INFO] Skipping creation."
    exit 0
fi

echo "[INFO] Creating kernel project: $KERNEL_NAME"

mkdir -p "$KERNEL_DIR/docs"
mkdir -p "$KERNEL_DIR/test"

cat > "$KERNEL_DIR/README.md" << EOF
# $KERNEL_NAME kernel

pyasc kernel implementation for the $KERNEL_NAME operation using the asc2 tile-based API.

## Files

- \`kernel.py\` — Kernel implementation (created in Phase 2)
- \`docs/design.md\` — Design document (created in Phase 1)
- \`docs/environment.json\` — Environment snapshot (created in Phase 0)
- \`test/\` — Test data and verification scripts

## Usage

\`\`\`bash
python3.10 kernel.py -r Model -v Ascend950PR_9599   # Run with simulator
python3.10 kernel.py -r NPU                         # Run with NPU hardware
pytest kernel.py --backend Model --platform Ascend950PR_9599
\`\`\`
EOF

cat > "$KERNEL_DIR/conftest.py" << 'EOF'
import pytest
from asc.runtime import config


def pytest_addoption(parser):
    parser.addoption("--backend", type=config.Backend, default=config.Backend.Model)
    parser.addoption("--platform", type=config.Platform, default=config.Platform.Ascend950PR_9599)


@pytest.fixture
def backend(request):
    return request.config.getoption("--backend")


@pytest.fixture
def platform(request):
    return request.config.getoption("--platform")
EOF

echo "[PASS] Kernel project initialized: $KERNEL_DIR"
echo "[INFO] Directory structure:"
echo "  $KERNEL_DIR/"
echo "  ├── docs/"
echo "  ├── test/"
echo "  ├── conftest.py"
echo "  └── README.md"
echo ""
echo "[NEXT] Run verify_environment.sh to save environment.json"
