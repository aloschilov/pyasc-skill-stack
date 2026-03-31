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

pyasc kernel implementation for the $KERNEL_NAME operation.

## Files

- \`kernel.py\` — Kernel implementation (created in Phase 2)
- \`docs/design.md\` — Design document (created in Phase 1)
- \`docs/environment.json\` — Environment snapshot (created in Phase 0)
- \`test/\` — Test data and verification scripts

## Usage

\`\`\`bash
python kernel.py -r Model        # Run with simulator
python kernel.py -r NPU          # Run with NPU hardware
\`\`\`
EOF

echo "[PASS] Kernel project initialized: $KERNEL_DIR"
echo "[INFO] Directory structure:"
echo "  $KERNEL_DIR/"
echo "  ├── docs/"
echo "  ├── test/"
echo "  └── README.md"
echo ""
echo "[NEXT] Run verify_environment.sh to save environment.json"
