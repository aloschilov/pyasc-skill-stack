#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: Capabilities matrix consistency (v2 schema)
# Validates capabilities.yaml against golden kernels and evidence artifacts.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Capabilities Matrix Validation (v2) ==="

CAPS_FILE="$SKILLS_DIR/capabilities.yaml"
CHECK_TOOL="$TOOLS_DIR/check_capabilities.py"

if [ ! -f "$CAPS_FILE" ]; then
    print_fail "capabilities.yaml not found at $CAPS_FILE"
    exit 1
fi

if [ ! -f "$CHECK_TOOL" ]; then
    print_fail "check_capabilities.py not found at $CHECK_TOOL"
    exit 1
fi

echo ""
echo "--- check_capabilities.py ---"
output=$($PYTHON "$CHECK_TOOL" --json 2>&1) || true

status=$(echo "$output" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
total=$(echo "$output" | $PYTHON -c "import json,sys; print(json.load(sys.stdin).get('total_cells',0))" 2>/dev/null || echo "0")
fail_count=$(echo "$output" | $PYTHON -c "import json,sys; print(len(json.load(sys.stdin).get('failures',[])))" 2>/dev/null || echo "0")
warn_count=$(echo "$output" | $PYTHON -c "import json,sys; print(len(json.load(sys.stdin).get('warnings',[])))" 2>/dev/null || echo "0")

golden_counts=$(echo "$output" | $PYTHON -c "
import json, sys
d = json.load(sys.stdin).get('golden_counts', {})
parts = [f'{k}: {v}' for k, v in sorted(d.items())]
print(', '.join(parts))
" 2>/dev/null || echo "n/a")

gen_counts=$(echo "$output" | $PYTHON -c "
import json, sys
d = json.load(sys.stdin).get('generative_counts', {})
parts = [f'{k}: {v}' for k, v in sorted(d.items())]
print(', '.join(parts))
" 2>/dev/null || echo "n/a")

echo "  Total cells: $total"
echo "  Golden:      $golden_counts"
echo "  Generative:  $gen_counts"
echo "  Failures: $fail_count"
echo "  Warnings: $warn_count"
echo ""

if [ "$status" = "pass" ]; then
    print_pass "Capabilities matrix is consistent (all confirmed cells have valid artifacts)"
else
    print_fail "Capabilities matrix has inconsistencies"
    echo "$output" | $PYTHON -c "
import json, sys
data = json.load(sys.stdin)
for f in data.get('failures', []):
    print(f'    {f[\"op\"]}/{f[\"dtype\"]}: {\"; \".join(f[\"issues\"])}')
" 2>/dev/null || true
    exit 1
fi

echo ""
echo "--- Evidence files ---"
EVIDENCE_DIR="$SKILLS_DIR/evidence"
if [ -d "$EVIDENCE_DIR" ]; then
    evidence_count=$(find "$EVIDENCE_DIR" -name '*.json' -type f 2>/dev/null | wc -l)
    golden_count=$(find "$EVIDENCE_DIR" -name '*-golden.json' -type f 2>/dev/null | wc -l)
    gen_count=$(find "$EVIDENCE_DIR" -name '*-generative.json' -type f 2>/dev/null | wc -l)
    print_pass "Evidence directory: $evidence_count file(s) ($golden_count golden, $gen_count generative)"
else
    print_warn "Evidence directory not found (no evidence collected yet)"
fi

echo ""
echo "[PASS] Capabilities matrix validation complete"
