#!/usr/bin/env bash
# =============================================================================
# L3 Integration Test: Full Kernel Generation (Agent-in-the-Loop)
# =============================================================================
# End-to-end test: OpenCode generates a pyasc vector add kernel, then the
# test harness verifies the output with static AST checks, automated scoring,
# and mandatory runtime verification (unless --static-only is explicit).
#
# Requires: opencode CLI, python3.10 with pyasc importable
# Estimated time: 5-15 minutes
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

STRICT=1
case "${1:-}" in
    --static-only) STRICT=0 ;;
    --strict|"") ;;
    *) echo "Usage: $0 [--strict|--static-only]"; exit 1 ;;
esac

echo "========================================"
echo -e " ${BOLD}L3 Integration: Kernel Generation E2E${NC}"
echo "========================================"
echo ""
echo "Tests full agent-driven pyasc kernel generation."
echo "Estimated time: 5-15 minutes"
echo "Requires: opencode CLI, python3.10"
echo ""

if ! check_opencode; then
    # Skip, not fail. This test needs a CLI and a funded API key that the
    # fast suite deliberately does not require, so an absent opencode means
    # "not exercised here", which run-tests.sh records as a skip. Exiting 1
    # would report a missing optional dependency as a product defect.
    print_skip "opencode CLI not found on PATH; generation not exercised"
    exit 2
fi

# ============================================
# Setup
# ============================================

TEST_PROJECT=$(mktemp -d -t "kernel-gen.XXXXXX")
ln -s "$SKILLS_DIR/skills" "$TEST_PROJECT/skills"
KERNEL_DIR="$TEST_PROJECT/kernels/add"
mkdir -p "$KERNEL_DIR/docs"

# Persist only allowlisted outputs, never rendered auth/config files.
finish_project() {
    if [ -n "${PYASC_GENERATION_OUTPUT:-}" ]; then
        mkdir -p "$PYASC_GENERATION_OUTPUT"
        for file in kernels/add/kernel.py kernels/add/docs/design.md kernel.py docs/design.md; do
            if [ -f "$TEST_PROJECT/$file" ]; then
                mkdir -p "$PYASC_GENERATION_OUTPUT/$(dirname "$file")"
                cp "$TEST_PROJECT/$file" "$PYASC_GENERATION_OUTPUT/$file"
            fi
        done
        "$PYTHON" - "$TEST_PROJECT" "$PYASC_GENERATION_OUTPUT" <<'PYLOG'
import os, pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
for name in ('agent-output.txt', 'prompt.txt'):
    path = source/name
    if not path.is_file(): continue
    text = path.read_text(errors='replace')
    for key in ('OPENCODE_API_KEY', 'OPENCODE_BASE_URL'):
        secret = os.environ.get(key)
        if secret: text = text.replace(secret, '<redacted>')
    (target/name).write_text(text)
PYLOG
    fi
    rm -rf "$TEST_PROJECT"
}
trap finish_project EXIT

# Render a tracked profile into the project rather than letting opencode fall
# back to the user's ~/.config/opencode/opencode.json. That global config is
# unknown to this repository and, on at least one machine, caps
# gateway/glm-5.3 at output=8192 with reasoning enabled -- the documented
# cause of the truncated generations in docs/verification-status.md. A
# harness that inherits an unknown ceiling is not testing what it claims to.
MODEL_PROFILE="${PYASC_MODEL_PROFILE:-cloud-glm-5.3}"
if ! $PYTHON "$TOOLS_DIR/opencode_profile.py" \
        --profile "$MODEL_PROFILE" \
        --skills on \
        --require-credentials \
        --out "$TEST_PROJECT/opencode.json"; then
    print_skip "profile '$MODEL_PROFILE' has no usable credentials; generation not exercised"
    exit 2
fi

echo "Test project: $TEST_PROJECT"
echo "Model profile: $MODEL_PROFILE"
echo ""

# ============================================
# Agent execution
# ============================================

print_section_header "Phase: Agent Execution"

# The environment paragraph is load-bearing, not politeness. Without it the
# agent opens with the workflow's "check environment" phase, finds no
# importable asctile on the host -- it lives in the pinned simulator image,
# not here -- and spends the entire budget excavating unrelated checkouts
# under /home and /tmp looking for one. Observed directly: 22 steps, 26 tool
# calls, zero files written, 900s exhausted. Telling it the truth up front
# is what makes this test measure authoring quality instead of filesystem
# archaeology.
PROMPT="You are a pyasc kernel development engineer.

ENVIRONMENT — read this first and do not try to verify it:
- asctile is NOT importable on this machine. It is compiled into the pinned
  simulator image, which is not running here. Do not search the filesystem
  for it, do not try to import it, and do not try to execute the kernel.
- Other asctile or pyasc checkouts may exist elsewhere on this host. They are
  stale and are not the reference. Ignore them.
- Your source will be compiled and executed by a separate deterministic harness. Write the
  kernel as if it will run; it will be executed elsewhere.
- The skills under ./skills are the authority for API and syntax. Read them
  rather than inferring the API from any installed package.

TASK:
Implement a vector add kernel for two float32 tensors of size 1024, under
kernels/add/:
1. A design document at kernels/add/docs/design.md
2. The kernel at kernels/add/kernel.py, using @asctile.jit with
   asctile.copy_in / asctile.copy_out for data movement
3. torch.allclose checks against x + y at sizes 1024 and 1025; use
   assert torch.allclose(...) or if not torch.allclose(...): raise ValueError(...)
   directly. Guard execution with if __name__ == \"__main__\" so import does not run it

Requirements:
- Use asctile.global_tensor, asctile.copy_in, asctile.copy_out, and
  asctile.range as needed
- Use kernel[CORE_NUM](...) launch syntax (no stream argument)
- Use only supported pyasc syntax (no print/break/continue/lambda inside
  @asctile.jit)

Write the files, then stop. Do not run them."

printf '%s\n' "$PROMPT" > "$TEST_PROJECT/prompt.txt"
OUTPUT_FILE="$TEST_PROJECT/agent-output.txt"

# Must stay below run-tests.sh's INTEGRATION_TIMEOUT (1800s), which bounds
# this whole file. When the two were both 300s the outer timer -- started
# earlier, and counting project setup -- always won, so the branches below
# and every verification phase after them were dead code.
AGENT_TIMEOUT="${PYASC_AGENT_TIMEOUT:-900}"

# -m is not optional. Without it opencode resolves the model from the config
# `model` key, and if that config came from the host's global file rather
# than the profile rendered above, the run silently adopts the host's limits.
# Naming the model makes the generation attributable to a specific one.
AGENT_MODEL="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model",""))' "$TEST_PROJECT/opencode.json")"

echo "Running OpenCode..."
echo "  Timeout: ${AGENT_TIMEOUT}s"
echo "  Model:   ${AGENT_MODEL:-<from config>}"
echo ""

AGENT_ARGS=(run "$PROMPT" --format json --dir "$TEST_PROJECT")
if [ -n "$AGENT_MODEL" ]; then
    AGENT_ARGS+=(-m "$AGENT_MODEL")
fi

AGENT_TIMED_OUT=false
if timeout "$AGENT_TIMEOUT" opencode "${AGENT_ARGS[@]}" > "$OUTPUT_FILE" 2>&1; then
    echo "  Agent completed successfully."
else
    ec=$?
    if [ "$ec" -eq 124 ]; then
        AGENT_TIMED_OUT=true
        print_warn "Agent timed out after ${AGENT_TIMEOUT}s"
    else
        print_info "Agent exited with code $ec"
    fi
fi

echo ""

# ============================================
# Artifact verification
# ============================================

print_section_header "Phase: Artifact Verification"

FAILED=0

# Look for kernel.py in expected and fallback locations
KERNEL_PY=""
for candidate in \
    "$KERNEL_DIR/kernel.py" \
    "$TEST_PROJECT/kernel.py" \
    "$TEST_PROJECT/kernels/add/kernel.py"; do
    if [ -f "$candidate" ]; then
        KERNEL_PY="$candidate"
        break
    fi
done

if [ -n "$KERNEL_PY" ]; then
    print_pass "kernel.py found: $KERNEL_PY"
else
    print_fail "kernel.py not found in any expected location"
    FAILED=$((FAILED + 1))
fi

DESIGN_MD=""
for candidate in \
    "$KERNEL_DIR/docs/design.md" \
    "$TEST_PROJECT/docs/design.md" \
    "$TEST_PROJECT/kernels/add/docs/design.md"; do
    if [ -f "$candidate" ]; then
        DESIGN_MD="$candidate"
        break
    fi
done

if [ -n "$DESIGN_MD" ]; then
    print_pass "design.md found: $DESIGN_MD"
else
    print_warn "design.md not found (agent may have skipped design phase)"
    if [ "$STRICT" = 1 ]; then FAILED=$((FAILED + 1)); fi
fi

echo ""

# ============================================
# Static verification
# ============================================

print_section_header "Phase: Static Verification"

if [ -n "$KERNEL_PY" ]; then
    echo "Running verify_kernel.py..."
    if $PYTHON "$TOOLS_DIR/verify_kernel.py" "$KERNEL_PY" 2>&1; then
        print_pass "Static verification passed"
    else
        print_fail "Static verification failed"
        FAILED=$((FAILED + 1))
    fi
else
    print_skip "No kernel.py to verify"
    FAILED=$((FAILED + 1))
fi

echo ""

# ============================================
# Automated scoring
# ============================================

print_section_header "Phase: Automated Scoring"

if [ -n "$KERNEL_PY" ]; then
    echo "Running score_kernel.py..."
    SCORE_OUTPUT=$($PYTHON "$TOOLS_DIR/score_kernel.py" "$KERNEL_PY" --json 2>&1) || true
    echo "$SCORE_OUTPUT" | $PYTHON -m json.tool 2>/dev/null || echo "$SCORE_OUTPUT"

    # Read the scale and the bar from the scorer's own JSON. They were
    # hardcoded here as "/ 10" and ">= 8.5" while score_kernel.py used
    # MAX_SCORE = 16 and THRESHOLD = 12, so every reported score was
    # printed against a denominator that did not exist and judged against a
    # threshold that was never applied.
    read -r score_val max_val thresh_val accepted gate_reason <<<"$(
        echo "$SCORE_OUTPUT" | $PYTHON -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    d = {}
print(d.get("score", 0),
      d.get("max_score", "?"),
      d.get("threshold", "?"),
      d.get("accepted", False),
      (d.get("failure_category") or "-"))
' 2>/dev/null || echo "0 ? ? False -")"

    echo ""
    echo "  Score: $score_val / $max_val  (threshold: $thresh_val)"
    if [ "$accepted" = "True" ]; then
        print_pass "Score meets the acceptance threshold (>= $thresh_val)"
    elif [ "$gate_reason" != "-" ]; then
        # A hard gate overrides the total: the kernel can score above the
        # threshold and still be rejected. Saying "below threshold" here
        # would misreport why.
        print_fail "Rejected by gate: $gate_reason"
        FAILED=$((FAILED + 1))
    else
        print_fail "Score $score_val is below the acceptance threshold $thresh_val"
        FAILED=$((FAILED + 1))
    fi
else
    print_skip "No kernel.py to score"
    FAILED=$((FAILED + 1))
fi

echo ""

# ============================================
# Runtime verification: mandatory in strict mode
# ============================================
if [ "$STRICT" = 1 ] && [ -n "$KERNEL_PY" ]; then
    runtime_exit=0
    "$PYTHON" "$TOOLS_DIR/qualify_generated.py" "$KERNEL_PY" \
        --results-dir "${PYASC_GENERATION_OUTPUT:?strict mode requires an external artifact directory}/qualification" || runtime_exit=$?
    if [ "$runtime_exit" -eq 2 ] && [ "$FAILED" -eq 0 ]; then
        print_skip "Runtime unavailable: strict generation is blocked"
        exit 2
    elif [ "$runtime_exit" -ne 0 ]; then
        FAILED=$((FAILED + 1))
    fi
elif [ "$STRICT" = 0 ]; then
    print_info "Static-only observation; does not establish generation (executed)"
fi

echo ""

# ============================================
# Summary
# ============================================

echo "========================================"
echo -e " ${BOLD}Kernel Generation E2E Results${NC}"
echo "========================================"
echo ""
echo "  Test project: $TEST_PROJECT"
echo "  Kernel file:  ${KERNEL_PY:-not found}"
echo "  Design doc:   ${DESIGN_MD:-not found}"
echo ""

if [ "$FAILED" -gt 0 ] && [ "$AGENT_TIMED_OUT" = true ] && [ -z "$KERNEL_PY" ]; then
    # The agent ran out of time before writing anything. Nothing about the
    # skill stack was measured, so this is `blocked`, not `failed`. Exit 124
    # so run-tests.sh records it as such rather than as a product defect.
    echo "  Agent timed out after ${AGENT_TIMEOUT}s and produced no kernel."
    echo "  Raise PYASC_AGENT_TIMEOUT (and PYASC_INTEGRATION_TIMEOUT above it) to retry."
    print_warn "BLOCKED: no generation to evaluate"
    exit 124
elif [ "$FAILED" -gt 0 ]; then
    echo "  Critical failures: $FAILED"
    print_status_failed
    exit 1
else
    print_status_passed
    exit 0
fi
