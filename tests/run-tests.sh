#!/usr/bin/env bash
# =============================================================================
# pyasc Skills Testing Framework
# L1 unit / L2 behavior / L3 integration pyramid
# Supports: --fast (unit only), --agentic (L2+L3 with OpenCode), --all
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/test-helpers.sh"

# Defaults
CATEGORY=""
RUN_INTEGRATION=false
RUN_ALL=false
RUN_AGENTIC=false
SINGLE_TEST=""
# Per-file wall-clock budget, by category. One number cannot serve both: a
# unit test that takes 300s is broken, while an integration test spends most
# of its budget waiting on a cloud agent and legitimately needs minutes.
#
# The single 300s budget that used to apply to everything collided with the
# agent timeout inside tests/integration/test-kernel-generation.sh, which was
# also 300s. The outer timer starts first and includes project setup, so it
# always expired first: the agent's own timeout handling and every
# verification phase after it were unreachable, and the run was recorded as a
# skip. Keep INTEGRATION_TIMEOUT comfortably above PYASC_AGENT_TIMEOUT.
TIMEOUT=300
INTEGRATION_TIMEOUT="${PYASC_INTEGRATION_TIMEOUT:-1800}"
VERBOSE=false
OUTPUT_FORMAT="text"
LIST_ONLY=false
PLATFORM="${DEFAULT_PLATFORM}"
RUNTIME_CHECK=false
EVAL_RESULTS=false

usage() {
    cat <<EOF
pyasc Skills Testing Framework

Usage: $0 [options]

Categories:
  --fast, -f            Run unit tests only (no agent, no runtime)
  --agentic, -a         Run L2 behavior + L3 integration (requires opencode)
  --integration, -i     Include integration tests (unit + behavior + integration)
  --all                 Run all tests (unit + behavior + integration)
  --category, -c CAT    Run specific category (unit/behavior/integration)

Options:
  --platform PLATFORM   Agent platform: opencode (default)
  --runtime             Enable pyasc runtime verification in L3 tests
  --test, -t NAME       Run a single test file (relative to tests/)
  --timeout SECONDS     Per-test timeout (default: 300)
  --verbose, -v         Show verbose output
  --output FORMAT       Output format (text/json)
  --eval-results        After tests, run eval-report.sh and save to .eval-history/
  --list, -l            List all available tests
  --help, -h            Display this help

Environment:
  PYASC_PYTHON          Python interpreter with pyasc (default: python3.10)
  DEFAULT_PLATFORM      Agent platform override (default: opencode)
  NO_COLOR              Disable colored output
  FORCE_COLOR=1         Force colored output in CI

Examples:
  $0 --fast                  Unit tests only (seconds)
  $0 --agentic               Agent-in-the-loop tests (minutes)
  $0 --all --runtime         Everything including pyasc runtime checks
  $0 --test integration/test-kernel-generation.sh
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast|-f)        CATEGORY="unit"; shift ;;
        --agentic|-a)     RUN_AGENTIC=true; shift ;;
        --integration|-i) RUN_INTEGRATION=true; shift ;;
        --all)            RUN_ALL=true; shift ;;
        --category|-c)    CATEGORY="$2"; shift 2 ;;
        --platform)       PLATFORM="$2"; shift 2 ;;
        --runtime)        RUNTIME_CHECK=true; shift ;;
        --test|-t)        SINGLE_TEST="$2"; shift 2 ;;
        --timeout)        TIMEOUT="$2"; shift 2 ;;
        --verbose|-v)     VERBOSE=true; shift ;;
        --output)         OUTPUT_FORMAT="$2"; shift 2 ;;
        --eval-results)   EVAL_RESULTS=true; shift ;;
        --list|-l)        LIST_ONLY=true; shift ;;
        --help|-h)        usage ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

export DEFAULT_PLATFORM="$PLATFORM"
export PYASC_RUNTIME_CHECK="$RUNTIME_CHECK"

list_tests() {
    echo "Available tests:"
    echo ""
    for cat_name in unit behavior integration; do
        if [ -d "$SCRIPT_DIR/$cat_name" ]; then
            echo "  $cat_name/"
            find "$SCRIPT_DIR/$cat_name" -name "test-*.sh" -type f 2>/dev/null | sort | while read -r tf; do
                local rel="${tf#$SCRIPT_DIR/}"
                echo "    $rel"
            done
        fi
    done
}

if $LIST_ONLY; then
    list_tests
    exit 0
fi

# ============================================
# Pre-flight checks
# ============================================

# When --output json, redirect all text to stderr so stdout is clean JSON
if [ "$OUTPUT_FORMAT" = "json" ]; then
    exec 3>&1 1>&2
fi

echo "========================================"
echo -e " ${BOLD}pyasc Skills Test Runner${NC}"
echo "========================================"
echo ""
echo "  Platform: $PLATFORM"
echo "  Runtime:  $RUNTIME_CHECK"
echo "  Python:   $PYTHON"
echo "  Timeout:  ${TIMEOUT}s"
echo ""

if $RUN_AGENTIC || $RUN_ALL || $RUN_INTEGRATION; then
    if ! check_opencode; then
        echo -e "${YELLOW}[WARN]${NC} opencode CLI not found; agent tests will fail/skip"
    fi
fi

# ============================================
# Test execution
# ============================================

declare -a TEST_RESULTS=()
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0
TOTAL_BLOCKED=0
START_TIME=$(date +%s)

run_test_file() {
    local test_file="$1"
    local rel_path="${test_file#$SCRIPT_DIR/}"

    echo ""
    echo -e "${BOLD}--- Running: $rel_path ---${NC}"

    local budget="$TIMEOUT"
    case "$rel_path" in
        integration/*) budget="$INTEGRATION_TIMEOUT" ;;
    esac

    local test_start
    test_start=$(date +%s)
    local exit_code=0

    if timeout "$budget" bash "$test_file"; then
        exit_code=0
    else
        exit_code=$?
    fi

    local test_end
    test_end=$(date +%s)
    local duration=$((test_end - test_start))

    if [ $exit_code -eq 0 ]; then
        TOTAL_PASS=$((TOTAL_PASS + 1))
        TEST_RESULTS+=("PASS|$rel_path|${duration}s")
    elif [ $exit_code -eq 124 ]; then
        # Blocked, not skipped. "Skipped" means we chose not to run this;
        # a timeout means it ran and we learned nothing, which is the
        # `blocked` state the rest of this repository reports separately
        # from `failed` (see docs/verification-status.md). Folding the two
        # together let an exhausted budget read as a deliberate omission.
        TOTAL_BLOCKED=$((TOTAL_BLOCKED + 1))
        TEST_RESULTS+=("BLOCK|$rel_path|timeout at ${budget}s")
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        TEST_RESULTS+=("FAIL|$rel_path|${duration}s")
    fi
}

run_category() {
    local cat_name="$1"
    local cat_dir="$SCRIPT_DIR/$cat_name"

    if [ ! -d "$cat_dir" ]; then
        echo "[WARN] Category directory not found: $cat_name"
        return
    fi

    print_section_header "Category: $cat_name"

    while IFS= read -r test_file; do
        run_test_file "$test_file"
    done < <(find "$cat_dir" -name "test-*.sh" -type f 2>/dev/null | sort)
}

if [ -n "$SINGLE_TEST" ]; then
    test_path="$SCRIPT_DIR/$SINGLE_TEST"
    if [ -f "$test_path" ]; then
        run_test_file "$test_path"
    else
        echo "[ERROR] Test not found: $SINGLE_TEST"
        exit 1
    fi
elif $RUN_ALL; then
    run_category "unit"
    run_category "behavior"
    run_category "integration"
elif $RUN_AGENTIC; then
    run_category "behavior"
    run_category "integration"
elif $RUN_INTEGRATION; then
    run_category "unit"
    run_category "behavior"
    run_category "integration"
elif [ -n "$CATEGORY" ] && [ "$CATEGORY" = "all" ]; then
    run_category "unit"
    run_category "behavior"
    run_category "integration"
elif [ -n "$CATEGORY" ]; then
    run_category "$CATEGORY"
else
    run_category "unit"
fi

# ============================================
# Optional eval report (before summary)
# ============================================

if $EVAL_RESULTS; then
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    EVAL_DIR="$REPO_ROOT/.eval-history"
    mkdir -p "$EVAL_DIR"
    EVAL_FILE="$EVAL_DIR/eval-$(date +%Y%m%d-%H%M%S).txt"
    bash "$SCRIPT_DIR/tools/eval-report.sh" > "$EVAL_FILE" 2>&1 || true
    echo ""
    echo "Eval report saved to: $EVAL_FILE"
    echo ""
fi

# ============================================
# Summary
# ============================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo -e " ${BOLD}Test Results Summary${NC}"
echo "========================================"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $TOTAL_PASS"
echo -e "  ${RED}Failed:${NC}  $TOTAL_FAIL"
echo -e "  ${YELLOW}Blocked:${NC} $TOTAL_BLOCKED"
echo -e "  ${YELLOW}Skipped:${NC} $TOTAL_SKIP"
echo "  Duration: ${TOTAL_DURATION}s"
echo ""

if [ ${#TEST_RESULTS[@]} -gt 0 ]; then
    echo "Details:"
    for result in "${TEST_RESULTS[@]}"; do
        IFS='|' read -r status name duration <<< "$result"
        case "$status" in
            PASS)  echo -e "  ${GREEN}[PASS]${NC} $name ($duration)" ;;
            FAIL)  echo -e "  ${RED}[FAIL]${NC} $name ($duration)" ;;
            BLOCK) echo -e "  ${YELLOW}[BLOCKED]${NC} $name ($duration)" ;;
            SKIP)  echo -e "  ${YELLOW}[SKIP]${NC} $name ($duration)" ;;
        esac
    done
fi

echo ""
if [ "$TOTAL_FAIL" -gt 0 ]; then
    print_status_failed
elif [ "$TOTAL_BLOCKED" -gt 0 ]; then
    echo -e "${YELLOW}STATUS: BLOCKED${NC} — $TOTAL_BLOCKED test(s) exhausted their budget."
    echo "No failure was observed, but no evidence was produced either."
else
    print_status_passed
fi

if [ "$OUTPUT_FORMAT" = "json" ]; then
    exec 1>&3 3>&-
    JSON_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [ "$TOTAL_FAIL" -gt 0 ]; then
        JSON_STATUS="failed"
    elif [ "$TOTAL_BLOCKED" -gt 0 ]; then
        JSON_STATUS="blocked"
    else
        JSON_STATUS="passed"
    fi

    json_escape_str() {
        "$PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
    }

    printf '{\n'
    printf '  "status": "%s",\n' "$JSON_STATUS"
    printf '  "passed": %d,\n' "$TOTAL_PASS"
    printf '  "failed": %d,\n' "$TOTAL_FAIL"
    printf '  "blocked": %d,\n' "$TOTAL_BLOCKED"
    printf '  "skipped": %d,\n' "$TOTAL_SKIP"
    printf '  "duration": %d,\n' "$TOTAL_DURATION"
    printf '  "timestamp": "%s",\n' "$JSON_TS"
    printf '  "tests": [\n'

    json_i=0
    json_n=${#TEST_RESULTS[@]}
    for result in "${TEST_RESULTS[@]}"; do
        IFS='|' read -r t_status t_name t_dur <<< "$result"
        case "$t_status" in
            PASS)  t_json_status="passed" ;;
            FAIL)  t_json_status="failed" ;;
            BLOCK) t_json_status="blocked" ;;
            SKIP)  t_json_status="skipped" ;;
            *)     t_json_status="unknown" ;;
        esac
        if [[ "$t_dur" =~ ^[0-9]+s$ ]]; then
            t_dur_json="${t_dur%s}"
        else
            t_dur_json="null"
        fi
        name_q=$(json_escape_str "$t_name")
        printf '    {"name": %s, "status": "%s", "duration": %s}' "$name_q" "$t_json_status" "$t_dur_json"
        json_i=$((json_i + 1))
        if [ "$json_i" -lt "$json_n" ]; then
            printf ','
        fi
        printf '\n'
    done

    printf '  ]\n'
    printf '}\n'
fi

# 1 = something failed. 2 = nothing failed but something is blocked, which is
# not success and must not be reported as such, yet is also not a defect in
# the code under test. Callers that only care about failures should test for
# exit 1 specifically rather than for non-zero.
if [ "$TOTAL_FAIL" -gt 0 ]; then
    exit 1
elif [ "$TOTAL_BLOCKED" -gt 0 ]; then
    exit 2
fi
