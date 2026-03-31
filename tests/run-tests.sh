#!/usr/bin/env bash
# =============================================================================
# pyasc Skills Testing Framework
# Mirrors the Ascend C test harness structure (L1/L2/L3 pyramid)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/test-helpers.sh"

# Defaults
CATEGORY="unit"
RUN_INTEGRATION=false
RUN_ALL=false
SINGLE_TEST=""
TIMEOUT=300
VERBOSE=false
OUTPUT_FORMAT="text"
LIST_ONLY=false
PLATFORM="auto"

usage() {
    cat <<EOF
pyasc Skills Testing Framework

Usage: $0 [options]

Options:
  --fast, -f            Run unit tests only (default)
  --integration, -i     Include integration tests
  --all                 Run all tests
  --category, -c CAT    Run specific category (unit/behavior/integration/all)
  --platform PLATFORM   Specify platform (claude/opencode/auto)
  --test, -t NAME       Run specific test
  --timeout SECONDS     Set timeout (default: 300)
  --verbose, -v         Show verbose output
  --output FORMAT       Output format (text/json)
  --list, -l            List all available tests
  --help, -h            Display this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast|-f)      CATEGORY="unit"; shift ;;
        --integration|-i) RUN_INTEGRATION=true; shift ;;
        --all)          RUN_ALL=true; shift ;;
        --category|-c)  CATEGORY="$2"; shift 2 ;;
        --platform)     PLATFORM="$2"; shift 2 ;;
        --test|-t)      SINGLE_TEST="$2"; shift 2 ;;
        --timeout)      TIMEOUT="$2"; shift 2 ;;
        --verbose|-v)   VERBOSE=true; shift ;;
        --output)       OUTPUT_FORMAT="$2"; shift 2 ;;
        --list|-l)      LIST_ONLY=true; shift ;;
        --help|-h)      usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

list_tests() {
    echo "Available tests:"
    echo ""
    for category in unit behavior integration; do
        if [ -d "$SCRIPT_DIR/$category" ]; then
            echo "  $category/"
            find "$SCRIPT_DIR/$category" -name "test-*.sh" -type f 2>/dev/null | sort | while read -r test_file; do
                local rel_path="${test_file#$SCRIPT_DIR/}"
                echo "    $rel_path"
            done
        fi
    done
}

if $LIST_ONLY; then
    list_tests
    exit 0
fi

# Test results tracking
declare -a TEST_RESULTS=()
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0
START_TIME=$(date +%s)

run_test_file() {
    local test_file="$1"
    local rel_path="${test_file#$SCRIPT_DIR/}"

    echo ""
    echo -e "${BOLD}--- Running: $rel_path ---${NC}"

    local test_start=$(date +%s)
    local exit_code=0

    if timeout "$TIMEOUT" bash "$test_file"; then
        exit_code=0
    else
        exit_code=$?
    fi

    local test_end=$(date +%s)
    local duration=$((test_end - test_start))

    if [ $exit_code -eq 0 ]; then
        TOTAL_PASS=$((TOTAL_PASS + 1))
        TEST_RESULTS+=("PASS|$rel_path|${duration}s")
    elif [ $exit_code -eq 124 ]; then
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        TEST_RESULTS+=("SKIP|$rel_path|timeout")
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        TEST_RESULTS+=("FAIL|$rel_path|${duration}s")
    fi
}

run_category() {
    local category="$1"
    local category_dir="$SCRIPT_DIR/$category"

    if [ ! -d "$category_dir" ]; then
        echo "[WARN] Category directory not found: $category"
        return
    fi

    print_section_header "Category: $category"

    find "$category_dir" -name "test-*.sh" -type f 2>/dev/null | sort | while read -r test_file; do
        run_test_file "$test_file"
    done
}

# Run specific test
if [ -n "$SINGLE_TEST" ]; then
    test_path="$SCRIPT_DIR/$SINGLE_TEST"
    if [ -f "$test_path" ]; then
        run_test_file "$test_path"
    else
        echo "[ERROR] Test not found: $SINGLE_TEST"
        exit 1
    fi
else
    # Run by category
    if $RUN_ALL; then
        run_category "unit"
        run_category "behavior"
        run_category "integration"
    elif $RUN_INTEGRATION; then
        run_category "unit"
        run_category "behavior"
        run_category "integration"
    elif [ "$CATEGORY" = "all" ]; then
        run_category "unit"
        run_category "behavior"
        run_category "integration"
    else
        run_category "$CATEGORY"
    fi
fi

# Summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo -e " ${BOLD}Test Results Summary${NC}"
echo "========================================"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $TOTAL_PASS"
echo -e "  ${RED}Failed:${NC}  $TOTAL_FAIL"
echo -e "  ${YELLOW}Skipped:${NC} $TOTAL_SKIP"
echo "  Duration: ${TOTAL_DURATION}s"
echo ""

if [ ${#TEST_RESULTS[@]} -gt 0 ]; then
    echo "Details:"
    for result in "${TEST_RESULTS[@]}"; do
        IFS='|' read -r status name duration <<< "$result"
        case "$status" in
            PASS) echo -e "  ${GREEN}[PASS]${NC} $name ($duration)" ;;
            FAIL) echo -e "  ${RED}[FAIL]${NC} $name ($duration)" ;;
            SKIP) echo -e "  ${YELLOW}[SKIP]${NC} $name ($duration)" ;;
        esac
    done
fi

echo ""
if [ "$TOTAL_FAIL" -gt 0 ]; then
    echo -e "${RED}${BOLD}STATUS: FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}${BOLD}STATUS: PASSED${NC}"
fi
