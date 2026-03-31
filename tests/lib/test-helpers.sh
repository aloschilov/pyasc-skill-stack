#!/usr/bin/env bash
# =============================================================================
# Helper functions for pyasc Skills tests
# Adapted from CANN Skills test harness
# =============================================================================

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$LIB_DIR/.." && pwd)"
SKILLS_DIR="$(cd "$LIB_DIR/../.." && pwd)"

# =============================================================================
# Color Output
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

setup_colors() {
    if [ -n "${NO_COLOR:-}" ]; then
        RED='' GREEN='' YELLOW='' BLUE='' CYAN='' BOLD='' NC=''
    fi
}

print_pass()  { echo -e "  ${GREEN}[PASS]${NC} $*"; }
print_fail()  { echo -e "  ${RED}[FAIL]${NC} $*"; }
print_skip()  { echo -e "  ${YELLOW}[SKIP]${NC} $*"; }
print_info()  { echo -e "  ${BLUE}[INFO]${NC} $*"; }
print_warn()  { echo -e "  ${YELLOW}[WARN]${NC} $*"; }
print_error() { echo -e "  ${RED}[ERROR]${NC} $*"; }

print_section_header() {
    echo ""
    echo -e "${BOLD}${CYAN}=== $1 ===${NC}"
    echo ""
}

setup_colors

# =============================================================================
# Skill keyword configuration (pyasc-adapted)
# =============================================================================

SKILL_KEYWORDS="pyasc|kernel|operator|API|syntax|JIT|build|verify|review|debug|test|Development|runtime|NPU"

# =============================================================================
# Assertions
# =============================================================================

assert_contains() {
    local output="$1" pattern="$2" test_name="${3:-test}"
    if echo "$output" | grep -qiE "$pattern"; then
        print_pass "$test_name"
        return 0
    else
        print_fail "$test_name"
        echo -e "  ${YELLOW}Expected to find:${NC} $pattern"
        return 1
    fi
}

assert_not_contains() {
    local output="$1" pattern="$2" test_name="${3:-test}"
    if echo "$output" | grep -qiE "$pattern"; then
        print_fail "$test_name"
        echo -e "  ${YELLOW}Did not expect:${NC} $pattern"
        return 1
    else
        print_pass "$test_name"
        return 0
    fi
}

assert_file_exists() {
    local file="$1" test_name="${2:-file exists}"
    if [ -f "$file" ]; then
        print_pass "$test_name"
        return 0
    else
        print_fail "$test_name"
        echo "  File not found: $file"
        return 1
    fi
}

# =============================================================================
# Skill & Team Queries
# =============================================================================

get_all_skills() {
    find "$SKILLS_DIR/skills" -maxdepth 2 -name "SKILL.md" -exec dirname {} \; 2>/dev/null | xargs -I{} basename {} | sort
}

get_all_teams() {
    find "$SKILLS_DIR/teams" -maxdepth 2 -name "AGENTS.md" -exec dirname {} \; 2>/dev/null | xargs -I{} basename {} | sort
}

# =============================================================================
# YAML Extraction
# =============================================================================

extract_team_skills() {
    local team_file="$1"
    local skills=()
    local in_skills=false

    while IFS= read -r line; do
        if [[ "$line" == "skills:" ]]; then
            in_skills=true
            continue
        fi
        if $in_skills; then
            if [[ "$line" =~ ^[[:space:]]+-[[:space:]]+(.+)$ ]]; then
                skills+=("${BASH_REMATCH[1]}")
            elif [[ ! "$line" =~ ^[[:space:]] ]] && [[ -n "$line" ]] && [[ "$line" != "---" ]]; then
                break
            fi
        fi
    done < "$team_file"
    echo "${skills[@]}"
}

# =============================================================================
# Structure Validation (adapted for pyasc- prefix)
# =============================================================================

validate_skill_structure() {
    local skill_file="$1"
    local skill_name=$(basename $(dirname "$skill_file"))
    local errors=()

    # S-STR-01: YAML format
    if ! head -1 "$skill_file" | grep -q "^---$"; then
        errors+=("S-STR-01: Missing opening ---")
    fi

    # S-STR-02: name field
    if ! grep -q "^name:" "$skill_file"; then
        errors+=("S-STR-02: Missing 'name' field")
    fi

    # S-STR-03: description field
    if ! grep -q "^description:" "$skill_file"; then
        errors+=("S-STR-03: Missing 'description' field")
    fi

    # S-STR-04: references directory not empty (if exists)
    local ref_dir="$(dirname "$skill_file")/references"
    if [ -d "$ref_dir" ]; then
        local ref_count=$(find "$ref_dir" -name "*.md" -type f 2>/dev/null | wc -l)
        if [ "$ref_count" -eq 0 ]; then
            errors+=("S-STR-04: Empty references directory")
        fi
    fi

    # S-STR-05: name length
    local yaml_name=$(grep "^name:" "$skill_file" | head -1 | sed 's/^name:[[:space:]]*//' | tr -d '[:space:]')
    if [ -n "$yaml_name" ]; then
        local name_len=${#yaml_name}
        if [ "$name_len" -lt 1 ] || [ "$name_len" -gt 64 ]; then
            errors+=("S-STR-05: name length must be 1-64 chars (got $name_len)")
        fi
    fi

    # S-STR-06: name format
    if [ -n "$yaml_name" ] && ! echo "$yaml_name" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
        errors+=("S-STR-06: name must match ^[a-z0-9]+(-[a-z0-9]+)*\$ (got '$yaml_name')")
    fi

    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$skill_name: ${#errors[@]} error(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        return 1
    else
        print_pass "$skill_name: Structure valid"
        return 0
    fi
}

validate_skill_content() {
    local skill_file="$1"
    local skill_name=$(basename $(dirname "$skill_file"))
    local errors=()

    # S-CON-01: name matches directory
    local yaml_name=$(grep "^name:" "$skill_file" | head -1 | cut -d: -f2 | tr -d ' ')
    if [ -n "$yaml_name" ] && [ "$yaml_name" != "$skill_name" ]; then
        errors+=("S-CON-01: name '$yaml_name' != directory '$skill_name'")
    fi

    # S-CON-02: description has trigger keywords
    local description=$(grep "^description:" "$skill_file" | head -1 | cut -d: -f2- | sed 's/^[[:space:]]*//')
    if [ -n "$description" ] && ! echo "$description" | grep -qiE "$SKILL_KEYWORDS"; then
        errors+=("S-CON-02: Description lacks trigger keywords")
    fi

    # S-CON-04: naming prefix
    if ! echo "$skill_name" | grep -qE "^(pyasc-|cann-|ascendc-|[a-z]+-)"; then
        errors+=("S-CON-04: Naming must have prefix (pyasc-, etc.)")
    fi

    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$skill_name: ${#errors[@]} error(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        return 1
    else
        print_pass "$skill_name: Content valid"
        return 0
    fi
}

validate_team_structure() {
    local team_file="$1"
    local team_name=$(basename $(dirname "$team_file"))
    local errors=()

    if ! head -1 "$team_file" | grep -q "^---$"; then
        errors+=("T-STR-01: Missing opening ---")
    fi
    if ! grep -q "^description:" "$team_file"; then
        errors+=("T-STR-02: Missing 'description' field")
    fi
    if ! grep -q "^mode:" "$team_file"; then
        errors+=("T-STR-03: Missing 'mode' field")
    fi
    if ! grep -q "^skills:" "$team_file"; then
        errors+=("T-STR-04: Missing 'skills' field")
    fi

    # Check skill dependencies exist
    local skills=$(extract_team_skills "$team_file")
    for skill in $skills; do
        local skill_file="$SKILLS_DIR/skills/$skill/SKILL.md"
        if [ ! -f "$skill_file" ]; then
            errors+=("T-STR-05: Missing skill dependency: $skill")
        fi
    done

    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$team_name: ${#errors[@]} error(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        return 1
    else
        print_pass "$team_name: Structure valid"
        return 0
    fi
}

# =============================================================================
# Test tracking
# =============================================================================

init_test_tracking() {
    TEST_PASSED=0
    TEST_FAILED=0
    TEST_SKIPPED=0
    TEST_START_TIME=$(date +%s)
}

record_test() {
    local result="$1"
    case "$result" in
        pass|PASS) ((TEST_PASSED++)) || true ;;
        fail|FAIL) ((TEST_FAILED++)) || true ;;
        skip|SKIP) ((TEST_SKIPPED++)) || true ;;
    esac
}

print_test_summary() {
    local end_time=$(date +%s)
    local duration=$((end_time - TEST_START_TIME))
    echo ""
    echo "========================================"
    echo -e " ${BOLD}Test Results${NC}"
    echo "========================================"
    echo -e "  ${GREEN}Passed:${NC}  $TEST_PASSED"
    echo -e "  ${RED}Failed:${NC}  $TEST_FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $TEST_SKIPPED"
    echo "  Duration: ${duration}s"
    if [ "$TEST_FAILED" -gt 0 ]; then
        echo -e "${RED}${BOLD}STATUS: FAILED${NC}"
        return 1
    else
        echo -e "${GREEN}${BOLD}STATUS: PASSED${NC}"
        return 0
    fi
}

# =============================================================================
# Exports
# =============================================================================

export -f print_pass print_fail print_skip print_info print_warn print_error print_section_header
export -f assert_contains assert_not_contains assert_file_exists
export -f get_all_skills get_all_teams extract_team_skills
export -f validate_skill_structure validate_skill_content validate_team_structure
export -f init_test_tracking record_test print_test_summary
export LIB_DIR TESTS_DIR SKILLS_DIR SKILL_KEYWORDS
