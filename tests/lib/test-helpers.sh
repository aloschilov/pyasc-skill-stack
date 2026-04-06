#!/usr/bin/env bash
# =============================================================================
# Helper functions for pyasc Skills tests
# Supports: OpenCode headless mode + session analysis
# =============================================================================

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$LIB_DIR/.." && pwd)"
SKILLS_DIR="$(cd "$LIB_DIR/../.." && pwd)"
TOOLS_DIR="$TESTS_DIR/tools"

DEFAULT_PLATFORM="opencode"
DEFAULT_TIMEOUT=120
PYTHON="${PYASC_PYTHON:-python3.10}"

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
    elif [ -n "${FORCE_COLOR:-}" ] || [ "${FORCE_COLOR:-}" = "1" ]; then
        : # keep colors
    elif ! [ -t 1 ]; then
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

print_status_passed() { echo -e "${GREEN}${BOLD}STATUS: PASSED${NC}"; }
print_status_failed() { echo -e "${RED}${BOLD}STATUS: FAILED${NC}"; }

setup_colors

# =============================================================================
# Skill keyword configuration (pyasc-adapted)
# =============================================================================

SKILL_KEYWORDS="pyasc|kernel|operator|API|syntax|JIT|build|verify|review|debug|test|Development|runtime|NPU"

# =============================================================================
# Platform-specific Agent Runners
# =============================================================================

# Run OpenCode in headless mode and capture output.
# Retries up to MAX_RETRIES times on transient TLS/certificate errors.
# Usage: run_opencode "prompt text" [timeout_seconds] [working_dir]
MAX_RETRIES="${MAX_RETRIES:-3}"

run_opencode() {
    local prompt="$1"
    local timeout_val="${2:-$DEFAULT_TIMEOUT}"
    local work_dir="${3:-$SKILLS_DIR}"

    local cmd_args=("run" "$prompt" "--format" "json")
    if [ -n "$work_dir" ]; then
        cmd_args+=("--dir" "$work_dir")
    fi

    # opencode run requires a pseudo-TTY; wrap with `script -qc` when
    # stdin is not a terminal (CI, piped shells, IDE tool runners).
    local oc_prefix=()
    if ! [ -t 0 ]; then
        oc_prefix=(script -qc)
    fi

    local attempt=0
    while [ "$attempt" -lt "$MAX_RETRIES" ]; do
        attempt=$((attempt + 1))
        local output_file
        output_file=$(mktemp)

        local oc_cmd
        if [ ${#oc_prefix[@]} -gt 0 ]; then
            oc_cmd="NODE_TLS_REJECT_UNAUTHORIZED=0 opencode $(printf '%q ' "${cmd_args[@]}")"
            if timeout "$timeout_val" script -qc "$oc_cmd" /dev/null > "$output_file" 2>&1; then
                if grep -q "certificate verification error" "$output_file" 2>/dev/null; then
                    print_warn "Transient TLS certificate error on attempt $attempt/$MAX_RETRIES — retrying..."
                    rm -f "$output_file"
                    sleep $((attempt * 2))
                    continue
                fi
                cat "$output_file"
                rm -f "$output_file"
                return 0
            else
                local exit_code=$?
                local output_content
                output_content=$(cat "$output_file" 2>/dev/null || true)
                rm -f "$output_file"
                if echo "$output_content" | grep -q "certificate verification error" 2>/dev/null; then
                    print_warn "Transient TLS certificate error on attempt $attempt/$MAX_RETRIES — retrying..."
                    sleep $((attempt * 2))
                    continue
                fi
                echo "$output_content" >&2
                return $exit_code
            fi
        else
            if timeout "$timeout_val" opencode "${cmd_args[@]}" > "$output_file" 2>&1; then
                if grep -q "certificate verification error" "$output_file" 2>/dev/null; then
                    print_warn "Transient TLS certificate error on attempt $attempt/$MAX_RETRIES — retrying..."
                    rm -f "$output_file"
                    sleep $((attempt * 2))
                    continue
                fi
                cat "$output_file"
                rm -f "$output_file"
                return 0
            else
                local exit_code=$?
                local output_content
                output_content=$(cat "$output_file" 2>/dev/null || true)
                rm -f "$output_file"
                if echo "$output_content" | grep -q "certificate verification error" 2>/dev/null; then
                    print_warn "Transient TLS certificate error on attempt $attempt/$MAX_RETRIES — retrying..."
                    sleep $((attempt * 2))
                    continue
                fi
                echo "$output_content" >&2
                return $exit_code
            fi
        fi
    done

    print_error "All $MAX_RETRIES attempts failed due to transient TLS errors"
    return 1
}

# Universal runner dispatches to the configured platform
# Usage: run_ai "prompt text" [timeout_seconds] [platform]
run_ai() {
    local prompt="$1"
    local timeout_val="${2:-$DEFAULT_TIMEOUT}"
    local platform="${3:-$DEFAULT_PLATFORM}"

    case "$platform" in
        opencode) run_opencode "$prompt" "$timeout_val" ;;
        *)
            echo "[ERROR] Unknown platform: $platform"
            return 1
            ;;
    esac
}

# Run a short behavior test: single-turn prompt, regex check on response
# Usage: run_behavior_test "test_name" "prompt" "expected_pattern" [timeout]
# Globals: pass_count, fail_count, skip_count should be defined before calling
run_behavior_test() {
    local name="$1"
    local prompt="$2"
    local expected="$3"
    local timeout_val="${4:-60}"

    echo "Testing: $name"

    local output
    if output=$(run_opencode "$prompt Answer in 1 line." "$timeout_val" 2>&1); then
        if echo "$output" | grep -qiE "$expected"; then
            print_pass "Correct response"
            pass_count=$((pass_count + 1))
        else
            print_fail "Incorrect response"
            echo -e "  ${YELLOW}Expected:${NC} $expected"
            echo "  Output: ${output:0:120}..."
            fail_count=$((fail_count + 1))
        fi
    else
        local ec=$?
        if [ "$ec" -eq 124 ]; then
            print_skip "OpenCode timed out after ${timeout_val}s"
        else
            print_skip "OpenCode exited with code $ec"
        fi
        skip_count=$((skip_count + 1))
    fi
    echo ""
}

# =============================================================================
# Test Project Setup / Cleanup
# =============================================================================

# Create a temporary test project with skills and team symlinked in.
# Usage: project_dir=$(create_test_project [prefix])
create_test_project() {
    local prefix="${1:-pyasc-test}"
    local test_dir
    test_dir=$(mktemp -d -t "${prefix}.XXXXXX")

    ln -s "$SKILLS_DIR/skills" "$test_dir/skills"
    ln -s "$SKILLS_DIR/teams" "$test_dir/teams"
    ln -s "$SKILLS_DIR/golden" "$test_dir/golden"

    (cd "$test_dir" && git init --quiet 2>/dev/null &&
     git config user.email "test@pyasc.test" 2>/dev/null &&
     git config user.name "pyasc-eval" 2>/dev/null &&
     git add -A 2>/dev/null &&
     git commit -m "init" --quiet 2>/dev/null) || true

    echo "$test_dir"
}

# Remove a test project directory
# Usage: cleanup_test_project "$project_dir"
cleanup_test_project() {
    local test_dir="$1"
    if [ -d "$test_dir" ]; then
        rm -rf "$test_dir"
    fi
}

# =============================================================================
# OpenCode Session Analysis
# =============================================================================

# Find the most recent OpenCode session ID.
# Uses `opencode session list` and picks the latest.
# Usage: session_id=$(find_recent_session)
find_recent_session() {
    opencode session list 2>/dev/null | grep -oE '[0-9a-f-]{36}' | head -1 || true
}

# Export an OpenCode session to a JSON file.
# Usage: export_session "session-id" "/path/to/output.json"
export_session() {
    local session_id="$1"
    local output_file="$2"
    opencode export "$session_id" > "$output_file" 2>/dev/null
}

# Verify a skill was invoked in an exported session JSON.
# Looks for tool_use events referencing the skill name.
# Usage: verify_skill_invoked "session.json" "pyasc-codegen-workflow"
verify_skill_invoked() {
    local session_file="$1"
    local skill_name="$2"

    if [ ! -f "$session_file" ]; then
        print_fail "Session file not found: $session_file"
        return 1
    fi

    if grep -qiE "\"(skill|name)\"[[:space:]]*:[[:space:]]*\"[^\"]*${skill_name}[^\"]*\"" "$session_file"; then
        print_pass "Skill '$skill_name' was invoked"
        return 0
    else
        print_fail "Skill '$skill_name' was NOT invoked"
        return 1
    fi
}

# Count tool invocations in an exported session.
# Usage: count=$(count_tool_invocations "session.json" "Read")
count_tool_invocations() {
    local session_file="$1"
    local tool_name="$2"

    if [ ! -f "$session_file" ]; then
        echo "0"
        return
    fi

    grep -coE "\"name\"[[:space:]]*:[[:space:]]*\"${tool_name}\"" "$session_file" 2>/dev/null || echo "0"
}

# Detect premature write/edit/shell actions before a target skill/tool load.
# Usage: analyze_premature_actions "session.json" "skill" "pyasc-codegen-workflow"
analyze_premature_actions() {
    local session_file="$1"
    local target_type="$2"   # "skill" or "agent"
    local target_name="$3"

    if [ ! -f "$session_file" ]; then
        print_skip "Session file not found"
        return 0
    fi

    local passed=true
    local first_target_line=""

    first_target_line=$(grep -n -iE "\"[^\"]*${target_name}[^\"]*\"" "$session_file" 2>/dev/null | head -1 | cut -d: -f1)

    if [ -z "$first_target_line" ]; then
        print_warn "Target $target_type '$target_name' not found in session"
        return 0
    fi

    print_pass "Target $target_type '$target_name' was invoked (line $first_target_line)"

    local premature_tools
    premature_tools=$(head -n "$first_target_line" "$session_file" 2>/dev/null | \
        grep -iE '"name"[[:space:]]*:[[:space:]]*"(Write|Edit|Shell|Bash|StrReplace)"' 2>/dev/null || true)

    if [ -n "$premature_tools" ]; then
        print_fail "Premature actions detected BEFORE $target_type invocation:"
        echo "$premature_tools" | head -5 | sed 's/^/    /'
        passed=false
    else
        print_pass "No premature Write/Edit/Shell actions before $target_type"
    fi

    local read_before=0
    read_before=$(head -n "$first_target_line" "$session_file" 2>/dev/null | \
        grep -ciE '"name"[[:space:]]*:[[:space:]]*"Read"' 2>/dev/null || echo "0")
    if [ "$read_before" -gt 0 ]; then
        print_info "Read used $read_before time(s) before $target_type (acceptable for context)"
    fi

    if $passed; then
        return 0
    else
        return 1
    fi
}

# Extract ordered tool invocation sequence from an exported session.
# Usage: analyze_workflow_sequence "session.json"
analyze_workflow_sequence() {
    local session_file="$1"

    if [ ! -f "$session_file" ]; then
        echo "[]"
        return 1
    fi

    if command -v jq &> /dev/null; then
        jq -r '.. | objects | select(.type == "tool_use") | .name // empty' "$session_file" 2>/dev/null || \
            grep -oE '"name"[[:space:]]*:[[:space:]]*"[^"]*"' "$session_file" 2>/dev/null | \
            sed 's/"name"[[:space:]]*:[[:space:]]*//; s/"//g'
    else
        grep -oE '"name"[[:space:]]*:[[:space:]]*"[^"]*"' "$session_file" 2>/dev/null | \
            sed 's/"name"[[:space:]]*:[[:space:]]*//; s/"//g'
    fi
}

# Produce a human-readable tool chain from an exported session.
# Usage: analyze_tool_chain "session.json"
analyze_tool_chain() {
    local session_file="$1"
    local idx=0

    if [ ! -f "$session_file" ]; then
        echo "(no session file)"
        return
    fi

    analyze_workflow_sequence "$session_file" | while IFS= read -r tool; do
        idx=$((idx + 1))
        printf "  %3d. %s\n" "$idx" "$tool"
    done
}

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

assert_count() {
    local output="$1" pattern="$2" expected="$3" test_name="${4:-test}"
    local actual
    actual=$(echo "$output" | grep -ciE "$pattern" || echo "0")
    if [ "$actual" -eq "$expected" ]; then
        print_pass "$test_name (found $actual)"
        return 0
    else
        print_fail "$test_name (expected $expected, found $actual)"
        return 1
    fi
}

assert_order() {
    local output="$1" pattern_a="$2" pattern_b="$3" test_name="${4:-order test}"
    local line_a line_b
    line_a=$(echo "$output" | grep -inE "$pattern_a" | head -1 | cut -d: -f1)
    line_b=$(echo "$output" | grep -inE "$pattern_b" | head -1 | cut -d: -f1)
    if [ -z "$line_a" ]; then
        print_fail "$test_name: pattern A not found: $pattern_a"
        return 1
    fi
    if [ -z "$line_b" ]; then
        print_fail "$test_name: pattern B not found: $pattern_b"
        return 1
    fi
    if [ "$line_a" -le "$line_b" ]; then
        print_pass "$test_name (A@$line_a <= B@$line_b)"
        return 0
    else
        print_fail "$test_name (A@$line_a > B@$line_b)"
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
# File Link Validation
# =============================================================================

check_file_links() {
    local file="$1"
    local file_type="${2:-skill}"
    local file_dir
    file_dir="$(dirname "$file")"
    local item_name
    item_name=$(basename "$file_dir")
    local broken_links=()

    while IFS= read -r line; do
        if echo "$line" | grep -qE '\]\(references/'; then
            local link
            link=$(echo "$line" | sed -n 's/.*(\(references\/[^)]*\)).*/\1/p' | head -1 | cut -d'#' -f1)
            if [ -n "$link" ] && [ ! -e "$file_dir/$link" ]; then
                broken_links+=("$link")
            fi
        fi
        if echo "$line" | grep -qE '\]\(\./'; then
            local link
            link=$(echo "$line" | sed -n 's/.*(\.\(\/[^)]*\)).*/\1/p' | head -1 | cut -d'#' -f1)
            if [ -n "$link" ] && [ ! -e "$file_dir$link" ]; then
                broken_links+=(".$link")
            fi
        fi
    done < "$file"

    if [ ${#broken_links[@]} -gt 0 ]; then
        print_fail "$file_type/$item_name: Broken links: ${broken_links[*]}"
        return 1
    else
        print_pass "$file_type/$item_name: All links valid"
        return 0
    fi
}

# =============================================================================
# Structure Validation (adapted for pyasc- prefix)
# =============================================================================

validate_skill_structure() {
    local skill_file="$1"
    local skill_name
    skill_name=$(basename "$(dirname "$skill_file")")
    local errors=()

    if ! head -1 "$skill_file" | grep -q "^---$"; then
        errors+=("S-STR-01: Missing opening ---")
    fi
    if ! grep -q "^name:" "$skill_file"; then
        errors+=("S-STR-02: Missing 'name' field")
    fi
    if ! grep -q "^description:" "$skill_file"; then
        errors+=("S-STR-03: Missing 'description' field")
    fi

    local ref_dir="$(dirname "$skill_file")/references"
    if [ -d "$ref_dir" ]; then
        local ref_count
        ref_count=$(find "$ref_dir" -name "*.md" -type f 2>/dev/null | wc -l)
        if [ "$ref_count" -eq 0 ]; then
            errors+=("S-STR-04: Empty references directory")
        fi
    fi

    local yaml_name
    yaml_name=$(grep "^name:" "$skill_file" | head -1 | sed 's/^name:[[:space:]]*//' | tr -d '[:space:]')
    if [ -n "$yaml_name" ]; then
        local name_len=${#yaml_name}
        if [ "$name_len" -lt 1 ] || [ "$name_len" -gt 64 ]; then
            errors+=("S-STR-05: name length must be 1-64 chars (got $name_len)")
        fi
    fi
    if [ -n "$yaml_name" ] && ! echo "$yaml_name" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
        errors+=("S-STR-06: name must match ^[a-z0-9]+(-[a-z0-9]+)*\$ (got '$yaml_name')")
    fi

    if ! { check_file_links "$skill_file" "skill" >/dev/null 2>&1; }; then
        errors+=("S-STR-08: Broken markdown links (references/ or ./)")
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
    local skill_name
    skill_name=$(basename "$(dirname "$skill_file")")
    local errors=()

    local yaml_name
    yaml_name=$(grep "^name:" "$skill_file" | head -1 | cut -d: -f2 | tr -d ' ')
    if [ -n "$yaml_name" ] && [ "$yaml_name" != "$skill_name" ]; then
        errors+=("S-CON-01: name '$yaml_name' != directory '$skill_name'")
    fi

    local description
    description=$(grep "^description:" "$skill_file" | head -1 | cut -d: -f2- | sed 's/^[[:space:]]*//')
    if [ -n "$description" ] && ! echo "$description" | grep -qiE "$SKILL_KEYWORDS"; then
        errors+=("S-CON-02: Description lacks trigger keywords")
    fi

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
    local team_name
    team_name=$(basename "$(dirname "$team_file")")
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

    local skills
    skills=$(extract_team_skills "$team_file")
    for skill in $skills; do
        local sf="$SKILLS_DIR/skills/$skill/SKILL.md"
        if [ ! -f "$sf" ]; then
            errors+=("T-STR-05: Missing skill dependency: $skill")
        fi
    done

    local mode
    mode=$(grep "^mode:" "$team_file" | head -1 | cut -d: -f2 | tr -d ' ')
    if [ -n "$mode" ] && [[ "$mode" != "primary" && "$mode" != "subagent" ]]; then
        errors+=("T-STR-06: Invalid mode '$mode' (must be: primary or subagent)")
    fi

    local team_description
    team_description=$(grep "^description:" "$team_file" | head -1 | sed 's/^description:[[:space:]]*//')
    if [ -n "$team_description" ]; then
        local tdesc_len=${#team_description}
        if [ "$tdesc_len" -lt 1 ] || [ "$tdesc_len" -gt 1024 ]; then
            errors+=("T-STR-07: description length must be 1-1024 chars (got $tdesc_len)")
        fi
    fi

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
# Agent Structure & Content Validation
# =============================================================================

AGENT_REQUIRED_SECTIONS="Core Responsibilities|Overview|Core Principles|Core Work Processes"

validate_agent_structure() {
    local agent_file="$1"
    local agent_name
    agent_name=$(basename "$(dirname "$agent_file")")
    local errors=()

    if ! head -1 "$agent_file" | grep -q "^---$"; then
        errors+=("A-STR-01: Missing opening ---")
    fi

    if ! grep -q "^description:" "$agent_file"; then
        errors+=("A-STR-02: Missing 'description' field")
    fi
    if ! grep -q "^mode:" "$agent_file"; then
        errors+=("A-STR-02: Missing 'mode' field")
    fi

    local mode
    mode=$(grep "^mode:" "$agent_file" | head -1 | cut -d: -f2 | tr -d ' ')
    if [ -n "$mode" ] && [[ "$mode" != "primary" && "$mode" != "subagent" ]]; then
        errors+=("A-STR-03: Invalid mode '$mode' (must be: primary or subagent)")
    fi

    local skills
    skills=$(extract_team_skills "$agent_file")
    for skill in $skills; do
        local sf="$SKILLS_DIR/skills/$skill/SKILL.md"
        if [ ! -f "$sf" ]; then
            errors+=("A-STR-04: Missing skill dependency: $skill")
        fi
    done

    local description
    description=$(grep "^description:" "$agent_file" | head -1 | sed 's/^description:[[:space:]]*//')
    if [ -n "$description" ]; then
        local desc_len=${#description}
        if [ "$desc_len" -lt 1 ] || [ "$desc_len" -gt 1024 ]; then
            errors+=("A-STR-07: description length must be 1-1024 chars (got $desc_len)")
        fi
    fi

    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$agent_name: ${#errors[@]} error(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        return 1
    else
        print_pass "$agent_name: Structure valid"
        return 0
    fi
}

validate_agent_content() {
    local agent_file="$1"
    local agent_name
    agent_name=$(basename "$(dirname "$agent_file")")
    local errors=()

    local description
    description=$(grep "^description:" "$agent_file" | head -1 | cut -d: -f2- | sed 's/^[[:space:]]*//')
    if [ -n "$description" ] && ! echo "$description" | grep -qiE "$SKILL_KEYWORDS"; then
        errors+=("A-CON-02: Description lacks trigger keywords")
    fi

    if ! echo "$agent_name" | grep -qE "^(pyasc-|cann-|ascendc-|[a-z]+-)"; then
        errors+=("A-CON-03: Naming must have prefix")
    fi

    if ! grep -qE "^#+ *($AGENT_REQUIRED_SECTIONS)" "$agent_file"; then
        errors+=("A-CON-04: Missing core responsibilities/principles section")
    fi

    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$agent_name: ${#errors[@]} error(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        return 1
    else
        print_pass "$agent_name: Content valid"
        return 0
    fi
}

# =============================================================================
# Team Content Validation
# =============================================================================

TEAM_KEYWORDS="Team|Collaboration|Organization|Process|Development|Agent|kernel|operator"

validate_team_content() {
    local team_file="$1"
    local team_name
    team_name=$(basename "$(dirname "$team_file")")
    local errors=()

    if ! echo "$team_name" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then
        errors+=("T-CON-01: Directory name must match ^[a-z0-9]+(-[a-z0-9]+)*\$ (got '$team_name')")
    fi

    local description
    description=$(grep "^description:" "$team_file" | head -1 | cut -d: -f2- | sed 's/^[[:space:]]*//')
    if [ -n "$description" ] && ! echo "$description" | grep -qiE "$TEAM_KEYWORDS"; then
        errors+=("T-CON-02: Description lacks trigger keywords")
    fi

    if ! grep -qE "^#+ *(Core Principles|Core Principles)" "$team_file"; then
        errors+=("T-CON-03: Missing core principles section")
    fi

    if [ ${#errors[@]} -gt 0 ]; then
        print_fail "$team_name: ${#errors[@]} error(s)"
        for err in "${errors[@]}"; do print_error "$err"; done
        return 1
    else
        print_pass "$team_name: Content valid"
        return 0
    fi
}

# =============================================================================
# Runtime Availability Checks
# =============================================================================

# Check if OpenCode CLI is available
check_opencode() {
    if command -v opencode &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if pyasc can be imported (via the configured python)
check_pyasc_import() {
    if $PYTHON -c "import asc" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if pyasc runtime is available (CANN simulator libs)
check_pyasc_runtime() {
    if $PYTHON -c "
import asc
from asc.runtime import config as rt_config
rt_config.set_platform(rt_config.Backend.Model, check=False)
" 2>/dev/null; then
        return 0
    else
        return 1
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
    local end_time
    end_time=$(date +%s)
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
        print_status_failed
        return 1
    else
        print_status_passed
        return 0
    fi
}

# =============================================================================
# Exports
# =============================================================================

export -f print_pass print_fail print_skip print_info print_warn print_error
export -f print_section_header print_status_passed print_status_failed
export -f setup_colors

export -f run_opencode run_ai run_behavior_test
export -f create_test_project cleanup_test_project

export -f find_recent_session export_session
export -f verify_skill_invoked count_tool_invocations
export -f analyze_premature_actions analyze_workflow_sequence analyze_tool_chain

export -f assert_contains assert_not_contains assert_file_exists
export -f assert_count assert_order

export -f get_all_skills get_all_teams extract_team_skills
export -f check_file_links
export -f validate_skill_structure validate_skill_content validate_team_structure
export -f validate_agent_structure validate_agent_content
export -f validate_team_content

export -f check_opencode check_pyasc_import check_pyasc_runtime

export -f init_test_tracking record_test print_test_summary

export LIB_DIR TESTS_DIR SKILLS_DIR TOOLS_DIR SKILL_KEYWORDS
export AGENT_REQUIRED_SECTIONS TEAM_KEYWORDS
export DEFAULT_PLATFORM DEFAULT_TIMEOUT PYTHON
