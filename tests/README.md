# pyasc-skill-stack test suite

This directory holds the **pyasc Skills Testing Framework**: a three-layer pyramid (L1 unit, L2 behavior, L3 integration), a shared Bash helper library, and auxiliary Python/Bash tools under `tests/tools/`.

The canonical entry point is [`run-tests.sh`](run-tests.sh).

---

## Quick start

From the repository root:

```bash
./tests/run-tests.sh --fast
```

| Command | What runs |
|--------|-----------|
| `./tests/run-tests.sh` | **Unit (L1) only** — same as `--fast` |
| `./tests/run-tests.sh --fast` / `-f` | Unit tests only (no OpenCode, no integration) |
| `./tests/run-tests.sh --agentic` / `-a` | **L2 behavior + L3 integration** (expects `opencode` on `PATH`) |
| `./tests/run-tests.sh --integration` / `-i` | **L1 + L2 + L3** |
| `./tests/run-tests.sh --all` | **L1 + L2 + L3** (same selection as `--integration`) |
| `./tests/run-tests.sh --category unit\|behavior\|integration` | Single category |
| `./tests/run-tests.sh --test integration/test-simple-kernel.sh` | One script, path relative to `tests/` |
| `./tests/run-tests.sh --list` / `-l` | List discovered `test-*.sh` files |

Useful options:

- `--runtime` — sets `PYASC_RUNTIME_CHECK=1` for subprocesses (see [Environment variables](#environment-variables)).
- `--platform PLATFORM` — agent platform (default `opencode`; exported as `DEFAULT_PLATFORM`).
- `--timeout SECONDS` — per-script timeout (default `300`; uses `timeout(1)`).
- `--verbose` / `-v` — more noise from the runner.
- `--output text|json` — when `json`, appends a machine-readable JSON summary to stdout after the text report. Shape: `{"status","passed","failed","skipped","duration","timestamp","tests":[{"name","status","duration"}]}`. Pipe through `jq` for clean output; see also `--output json` in Step 4 below for clean JSON-only mode.

**OpenCode:** For `--agentic`, `--integration`, or `--all`, install the `opencode` CLI and ensure it is on `PATH`. If it is missing, the runner warns; behavior and integration scripts may fail or skip.

---

## Directory structure

```
tests/
├── unit/
│   ├── skills/
│   │   ├── test-structure.sh    # Skill YAML/layout rules (S-STR-*)
│   │   └── test-content.sh      # Skill naming/description rules (S-CON-*)
│   ├── agents/
│   │   ├── test-structure.sh    # Team AGENTS.md structure (A-STR-*)
│   │   └── test-content.sh      # Agent content rules (A-CON-*)
│   └── teams/
│       ├── test-structure.sh    # Team AGENTS.md structure (T-STR-*)
│       └── test-content.sh      # Team content rules (T-CON-*)
├── behavior/
│   ├── skills/
│   │   ├── test-trigger-correctness.sh
│   │   ├── test-premature-action.sh
│   │   └── test-workflow-enforcement.sh
│   └── agents/
│       ├── test-trigger-correctness.sh
│       └── test-premature-action.sh
├── integration/
│   ├── test-simple-kernel.sh
│   ├── test-kernel-generation.sh
│   ├── test-golden-comparison.sh
│   └── test-workflow-execution.sh
├── tools/
│   ├── verify_kernel.py          # Static AST verifier for kernels
│   ├── score_kernel.py           # Code-review checklist scorer
│   ├── run_and_verify.py         # Runtime verification (jit / simulator / auto)
│   ├── pytest_verify_kernel.py   # JIT-focused pytest-style verifier
│   ├── gen_golden.py             # Golden numpy data generator
│   ├── eval-report.sh            # Aggregated evaluation report
│   ├── analyze-session.sh        # OpenCode session inspection
│   ├── analyze-workflow.sh       # Workflow / ordering heuristics
│   ├── analyze-tokens.sh         # Token usage (delegates to Python when possible)
│   └── analyze-token-usage.py    # Token usage parser
├── lib/
│   └── test-helpers.sh           # Shared library (sourced by tests and some tools)
└── run-tests.sh                  # Main test runner
```

---

## Test layering (L1 / L2 / L3)

| Layer | Directory | Purpose | Typical dependencies |
|-------|-----------|---------|----------------------|
| **L1** | `unit/` | Fast, deterministic checks on repo files (skills, teams, `AGENTS.md`) | Bash, repo checkout |
| **L2** | `behavior/` | Agent-in-the-loop checks via OpenCode headless runs | `opencode`, network (model provider) |
| **L3** | `integration/` | End-to-end scripts (init scripts, golden assets, workflows) | Bash; optional Python/pyasc for deeper paths |

The runner discovers tests with:

`find <category-dir> -name 'test-*.sh' -type f`

So new suites must use the `test-*.sh` naming convention.

---

## Rule IDs by category

Validators live in [`lib/test-helpers.sh`](lib/test-helpers.sh). Unit scripts call these functions for every skill or team under `skills/` and `teams/`.

### Skills (`unit/skills/`)

| ID | Meaning |
|----|---------|
| **S-STR-01** | Front matter: opening `---` |
| **S-STR-02** | YAML: `name` present |
| **S-STR-03** | YAML: `description` present |
| **S-STR-04** | If `references/` exists, it must contain at least one `.md` file |
| **S-STR-05** | `name` length 1–64 characters |
| **S-STR-06** | `name` matches `^[a-z0-9]+(-[a-z0-9]+)*$` |
| **S-STR-08** | No broken `references/` or `./` markdown links |
| **S-CON-01** | YAML `name` matches parent directory name |
| **S-CON-02** | Description matches trigger keyword regex (`SKILL_KEYWORDS` in helpers) |
| **S-CON-04** | Directory/skill naming uses an allowed prefix pattern (e.g. `pyasc-`, `cann-`, …) |

The structure test script header refers to S-STR-01–06; **S-STR-08** is still enforced by `validate_skill_structure` when links are invalid. There is no **S-STR-07** in the current helper implementation. There is no **S-CON-03** code path today.

### Agents (`unit/agents/` — `teams/*/AGENTS.md`)

| ID | Meaning |
|----|---------|
| **A-STR-01** | Opening `---` |
| **A-STR-02** | `description` present *(and separately: `mode` present — same label in code)* |
| **A-STR-03** | `mode` is `primary` or `subagent` |
| **A-STR-04** | Each listed skill exists under `skills/<name>/SKILL.md` |
| **A-STR-07** | `description` length 1–1024 characters |
| **A-CON-02** | Description matches `SKILL_KEYWORDS` |
| **A-CON-03** | Agent directory name uses allowed prefix pattern |
| **A-CON-04** | Markdown contains expected core sections (see `AGENT_REQUIRED_SECTIONS` in helpers) |

**A-STR-05** and **A-STR-06** are not defined in the current validators. **A-CON-01** is not emitted by `validate_agent_content` today.

### Teams (`unit/teams/` — same `AGENTS.md` files as agents)

| ID | Meaning |
|----|---------|
| **T-STR-01** | Opening `---` |
| **T-STR-02** | `description` present |
| **T-STR-03** | `mode` present |
| **T-STR-04** | `skills` list present |
| **T-STR-05** | Each skill dependency exists on disk |
| **T-STR-06** | `mode` is `primary` or `subagent` |
| **T-STR-07** | `description` length 1–1024 characters |
| **T-CON-01** | Team directory name matches `^[a-z0-9]+(-[a-z0-9]+)*$` |
| **T-CON-02** | Description matches `TEAM_KEYWORDS` |
| **T-CON-03** | Core principles heading present |

**T-CON-04** and **T-CON-05** are not implemented in `validate_team_content` yet; the team content test file header may still reference a future range.

### Behavior (`behavior/`)

| Script | Intent |
|--------|--------|
| `skills/test-trigger-correctness.sh` | OpenCode returns pyasc-relevant content for domain prompts |
| `skills/test-premature-action.sh` | Session export: no destructive tools before skill load |
| `skills/test-workflow-enforcement.sh` | Responses reflect phased workflow (env → design → implement → verify) |
| `agents/test-trigger-correctness.sh` | Same class of checks, agent-oriented prompts |
| `agents/test-premature-action.sh` | Premature action checks against agent/session exports |

### Integration (`integration/`)

| Script | Intent |
|--------|--------|
| `test-simple-kernel.sh` | Init script, golden assets, basic kernel project layout |
| `test-kernel-generation.sh` | Kernel generation / workflow smoke tests |
| `test-golden-comparison.sh` | Golden docs/code consistency |
| `test-workflow-execution.sh` | Workflow scripts and execution paths |

---

## Running parameters (`run-tests.sh`)

| Flag | Description |
|------|-------------|
| `--fast` / `-f` | `unit` only |
| `--agentic` / `-a` | `behavior` + `integration` |
| `--integration` / `-i` | `unit` + `behavior` + `integration` |
| `--all` | Same as `--integration` |
| `--category` / `-c` | `unit`, `behavior`, `integration`, or `all` |
| `--test` / `-t` | Single file under `tests/` |
| `--platform` | Agent platform (default `opencode`) |
| `--runtime` | Export `PYASC_RUNTIME_CHECK=1` |
| `--timeout` | Seconds per test script (default `300`) |
| `--verbose` / `-v` | Verbose runner |
| `--output` | `text` or `json` (see Quick start) |
| `--list` / `-l` | List tests |
| `--help` / `-h` | Help |

---

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `PYASC_PYTHON` | `test-helpers.sh`, `analyze-tokens.sh` | Python interpreter (default `python3.10`) |
| `DEFAULT_PLATFORM` | `run-tests.sh` → helpers | Agent backend; overridden by `--platform` |
| `PYASC_RUNTIME_CHECK` | Exported from `run-tests.sh` when `--runtime` is set | Intended for tests/tools that should gate simulator or hardware checks (convention; not all scripts read it yet) |
| `NO_COLOR` | `test-helpers.sh` | Disable ANSI colors |
| `FORCE_COLOR` / `FORCE_COLOR=1` | `test-helpers.sh` | Keep colors when stdout is not a TTY |
| `MAX_RETRIES` | `run_opencode` in helpers | Retries on transient TLS errors (default `3`) |

Kernel/runtime tools may also honor `ASCEND_HOME_PATH`, `LD_LIBRARY_PATH`, and `PYASC_DUMP_PATH` where documented in each script (see `run_and_verify.py`).

---

## Test helper library API (`lib/test-helpers.sh`)

Source the library at the top of a test script (paths vary by depth):

```bash
source "$SCRIPT_DIR/../../lib/test-helpers.sh"
```

### Paths (read-only globals)

- `SKILLS_DIR` — repository root (parent of `skills/` and `teams/`)
- `TESTS_DIR` — `tests/`
- `TOOLS_DIR` — `tests/tools/`
- `PYTHON` — resolved from `PYASC_PYTHON`

### Output helpers

- `print_pass`, `print_fail`, `print_skip`, `print_info`, `print_warn`, `print_error`
- `print_section_header`, `print_status_passed`, `print_status_failed`
- `setup_colors`

### Agent runners

- `run_opencode "prompt" [timeout_sec] [work_dir]` — headless `opencode run … --format json`
- `run_ai "prompt" [timeout_sec] [platform]` — dispatches by `DEFAULT_PLATFORM`
- `run_behavior_test "name" "prompt" "regex" [timeout]` — increments `pass_count` / `fail_count` / `skip_count` (define those before use)

### Temp projects

- `create_test_project [prefix]` — temp dir with symlinks to `skills`, `teams`, `golden` and a throwaway git repo
- `cleanup_test_project "$dir"`

### Session analysis

- `find_recent_session`, `export_session id file.json`
- `verify_skill_invoked session.json skill-name`
- `count_tool_invocations session.json ToolName`
- `analyze_premature_actions session.json skill\|agent target-name`
- `analyze_workflow_sequence session.json`, `analyze_tool_chain session.json`

### Assertions

- `assert_contains`, `assert_not_contains`, `assert_file_exists`, `assert_count`, `assert_order`

### Discovery / YAML

- `get_all_skills`, `get_all_teams`, `extract_team_skills file`
- `check_file_links file [skill|team]`

### Validators (return non-zero on failure)

- `validate_skill_structure`, `validate_skill_content`
- `validate_agent_structure`, `validate_agent_content`
- `validate_team_structure`, `validate_team_content`

### Environment probes

- `check_opencode`, `check_pyasc_import`, `check_pyasc_runtime`

### Counters (optional pattern for custom tests)

- `init_test_tracking`, `record_test pass|fail|skip`, `print_test_summary`

Many helpers are `export -f`’d for subshells; prefer sourcing once per script.

---

## Analysis and evaluation tools (`tests/tools/`)

Run from repo root or `tests/tools/`; use `PYASC_PYTHON` where noted.

| Tool | Role |
|------|------|
| [`analyze-session.sh`](tools/analyze-session.sh) | `analyze-session.sh <session.json> [--brief\|--full\|--json\|--tools]` |
| [`analyze-workflow.sh`](tools/analyze-workflow.sh) | Skill order and workflow heuristics on an export |
| [`analyze-tokens.sh`](tools/analyze-tokens.sh) | Token stats; prefers `analyze-token-usage.py` |
| [`analyze-token-usage.py`](tools/analyze-token-usage.py) | Python parser for token fields in session JSON |
| [`eval-report.sh`](tools/eval-report.sh) | Combined report; `bash eval-report.sh [--json] [--kernel path] [--session path]` |
| [`verify_kernel.py`](tools/verify_kernel.py) | Static AST rules for `@asc.jit` kernels |
| [`score_kernel.py`](tools/score_kernel.py) | Checklist score / `--json` |
| [`run_and_verify.py`](tools/run_and_verify.py) | `--mode jit\|simulator\|auto`, optional `--json` |
| [`pytest_verify_kernel.py`](tools/pytest_verify_kernel.py) | JIT-oriented verification |
| [`gen_golden.py`](tools/gen_golden.py) | Golden numpy tensors for ops/shapes |

---

## How to add new tests

1. **Pick a layer** — L1 under `unit/`, L2 under `behavior/`, L3 under `integration/`.
2. **Name the file** `test-<topic>.sh` so `run-tests.sh` discovers it.
3. **Start with** `set -euo pipefail` and `source` [`lib/test-helpers.sh`](lib/test-helpers.sh) using the correct relative path.
4. **Exit code** — `0` pass, non-zero fail. The runner treats exit `124` from `timeout` as **SKIP** in the summary.
5. **Reuse** validators and assertions instead of duplicating grep/sed logic.
6. **Document rule IDs** — if you add new checks to `validate_*` functions, add a stable `X-STR-nn` / `X-CON-nn` string and mention it in the unit test header comment.
7. **Optional** — for OpenCode-dependent tests, guard with `check_opencode` and skip or fail clearly; use `run_opencode` / `run_ai` for consistent timeouts and retries.

For new **Python** checks, add a script under `tests/tools/` and invoke it from an integration or evaluation script; keep CLI and exit codes documented in the file docstring.

---

## See also

- Repository [`README.md`](../README.md) for skill-stack overview
- Skill-specific verification: `skills/pyasc-build-run-verify/` and `skills/pyasc-codegen-workflow/`
