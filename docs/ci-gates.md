# CI Gate Tiers

Three gate tiers ensure fast feedback on PRs while reserving expensive checks for merge and nightly runs.

## Tiers

| Tier | Trigger | Time budget | What runs |
|------|---------|-------------|-----------|
| **pr** | Every push / PR | < 30s | L1 unit tests + JIT verification of golden kernels |
| **merge** | Merge to main | < 5 min | PR tier + simulator execution of golden kernels |
| **nightly** | Scheduled (daily) | 15-30 min | Merge tier + L2 behavior + L3 agentic integration |

## Entry point

```bash
bash tests/ci-gate.sh --tier pr        # Fast PR gate
bash tests/ci-gate.sh --tier merge     # Merge gate (includes simulator)
bash tests/ci-gate.sh --tier nightly   # Full nightly run
```

## PR gate (`--tier pr`)

Runs in under 30 seconds. Suitable for pre-commit hooks and PR checks.

1. `run-tests.sh --fast` -- L1 structural and content validation (skills, agents, teams)
2. JIT verification of all golden kernels via `pytest_verify_kernel.py` -- confirms pyasc JIT compilation works without needing the simulator

No network, no simulator, no opencode required.

## Merge gate (`--tier merge`)

Runs in under 5 minutes. Requires CANN simulator environment.

1. Everything in PR gate
2. Simulator execution of all golden kernels via `run_and_verify.py --mode simulator` -- confirms numerical correctness with `torch.allclose`

Requires: `source $HOME/Ascend/cann/set_env.sh` and `LD_LIBRARY_PATH` set. See [cann-setup.md](cann-setup.md).

## Nightly gate (`--tier nightly`)

Runs in 15-30 minutes. Requires opencode CLI and CANN simulator.

1. Everything in merge gate
2. `run-tests.sh --all` -- L2 behavior tests (agent trigger correctness, premature action detection) and L3 integration tests (full agent-in-the-loop kernel generation)

Requires: opencode CLI on PATH, CANN simulator environment.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYASC_PYTHON` | `python3.10` | Python interpreter with pyasc |
| `ASCEND_HOME_PATH` | (from set_env.sh) | CANN toolkit root |
| `LD_LIBRARY_PATH` | (must include simulator) | Simulator libraries |
| `NODE_TLS_REJECT_UNAUTHORIZED` | `0` (for opencode) | Bypass TLS issues |

## Exit codes

- `0` -- all checks passed
- `1` -- one or more checks failed
- `2` -- environment prerequisites missing (e.g., simulator not available for merge tier)
