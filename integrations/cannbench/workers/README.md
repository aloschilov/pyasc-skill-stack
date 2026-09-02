# CANN Bench worker campaign

Automates perf tuning and operator generation with local opencode/GLM-5.2
workers plus a serialized private CANNBench evaluation queue on 950PR.

## Pieces

- `prompts.py` — renders definite prompts from `templates/` + vendored specs
  in `../tasks/` + archived reports in `../../../evidence/cannbench/`.
- `templates/constraints.md` — the pyasc asc2 contract digest (kernel rules,
  UB budget, numerical-stability recipes, anti-cheat rules) shared by both
  prompt kinds.
- `driver.py` — work-item loop: spawn worker → static-check `candidate.py` →
  deploy → harness eval → accept/rollback → feedback resume (opencode `-c`).
- `evalqueue.py` — credit-aware private site submission, streaming upload,
  job polling, report parsing, and immutable candidate staging. It never uses
  PR comments as a data or command channel.

## Usage

```bash
cd integrations/cannbench/workers
python3 driver.py --items sigmoid:tune --iterations 1 --workers 1        # sanity
python3 driver.py --items sigmoid:tune,exp:tune,mish:tune,gelu:tune \
                  --iterations 3 --workers 2                             # M2
python3 driver.py --items masked_scale:generate,swi_glu:generate \
                  --iterations 3 --workers 2                             # M3a
python3 driver.py --items foreach_addcdiv_scalar:generate,foreach_norm:generate \
                  --iterations 3 --workers 2                             # M3b
python3 driver.py --items <op>:<kind> --dry-run                          # prompt only
python3 ../sync-benchmark-task.py RmsNorm                                # vendor next task
```

Artifacts land in `runs/<timestamp>/<item>/` (prompt, per-iter candidate,
worker log, private submission/job metadata, report.json, digest.json).
Accepted candidates overwrite the canonical module in
`../submission/cann_bench/` and refresh `evidence/cannbench/<op>_final_eval.json`.

Each evaluated candidate consumes one CANNBench submission credit. Jobs are
serialized even when several GLM workers generate code concurrently, and are
private by default.

If the configured local OpenCode/GLM endpoint exits successfully without
writing `candidate.py`, the driver retries three times without reusing stale
output. A code-driven candidate may then be placed in the run directory and
evaluated through the same immutable queue; model generation is an input to
the loop, not a requirement for measurement or acceptance.

## Acceptance rules

- `tune`: all cases pass AND harness score beats the incumbent; otherwise the
  remote submission is rolled back to the canonical module.
- `generate`: all cases pass (perf recorded, tuning can follow later); the
  package `__init__.py` gains the new export on accept and is rolled back on
  reject so a broken module can never poison other ops' evals.
