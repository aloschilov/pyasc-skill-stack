# CANN Bench worker campaign

Automates perf tuning and full-coverage operator generation with local OpenCode
workers, a credit-free exact-v2 compile gate, and an optional serialized private
CANNBench evaluation queue on 950PR.

## Pieces

- `prompts.py` — renders definite prompts from `templates/` + vendored specs
  in `../tasks/` + archived reports in `../../../evidence/cannbench/`.
- `templates/constraints.md` — the pyasc asc2 contract digest (kernel rules,
  UB budget, numerical-stability recipes, anti-cheat rules) shared by both
  prompt kinds.
- `driver.py` — provenance-gated design/implementation/review loop with static
  checks, measured repair feedback, and local or explicitly remote evaluation.
- `local_compile_gate.py` / `run_local_compile_gate.sh` — replay all official
  case routes against the self-contained pinned-v2 runtime under QEMU.
- `staged_local_repair.py` — finish a model seed with measured repair and an
  independent model review without touching the canonical submission.
- `assemble_local_bundle.py` / `run_local_matrix.py` — assemble and recheck the
  full nine-operator, 180-route generated bundle.
- `run_camodel_smoke.py` — execute either the basic float32 suite or selected
  high-risk dtype/control routes with a native exact-v2 wheel; raw ticks are
  not speedups.
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

Local generation is the default and does not consume submission credits:

```bash
python3 driver.py --items all --evaluation local --iterations 2 --workers 2 \
  --models dashscope/glm-5.2,dashscope/qwen3.7-max
```

Artifacts land in `runs/<timestamp>/<item>/` (prompt, design, candidate,
worker JSON-event logs, phase skill traces, exact-v2 compile report,
`provenance.json`, and `digest.json`). Candidates that route and compile all
20 cases are copied to `runs/<timestamp>/locally_qualified/cann_bench/`; this
bundle does not overwrite the canonical submission because the local gate does
not execute numerics or measure performance.

Remote evaluation is explicit (`--evaluation remote`), consumes credits, and
is the only driver mode that can promote a 20/20 candidate into the canonical
module and refresh `evidence/cannbench/<op>_final_eval.json`.

Every worker run is pinned to the `skills/` directory in this checkout and
must complete native OpenCode `skill` calls. Design and implementation use the
compact CANNBench and syntax skills; measured repair and review deliberately
use only the compact CANNBench skill to avoid loading contradictory standalone
profiles. The implementation model and independent review model are rotated
from `--models`. The driver
parses the structured OpenCode event stream and rejects a candidate before any
evaluation when a required skill call or completion marker is missing. Scratch
work stays in the gitignored in-repo `.scratch/` directory so project
permissions and skill discovery are deterministic.

When a locally validated candidate must wait for the daily site quota, resume
the exact immutable run with:

```bash
python3 submit_skill_candidate.py runs/<timestamp>/<item>/iter<N>
```

The resume command verifies candidate, prompt, skill-source and local-lowering
evidence before uploading. It embeds `PROVENANCE.json` in the submission ZIP
and updates the canonical module only after 20/20 correctness with zero
anti-cheat failures.

Each evaluated candidate consumes one CANNBench submission credit. Jobs are
serialized even when several GLM workers generate code concurrently, and are
private by default.

If the configured local OpenCode/GLM endpoint exits successfully without
writing `candidate.py`, or times out without its phase marker, the driver
rejects that phase and can fail over to the next configured model without
reusing stale output.

For native CAModel validation, run the basic suite first and then the critical
suite in short operator groups when necessary:

```bash
python3.10 run_camodel_smoke.py --candidate-root <bundle>/cann_bench \
  --suite basic --output <basic-evidence.json>
python3.10 run_camodel_smoke.py --candidate-root <bundle>/cann_bench \
  --suite critical --ops masked_scale,swi_glu,foreach_norm,rms_norm \
  --output <critical-evidence.json>
python3.10 run_camodel_smoke.py --candidate-root <bundle>/cann_bench \
  --suite adversarial --ops gelu,foreach_addcdiv_scalar \
  --output <adversarial-evidence.json>
```

## Acceptance rules

- `tune`: all cases pass AND harness score beats the incumbent; otherwise the
  remote submission is rolled back to the canonical module.
- `generate`: all cases pass (perf recorded, tuning can follow later); the
  package `__init__.py` gains the new export on accept and is rolled back on
  reject so a broken module can never poison other ops' evals.
