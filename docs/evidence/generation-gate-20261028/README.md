# Generation gate — first executed evidence, 2026-10-28

The agent-generation gate had never been run. It was recorded as `blocked`
on a provider output ceiling, and every claim about it — including the
ceiling itself — was diagnosis rather than observation. These are the
artifacts from the first three runs that actually executed.

## Provenance

| | |
|---|---|
| Repository commit | `7fc30cc0a35a01abdd61e52e96ae23f06563264b` (v0.1.0) |
| Harness | `tests/integration/test-kernel-generation.sh` |
| Model profile | `cloud-glm-5.3` (`gateway/glm-5.3`, `output: 32768`, reasoning low) |
| opencode | 1.19.0 |
| Agent timeout | 900 s (`PYASC_AGENT_TIMEOUT`) |
| Host | Linux aarch64, no NPU, no importable `asctile` |
| pyasc pin | `d0f64a63c432761aa57648e67be7582780a17561` |

The task was identical in all three runs: a float32 vector add of size 1024
under `kernels/add/`, with a design document.

## Results

| Run | Elapsed | Artifacts | Static verification | Score | Runtime | Harness exit |
|---|---|---|---|---|---|---|
| 1 | 129 s | kernel + design | passed, 7/7 | **12.0 / 16** | `SKIP` | 0 (pass) |
| 2 | 145 s | kernel + design | passed, 7/7 | **12.0 / 16** | `SKIP` | 0 (pass) |
| 3 | 51 s | kernel + design | passed, 7/7 | **11.0 / 16** | `SKIP` | 1 (fail) |

Threshold is 12. Two of three runs cleared it, by exactly zero margin.

### The one dimension that moved

Every run produced a kernel that passed all seven static checks. The only
score dimension that differed between a pass and a failure was
`multi_shape_verification`:

- runs 1 and 2 — `verify x2, shape loop=no` (passed)
- run 3 — `single verify call, no shape loop` (failed)

So the spread is 11 vs 12 on a threshold of 12, decided by whether the agent
happened to verify one shape or two. Describing this gate as "passing" would
overstate it: a one-point rubric item it is not reliably instructed to
satisfy sits exactly on the acceptance boundary. The honest reading is that
the gate now *runs*, and that it is marginal.

### Runtime is not covered

`run_and_verify.py` reported `SKIP — pyasc (asc) not importable` in all three
runs. `asctile` lives in the pinned simulator image, not on this host, and
that script has no docker path. **No generated kernel here has been
executed.** These are static results only.

## What this corrected

Running the gate for the first time exposed two defects that could only have
been found by running it, both now fixed:

1. **`verify_kernel.py` did not recognize `@asctile.jit`.** It accepted only
   `asc` and `asc2`. Since the skills and the prompt both mandate `asctile`,
   every correctly written kernel failed `jit_decorator` and cascaded into
   three further "skipped (no JIT function)" failures. All three logs here
   are post-fix, so they show the seven passes rather than the four failures
   that prompted the change.

2. **The prompt sent the agent hunting for an environment that does not
   exist.** Asked to "check environment" first, the agent found no importable
   `asctile`, then spent 22 steps and 26 tool calls excavating unrelated
   checkouts under `/home` and `/tmp`, wrote nothing, and exhausted 900 s. The
   prompt now states the environment as fact. That single change took the run
   from a 900 s timeout to a 129 s success.

Neither was the diagnosed output ceiling. The ceiling was real — see
[`../../verification-status.md`](../../verification-status.md) — but it was
not the only thing standing between this gate and a result, and it was not
the first thing to hit.

## Files

Per run: `kernel.py` and `design.md` as the agent wrote them, `score.json`
from `tests/tools/score_kernel.py --json`, and `harness.log` with the
temporary project path rewritten to `<project>`.

Reproduce with:

```bash
export OPENCODE_BASE_URL=... OPENCODE_API_KEY=...
PYASC_KEEP_PROJECT=1 bash tests/integration/test-kernel-generation.sh
```

A generation is not deterministic, so a rerun will not reproduce these files
byte for byte. That is the reason they are committed rather than regenerated
on demand.
