# Verification status

What has actually been verified, and what has not. Undated on purpose: this
file is updated in place, so the latest state is the only state. The history
is in `git log`.

Gate vocabulary is `passed | failed | blocked | not-run`. A missing runtime,
provider or device is an honest `blocked`, never a fabricated pass.

## Gates

| Gate | Status | Basis |
|---|---|---|
| install | `passed` | `pip install --pre pyasc-dev` into a clean venv, in an external project whose path contains spaces |
| skill discovery | `passed` | `opencode debug skill --pure` resolves the skill from the project |
| generation | `blocked` | provider output ceiling — see below |
| review | `not-run` | depends on generation |
| compile | `passed` | codegen produced a kernel binary; output left untouched under `PYASC_DRY_RUN=1` |
| numerical | `passed` | simulator execution matches the NumPy reference exactly, tail included |
| target suite (collection) | `passed` | 1155 cases across 24 modules collect against the published wheel |
| target suite (compile) | `passed` | 23 of 24 modules compile-check in the pinned image; 1 `blocked`, 0 failed — see below |
| simulator image | `passed` | pinned digest pulls anonymously and `asc.experimental.asctile.jit` imports |
| hardware | `not-run` | no `/dev/davinci*` and no `npu-smi` on this host; simulator only |
| performance | `not-run` | a simulator result is not a hardware measurement |

**The kernel in the compile and numerical gates was written by hand.** The
generation gate is about an *agent* producing one from a plain-language
request, and that is blocked. So `passed` there says the runtime and the
documented API work — not that the agent workflow does.

## The target-suite compile gate

Every module under `reference/asctile/target/` is compile-checked inside the
image pinned by [`../reference/images/ENV.json`](../reference/images/ENV.json).
Full sweep on Linux aarch64, 39 GB RAM:

```
compile-checked 24 module(s); 0 failed; 1 blocked
```

Roughly 12 minutes end to end. The one blocked module is `test_kl_div.py`,
and it is blocked on host memory, not on anything about the kernel: its
largest case has shape `[1000, 997, 1000, 2, 1]`, which is 1.99e9 float32
elements — **7.4 GiB for a single input tensor**, with several allocated per
case. The process is killed with SIGKILL (exit 137) by the OOM killer even
with 27 GB free.

That is `blocked`, not `failed`, and the gate keeps the two apart: it fails
the job on a compile error and only reports a resource blockage. Collapsing
them would either turn the gate permanently red — at which point nobody reads
it — or hide a real regression behind an expected failure.

Two harness defects were found and fixed while establishing this, both of them
cases where a module looked broken but nothing had actually been checked:

- `run_target_test.py` built its `-k` selector as `<name>-<block_num>-`, which
  only matches modules whose ids are generated from parameters. Modules using
  an explicit `id=` on `pytest.param` — `test_gather.py` — matched nothing,
  pytest exited 5 with "14 deselected", and the module was reported as failed
  having run zero cases. It now probes candidate patterns and falls back.
- Whole-module runs compiled every case in one interpreter, and memory
  accumulates across compiles. `matmul_v3/test_matmul_v3_cube_only.py` (184
  cases) was OOM-killed. Cases now run in batches of 24, each in a fresh
  process, and it passes in 288 s with no loss of coverage.

Note what this gate does **not** establish: it is compile-only. Numerical
execution of the suite on the camodel remains `not-run`.

## What the numerical gate covered

`add`, float32, shape `[1000]`, tile 128, one block. 1000 = 7×128 + 104, so the
final tile is a 104-element tail, handled the way the skills document it:
allocate a full physical tile, pass the logical extent through `real_shape`,
pad with the operation-neutral value.

`max_abs_err = 0.0` against `x + y`. The tail region `[896, 1000)` is asserted
separately, so a kernel that silently skipped it could not pass. Four
consecutive clean-cache runs, all exact.

The runnable version of this is
[`../skills/pyasc-build-run-verify/references/execution-example.md`](../skills/pyasc-build-run-verify/references/execution-example.md).

### Environment

- Runtime `pyasc-dev 1.1.1.dev20261021205441`, installed from PyPI, not a local build
- Import path `asc.experimental.asctile`
- Python 3.10, Linux aarch64
- CANN 9.0.0, simulator `Ascend950PR_9599`, Bisheng present
- No NPU

### One unexplained failure

The first simulator execution segfaulted (exit 139) before writing a result.
It has not recurred in four subsequent runs with cleared caches.

It was not the environment and not the kernel: the official `runtime-smoke.py`
passes in the same venv and path; the construct was bisected into four variants
— loop without tail, `real_shape` without loop, loop with constant
`real_shape`, and loop with per-iteration `min()` `real_shape` — and all four
pass, including the exact idiom that crashed; and `ulimit -n` is 1048576, so
this is not the simulator file-descriptor exhaustion the README warns about.

Recorded rather than explained. One unreproducible segfault is not enough to
call the numerical gate flaky, but it is not something to leave undocumented
either.

## The open blocker: an 8192-token output ceiling

Diagnosis only. No provider call was made to produce this section.

Eight `gateway/glm-5.3` calls failed to produce their artifact, reported as
`incomplete-length`. Every length-terminated step reports `reasoning` and
`output` token counts summing to exactly 8192:

| reasoning | output | sum |
|---|---|---|
| 8190 | 2 | 8192 |
| 8191 | 1 | 8192 |
| 8191 | 1 | 8192 |
| 8192 | 0 | 8192 |
| 8187 | 5 | 8192 |
| 8185 | 7 | 8192 |

Six independent runs landing on the same total is a configured cap, not a
verbose model. Steps in the same sessions that end with `reason: "tool-calls"`
show reasoning counts of 53, 123 and 297 and complete normally — the model
fails precisely on the steps that need a long answer.

### Root cause

The cap is declared locally, in `~/.config/opencode/opencode.json`:

```json
"limit": { "context": 256000, "output": 8192 },
"reasoning": true
```

**Not a provider limit.** `gateway` is a custom provider defined only in that
local config; it is absent from the models.dev catalog OpenCode caches at
`~/.cache/opencode/models.json`. Every catalogued provider serving GLM-5.3
declares `output: 131072` — 16× higher — and `context: 1000000`.

**Not an OpenCode default and not the harness.** The runner invokes
`opencode run --pure --format json -m gateway/glm-5.3` and passes no token
flag, so the config entry is the only possible source.

**`reasoning: true` explains the shape of the failure.** GLM-5.3 spends
reasoning tokens from the same output budget. With 8192 total, a hard
authoring or review step exhausts the budget while still thinking and has 0–7
tokens left to write with. That is why artifacts were never written despite
the model working for over two minutes (132.2 s and 129.0 s, both ending
empty).

### Options

1. **Correct the declaration to the model's real capability** (`output:
   131072`). One line, no new spend to apply, matches every catalogued
   provider. If the gateway deployment genuinely caps lower, the right value
   is whatever it enforces — but 8192 should be confirmed, not assumed.
2. **Raise it partially**, e.g. 32768. Lower risk if gateway billing is a
   concern, still ~4× the headroom that starved these runs.
3. **Leave it and split the work into smaller steps.** Already tried: a
   two-part review split still hit the ceiling at 8187 + 5. Step size is not
   the binding constraint. Not recommended.

### A second, independent blocker

Fixing the token ceiling is necessary but may not be sufficient. The harness
caps the review call at 60 seconds (`rc = p.wait(timeout=60)`), and one review
failed on exactly that timeout, separately from the token issue. Observed
review steps took 132.2 s and 129.0 s, so a reasoning model will very likely
exceed 60 s even with tokens to spare. The authoring call is given 240 s.
Expect to raise the review timeout before the end-to-end path can complete.

No fix is applied here: `~/.config/opencode/opencode.json` is outside this
repository, and changing it is the maintainer's decision.

## What none of this establishes

One operator, one dtype, one shape, one block count, on a simulator. Not a
numerical suite, not hardware qualification, not a performance result. The
agent-generation path is untested because it is blocked upstream.

For what *would* constitute evidence, see
[`evidence-sources.md`](evidence-sources.md).
