# pyasc v2 target-test × CANNBench comparison

## Scope

The filename/semantic intersection between
`compiler-team/pyasc@v2:python/test/asc2/target/` at commit
`4d1db41d61cabf565bca1cfb0b11ef5ec4f84c7f` and the vendored CANNBench
L1 tasks is:

- `test_gelu.py` → `gelu`;
- `test_addcdiv.py` → `foreach_addcdiv_scalar` (same elementwise formula,
  adapted to the benchmark's TensorList and runtime-scalar interface).

RMSNorm is not included because it is under `python/test/asc2/kernels/`, not
the requested `target/` suite. All three runs select exactly these same two
operators on private 950PR evaluation.

## Arms

1. `handwritten` is the requested run label. More precisely, this is the
   **repository-target-derived** arm: arithmetic and tiling policy come from
   the two v2 target tests, with only benchmark transport/interface/tail
   adapters. Human authorship is not asserted; pyasc history itself describes
   the Addcdiv target addition as AI-generated. The target GELU lacks
   `approximate=none`, and both target kernels lack BF16 arithmetic coverage;
   these gaps are intentionally left visible to CANNBench.
2. `no_skills` uses OpenCode Qwen 3.7 Max for design/implementation and GLM
   5.2 for review, with skill discovery disabled. Structured traces prove zero
   skill calls. GELU's GLM review returned without a completion marker or
   edits, so its final artifact is the Qwen implementation; this is recorded
   rather than hidden.
3. `with_skills` uses the provenance-gated worker artifacts from
   `workers/runs/20260902_full_skill_generated/`. Native skill calls, session
   IDs, skill paths and source hashes are retained per operator.

All arms use the same self-contained submission runtime at pyasc commit
`ac1222a48c8914d3f81297c7570d1a84f0f26778`, which was the locally and
previously CANNBench-verified v2 runtime available for CPython 3.12/x86_64.
The newer `4d1db41d` identifies the target-test source used to determine and
construct the repository-target arm; this source/runtime distinction is a
known limitation of the comparison.

## Local preflight

The exact-v2 QEMU gate exercises all 20 benchmark routes per operator without
executing numerics:

- repository-target-derived: GELU 14/20 and ForeachAddcdivScalar 14/20;
  all six rejected routes per operator are BF16, unsupported by the original
  target arithmetic;
- no-skills: 20/20 + 20/20;
- with-skills: 20/20 + 20/20.

Compile coverage is not a correctness or performance result. CANNBench is the
authoritative NPU measurement.

## Submission order

`submit_runs.py` waits for pre-existing active account jobs and then uploads
the immutable ZIPs in this order:

1. `handwritten`;
2. `no_skills`;
3. `with_skills`.

It requires enough credits for every still-missing run before uploading and is
idempotent across retries. Site responses and final job/log payloads are saved
under `remote_runs/`.

## CANNBench 950PR results

All three private jobs ran against `official-tasks` benchmark version 1.1.1 on
2 September 2026. CANNBench charged one credit per selected operator (six
credits total), leaving four credits after the comparison. No arm triggered an
anti-cheat failure.

- Repository-target-derived (`job_79c05b96ed9d`): 14/40 cases, score
  69.794947, geometric-mean speedup 1.023126x (0.537783x versus the hardware
  limit). ForeachAddcdivScalar passed 13/20 with 1.189408x average speedup;
  GELU passed 1/20 with 0.880090x average speedup.
- Without skills (`job_75a7fee4ae6f`): 35/40 cases, score 125.452428,
  geometric-mean speedup 0.616455x (0.290534x versus the hardware limit).
  ForeachAddcdivScalar passed 20/20 with 1.082186x average speedup; GELU passed
  15/20 with 0.351157x average speedup.
- With skills (`job_ae3bfdefd087`): 36/40 cases, score 124.607094,
  geometric-mean speedup 0.595558x (0.279670x versus the hardware limit).
  ForeachAddcdivScalar passed 20/20 with 0.920270x average speedup; GELU passed
  16/20 with 0.385419x average speedup.

The skill-guided arm improves coverage by one GELU case over the no-skill arm,
but does not improve the aggregate score or speed. Consequently, this run is
evidence of a coverage benefit, not yet a performance-tuning win. The
repository-target-derived arm is not directly competitive because target-test
coverage gaps intentionally remain in that arm; its higher geometric-mean
speedup is calculated only over the much smaller set of passing cases.
