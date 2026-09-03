# CANNBench: four-operator three-arm comparison

Date: 2026-09-03 (Europe/Moscow). Benchmark: `official-tasks` v1.1.1, private submissions, 950PR.

## Scope and provenance

All arms use the same current pyasc `v2` snapshot `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d` and the same four CANNBench operators: `gelu`, `rms_norm`, `softmax`, `transpose`.

- handwritten target-derived: exact JIT/device definitions extracted from `python/test/asctile/target/test_<op>.py`; only host signature/allocation/launch transport is generated or assembled deterministically;
- generated without skills: OpenCode workers have no skill paths and any skill call fails provenance acceptance;
- generated with skills: the same generation/review structure, with native skill calls required and verified from OpenCode event traces.
- controlled exceptions: skill-guided Softmax is a review/tuning pass over the 20/20 no-skill baseline and produced the same hash; both generated Transpose implementation workers exhausted their context before writing, so both comparison arms use the same target-derived, deterministically remediated rank-5/int64 candidate and differ only in their recorded review path. No Transpose delta may therefore be attributed to skills.

Verified required skill loads: **24/24**. Accepted skill-arm sessions: `ses_f9a057a23ffe1W8pco4Rf1AU7Y`, `ses_f99fab716ffeq0B6GjWP1OLat4`, `ses_f99f3c9e3ffeM1e1TVka79xfnV`, `ses_f99fd30d7ffe5fAC5Qst8KhLJ4`, `ses_f99e91256ffe8Psm8R1O3Z6opI`, `ses_f99e5abfcffe9B1IgU7qHqYqLb`, `ses_f99d5421dffe0KcbdYncuWXsBc`, `ses_f999f55c4ffeeMGESc0N4F7vce`, `ses_f99df7100ffeXn9a0Ncz6iG1nH`, `ses_f99910755ffeWC7Gx2oAd1K4kO`.

CANNBench receives only the built wheel and cannot infer generation history. Arm identity is therefore established by candidate hashes, source-definition hashes, prompts, model sessions, and native skill-call traces retained beside this report.

### Generated-arm provenance

| Arm | Operator | Candidate SHA-256 | Phase models | Session IDs | Verified/required skill loads | Skill calls |
|---|---|---|---|---|---:|---:|
| Generated without skills | `gelu` | `d74d059af15df90e` | dashscope/qwen3.7-max, dashscope/qwen3.7-max, dashscope/qwen3.7-max | `ses_f9a1c4471ffe6qXDkZZUYiszJf, ses_f9a0c38c3ffealzKwgHGwsLImm, ses_f9a095e94ffeVo2Rhbj9BIK4VZ` | 0/0 | 0 |
| Generated without skills | `rms_norm` | `d99df6e42d5a7959` | dashscope/glm-5.2, dashscope/glm-5.2, dashscope/qwen3.7-max | `ses_f9a237f9effeQ6xafR3AexDTOz, ses_f9a1e33dfffehSJZJCw0oNuDu6, ses_f9a1d603affeGBmzoSXp98MlXW` | 0/0 | 0 |
| Generated without skills | `softmax` | `ce232b6d4b938667` | dashscope/qwen3.7-max, dashscope/qwen3.7-max, dashscope/qwen3.7-max | `ses_f99dd6d0affea4eIRvuWCa9Q75, ses_f99d158bcffe8Dq6iUihvUJOso, ses_f99cbf58dffeAgaNPH0Up13h6K` | 0/0 | 0 |
| Generated without skills | `transpose` | `887e9d7cca7134bb` | dashscope/qwen3.7-max, deterministic:human, dashscope/qwen3.7-max | `ses_f99fbaf37ffeyhT3mhaOoAkvwi, —, ses_f99916d81ffeccorcZm6Iu813n` | 0/0 | 0 |
| Generated with skills | `gelu` | `9e962802b651f836` | dashscope/glm-5.2, dashscope/glm-5.2, dashscope/qwen3.7-max | `ses_f9a057a23ffe1W8pco4Rf1AU7Y, ses_f99fab716ffeq0B6GjWP1OLat4, ses_f99f3c9e3ffeM1e1TVka79xfnV` | 7/7 | 7 |
| Generated with skills | `rms_norm` | `8f976fc4c9c5d00e` | dashscope/qwen3.7-max, dashscope/qwen3.7-max, dashscope/qwen3.7-max | `ses_f99fd30d7ffe5fAC5Qst8KhLJ4, ses_f99e91256ffe8Psm8R1O3Z6opI, ses_f99e5abfcffe9B1IgU7qHqYqLb` | 7/7 | 7 |
| Generated with skills | `softmax` | `ce232b6d4b938667` | dashscope/qwen3.7-max, deterministic:no_skills-candidate, dashscope/qwen3.7-max | `ses_f99d5421dffe0KcbdYncuWXsBc, —, ses_f999f55c4ffeeMGESc0N4F7vce` | 5/5 | 5 |
| Generated with skills | `transpose` | `887e9d7cca7134bb` | dashscope/qwen3.7-max, deterministic:no_skills-remediated-candidate, dashscope/qwen3.7-max | `ses_f99df7100ffeXn9a0Ncz6iG1nH, —, ses_f99910755ffeWC7Gx2oAd1K4kO` | 5/5 | 5 |

## Local pinned-v2 compile gate

| Arm | Operator | Dispatch | Compile | Status |
|---|---|---:|---:|---|
| Handwritten target-derived | `gelu` | 14/20 | 14/20 | failed |
| Handwritten target-derived | `rms_norm` | 20/20 | 20/20 | passed |
| Handwritten target-derived | `softmax` | 14/20 | 14/20 | failed |
| Handwritten target-derived | `transpose` | 17/20 | 17/20 | failed |
| Generated without skills | `gelu` | 20/20 | 20/20 | passed |
| Generated without skills | `rms_norm` | 20/20 | 20/20 | passed |
| Generated without skills | `softmax` | 20/20 | 20/20 | passed |
| Generated without skills | `transpose` | 20/20 | 20/20 | passed |
| Generated with skills | `gelu` | 20/20 | 20/20 | passed |
| Generated with skills | `rms_norm` | 20/20 | 20/20 | passed |
| Generated with skills | `softmax` | 20/20 | 20/20 | passed |
| Generated with skills | `transpose` | 20/20 | 20/20 | passed |

The QEMU gate checks host dispatch plus pyasc AST/codegen, lowering, AscendC translation, and 950PR UB limits. It does not execute numeric code or measure silicon latency.
Worker prose that called a local compile result `verified-cannbench` is rejected by this report; only a completed remote job receives that label.

## CANNBench results

Run pages:

- Handwritten target-derived: [`job_00385e9c7a19`](https://cannbench.com/workspace/jobs/job_00385e9c7a19)

| Arm | Job | Cases | Score | GMean speedup | Anti-cheat | Status |
|---|---|---:|---:|---:|---:|---|
| Handwritten target-derived | [`job_00385e9c7a19`](https://cannbench.com/workspace/jobs/job_00385e9c7a19) | — | — | — | — | queued |
| Generated without skills | [pending](https://cannbench.com/workspace/jobs) | — | — | — | — | pending |
| Generated with skills | [pending](https://cannbench.com/workspace/jobs) | — | — | — | — | pending |

### Per-operator measurements

| Arm | Operator | Cases | Average speedup | Score |
|---|---|---:|---:|---:|
| — | — | — | — | pending |

## Interpretation rules

Correctness is a hard gate: speedups from a partial pass must not be compared as if they represented complete solutions. The target-derived arm measures how far upstream development targets cover the broader CANNBench contract; it is not silently repaired into a fourth implementation. Generated arms are compared only after preserving their skill/no-skill provenance boundary.

## Artifacts

- `MANIFEST.json`: runtime and candidate hashes;
- `generated/*/*/evidence`: prompts, events, accepted/rejected attempts, model sessions and skill traces;
- `handwritten/*/provenance.json`: exact upstream source paths and definition hashes;
- `*/local_validation`: QEMU gate reports;
- `remote_runs`: CANNBench job payloads and logs;
- `submissions/*.zip`: immutable submitted source archives.
