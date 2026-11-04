# External hardware qualification

CANNBench evaluates a packaged operator without requiring access to an NPU host.
Use the [submission procedure](../integrations/cannbench/README.md). Shared CI
handles task snapshots, packaging, submission and evidence; a checked-in operator
recipe supplies its prompt and numerical preflight. SwiGlu is the currently
qualified recipe, not evidence of arbitrary-operator coverage.

## Recorded generated result

The OpenCode 1.19.0 / `gateway/glm-5.3` candidate passed on 2026-11-03 UTC.

| Check | Recorded result |
|---|---|
| Task and runner | official-tasks 2.0.0, SwiGlu; Ascend950PR_9589 / CANN 9.2.0 |
| Official numerical and measured cases | 20/20; zero reported anti-cheat failures |
| Device execution | `_swiglu_impl` kernel records and positive finite timings for all cases |
| Service `overall_score` | 32.344009230305275 |
| Geometric mean baseline/candidate timing ratio | 0.017450287921005128x — slower than baseline |
| Queue / execution | 16478.414 s / 303.092 s |
| Preflight | Exact-runtime lowering 20/20; separate simulator numerical checks 12/12 |
| Generation stack / PyASC | `195c3dc570d54e9925426ff9418ebadb31c8d623` / `030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d` |

The service score and timing ratio are different metrics. This establishes
correctness and measurement for one generated candidate; it demonstrates no
speedup, other-operator coverage or full simulator matrix on hardware. Provider
model revision is unknown. The archived configuration requested low reasoning
effort; the current high-effort profile has not received a new hardware run.
A separate reviewer session was not automated in this qualification.

## Durable evidence

The [qualification Release](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-swiglu-41673353296)
contains `RESULT.md`, the complete qualification archive and its checksum.
CI published and downloaded the assets to verify them. The archive retains
native model sessions, skill/configuration hashes, repairs, task snapshots,
exact evaluated package, runtime provenance and per-case device results.

`run-41673353296-qualification.tar.gz` SHA256:
`aa86a82a28f4a13edf5858059495119cf083c0df876c81e9641a16b8b3a52d56`.

Earlier reference and failed-run records remain preserved outside checkout;
their [historical evidence index](https://github.com/aloschilov/pyasc-skill-stack/blob/654a05556107c538dca11fc71b02fd1378543ded/docs/hardware-lane.md)
is retained at an immutable revision. They are not current product instructions.
