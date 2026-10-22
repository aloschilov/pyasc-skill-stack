# Why the skill E2E never completed: the 8192 output ceiling

Diagnosis only. No provider call was made to produce this document; every
number below comes from logs already preserved under
`release/20261018-user-release-v1/evidence/`.

## Symptom

Eight `gateway/glm-5.3` calls across cycles 6-12 of the user-release
development run failed to produce their artifact. `review` stayed `blocked`,
and `compile`, `numerical` and `hardware` stayed `not-run`. The run report
describes the failures as `incomplete-length`.

## The signature

Every length-terminated step reports `reasoning` and `output` token counts that
sum to exactly 8192:

| Evidence run | reasoning | output | sum |
|---|---|---|---|
| `dev-run-20261019T004712-3956fbca/authoring.jsonl` | 8190 | 2 | 8192 |
| `dev-run-20261019T005035-e5d17dd8/authoring.jsonl` | 8191 | 1 | 8192 |
| `dev-run-20261019T013921-b1cefc04/authoring.jsonl` | 8191 | 1 | 8192 |
| `dev-run-review-recovery-20261019T110824-eb96c3d7` | 8192 | 0 | 8192 |
| `dev-run-review-split-20261019T151855-31585e37` | 8187 | 5 | 8192 |
| `doc-worker-D1-20261020T125703-1d93ac8e` | 8185 | 7 | 8192 |

Six independent runs landing on exactly the same total is a configured cap, not
a model that happens to be verbose.

Steps in the same sessions that finish for other reasons are unremarkable:
`reason: "tool-calls"` steps show reasoning counts of 53, 123 and 297 and
complete normally. The model is not failing in general — it fails precisely on
the steps that need a long answer.

## Root cause

The cap is declared locally, in `~/.config/opencode/opencode.json`:

```json
"limit": {
  "context": 256000,
  "output": 8192
},
"reasoning": true
```

This entry defines `gateway/glm-5.3`. Three things follow.

**It is not a provider limit.** `gateway` is a custom provider that exists only
in this local config; it is absent from the models.dev catalog that OpenCode
caches at `~/.cache/opencode/models.json`. Every provider in that catalog that
serves GLM-5.3 declares `output: 131072`. The local declaration is 16x lower
than the model's actual capability, and the declared `context` of 256000 is
likewise well under the catalogued 1000000.

**It is not an OpenCode default and not the harness.**
`evals/run_dev_attempt.py` invokes `opencode run --pure --format json -m
gateway/glm-5.3` and passes no token flag at all, so the only place 8192 can
come from is the config entry above.

**The `reasoning: true` flag next to it explains the shape of the failure.**
GLM-5.3 spends reasoning tokens out of the same output budget. With 8192 total,
a hard authoring or review step exhausts the budget while still thinking and
has 0-7 tokens left to write anything. That is why the artifacts were never
written despite the model working for over two minutes: cycle 9 ran 132.2 s and
cycle 10 ran 129.0 s, both ending with an empty answer.

## Options

1. **Correct the declaration to the model's real capability** (`output:
   131072`). This is a one-line config change requiring no new provider spend
   to apply, and it matches what every catalogued GLM-5.3 provider advertises.
   If the gateway deployment genuinely caps lower than upstream, the right
   value is whatever that deployment enforces — but 8192 should be confirmed
   rather than assumed, since nothing in the catalog supports it.
2. **Raise it partially**, e.g. to 32768. Lower risk if gateway billing is a
   concern, and still roughly 4x the headroom that starved these runs.
3. **Leave it and restructure the work into smaller steps.** Cycle 10 already
   tried this with a two-part review split and still hit the ceiling at
   8187 + 5, so the step size is not the binding constraint. Not recommended.

## A second, independent blocker

Fixing the token ceiling is necessary but may not be sufficient. The harness
also caps the review call at 60 seconds:

```
418:            rc = p.wait(timeout=60)
```

Cycle 8's review failed on exactly this timeout, separately from the token
issue. Observed review steps took 132.2 s and 129.0 s, so a reasoning model
will very likely exceed 60 s even once it has tokens to spare. The authoring
call is given 240 s by comparison. Expect to raise the review timeout before
the end-to-end path can complete.

## Status

Root cause identified; no fix applied, and no provider call made. Applying
option 1 or 2 is a configuration change to a file outside this repository
(`~/.config/opencode/opencode.json`) and is left to the maintainer's decision.
