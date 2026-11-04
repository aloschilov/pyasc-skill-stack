# Verification status

Use `passed | failed | blocked | not-run` per acceptance scope. Missing
prerequisites or resource blocks do not count as successful coverage.

| Scope | Evidence and limit |
|---|---|
| Offline checks and shipped simulator fixtures | Required on each PR head by `pr-gate` and `examples-gate`; inspect that head's Actions results. These do not invoke a model or NPU. |
| Fresh generation and hardware measurement | **Passed** for one GLM-5.3 SwiGlu candidate: 20/20 official cases, zero reported anti-cheat failures, device timings. [Durable result, revisions and checksum](hardware-lane.md). |
| Performance improvement | **Not established**: the generated candidate's baseline/candidate geometric timing ratio is 0.017450287921005128x (slower). Service score is a separate metric. |
| New profile and recipe plumbing | Offline regression scope only. The high-effort profile and shared recipe driver have no new generation/hardware qualification. Historical results retain their original configurations. |
| Separate reviewer / tuner automation | **Not-run** in the recorded generated CI qualification. Skills retain these phases; [issue #20](https://github.com/aloschilov/pyasc-skill-stack/issues/20) tracks automatic handoffs. |
| Full upstream target suite | No complete numerical pass claimed. `merge-gate` is compile-only and now requires every discovered module to complete; resource blocks fail coverage. |
| Other operators or nightly generation | **Not-run** by the bounded hardware lane; no automatic campaign or nightly schedule. |

The checkout and installed-skill workflows also have earlier simulator-only
qualifications. Their source hashes, results, archive locations and checksums
remain in the [immutable historical index](https://github.com/aloschilov/pyasc-skill-stack/blob/654a05556107c538dca11fc71b02fd1378543ded/docs/verification-status.md).
Those records do not qualify later revisions or expand hardware coverage.
Full run artifacts remain outside checkout; see the [storage and attribution
contract](verification-contract.md).
