# Evidence sources

| Claim | Authoritative evidence | Limit |
|---|---|---|
| Upstream API and target behavior | Pinned Huawei-authored tests in `reference/asctile/target/` and their actual run reports | Vendoring or collecting tests is not executing them. |
| Generated-kernel correctness | Task oracle and per-case numerical results for the exact candidate | Candidate agreement with another generated kernel is not an independent oracle. |
| Simulator environment | `reference/images/ENV.json`, image digest and upstream qualification report | A published/importable image does not imply the whole target suite passed. |
| Hardware correctness and performance | Official CANNBench task snapshot, exact evaluated package, terminal job and device timings | Scope is the recorded cases, runtime and runner. |

## Upstream target suite and simulator

`reference/asctile/PIN.json` records source revision and hashes;
`scripts/check-target-pin.sh` checks drift offline. Local import compatibility
changes are documented in [reference/README.md](../reference/README.md).
Derive case counts from the pinned tree instead of copying a historical total.
The upstream `kimi_k3/` name does not change its recorded provenance.

Repository CI uses `.dev_image.ref` from `reference/images/ENV.json`, including
its pinned torch/pytest dependencies. `verify_pyasc_image.sh` verifies the
runtime contract, and `check-image-pin.sh` checks the overlay/base relationship.
The repository consumes these images; it does not build them.

For upstream execution coverage, read the qualification report linked by the pin
and select the actual architecture/cell. Distinguish environment qualification,
module collection, compilation and numerical execution. Read skipped/blocked
counts as well as failures; absence of failures does not prove complete coverage.
A killed process or exit 137 alone does not establish an OOM diagnosis.

`merge-gate` invokes `run_target_test.py`, defaulting to compile-only. It selects
a representative case when the module's declarative case list permits it;
otherwise it processes the module's cases. Its green result means all discovered
modules completed that compile policy, not that every possible specialization
or numerical case passed. Resource-blocked modules make this gate nonzero.
Use explicit `--full --run` for a full module numerical attempt and report its
actual coverage. See [manual review](review-order.md).

## CANNBench results

The bounded lane snapshots the live official task/version before preparation and
rejects drift before submission. Vendored task files support development; they
do not override that live contract. Model-produced kernels are test subjects,
not authoritative correctness or CANN performance baselines.

Retain the service `overall_score` with the complete per-case report. Compute a
separate timing ratio only from matching positive finite baseline/candidate
measurements. The hardware lane reports their unweighted geometric mean; it is
not the service score and is not simulator ticks. Do not apply a historical
campaign threshold to this measurement.

The [recorded generated qualification](hardware-lane.md) includes model/skill
provenance, runtime, exact package continuity, official numerical results and
device-kernel timings in a durable Release. This is an implemented CI path,
limited to the qualified recipe; it is not a broad benchmark campaign.

TTK case definitions can inform separate CANN-side reproductions; see the
[CANNBench skill reference](../skills/pyasc-cannbench-kernel/references/ttk.md).
A private TTK export is not needed to read or validate the current service report.

## Retention and status

Use `passed | failed | blocked | not-run` for each gate independently.
[Verification status](verification-status.md) states current qualified scope;
[the storage contract](verification-contract.md#storage) defines accepted
archives. Older image-status snapshots and campaign assumptions remain in the
[historical source guide](https://github.com/aloschilov/pyasc-skill-stack/blob/654a05556107c538dca11fc71b02fd1378543ded/docs/evidence-sources.md),
not as current acceptance evidence.
