# Pinned upstream numerical census

This lane concerns the upstream runtime fixtures, not model generation. Its
inventory contains **1,155 cases in 24 modules** at PyASC
`d0f64a63c432761aa57648e67be7582780a17561`. That count proves collection only.
At stack source `a3e3075`, **six selected official cases qualified**: float32 add
with a tail, int32 reduction and bfloat16 broadcast, each in static and dynamic
form. They produced six completed Model launches, six official assertions and
six rejecting controls. The other **1,149 cases remain unevaluated**.

The [durable subset archive](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-target-subset-a3e3075)
is `qualification-target-subset-a3e3075.tar.gz`, SHA256
`8da0a446302304cda44b934b4a0abb8bac3e219d0ac289b2aa6a4924c3c1e6ce`.
Its 107 member hashes and 38 source hashes were verified; copies are retained on
the qualification Mac and Linux VM. Real phases took 34.70 s and controls 12.89 s
including container startup/cleanup. All owned containers were confirmed absent,
and resuming an existing shard preserved every case artifact without replay.
The earlier package-import failure is retained separately in the same release;
it executed no kernels and does not count as a kernel defect.

`inventory.json` versions the derivation inputs, module counts and exact sorted
pytest-ID hashes. The complete IDs are reproducibly collected from the pinned
source and retained externally with stdout, stderr and checksums. Every upstream
file and local fixture is hashed; empty, duplicate, missing or changed IDs fail
validation. Collection uses the pinned development image, one CPU, 2 GiB,
180 seconds, no network and no device access.

```sh
python3 tests/tools/target_inventory.py collect --output /outside/collection
python3 tests/tools/target_suite.py plan --collection /outside/collection \
  --output /outside/plan.json
python3 tests/tools/target_suite.py run --plan /outside/plan.json \
  --root /outside/run --shard 0
python3 tests/tools/target_suite.py report --root /outside/run
```

The plan includes every official case. By default it selects all cases and
partitions them into shards of up to eight, with 30 seconds per execution or
control phase and a 900-second shard deadline. Each phase reserves an additional
20 seconds for process/container cleanup. Resource limits are one CPU and 2 GiB
per container; timeouts and missing worker receipts remain explicit gaps. Large
production shapes may exceed these limits. The plan can set smaller/larger
bounded shards; limits and selected cases cannot change during resume.

An explicit `--selected /outside/cases.json --excluded-reason "..."` selects a
bounded subset from the full inventory. Exclusions retain their reason and
remain in the full-suite denominator. All generated plans, snapshots, receipts,
logs and coverage reports belong outside the checkout.

Run another shard with the same plan and result directory to continue. A lock
prevents concurrent writers. Previously attempted cases are retained, including
failures; they are never replayed. An interrupted case or unconfirmed container
cleanup blocks further execution until its exact work is reconciled. A changed
source/runtime/selection requires a new plan and external run directory. Reports
reject missing or altered artifacts and mismatched case/runtime provenance.

For each case, the original pytest fixture executes in the Model backend. A
call-through wrapper records completed `Launcher.run` calls and the original
`torch.testing.assert_close` calls, including their source locations and explicit
tolerances. Omitted assertion options retain the recorded Torch version's
defaults. No oracle, tolerance or test input is rewritten. A separate run
suppresses launches and must fail at the same numerical assertion; unrelated
errors, skips, expected failures, zero launches or an accepting control cannot
qualify the case. The retained runtime version and source hashes bind both runs.

Coverage means the assertions actually present upstream. For example,
`test_rms_norm.py` currently checks `y` but leaves the `rstd` assertion commented
out; passing that case cannot establish `rstd` correctness. The harness does not
invent missing upstream output checks. Source-level gaps and reproducible runtime
failures should have focused follow-up issues.

`coverage.json` distinguishes qualified, failed, unsupported, timed-out, blocked,
unresolved, not-run and excluded cases. `all_selected_qualified` describes only
the planned subset; `full_suite_qualified` requires all 1,155 cases. The command
returns nonzero for incomplete full-suite coverage even if the selected subset
passes. Compile-only, NPU and model-generation coverage are explicitly separate.
This JSON is a dashboard input, not a blanket green badge.

Ordinary PR checks run offline contract tests only. Larger numerical shards are
manual and separately bounded; no schedule or automatic NPU use is enabled.
Any future periodic policy must specify its shard/resource budget and record
the complete denominator, including shards that were not run.
