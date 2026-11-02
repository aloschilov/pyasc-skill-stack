# One-operator CANNBench hardware smoke

CANNBench provides external NPU evaluation without access to a hardware host.
The minimal supported path checks one **SwiGlu** candidate against the current
public task contract. It uses the existing service adapter and strict hardware
result checker; the older full-catalog worker campaign is not required.

This is a manually invoked, credit-consuming check, not a PR CI job. The shipped
kernel is a historical PyASC reference, not a newly generated candidate. A pass
establishes packaged-kernel correctness and timings on the reported runner only.
It does not qualify fresh generation, all operators, local hardware, or the
simulator's suppressed-launch numerical-control contract.

## Requirements

- A full checkout, Python 3.10+ on Linux/macOS, `curl`, and the official BenchSite
  adapter (`bash integrations/cannbench/setup-site-mcp.sh`).
- An existing aggregation scheme owned by or shared with the account. Pass its
  ID as `--aggregation-token` (or `CANNBENCH_AGGREGATION_TOKEN`); this is required
  for private submissions too ([current API contract](https://cannbench.com/docs/mcp)). The smoke never creates schemes or opts into
  third-party independent runner environments.
- `BENCHSITE_API_TOKEN` in the environment, or an existing private environment
  file passed as `--credentials-file`. Never commit or print it.
- A self-contained PyASC runtime wheel for **CPython 3.12 / Linux x86_64**, with
  `asc`, `asctile`, `pybind11` and their licenses. Supply its SHA256 explicitly.
  Runtime binaries are external inputs, not checked into this repository.
  The packager copies only runtime/license files; it replaces candidate code,
  wheel metadata and RECORD. The ZIP contains the candidate sources, `setup.py`
  and `build.sh`; the build stages the exact hash-verified prebuilt wheel. Arbitrary PyASC revisions are not presumed compatible.

The qualification runtime bundle is available in the preserved history at
`origin/archive/campaign-2026`, under
`integrations/cannbench/submission/` (build recipe and vendored runtime).
See [the smoke result](../../docs/hardware-lane.md) for the exact tested binary
hash and retained archive. Use that binary or build and qualify a compatible
bundle; the pinned simulator image is not a CANNBench submission runtime.

## Prepare, submit once, collect

Use one persistent state directory per service account on the submitting machine.
Results must be outside checkout. `prepare` takes a fresh live task snapshot,
syntax-checks candidate sources and produces a deterministic package. It does
not claim local compilation or NPU correctness.

```bash
PYTHON=.tools/benchsite-mcp/bin/python
RUN="$HOME/pyasc-qualification-runs/swiglu-smoke-$(date +%Y%m%d-%H%M%S)"
"$PYTHON" integrations/cannbench/smoke.py prepare --run-dir "$RUN" \
  --runtime-bundle /absolute/path/runtime-cp312-cp312-linux_x86_64.whl \
  --runtime-sha256 <verified-sha256> --aggregation-token <existing-scheme-id>
"$PYTHON" integrations/cannbench/smoke.py submit --run-dir "$RUN"
"$PYTHON" integrations/cannbench/smoke.py collect --run-dir "$RUN"
```

Replace the hash placeholder before running. Each deliberate new submission
consumes a credit; inspect the service balance and keep within the available
10/day working budget. `submit` checks credits, active jobs and task drift;
it saves a durable intent **before** its only possible upload. It never retries
POST automatically. An explicit HTTP rejection is retained as `rejected`; prepare
corrected inputs before a deliberate new submission. Unknown delivery remains
blocked until GET reconciliation. The default ledger is `~/.local/state/pyasc-cannbench-smoke`.
Keep it and the run directory after interruptions. Do not work around an
unresolved intent by changing state directories.

`collect` only issues GETs, reconciles an uncertain upload by its original tag,
and saves `job.json` and, when terminal, `report.json`. Repeat `collect` while
queued/running (exit 2), not `submit`. Exit 0 from `submit` confirms an intent/job,
not hardware success. Exit 0 from terminal `collect` requires the exact official
case set, correct benchmark/version/hardware, successful numerics, zero anti-cheat
failures and positive finite timings. A failed hardware gate exits 1 and retains
the service report. No automatic repair/resubmission occurs.

To check a generated SwiGlu candidate later, pass `--kernel-dir` containing
`swi_glu.py` and `_pyasc_runtime.py` to `prepare`. Use the task's callable contract,
perform numerical work in the PyASC kernel, and record generation provenance
separately. No second model is required by this smoke path. The legacy campaign's
multi-model requirement and archived resources remain outside this scope.

Keep full evidence privately outside Git. For accepted qualification, retain a
durable archive and publish only a short result, archive checksum and reference.
