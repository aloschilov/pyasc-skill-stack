# CI and CANNBench acceptance

## Current pipeline

`CI` runs static/L1 checks and CANNBench acceptance unit tests on a GitHub-hosted Ubuntu runner. These checks require no CANN runtime. CANNBench execution belongs on the Ubuntu VM, through the dedicated `cannbench-vm` runner. That runner deliberately has no default labels, so historical queued `arm64` jobs cannot execute there.

`CANNBench gate` captures the complete official site catalog and checks hardware evidence for every listed operator and case. On 2026-10-01, the accepted submission version was `official-tasks/1.1.2`: **53 operators, 1060 cases**. The checked-in snapshot includes the raw catalog, exact task inputs and SHA256 identities. A live capture checks the catalog before and after collecting all tasks. No missing operator, case, unsupported dtype or failed generation is removed from the denominator.

A manual run can generate the entire catalog using `gateway/glm-5.3` plus an independent model review. Workers receive the complete task and pinned API guide; they can edit only their own candidate/design files. A reviewer receives measured compiler feedback, and the reviewed source is compiled again. The daily schedule reads explicitly selected hardware jobs or the pending/latest durable job; it does not create submissions. `OPENCODE_API_KEY` is encrypted in GitHub Secrets, and `OPENCODE_BASE_URL` is a repository variable. They were imported from OpenCode on the VM without saving the key to this repository.

Hardware results must match the catalog version and hardware, include the exact complete case set, final correctness and anti-cheat results, and finite positive candidate/reference timings. Duplicate measurements are rejected rather than merged by fastest case. The true geometric mean is recomputed from `baseline_perf_us / elapsed_us`; the misleading API aggregate is ignored. Local dispatch/lowering and Model smoke results do not satisfy hardware acceptance.

The workflow has two separate results: hardware evidence and skill-stack acceptance. Skill-stack acceptance additionally requires all 53 fresh candidates, native skill traces, a different implementation/review model, exact task and candidate hashes, complete compiler coverage, an installed evaluator-wheel replay, and a durable identity binding to the submitted archive and returned hardware job. An imported historical job alone cannot satisfy this gate.

A manual run with `generate=true` and `submit=true` requests one full-catalog hardware submission after every earlier qualification passes. A failed or partial generation creates no submission. POST has no retries; durable upload intent is written under the existing shared upload lock before sending. After an ambiguous response, the tool reconciles the unique tag by GET and refuses to repeat POST even when no match is visible. State and the hashed submitted archive live outside the runner checkout under `/home/aloschilov/ci-campaign-state/pyasc-cannbench`. Schedules reconcile pending uploads and refresh existing jobs using GET only, including after runner cancellation.

These stages are implemented but full-catalog generation, installed-wheel replay and hardware acceptance have not yet completed. Infrastructure tests are not evidence of 53 correctly generated/accepted kernels.

## Dashboard

Pages downloads the evidence artifact from the matching CANNBench run, including failed gates, and renders every official operator. Missing and failed cases remain visible. Simulator comparison and historical skill-intervention results remain separate diagnostics. No evidence commit is needed for metrics publication. Generation artifacts are retained separately for 90 days. Each run has a unique artifact directory. A run that fails before uploading hardware evidence is published as missing/failed, preserving the full denominator.

## Commands

On the Mac, for static verification:

```bash
python -m unittest discover -s tests/unit -p 'test_cannbench_*.py' -v
python integrations/cannbench/ci/validate_catalog.py integrations/cannbench/catalogs/official-tasks-1.1.2-20261001
bash tests/ci-gate.sh --tier pr
python tests/tools/generate_dashboard.py --output-dir _site
```

On the VM, with its existing CANNBench credentials:

```bash
python3 integrations/cannbench/ci/run.py --output evidence/cannbench/ci-current --job-id job_13cb06f8dff9
```

The GeLU-only example intentionally fails the complete 1060-case gate. It is an example of retaining partial measured evidence, not acceptance of the catalog.

A full campaign is explicitly requested on the VM with:

```bash
python3 integrations/cannbench/ci/run.py --output evidence/cannbench/run-unique --generate --submit
```

Compilation and installed-wheel replay use the existing shared execution locks. The runtime wheel and compiler image are pinned by SHA256 in `integrations/cannbench/ci/runtime.json`; a changed image or wheel requires requalification. Runtime packaging includes `asc` and the current public `asctile` package; the historical packaging helper's `asc2` pin is not reused. The hardware job is the full numerical and performance oracle. No canonical module is promoted by this campaign.

The dedicated runner is managed by `pyasc-cannbench-runner.service` in the VM user manager, with linger enabled for reboot persistence. `runner_service.py` refuses to replace a different service or interrupt a busy runner.

The output directory must be new. Never overwrite prior snapshots or durable submission state. Credentials stay on the VM. Preserve the historical shared simulator and upload locks. Reconcile an ambiguous submission by GET; never repeat POST automatically. No automatic commit/push.

## Legacy diagnostics

The previous simulator/OpenCode intervention workflow is archived as `docs/ci/legacy-ci.yml` for reference. Its manually selected capability cells, simulator ticks, and report-only 0.70 threshold are historical diagnostics, not the current CANNBench kernel set or hardware gate. The standalone legacy helper scripts and evidence are preserved.
