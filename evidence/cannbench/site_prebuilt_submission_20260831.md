# CANNBench private submission: vendored pyasc v2

- Source: `compiler-team/pyasc`, branch `v2`
- Commit: `ac1222a48c8914d3f81297c7570d1a84f0f26778`
- Runtime ABI: CPython 3.12 / x86_64
- Runtime wheel SHA-256: `0e9ef3873ffdc0f6926f4fb05ce3b07e4620602aaea0ff707d3e301a8f33e1dc`
- Submission ZIP SHA-256: `5a94b247441f0d10461be14078676f1309f88b0f61a5a57713674041de5ba18b`
- Hardware / benchmark: `950pr` / `official-tasks` 1.1.1
- Operator: `masked_scale`
- Private submission: `sub_a395400f9c50`
- Job: `job_5a902df0e8c1`
- Status: `succeeded`
- Result: 20/20 cases, score 82.60741992933424
- Geometric-mean speedup: 1.553853403049041
- Anti-cheat failures: 0

The first prebuilt job, `job_9cb9913ccd4a`, proved that the wheel compiled
offline but exposed a hard-coded SoC mismatch (`9599` vs runner revision
`957c`). The wrapper now lets pyasc detect the active NPU revision. The next
job succeeded without changing the operator implementation.

Artifacts:

- `site_prebuilt_job_5a902df0e8c1.json`: full successful job and result
- `site_prebuilt_job_5a902df0e8c1_logs.json`: successful stage log
- `site_prebuilt_job_9cb9913ccd4a.json`: failed-run diagnosis
- `site_prebuilt_job_9cb9913ccd4a_logs.json`: failed-run stage log
