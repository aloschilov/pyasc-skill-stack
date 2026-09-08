# Independent review adjudication

OpenCode `dashscope/qwen3.7-max` reviewed the exact supplied source and harness;
see [provenance](workers-submission/provenance.json). It executed no tests.

- Accepted: UB capacity is a feasibility constraint, not a throughput metric.
  Compare identical dtype, mode, mathematics and logical input size; do not infer
  causation from the previous table mixing exact/tanh and different shapes.
- Accepted: one-core CaModel timings do not predict 72-block hardware latency.
  They isolate tiling/allocation effects at fixed launch count. Official dispatch
  is independently compiled and one diagnostic hardware run remains necessary.
- Rejected: changing the number of tiles invalidates a matched operator workload.
  With identical logical work and fixed cores, tile count/padding are precisely
  the geometry effects being measured; label the scope, rather than conceal them.
- Rejected: multi-core tails are absent. `tune.py qualify` explicitly passes
  `--cores 8`, checks the output guard and repeats cached launches. Full official
  shapes are dispatch/compile tests; reduced Model execution is labeled separately.
- Rejected: failed timing accuracy remains eligible. `execute()` requires
  `numerical_passed` and every `repeat_accuracy` before marking a result valid.
- Rejected as an established fact: mode 2 necessarily aliases every second
  iteration and introduces the specific claimed hazard. The reviewer supplied no
  compiler/pass evidence. Synchronization verification, multi-tile repeated Model
  tests and hardware correctness are required; no unmeasured safety claim is made.
- Accepted with limits: three warmed deterministic Model samples are not a
  hardware confidence interval. Preserve raw samples and report no NPU speedup
  from them. Requiring ten samples does not itself validate simulator fidelity.

Bounded-search adjustment: initial 65536-element FP32 probes were stopped before
completion because simulator wall time was excessive. They are not timing results.
The final matched screen uses 8192 logical elements for FP32 exact and 32768 for
the other routes; comparisons are only within a route and workload. All source
math remains unchanged. The table reports launched blocks separately from blocks
with useful work (analytically reconstructed, not sampled device occupancy).
