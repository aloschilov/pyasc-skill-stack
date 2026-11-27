# Supplied SwiGLU tuning task

The supplied baseline implements `swi_glu(input, dim=-1)`. This task covers finite, contiguous 2D FP16 inputs in [-1, 1] with `(rows, half-width)` equal to `(5, 61)`, `(29, 191)` and `(47, 509)`. The trusted evaluator uses seeds 19 and 47. It owns all input generation, CPU oracles, numerical thresholds and canaries; do not reproduce or replace them.

Load all four provided skills. Qualify the baseline before proposing a change. Its receipt reports observed resource queries, launch grids, effective compiler options, compiler UB and source/codegen bindings. Use that evidence to identify one opportunity. No particular parameter value or winner is prescribed.

Only `_row_geometry` parameters may change. Everything else must remain AST-identical, including the kernel, arithmetic, resource query, width calculation, UB shrink loop and launch-count formula. You may choose the initial height from 1, 2, 4, 8, 16, 32, 64 or 128, and change/add up to five scalar caps of the form `T = min(T, expression)` before the existing positive clamp. Expressions may use R, H, aiv, T, W, the existing UB bound, positive bounded integer literals, integer +, -, *, //, and min/max/_pow2_ceil. No lookup tables, attributes, new loops, imports, extra launches or host tensor computation are permitted. This restriction defines the small study, not a universal tuning API.

Before calling `variant`, decision.json must contain:

- `hypothesis`: a concrete opportunity inferred from the baseline receipt;
- `parameter_change`: the single intended control;
- `baseline_receipt_sha256`: SHA256 of that baseline receipt (returned by the interface).

The evaluator snapshots this document and candidate.py. There is one evaluated variant; no repair, substitution or later edit of that candidate. Both snapshots must pass all six numerical checks and all six rejecting suppressed-launch controls before comparison.

Comparison uses both ABBA and BAAB orders. Every series has one warmup and three measured invocations, with the same module/JIT objects, compiler/runtime/source bindings and original numerical checks. Timing brackets only the synchronized device launch in CaModel. It does not measure Python or whole-function latency. The report includes per-shape results; no shape pooling.

Selection rule: keep the baseline unless all results and bindings are complete, both orders agree in direction for every shape, the variant has no greater than 1% tick regression on any shape in either order, and at least one shape has more than 1% tick reduction in both orders. The 1% engineering margin was fixed from earlier retained simulator variability. It is not a confidence interval or hardware-noise estimate. Unknown outcomes, timeout, incomplete evidence, ambiguity or a tie retain the qualified baseline; an unresolved process also stops further work.

After the comparison (or a known nonqualifying variant), update decision.json without changing its hypothesis/control/baseline receipt fields. Add `selected_candidate_sha256`, `baseline_sha256`, `variant_sha256` (when a snapshot exists), `comparison_receipt_sha256` (null if no comparison), and `rationale`. The decision command checks the selected hash and evidence references; it does not write your decision.

Bounds: 12 provider requests, 200,000 output tokens, profile-native 32,768 tokens per response; 900 seconds of non-tool elapsed generation time within a 2,700-second total session. The client watchdog enforces idle time as well as provider dispatch; only validated bounded owned evaluator time is excluded. Baseline and variant qualification each have a 240-second owned bound. Comparison has a 540-second aggregate measurement bound, a 560-second inner/client bound and a 580-second owner bound. Its outer wait is 585 seconds; actual preparation plus that wait and five seconds of postprocessing reserve must fit the 595-second action cap inside the native 600-second timeout. Insufficient remaining time stops before dispatch, and budget failures are distinct from numerical failures. Failures and actual consumption are retained; no retries or additional variant are scheduled by this recipe.
