# Independent review adjudication

OpenCode `dashscope/qwen3.7-max`, session `ses_f83372b82ffevFC3dkTWaJUJjP`;
full skill/candidate/issue/evidence supplied inline, tools denied.
Prompt SHA256: `8e69866a62e601cfdfaa65d83b2738f34f920bef7131ae6647825b3013ea5645`.
The reviewer is advisory, not the acceptance oracle.

- Accepted: final combined package needs its own numerical and 20-case dispatch/compile qualification. Previous isolated functions are insufficient.
- Accepted: the Erf API request is appropriately scoped; the subsection algorithm was not measured and is not a proven fix.
- Rejected: the alleged `-Inf` correctness blocker. The official Torch oracle returns NaN for GeLU(-Inf), in both modes. This is already executed by the harness. Clamping the tail to force a limiting zero would change the required contract.
- Rejected: claimed SciPy spot-check execution. Tools were denied and no execution evidence exists. The review's explanation also loses the negative sign and introduces a spurious factor 2. The actual identity is `x*sigmoid(sqrt(8/pi)*(x+0.044715*x^3))`.
- Corrected: case 5 has 67,108,864 elements, not 268M elements; 268,435,456 is its FP32 byte count. Three configurations cover six dtype/mode routes; these are not three missing routes.
- Context supplied: 15872 is the measured maximum fitting this target FP32 tanh configuration (253952 B UB), not an arbitrary power-of-two requirement. Historical promoted/exact low-precision samples exist; nevertheless the final module is rechecked.
- Confirmed: both branches of `where` are evaluated; negative-tail rarity does not avoid CF arithmetic.

After the inline source snapshot, host geometry was changed to accept an explicitly annotated `is_fp32: bool` rather than an unannotated dtype object. This satisfies the conservative source gate without weakening it; kernel mathematics/geometry are unchanged. Final source hash and revalidation evidence are recorded in package.json and final checks. No review statement is counted as measured correctness or performance.

## Final bounded-loop revision

The first combined source emitted zero-length DMA on unused end tiles of a
core partition. FP16 `N=73729`, tile8192, unroll2, eight cores passed values
and guards but logged `mte_gdma_illegal_burst_len_t0`; that run is rejected.
The final source bounds the loop by each core's actual remaining tile count,
so empty tiles do not execute. This is a host/kernel partition fix, not a
compiler patch. Final checks retain diagnostics and reject unexpected errors.
`vec_err_idata_inf_nan_t0` alone is separately counted during deliberate
NaN/Inf numerical stress, and only with a passing special-value oracle check;
it is not conflated with invalid DMA or synchronization errors.

OpenCode reviewed the exact final source in session
`ses_f8329847bffee0yi5gbyJuEoSu` (`dashscope/qwen3.7-max`, tools denied).
Prompt SHA256 `a10ab9460704a7f2027a21929b10c72b0f4df11bff24c5a51d03e80379aebeb3`;
source SHA256 `3f068907fe7437b8c0daeb5310e8ea2cb3617bd11d242747e093045e5a943d52`.
The reviewer correctly checked contiguous non-overlapping partitions, zero
iterations for idle cores, six dtype/mode routes, standard tanh constants and
Torch special-value semantics. Its statement that final qualification is
"optional" is not adopted: the coordinator enforces all local gates before
the single upload. No full hardware result is inferred from this review.
