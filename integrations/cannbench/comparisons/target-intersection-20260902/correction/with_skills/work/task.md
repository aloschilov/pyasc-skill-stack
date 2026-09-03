# Repair task: CANNBench GELU after real 950PR feedback

Repair `candidate.py` for the vendored CANNBench GELU contract. Numerical work
must remain in pyasc v2 `@asc2.jit` kernels and the public signature remains
`gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor`.

The candidate already passes the exact-v2 compile matrix, but private 950PR job
`job_ae3bfdefd087` passed only 16/20. All failures are FP32 precision failures:

- case 5, tanh, range [-100, 100]: MERE 9.008301e-4;
- case 8, tanh, range [-5, 10]: MERE 0.003110;
- case 11, exact, range [-88, 88]: MERE 0.002386;
- case 20, exact, range [-20, 40]: MERE 0.006894.

BF16 Inf/NaN case 12 passes in this seed and must remain passing. The direct
`1 + erf/tanh` forms cancel in negative tails. Apply the updated
`pyasc-cannbench-kernel` skill's 950PR-verified stable GELU formulations. Do not
copy or inspect the repository's canonical submission implementation. Preserve
all dtypes, tails, attributes and special-value semantics. After editing, run
only `python3 -m py_compile candidate.py` and end with `REPAIR_DONE`.

The first repair attempt is rejected. It substituted a different
Abramowitz-Stegun-style polynomial, evaluated it at `abs(x)` instead of
`abs(x)/sqrt(2)`, and did not implement the required Numerical Recipes Horner
chain. Replace that rejected exact-mode body using the now-explicit coefficient
sequence in the skill reference; do not merely review it for syntax.

The second repair attempt implemented the required formula, but the exact-v2
compile gate rejected every exact-mode specialization. With
`tile_size=2048`, `unroll_factor=2`, reported UB use is 819968 bytes for
FP16/BF16 and 795392 bytes for FP32, above the 253952-byte limit. The three
tanh specializations pass at tile 1024 and use 172032--184320 bytes. Apply the
updated measured tile rule from the skill reference: exact mode must use the
safe 512-element tile with unroll factor 2. Keep the stable formulas and do not
regress the already-passing tanh branch.
