# SwiGLU CANNBench review task

Review the existing generated `candidate.py`; do not redesign it unless a
concrete defect is found. The exact public schema is:

`swi_glu(input: Tensor, dim: int = -1) -> Tensor`

Split the even-sized selected dimension into `x0`, `x1` and compute in f32:
`(x0 * sigmoid(x0)) * x1`; cast once to the input dtype. Output shape equals
input shape with the selected dimension halved. Numerical work must remain in
`@asc2.jit`; torch is limited to allocation, contiguity and metadata views.

The 20 official routes cover f16/f32/bf16, rank 2 through 5, dimensions
`-1/0/1/2`, aligned and prime tails, 150K through 134M input elements, ranges
from tiny values through f16 boundaries and ±1000, plus zero, Inf and NaN.
Critical routes are `[363,367,14]` bf16 dim=-1; `[1000003,2]` bf16 dim=1;
`[11,13,16,67]` f32 NaN dim=2; and row-axis splits where contiguous halves
cannot be assumed. The pinned v2 2-D copy path requires the final copied
dimension to be at least 32 bytes; a separate 1-D metadata-split path may be
used for narrower layouts.

The existing seed already passes all 20 exact-v2 dispatch/lowering routes.
Preserve that status. Local compilation proves neither numerical correctness
nor NPU performance. Run only `python3 -m py_compile candidate.py` and finish
with `REVIEW_DONE`.
