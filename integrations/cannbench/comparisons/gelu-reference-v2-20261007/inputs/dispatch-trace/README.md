# GeLU reference: per-shape source dispatch trace

Investigation date: 2026-10-07. **All20 cases resolve to SIMD on the inspected
Ascend950 eager Torch-NPU/CANN source path. No shape selects SIMT.**

[All20 shapes, dtype/mode, template arguments and implementation](cases.csv)
are also recorded in [trace.json](trace.json), including source SHA256 hashes.
Reproduce the inventory/assertions with `python3 trace.py` from this directory.
This is static source tracing, not a hardware execution or a recovered binary
trace of CANNBench's historical baseline timing measurements.

## Source checkouts and scope

- `/home/aloschilov/workspace/torch-npu-gelu-trace-20261007`:
  [4bf6fb09fc85314ce492941788c5104fcebeb250](https://github.com/Ascend/pytorch/tree/4bf6fb09fc85314ce492941788c5104fcebeb250).
  Selected from v2.10.0 as the latest public commit before the saved job's
  2026-09-16 17:39:53 UTC timestamp. Not claimed to equal its wheel build commit.
- `/home/aloschilov/workspace/op-plugin-gelu-trace-20261007`:
  [daa248da671c9620a5753a17f348525092c0a715](https://github.com/Ascend/op-plugin/tree/daa248da671c9620a5753a17f348525092c0a715),
  exactly the gitlink of that Torch-NPU revision, not an unrelated master tip.
- Existing `/home/aloschilov/workspace/ops-nn`: commit
  `f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307`; relevant GeLU files unmodified.
- Existing `ops-nn/third_party/opbase`: commit
  `f59a9cb9f414c4b80cf4eccc58006798c1af15de`; relevant ATVOSS files unmodified.
  Other existing changes in these repositories were preserved.
- Installed CANN9.0.0 headers establish the Erf implementation/defaults below.
  The saved runner reported CANN9.1.0 and torch_npu2.10.0, without source hashes.
  No global installation was changed and no submissions were made.

## Complete dispatch chain

1. [CANNBench golden](../golden.py) calls
   `torch.nn.functional.gelu(x, approximate=approximate)`.
2. [PyTorch2.10 schema](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/native_functions.yaml#L4701)
   delegates functional `gelu` to structured `gelu.out`.
3. op-plugin `op_plugin/config/op_plugin_functions.yaml` registers `gelu.out`
   for the op_api backend. This analysis assumes ordinary eager NPU execution,
   not separately enabled DVM, torch.compile, custom operators or CPU golden.
4. [Torch-NPU NpuVariables.cpp](https://github.com/Ascend/pytorch/blob/4bf6fb09fc85314ce492941788c5104fcebeb250/torch_npu/csrc/core/npu/NpuVariables.cpp#L44)
   recognizes the `Ascend950` prefix (including `Ascend950PR_9589`).
   `IsAclnnOnly()` returns true for Ascend950 and above.
5. [GeluOutKernelNpuOpApi.cpp](https://github.com/Ascend/op-plugin/blob/daa248da671c9620a5753a17f348525092c0a715/op_plugin/ops/opapi/GeluOutKernelNpuOpApi.cpp#L21)
   therefore takes `aclnnGeluV2`, not the legacy `aclnnGelu` branch, independent
   of shape and compatibility flag. `KernelNpuNewParams.cpp` maps none to0,
   tanh to1. These API enum values differ from kernel template mode values.
6. `ops-nn/activation/gelu_v2/op_api/aclnn_gelu_v2.cpp` checks dtype/shape,
   handles empty inputs, normalizes contiguity, and calls `l0op::GeluV2`.
   None of our20 shapes is empty. Arbitrary strided views can require extra
   copies; the shapes alone do not specify strides.
7. `op_api/gelu_v2.cpp` registers the AICore launcher. The arch35 host tiler
   selects `GET_TPL_TILING_KEY(1, approximate, dType)`; the entry point is
   `gelu_v2<schMode, approximate, dType>` in `op_kernel/gelu_v2_apt.cpp`.
8. The entry point selects `ElementwiseSch16B<1, OpDag>` with the DAG below.
   AIV_ONLY is not by itself SIMD proof: the vector body and Erf callee are.

## All six routes and coverage

- FP16 exact, cases1/7: `GeluV2Erf16BDag<half>`.
- BF16 exact, cases3/9/16: `GeluV2Erf16BDag<bfloat16_t>`.
  Both promote to float, multiply by1/sqrt(2), use `ErfFast<float>` (default
  AscendC Erf, Padé in inspected SDK), execute `GeluV2ErfPost<float>`, then
  cast back. Postprocessing uses a SIMD `__VEC_SCOPE__` with masked registers.
- FP32 exact, cases2/11/13/15/18/20: `GeluV2Erf32BDag<float>`.
  Uses `Vec::Erf<float>` from ATVOSS, explicitly configured with
  `ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION`, followed by the SIMD post-op.
- FP16 tanh, cases4/10/14/17: `GeluV2TanhDag<half>`.
- BF16 tanh, cases6/12/19: `GeluV2TanhDag<bfloat16_t>`.
- FP32 tanh, cases5/8: `GeluV2TanhDag<float>`.
  All tanh routes calculate in FP32 via `GeluV2Tanh<float>`: vector square/cube,
  Axpy, multiply, Exp, Adds, Div. Low-precision routes have conversion stages.

The decisive math source is
`ops-nn/activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h`.
The inspected CANN Erf implementation ends in `__simd_vf__ ErfCoreImpl`, which
selects `ErfPadeCompute` versus `ErfSubsectionCompute` at compile time, using
`Reg::RegTensor`, masks and vector loads/stores. Neither branch is SIMT.

## What changes with shape

In `opbase/src/op_common/atvoss/elewise/elewise_tiling.cpp`, dimensions flatten
to N=product(shape). Platform AIV count C and UB size U are queried; no tensor
shape threshold switches the computational backend to SIMT.

Source formulas (not measurements of the remote runner):

```text
cores = min(C, ceil(N * min_dtype_bits / 32768))
blockFormer = align_up(ceil(N / cores), 512)  # source aligns element count
blocks = ceil(N / blockFormer)
blockTail = N - (blocks - 1) * blockFormer
tile = align_down(floor((U - 122880) / (BufferNum * max_dtype_bytes)),
                  256 / min_dtype_bytes)
```

`BufferNum` comes from the instantiated DAG; it is not the number of visible
Python tensors. The GeLU host tiler reserves122880 bytes for every route.
Numeric tiles/blocks cannot be assigned to the historical baseline without its
platform values and matching DAG/library build. No invented numeric tiling key
is provided: CSV records the actual symbolic template arguments instead.

The scheduler uses distinct full/tail loop counts, masked vector operations,
and `i & 1` to select ping/pong buffer IDs. Its buffer-ID calculation may share
some intermediates across slots. Source structure enables pipelining, but does
not prove a particular measured overlap or speedup.

## Implications and remaining evidence boundary

This closes the shape-dependent SIMD-versus-SIMT question for the inspected
standard source route: all20 use SIMD, including prime dimensions and tails.
It also removes the prior ambiguity between Gelu and GeluV2 on Ascend950.

It does NOT establish that all20 historical `baseline_perf_us` values were
generated by these exact builds. Case notes say `Gelu×1`, which is not a
binary name/hash or a tiling-key capture. CPU golden accuracy, NPU dispatch
and stored baseline timings must not be conflated.

Importantly, the reference exact DAG itself also contains an Erf operation and
separate postprocessing. A special-function boundary alone cannot explain why
our kernel loses. Relevant comparisons are the selected Erf algorithm (versus
our FP32 Erfc), conversion stages, DAG buffer reuse, pipelining and emitted
instruction schedules. No speedup attribution is proven by this source audit.

No CANNBench credit, kernel change, commit or push was used for this analysis.
