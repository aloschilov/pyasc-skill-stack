# Source-driven, top-down AscTile tuning

Use this reference for performance iterations, not as a replacement operator
specification. Source review: local `cann-samples`, clean commit
`9e144a01e265a74cf93954670377f2e21ce83a99`, `Samples/2_Performance`.
[Pinned source tree](https://gitcode.com/cann/cann-samples/tree/9e144a01e265a74cf93954670377f2e21ce83a99/Samples/2_Performance).
Samples carry the CANN Open Software License Agreement v2.0; the guidance here
is a synthesis, not copied kernel implementation. Preserve licensing if later
vendoring source. No sample was built or timed for this review.

## Scope and portability

The index covers vector, reduction, layout, matmul, attention, and SIMT stories.
The actionable review concentrates on GeLU, softmax, RMSNormQuant, scalar,
SIMD/VF, data transforms, matmul and scatter/histogram. The attention, grouped
matmul and distributed communication entries are routing references, not
reproduced implementations. Read their specific contracts before using them.

- `gelu_eltwise_regbase_story` computes `exp(-0.5*(GELU(x)+1)^2+2)`, **not
  standalone GeLU**. Its generator uses the tanh/logistic approximation with
  rounded coefficient `-1.595769`, FP32, uniform[-3,3], one 8192x8192 shape,
  and absolute tolerance 1e-3. It does not qualify exact GeLU, low-precision
  routes, CANNBench tolerances, exceptional inputs, or tails.
- GeLU and softmax examples specify CANN >=9.1 and dav-3510. Softmax reported
  hardware data uses CANN9.2; RMSNormQuant labels a 64-vector-core simulation;
  data-transform/SIMD stories report CANNsim traces. Neither those times nor
  their fixed 32/64-block launches describe our pinned CANN9.0 runtime/runner.
- Keep public AscTile math/copy/types/JIT. `Reg::*`, manual mutexes, pragmas,
  `asc_vf_call`, direct VF calls and UnitFlag are **lowering comparison targets**,
  not permission to embed those APIs in kernels. An explicitly authorized
  inline Erf/Erfc exception stays limited to that family and the approved
  source gate; never broaden it to unrelated register math or disable sync.

## One operator, one route, one falsifiable change

1. Freeze the task golden/tolerances, source/runtime hashes, input bytes and
   dtype/mode. Record tensor shape, logical/physical tile shape, tile count,
   unroll, reported vector cores, applicable resource limits, launched/useful
   blocks, per-block workload, requested/effective JIT, compiler UB and model.
   Query the active device using the qualified host resource helper; do not
   derive resources from a runner name. Missing resources mean a labeled
   single-block compatibility fallback, not an optimized benchmark baseline.
2. Work top-down: workload coverage and launch geometry; kernel envelope and
   per-block tails; engine overlap/critical path; then instruction, allocation
   and lowering details. Choose one bottleneck hypothesis and one intervention.
   Keep mathematics fixed while comparing geometry or JIT. Change algorithms
   separately and requalify all affected routes.
3. Start with defaults and reuse_alloc=1 at matched geometry. Track the public
   reuse {0,1,2}, static {None,False,True}, VF {False,True} matrix, but prioritize
   cells relevant to the observed bottleneck. Deduplicate only after proving
   equivalent lowering. Keep insert_sync=True, opt_level=3; fresh cache/process,
   verify_sync/forced compile diagnostics are separate from warmed timing.
4. Reject measured overflows; unknown UB is not safe. Compile all dispatches,
   then execute changed routes, boundary values, tails, odd tile counts, idle
   blocks, canaries and repeats. A source simplification is not a numerical proof.
5. Compare repeated A/B in the same model/device configuration; record noise
   and failures. For the GeLU campaign require reproducible >=5% local gain to
   promote an optimization, separately allowing an authorized portability
   control. Then hardware validates the candidate; model ticks are not NPU us.
   Do not run new hardware jobs or restart schedules without task authorization.

## Read the metrics without overclaiming

- Engine busy ratios can overlap. Do not add vec/scalar/MTE ratios to obtain
  total time, or report `1-sum(ratios)` as idle. Validate each counter's scope
  (one core vs aggregate, sampled vs all events) and denominator. Ratio sums
  near 100% do not prove serialization; use timestamped intervals on the same
  block/clock domain. Union/intersection of those intervals establishes observed
  occupancy/overlap only for that captured scope.
- Separate full-kernel envelope, VF duration, queue residence (`PUSHQ`), issue
  counts, and accumulated instruction cycles. Counted cycles are not wall time.
  Report absolute engine time alongside ratios: a ratio can grow simply because
  another engine became faster. Queue time may include waiting, not execution.
- A high busy ratio does not establish the critical path. Check when each block
  finishes, dependencies, and whether shortening that engine shortens the whole
  envelope. Small grid, tail imbalance and launch overhead need different fixes.
  Device-only simulator traces cannot measure host launch overhead.
- Record useful GM bytes separately from padded/actual transfers; derive any
  effective bandwidth using a matching scope/time. UB allocation is capacity
  consumption, not UB bandwidth, bank-conflict rate, occupancy or utilization.
  Assembly can reveal spills or instruction patterns; missing stall counters
  mean `unknown`, not a proven conflict or zero conflicts.

## Techniques to test, and when to stop

### GeLU / elementwise: transfers versus dependency depth

Read `gelu_eltwise_regbase_story/src/0_membase.asc` through
`4_scalar_tuning.asc`. The progression is VF fusion, split dependent loops,
VF-level unrolling, then hoisted constants. Apply through public expressions
and supported JIT settings, verifying emitted VF groups and UB loads/stores.

- VF fusion may remove intermediate UB round trips, but a long dependency chain
  can reduce instruction-level parallelism. The sample's loop split adds an
  intermediate UB store/load to expose shorter chains. Do not mandate either
  maximal fusion or minimal transfer count. For standalone exact GeLU with an
  opaque Erf/Erfc call, this is a hypothesis, not proof the same split helps.
- Distinguish host/block tiling, AscTile tile-loop unrolling, and compiler VF
  vector-loop unrolling. Sample `#pragma unroll 6` is not evidence to set AscTile
  unroll=6. Inspect remainder bodies, register pressure/spills, code size and
  live allocations. Compare guarded fixed groups against dynamic remainder
  lowering only where safe at arbitrary tile counts.
- Hoist host constants and invariant scalar arithmetic first. Hoisting a tile
  can extend UB/register liveness and expose reuse/alias hazards. Public syntax
  does not guarantee register-resident `Duplicate`: inspect generated code.
  Keep FP32 intermediates and verified exact/tanh formulas unchanged initially.

### Softmax / reductions / norms: independent rows, reuse, real overlap

Read `softmax_regbase_story/src/{2_binary_fold,3_multi_row,4_pipeline,
5_bigtile,6_merged_vf,7_merged_vf_direct}.asc` and
`rms_norm_quant_story/Story.md`, `src/1_preload_gamma.asc`,
`src/4_double_buffer.asc`, `src/5_ub_utilization.asc`, `src/6_binary_sum.asc`.

- Batch contiguous independent rows to amortize transfers/VF calls, while
  retaining enough blocks and iterations. Compare independent-row scheduling
  with within-row partial sums; stacking both can raise register pressure.
  A wider tile may reduce DMA setup but eliminate useful double buffering.
- Reuse gamma/shared tiles across rows only when the avoided loads pay for
  longer lifetimes. Budget fixed resident data **plus** per-row data, casts,
  scratch, alignment and actual buffering. The sample's ~49KB gamma resident
  footprint/large UB assumption is not our runtime's allocation budget.
- `unroll_factor=2` does not prove ping-pong overlap. The softmax story records
  unsuccessful prefetch/merged-VF experiments: inspect actual MTE2/V schedules
  and blocking calls. It attributes a direct-VF improvement to its own SDK;
  reproduce at our pin before blaming AscTile or asking for compiler changes.
  Do not remove required barriers or substitute direct VF code.
- Pairwise sums/rsqrt refinement are separate accuracy experiments. Reduction
  order, quantization rounding and epsilon placement affect output. The norm
  sample is RMSNorm **plus int8 quantization**, not the CANNBench RMSNorm contract.
  Preserve the real reduction axis, neutral padding, and dtype conversion order.

### Scalar, broadcast, casts and bank conflicts

Read `scalar_story/README.md`, `simd_vf_story/README.md` and
`vf_data_transform_story/vf_data_transform_tutorials/`.

- Distinguish instruction-fetch misses, scalar dependency/spill stalls, and
  back-pressure from a full downstream queue. Fewer scalar instructions alone
  need not help. Test static_alloc through public JIT, shorter live ranges,
  constant metadata and simpler address arithmetic. Do not insert manual I-cache
  prefetch or force static UB offsets; inspect the compiler output instead.
- For broadcast, tile multiple rows and keep broadcast logical (no host/device
  materialized expansion). Check VF launch count, PB setup and address arithmetic
  in emitted traces; don't mistake more VF launches for more useful parallelism.
- Cast/layout examples optimize register distribution, packing and bank access.
  AscTile candidates may remove redundant casts or change physical layout using
  public operations, preserving rounding, saturation and logical indexing. Packed
  FP4/FP8 transformations are not generic numerical casts. Unsupported RegLayout
  or padding controls are compiler/API gaps, not licenses for hidden helpers.

### Matmul, attention and irregular operators

- `matmul_story/docs/matmul_performance.md`: L1 operand reuse, M/N traversal
  for L2 locality (SWAT), tail balancing, and split-K. Express supported changes
  with high-level tiled loops/copy/matmul. Measure extra workspace, reduction,
  padding and synchronization costs; split-K is not free parallelism. UnitFlag
  and L1 bank placement require supported lowering evidence, not inline patches.
- `grouped_matmul_story`, attention and KV/RoPE stories guide fusion boundaries,
  persistent operands, mixed Cube/Vector dependencies and workload scheduling.
  Their specialized packed layouts/state contracts do not apply automatically
  to standalone operators. Account for preparation and all launched kernels.
- `simt_scatter_story` and `simt_histogram_story`: data distribution and collision
  semantics matter. Last-writer-wins, sum and max are different contracts.
  Sorting/grouping/preprocessing must be implemented and timed if needed, never
  assumed because sample inputs are pregrouped. SIMT launch_bounds/float4/direct
  GM are not public AscTile tuning flags; retain them as backend comparisons.

## Reusable profiling harness patterns

The data-transform tutorials provide `scripts/profile_tutorial.sh` and
`scripts/summarize_msopprof.py`. Reuse the ideas of round-robin candidate
measurement, explicit warm-up, per-run raw files, schema-checked parsing and
reporting both individual times and their median. Their parser expects exactly
one `OpBasicInfo.csv` row with `Task Duration(us)`; do not apply it blindly to
multi-kernel CANNBench or different profiler versions. Reject nonfinite/zero
times and associate rows with the actual kernel and launch being compared.

The script uses `npusim` (falling back to `cannsim`) `record ... -g vf -n 0`
for a scoped VF trace, and `msopprof --warm-up=5 --launch-count=1
--launch-skip-before-match=1 --aic-metrics=BasicInfo --replay-mode=kernel`
for sample hardware runs. These are version-specific sample invocations, not
aliases for the project's `npu-sim --gen-report`. Check the installed tool's
help, capture scope, selected device and raw output schema before adapting.
`--case trace` is a reduced diagnostic workload, not the full/perf input.
Profiler warm-up/replay may alter caches and launch count; match those between
variants and keep correctness checks outside the timed region.

## Next GeLU iteration, after existing F/E hardware verification

Keep the frozen F/E candidates unchanged. Start with the FP32 exact case15
matched pair: fixed source math, tile4160, unroll2, reuse0, VF on, 64 requested
logical blocks only if resource-qualified. Test one of: reduced scalar setup,
changed full-tile/tail grouping, or a neighboring tile size with measured UB.
Choose using actual traces, not UB fill. Classify the remaining Erf/Erfc call
boundary and first surviving transfer in IR before proposing a VF split.
Then check small/large exact cases and all low-precision/tanh routes for
regression. Existing local F gains do not predict >1x on hardware.

Record each experiment as: hypothesis; source/pin/input/model hashes; baseline;
single changed variable; expected trace signature; measured UB and correctness
scope; repeated envelope and counter evidence; decision/rejected alternatives;
unknowns; next test. A missing public control gets a minimal high-level
reproducer and before/after lowering artifacts, not a compiler monkeypatch.
