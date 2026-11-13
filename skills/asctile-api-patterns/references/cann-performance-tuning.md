# Source-driven, top-down AscTile tuning

Use this reference for performance iterations, not as a replacement operator
specification. Source review: local `cann-samples`, clean commit
`9e144a01e265a74cf93954670377f2e21ce83a99`, `Samples/2_Performance`.
[Pinned source tree](https://gitcode.com/cann/cann-samples/tree/9e144a01e265a74cf93954670377f2e21ce83a99/Samples/2_Performance).
Samples carry the CANN Open Software License Agreement v2.0; the guidance here
is a synthesis, not copied kernel implementation. Preserve licensing if later
vendoring source. No sample was built or timed for this review.

## Worked example: Maximum on three vector cores

Start with the retained FP16 Maximum 01 candidate (source SHA256 begins
`a517bbb8`). It launches 32 blocks, walks tiles of 1,024 elements in a
grid stride, pads input loads and stores only the valid extent. Its original
CaModel qualification is in the [breadth pilot](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-cannbench-breadth-49).
For this experiment set the launch count to one, then three. Keep the
math, FP16 storage, tile and comparator fixed. The compiler target is
`Ascend950PR_9599`, PyASC `d0f64a63`, CANN 9.2.0. The Model loads dav_3510;
its driver stub reports `Ascend950PR_9589`.

**Distribute the work.** With `T = ceil(numel/tile)`, tile `q` belongs to
block `q % blocks`. Its valid extent is `min(tile, numel - q*tile)`.
For case 01 (1,024 × 1,024), one block executes all 1,024 tiles. Three
blocks execute 342, 341 and 341. The last tile is full and belongs to block 0.
For case 07 (1,023 × 1,023), the same tile gives 341 tiles to each block.
Block 2 owns the last tile with **one valid element**. A full-tile store
there exceeds the logical output.

**Inspect the compiled specialization.** Set `PYASC_DRY_RUN=1` and use a
fresh private cache per compiler comparison. Read `kernel.meta.memory_consumed["UB"]`
and the byte offsets of `LocalTensor<half>` in dumped `ascendc.cpp`. For
this pure-vector kernel the compiler emitted the following layouts.

| Blocks | Tile elements | Reuse | Unroll | UB bytes | Emitted starting byte offsets |
|---:|---:|---:|---:|---:|---|
| 1 | 1,024 | 0 | 1 | 6,144 | 0, 2,048, 4,096 |
| 3 | 1,024 | 0 | 1 | 6,144 | 0, 2,048, 4,096 |
| 3 | 1,024 | 1 | 1 | 4,096 | 0, 2,048 |
| 3 | 1,024 | 2 | 1 | 4,096 | 0, 2,048 |
| 3 | 1,024 | 0 | 2 | 18,432 | Nine slots, 2,048 bytes apart |
| 3 | 1,536 | 1 | 1 | 6,144 | 0, 3,072 |

Reusing a dead input slot saved 2,048 bytes. Try spending those bytes on
two 1,536-element buffers: the compiled specialization fits in the original
6,144 bytes. At that tile case 01 has 683 tiles and its last 1,024 elements
belong to block 1. Case 07 has 682 tiles and its last 513 elements belong
to block 0. Unrolling by two tripled reported UB at reuse 0; inspect live
buffers before combining unrolling with a larger tile. Reuse 1 places the second input and result at offset 0; reuse 2 places them
at offset 2,048. Their equal peak UB hides a different assignment.

The allocation trade-off shows up in execution too. At tile 1,024 on three
blocks, seed 23 took 142,295 ticks (case 01) and 141,996 (07) with reuse 0
and unroll 2. Reuse 2 with unroll 1 took 403,904 and 402,789, respectively.
These are single synchronized observations after seed 19; the two variants
change different controls relative to reuse 0/unroll 1. Use repeated
measurements at fixed geometry to choose between close alternatives.

**Check the decision.** Unset `PYASC_DRY_RUN`. Check seeds 19 and 23
against the official comparator, then suppress launch in a separate process
and require both comparisons to fail. For timing use the unchanged candidate, a checked warm-up and repeated
synchronized Model tick deltas.
Compare 1,024 and 1,536 with three blocks, reuse 1 and unroll 1. Changing
reuse simultaneously would confound the tile comparison.

All six variants passed both official seeds on both shapes. Each suppressed
launch control rejected both seeds. For the matched retile comparison, seed
19 warmed the specialization and also passed the official comparator. Seed
23 ran twice on the same inputs; the outputs agreed and passed the comparator.
Both tiles for a given shape used the same host and pinned Model image.

| Case | Tile | Warmed ticks, seed 23, repeat 1 | Repeat 2 |
|---|---:|---:|---:|
| 01 | 1,024 | 403,905 | 403,834 |
| 01 | 1,536 | 274,860 | 274,521 |
| 07 | 1,024 | 402,790 | 402,419 |
| 07 | 1,536 | 274,682 | 274,317 |

The 1,536-element tile used fewer ticks in every matched observation at
reuse 1. Prefer it for this reuse-1 specialization.
At tile 1,024 the reuse-0 baseline also used fewer ticks on seed 23
than the larger reuse-1 tile. The compiler’s allocation saving alone
therefore gives no reason to switch away from the baseline.

One-core repeated timing reached its 5,400-second wall limit on cases 01 and 07.
The independent two-seed numerical runs completed. For seed 23, the warmed
single-core launches took 659,400 ticks (01) and 658,226 (07).
With three cores, tile 1,024, reuse 0 and unroll 1, the seed-23 medians
were 221,759 and 221,437 ticks, respectively.
Each one-core figure is a single observation. The archive preserves the
interrupted repeats.
At reuse 1 the initial eight-launch timing run also exceeded the limit
for tile 1,024 on both shapes. The shorter matched comparison above covers all four cells.
These tick counts describe this simulator. Device latency remains unmeasured.

**Reproduce.** In the [qualification archive](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-cannbench-transfer-52)
use the worked-example directory. Pull the ARM64 image pinned in the manifest
before going offline. Run `python3 restore-catalog.py /path/to/cann-bench`
to export catalog revision `08d519c5`, then run `python3 variants.py`. Compile
with `python3 probe-host.py dry maximum/1 variants/c3-r1-u1-t1536.py my-dry`.
The dry probe uses the 1,024 × 1,024 input. Both shapes share this compiled
specialization because the element count is a runtime argument.
Use `python3 decision-probe/probe-host.py timing maximum/7 variants/c3-r1-u1-t1536.py my-time-1536`
for the matched timing. Repeat at tile 1,024 and for case 01 in distinct directories.
Run `probe-host.py suppressed` for each variant in a new output directory.
The probe records source hashes, compiler layout, launch counts and both seed
verdicts. For other operators, first identify which property changes the
tile budget: dtype conversion, spatial footprint, accumulated outputs or
indexing may dominate. The Maximum tile size applies only to this example.
[Control details and runtime pins](api-jit-options.md#controls-used-in-the-maximum-example)

The GeLU geometry experiment on PyASC `9069108`, Ascend950PR_957c, CANN 9.1,
and the legacy `asc2.range(parallel=True, unroll_factor=2)` wide-reduction
regression on Ascend910B1 motivate checking block counts and unroll separately.
Their timings belong to their original operators and targets.
[GeLU measurements](https://github.com/aloschilov/pyasc-skill-stack/issues/13) ·
[Legacy reduction measurements](https://github.com/aloschilov/pyasc-skill-stack/issues/1)

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
  their fixed 32/64-block launches qualify the historical CANN 9.0 run or this CANN 9.2 Model.
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

## Simulator timing protocol

- Keep simulator ticks, simulator host wall time and measured NPU microseconds
  as separate metrics. Do not convert ticks to hardware latency using an assumed
  clock rate or treat JIT/import wall time as kernel time.
- For a per-launch comparison, run one kernel/shape per fresh process, or use
  per-launch trace boundaries. A process-wide cumulative counter cannot identify
  individual launch costs without those boundaries.
- Hold runtime/image digest, platform, inputs, dtype, all geometry except the control under test, and counter scope
  fixed across baseline and candidate. Record warm-up/cache policy and repeated
  samples; retain failures and report dispersion rather than selecting the best.
- Keep numerical verification separate from the timed interval. Compare complete
  logical outputs under the task tolerances before accepting a timing result.
- Record raw counters and exact collection commands outside checkout. API/JIT
  compatibility belongs to the pinned API references, not to this timing protocol.

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
