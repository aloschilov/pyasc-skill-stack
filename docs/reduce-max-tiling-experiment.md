# ReduceMax tiling-selection experiment

Experiment: encode last-axis reduction tiling-selection heuristics into the skill
stack (see [`skills/pyasc-api-patterns/references/reduction-tiling.md`](../skills/pyasc-api-patterns/references/reduction-tiling.md)
and the reductions section of [`skills/pyasc-target-operator/SKILL.md`](../skills/pyasc-target-operator/SKILL.md)),
then drive a **single short prompt** (skills supply the tiling detail) to a
performant `reduce_max` kernel. Target: the 11-case table in the pyasc-fork
`test_reduce_max.py`, for direct comparison with pyasc PR #353.

## Problem

The pyasc-fork `test_reduce_max.py` pins degenerate host-side tiling: for all 11
cases `tiling_values[2:4]` (`tile_rows, tile_cols`) align to `[8, 8]` = 64 fp32
elements = 256 bytes, versus a ~64 KB usable UB budget — about **0.4% UB
utilization**. This is the "host-side tiling implemented wrong" symptom behind
the low CANN ratio. The kernel itself already supports an arbitrary
`[tile_rows, tile_cols]`; the lever is purely the host-side tile selection.

## The short prompt (experiment artifact)

```
Develop a high-performance last-axis reduce_max kernel for Ascend950PR_9599
(C310), float32, covering the 11 shapes in test_reduce_max.py (flatten each to
[rows, C] and reduce axis 1 to [rows]). The current host-side tiling pins a
degenerate [8, 8] tile. Choose the tiling to maximize UB utilization while
preserving double buffering. Verify all shapes on the Model backend against
torch.amax(x, dim=1).
```

Everything about *how* to pick the tile (UB-utilization quality metric,
double-buffer "2*(1+) iterations" rule, small-C row-packing vs tiny-C transpose
vs large-C column tiling, reshape short-circuit) is deferred to the skill.

## Tiling selection (from the skill)

Flatten to `[R, C]`. With `UB_BUDGET_BYTES = 64*1024`, `BUFFER_NUM = 2`, fp32:
`per_buffer = (UB_BUDGET_BYTES/4)/2 = 8192` elems. For the dominant small-C
regime: `tile_cols = align(C)` (whole row), `tile_rows = per_buffer/tile_cols`,
then capped so the double-buffered row loop keeps `>= 2*unroll_factor` iterations.

## Correctness result (Model / camodel simulator)

All 11 shapes pass `torch.amax` (`atol=rtol=1e-3`) on the `Ascend950PR_9599`
Model backend, with UB utilization lifted from ~0.4% to ~50% on every
substantial case while double buffering is preserved (row-iters/block >= 4):

| input_shape       |  C  | core | selected tile | row-iters/blk | UB util |
|-------------------|-----|------|---------------|---------------|---------|
| [200, 10]         |  10 |  8   | [8, 16]       | 4             |  0.8%   |
| [13, 2048, 32]    |  32 |  4   | [256, 32]     | 26            | 50.0%   |
| [10, 2048, 64]    |  64 |  4   | [128, 64]     | 40            | 50.0%   |
| [45, 2048, 4]     |  4  |  4   | [1024, 8]     | 23            | 50.0%   |
| [64, 2048, 8]     |  8  |  4   | [1024, 8]     | 32            | 50.0%   |
| [70, 2048, 16]    |  16 |  4   | [512, 16]     | 70            | 50.0%   |
| [2048, 83, 18]    |  18 |  8   | [336, 24]     | 64            | 49.2%   |
| [1500, 1, 61]     |  61 |  4   | [88, 64]      | 5             | 34.4%   |
| [3072, 113, 24]   |  24 |  8   | [336, 24]     | 130           | 49.2%   |
| [4608, 115, 12]   |  12 |  8   | [512, 16]     | 130           | 50.0%   |
| [1500, 61, 61]    |  61 |  8   | [128, 64]     | 90            | 50.0%   |

`[200, 10]` stays small because the whole problem is only 200 rows — there is not
enough data to fill UB without dropping below the double-buffer iteration floor.

Verification was run in `ghcr.io/aloschilov/pyasc-sim:py3.11` (the same camodel
image the evidence harness uses) with
`LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib` and
`PYASC_DUMP_PATH` set, since the local host has no CANN install.

## NPU performance A/B (real hardware)

The `gcpty` tunnel box (`f38723bfa99d`, checkout `/home/l00958488/pyasc-fork1`) is a
**real NPU**, not a camodel: real device nodes `/dev/davinci0`,
`/dev/davinci_manager`, `/dev/hisi_hdc`; real driver
`/usr/local/Ascend/driver/lib64/driver/libascend_hal.so`; CANN 8.5.0
(`ASCEND_HOME_PATH=/usr/local/Ascend/latest`). A runtime `plog` showed real HBM
alloc/free on `dev0`, so `--backend NPU --profile --runs N` measures real-hardware
microseconds here (via the profiler fixture's `task_time_median`).

### Result (`--backend NPU --profile --runs 10`, median µs)

Baseline degenerate `[8, 8]` tiling vs the UB-selected tiling, same harness, only
the tiling changed. All 11 shapes pass `torch.amax` (`atol=rtol=1e-3`):

| input_shape       | baseline µs | selected tile | selected µs | speedup |
|-------------------|-------------|---------------|-------------|---------|
| [200, 10]         | 4.11        | (8, 16)       | 2.18        | 1.9x    |
| [13, 2048, 32]    | 974.24      | (256, 32)     | 12.51       | 78x     |
| [10, 2048, 64]    | 1334.54     | (128, 64)     | 17.82       | 75x     |
| [45, 2048, 4]     | 475.08      | (1024, 8)     | 11.18       | 42x     |
| [64, 2048, 8]     | 675.63      | (1024, 8)     | 14.75       | 46x     |
| [70, 2048, 16]    | 3213.54     | (512, 16)     | 37.18       | 86x     |
| [2048, 83, 18]    | 2554.31     | (336, 24)     | 30.15       | 85x     |
| [1500, 1, 61]     | 98.84       | (120, 64)     | 2.94        | 34x     |
| [3072, 113, 24]   | 5194.92     | (336, 24)     | 58.90       | 88x     |
| [4608, 115, 12]   | 5939.14     | (512, 16)     | 69.50       | 85x     |
| [1500, 61, 61]    | 3053.38     | (128, 64)     | 38.75       | 79x     |

The degenerate `[8, 8]` tile (~0.4% UB) is ~75–88x slower on every substantial shape.
`[200, 10]` only improves ~1.9x because 200 rows cannot fill UB without dropping below
the double-buffer iteration floor. Shipped as gitcode MR
[#391](https://gitcode.com/compiler-team/pyasc/merge_requests/391) (follow-up to #353).

Staged on the remote (compile-clean), ready for the A/B — no branch switch needed:

- `python/test/asc2/target/test_reduce_max_base.py` — baseline degenerate `[8, 8]`
  tiling (materialized from the local `reducemax-target` branch).
- `python/test/asc2/target/test_reduce_max_sel.py` — the UB-optimized
  `_select_reduce_tile` version.

NPU environment (required — the runtime needs the driver HAL on `LD_LIBRARY_PATH`):

```bash
cd /home/l00958488/pyasc-fork1
source /usr/local/Ascend/latest/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
```

Run (smoke one shape, then full A/B — the "Profiling results" section prints per-shape μs):

```bash
python3 -m pytest -q -rs python/test/asc2/target/test_reduce_max_base.py -k 200 \
  --backend NPU --platform Ascend910_9599 --profile --runs 10
python3 -m pytest -q -rs python/test/asc2/target/test_reduce_max_base.py \
  --backend NPU --platform Ascend910_9599 --profile --runs 10 | tee /tmp/rmax_base.log
python3 -m pytest -q -rs python/test/asc2/target/test_reduce_max_sel.py \
  --backend NPU --platform Ascend910_9599 --profile --runs 10 | tee /tmp/rmax_sel.log
```

**Device-wedge caveat.** During this session the device wedged:
`aclrtSetDevice(0)` hung 60–190 s then returned `507033` (even for a bare
`set_device(0)`), with `dmesg` spamming
`[ascend][apm] apm_fops_query_slave_status … Query status failed (ret=-3)`. No
holding process was visible inside the container and there is no `npu-smi` /
`ascend-dmi` to reset it, so recovery is host-side: reset the device
(`npu-smi -r -i 0`) or relaunch the container with a healthy `davinci0`. This is
the "a `timeout`-killed NPU process wedges the device; subsequent runs all err"
hazard — avoid killing an in-flight NPU test mid-`SetDevice`.

## Update: three-lever parity tiling (v2)

The UB-selector above (branch `reducemax-target-tiling`, MR #391) reached a
**geomean of 0.426 of CANN** across the eight #353-referenced shapes — a huge win
over the degenerate `[8, 8]` (0.011), but still ~2.3x off CANN. A later TTK
near-parity export on MR #391 exposed three tiling levers the UB-selector still
missed. Encoding them in the [reduction-tiling
hint](../skills/pyasc-api-patterns/references/reduction-tiling.md) and
regenerating the kernel via opencode lifts the geomean to **~1.04 of CANN
(parity)**.

### The three levers (the actual experiment signal)

1. **Use every AI core.** The UB-selector pinned `core_num=4/8`; parity spreads
   `R` across all **72** cores (`rows_per_block = ceildiv(R, core_num)`). This is
   the dominant lever — a 9-18x deficit on the big-`R` shapes.
2. **Keep the reduce axis contiguous and unpadded (`tile_cols = C`).** The
   UB-selector aligned `tile_cols` up to 32 B (`C=4 -> 8` doubles DMA on a
   memory-bound op). v2 uses `tile_cols = C` where asc2's 2-D `copy_in` 32-byte
   last-dim rule allows (`C` a multiple of 8); the 4/10/12/18/61 widths still pad
   the axis up to 32 B (a fully unpadded `C` needs a flat 1-D load + in-UB view —
   future work).
3. **Size `tile_rows` to the per-core block against physical UB.** v2 budgets
   against ~192 KB and, critically, for **4 live buffers** (2 copy_in ping-pong x
   `unroll_factor=2`) — a naive `/2` budget overflowed UB on `[3072,113,24]`
   (375584 B used vs 253952 B available).

Opencode reproduced this signal directly: with the *first* hint version it copied
the `reduce_sum` teaching pattern (one row per iteration, `core_num=16`) and only
adopted lever 2; after the hint was strengthened with an explicit "one row per
iteration is wrong for large R" anti-pattern and a core-count clarification, the
regenerated kernel independently arrived at all three levers.

### Result (`--backend NPU --profile --runs 10`, median µs, `Ascend950PR_9599`)

`ratio = CANN CST µs / pyasc µs` (fraction of CANN; `1.0` = parity):

| input_shape     | CANN CST µs | UB-sel µs (ratio) | v2 tile tr×tc | v2 µs (ratio) |
|-----------------|-------------|-------------------|---------------|---------------|
| [200, 10]       | 2.010       | 2.18 (0.922)      | 3×16          | 3.084 (0.652) |
| [1500, 1, 61]   | 2.357       | 2.94 (0.802)      | 21×64         | 2.127 (1.108) |
| [45, 2048, 4]   | 3.971       | 11.18 (0.355)     | 1280×8        | 3.529 (1.125) |
| [13, 2048, 32]  | 4.430       | 12.51 (0.354)     | 370×32        | 3.990 (1.110) |
| [64, 2048, 8]   | 4.614       | 14.75 (0.313)     | 911×8         | 4.386 (1.052) |
| [10, 2048, 64]  | 5.853       | 17.82 (0.328)     | 143×64        | 5.037 (1.162) |
| [70, 2048, 16]  | 8.133       | 37.18 (0.219)     | 664×16        | 8.561 (0.950) |
| [2048, 83, 18]  | 15.656      | 30.15 (0.519)     | 473×24        | 11.912 (1.314)|
| **geomean (8)** |             | **0.426**         |               | **1.041**     |

Large-bucket shapes (no CANN CST; static CST compile fails on the `axes`
const-input): `[3072,113,24]` 58.90→22.30 µs, `[4608,115,12]` 69.50→23.87 µs,
`[1500,61,61]` 38.75→16.16 µs (all ~2.4-2.9x over the UB-selector). All 11 shapes
pass `torch.amax` (`atol=rtol=1e-3`).

The only non-improvement is the tiny `[200, 10]` (3.08 vs 2.18 µs): 200 rows over
72 cores is launch-overhead bound, where fewer cores were better — a candidate for
a small-`R` core-count backoff.

This run used the recreated `gcpty` box (`c50f972f7149`, `/dev/davinci0`); the
tiling change is Python-only, so no C++ rebuild was needed (the remote
`libpyasc*.so` was reused). Posted to MR #391; branch
`reducemax-target-tiling-v2`.

## Follow-ups

- v2 (three-lever) tiling reaches CANN parity (geomean ~1.04) and supersedes the
  UB-selector (0.426) and the degenerate `[8, 8]` (0.011).
- Remaining: a flat 1-D contiguous load + in-UB view to make `tile_cols = C` legal
  for unaligned `C` (4/10/12/18/61), and a small-`R` core-count backoff for tiny
  shapes like `[200, 10]`.
- Branches: `reducemax-target-tiling-v2` (v2, parity) stacked on
  `reducemax-target-tiling` (UB-selector, MR
  [#391](https://gitcode.com/compiler-team/pyasc/merge_requests/391)) / `#353`.
