You are optimizing the PERFORMANCE of a working CANN Bench operator kernel (pyasc asctile, Ascend 950PR NPU, 72 AIV cores). You have NO NPU access — the official harness measures your file on real hardware after you finish.

Correctness is currently 100% ({{N_CASES}}/{{N_CASES}} cases pass) and MUST stay 100% — a single failed case loses more score than any speedup can win back. Only kernel speed can improve the score now: the performance sub-score (0-50) grows with measured kernel time relative to the per-case `baseline_us` (aclnn reference) and saturates near the analytical hardware limit `t_hw_us`. Speedup = baseline_us / elapsed_us; current average is {{AVG_SPEEDUP}}x, leaders on this hardware reach 1.0-3.4x.

# Current module — operator {{OP_NAME}} ({{SCORE_SUMMARY}})

```python
{{CURRENT_SOURCE}}
```

# Measured per-case kernel timings (torch_npu profiler, kernel time only)

{{TIMINGS_TABLE}}

# Known optimization levers for this hardware (priority order, all previously measured)

1. **Wider tiles** amortize per-tile DMA/loop setup: TILE=2048 measured ~0.9x
   of hand-written AscendC for a simple elementwise op, vs ~0.2x at TILE=128.
   Constraint: UB budget — roughly `num_f32_temporaries * 4 * TILE * 2 <= 250000`
   bytes (the trailing *2 is unroll double-buffering).
2. **Fewer tile temporaries** (algebraic simplification / fusing constants)
   directly buys a wider tile. This is the main lever for long op chains.
3. **Small-case parallelism**: `cores = min(72, num_tiles)` means small shapes
   underuse the 72 cores when TILE is wide. Consider selecting between two
   compiled tile variants at the host by size (e.g. wide tile when
   `numel >= 72 * WIDE_TILE`, narrow otherwise). Look at the per-case table:
   cases with small `numel` and low speedup are parallelism-starved.
4. `unroll_factor=2` on the grid-stride loop is already present — keep it.
{{EXTRA_LEVERS}}

{{CONSTRAINTS}}

# Deliverable

Write the improved COMPLETE module to `candidate.py` in the current working directory. Keep the public callable name and signature EXACTLY `{{CALLABLE}}`. The numerically-stable forms in the current source exist because naive forms failed the harness — keep the math cancellation-free. Do not change behavior, only speed.

IMPORTANT — no local execution: pyasc/asc/asctile/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
