The harness evaluated your candidate.py on the NPU. Result:
- score 58.0/100 (compile 20.0/20, accuracy 28.5/30, perf 9.5/50)
- 19/20 cases passed, avg speedup 0.421x

Failed cases:
- level1/swi_glu_12: 精度不达标:   - bfloat16[output]: ❌ MERE=0.000000, MARE=0.000000 (threshold=7.812500e-03, mare_threshold=7.812500e-02), NaN位置不匹配

Per-case timings (elapsed_us / baseline_us / speedup):
- swi_glu_1: 14.11 / 6.75 / 0.478x
- swi_glu_2: 42.79 / 27.47 / 0.642x
- swi_glu_3: 167.24 / 57.455 / 0.344x
- swi_glu_4: 673.53 / 294.38 / 0.437x
- swi_glu_5: 854.42 / 620.68 / 0.726x
- swi_glu_6: 14.54 / 6.3 / 0.433x
- swi_glu_7: 19.25 / 6.985 / 0.363x
- swi_glu_8: 15.18 / 9.55 / 0.629x
- swi_glu_9: 1462.1 / 21.21 / 0.015x
- swi_glu_10: 30.23 / 6.725 / 0.222x
- swi_glu_11: 7.95 / 3.83 / 0.482x
- swi_glu_13: 5.58 / 3.52 / 0.631x
- swi_glu_14: 48.62 / 9.06 / 0.186x
- swi_glu_15: 14.11 / 8.73 / 0.619x
- swi_glu_16: 25.41 / 9.47 / 0.373x
- swi_glu_17: 24.47 / 9.72 / 0.397x
- swi_glu_18: 42.81 / 27.47 / 0.642x
- swi_glu_19: 47.8 / 16.15 / 0.338x
- swi_glu_20: 599.24 / 21.225 / 0.035x

Improve candidate.py accordingly and overwrite it in place. Keep the public callable name/signature identical. Remember: correctness first — a failed case costs more than any speedup gains.