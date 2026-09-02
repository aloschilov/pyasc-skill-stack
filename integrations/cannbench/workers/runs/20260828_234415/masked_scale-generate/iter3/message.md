The harness evaluated your candidate.py on the NPU. Result:
- score 80.35/100 (compile 19.0/20, accuracy 28.5/30, perf 32.85/50)
- 19/20 cases passed, avg speedup 1.691x

Failed cases:
- level1/masked_scale_3: AI算子执行失败: compile failed! | Error message is In file included from <built-in>:1: | In file included from /home/f00816836/Ascend0615/cann-9.1.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_runtime_wrapper.h:43:

Per-case timings (elapsed_us / baseline_us / speedup):
- masked_scale_1: 5.91 / 11.495 / 1.945x
- masked_scale_2: 26.23 / 31.83 / 1.213x
- masked_scale_4: 353.64 / 817.845 / 2.313x
- masked_scale_5: 382.25 / 1353.855 / 3.542x
- masked_scale_6: 10.51 / 11.485 / 1.093x
- masked_scale_7: 6.02 / 11.59 / 1.925x
- masked_scale_8: 7.39 / 14.5 / 1.962x
- masked_scale_9: 253.63 / 588.285 / 2.319x
- masked_scale_10: 9.89 / 11.46 / 1.159x
- masked_scale_11: 6.75 / 13.24 / 1.961x
- masked_scale_12: 6.36 / 9.21 / 1.448x
- masked_scale_13: 65.47 / 87.7 / 1.34x
- masked_scale_14: 12.47 / 17.95 / 1.439x
- masked_scale_15: 9.86 / 13.07 / 1.326x
- masked_scale_16: 9.61 / 14.21 / 1.479x
- masked_scale_17: 15.41 / 13.92 / 0.903x
- masked_scale_18: 11.36 / 19.52 / 1.718x
- masked_scale_19: 11.74 / 17.62 / 1.501x
- masked_scale_20: 44.6 / 69.12 / 1.55x

Improve candidate.py accordingly and overwrite it in place. Keep the public callable name/signature identical. Remember: correctness first — a failed case costs more than any speedup gains.