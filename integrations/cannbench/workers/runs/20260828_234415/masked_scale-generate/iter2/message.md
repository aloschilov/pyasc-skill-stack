The harness evaluated your candidate.py on the NPU. Result:
- score 80.44/100 (compile 19.0/20, accuracy 28.5/30, perf 32.94/50)
- 19/20 cases passed, avg speedup 1.691x

Failed cases:
- level1/masked_scale_3: AI算子执行失败: compile failed! | Error message is In file included from <built-in>:1: | In file included from /home/f00816836/Ascend0615/cann-9.1.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_runtime_wrapper.h:43:

Per-case timings (elapsed_us / baseline_us / speedup):
- masked_scale_1: 5.98 / 11.495 / 1.922x
- masked_scale_2: 25.82 / 31.83 / 1.233x
- masked_scale_4: 353.01 / 817.845 / 2.317x
- masked_scale_5: 383.88 / 1353.855 / 3.527x
- masked_scale_6: 9.65 / 11.485 / 1.19x
- masked_scale_7: 6.23 / 11.59 / 1.86x
- masked_scale_8: 7.65 / 14.5 / 1.895x
- masked_scale_9: 255.67 / 588.285 / 2.301x
- masked_scale_10: 9.68 / 11.46 / 1.184x
- masked_scale_11: 6.92 / 13.24 / 1.913x
- masked_scale_12: 6.45 / 9.21 / 1.428x
- masked_scale_13: 64.61 / 87.7 / 1.357x
- masked_scale_14: 12.07 / 17.95 / 1.487x
- masked_scale_15: 9.72 / 13.07 / 1.345x
- masked_scale_16: 9.33 / 14.21 / 1.523x
- masked_scale_17: 15.76 / 13.92 / 0.883x
- masked_scale_18: 11.64 / 19.52 / 1.677x
- masked_scale_19: 11.76 / 17.62 / 1.498x
- masked_scale_20: 43.3 / 69.12 / 1.596x

Improve candidate.py accordingly and overwrite it in place. Keep the public callable name/signature identical. Remember: correctness first — a failed case costs more than any speedup gains.