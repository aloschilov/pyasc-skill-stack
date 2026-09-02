The harness evaluated your candidate.py on the NPU. Result:
- score 79.13/100 (compile 19.0/20, accuracy 28.5/30, perf 31.63/50)
- 19/20 cases passed, avg speedup 1.583x

Failed cases:
- level1/masked_scale_3: AI算子执行失败: compile failed! | Error message is In file included from <built-in>:1: | In file included from /home/f00816836/Ascend0615/cann-9.1.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_runtime_wrapper.h:43:

Per-case timings (elapsed_us / baseline_us / speedup):
- masked_scale_1: 6.65 / 11.495 / 1.729x
- masked_scale_2: 27.91 / 31.83 / 1.14x
- masked_scale_4: 377.75 / 817.845 / 2.165x
- masked_scale_5: 409.7 / 1353.855 / 3.305x
- masked_scale_6: 10.17 / 11.485 / 1.129x
- masked_scale_7: 6.62 / 11.59 / 1.751x
- masked_scale_8: 8.06 / 14.5 / 1.799x
- masked_scale_9: 275.06 / 588.285 / 2.139x
- masked_scale_10: 10.27 / 11.46 / 1.116x
- masked_scale_11: 7.25 / 13.24 / 1.826x
- masked_scale_12: 6.67 / 9.21 / 1.381x
- masked_scale_13: 66.47 / 87.7 / 1.319x
- masked_scale_14: 14.62 / 17.95 / 1.228x
- masked_scale_15: 10.05 / 13.07 / 1.3x
- masked_scale_16: 10.85 / 14.21 / 1.31x
- masked_scale_17: 16.97 / 13.92 / 0.82x
- masked_scale_18: 12.01 / 19.52 / 1.625x
- masked_scale_19: 11.82 / 17.62 / 1.491x
- masked_scale_20: 45.93 / 69.12 / 1.505x

Improve candidate.py accordingly and overwrite it in place. Keep the public callable name/signature identical. Remember: correctness first — a failed case costs more than any speedup gains.