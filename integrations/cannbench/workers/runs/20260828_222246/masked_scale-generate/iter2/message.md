The harness evaluated your candidate.py on the NPU. Result:
- score 78.01/100 (compile 19.0/20, accuracy 28.5/30, perf 30.51/50)
- 19/20 cases passed, avg speedup 1.516x

Failed cases:
- level1/masked_scale_3: AI算子执行失败: compile failed! | Error message is In file included from <built-in>:1: | In file included from /home/f00816836/Ascend0615/cann-9.1.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_runtime_wrapper.h:43:

Per-case timings (elapsed_us / baseline_us / speedup):
- masked_scale_1: 6.46 / 11.495 / 1.779x
- masked_scale_2: 38.93 / 31.83 / 0.818x
- masked_scale_4: 378.42 / 817.845 / 2.161x
- masked_scale_5: 409.69 / 1353.855 / 3.305x
- masked_scale_6: 13.29 / 11.485 / 0.864x
- masked_scale_7: 6.45 / 11.59 / 1.797x
- masked_scale_8: 7.8 / 14.5 / 1.859x
- masked_scale_9: 274.51 / 588.285 / 2.143x
- masked_scale_10: 13.06 / 11.46 / 0.877x
- masked_scale_11: 7.5 / 13.24 / 1.765x
- masked_scale_12: 6.76 / 9.21 / 1.362x
- masked_scale_13: 65.17 / 87.7 / 1.346x
- masked_scale_14: 14.67 / 17.95 / 1.224x
- masked_scale_15: 13.03 / 13.07 / 1.003x
- masked_scale_16: 10.96 / 14.21 / 1.297x
- masked_scale_17: 22.06 / 13.92 / 0.631x
- masked_scale_18: 12.02 / 19.52 / 1.624x
- masked_scale_19: 12.09 / 17.62 / 1.457x
- masked_scale_20: 46.25 / 69.12 / 1.494x

Improve candidate.py accordingly and overwrite it in place. Keep the public callable name/signature identical. Remember: correctness first — a failed case costs more than any speedup gains.