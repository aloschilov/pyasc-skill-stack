The harness evaluated your candidate.py on the NPU. Result:
- score 0.0/100 (compile 0.0/20, accuracy 0.0/30, perf 0.0/50)
- 0/20 cases passed, avg speedup 0.0x

Failed cases:
- level1/foreach_norm_1: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_2: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_3: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_4: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_5: AI算子执行失败: at <source>:12:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_6: AI算子执行失败: at <source>:12:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_7: AI算子执行失败: at <source>:12:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_8: AI算子执行失败: at <source>:12:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_9: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_10: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_11: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_12: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_13: AI算子执行失败: at <source>:12:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_14: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_15: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_16: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_17: AI算子执行失败: at <source>:12:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_18: AI算子执行失败: at <source>:11:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_19: AI算子执行失败: at <source>:12:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])
- level1/foreach_norm_20: AI算子执行失败: at <source>:12:4: | ): |     x_gm = asc2.global_tensor(x_ptr, [size])

Improve candidate.py accordingly and overwrite it in place. Keep the public callable name/signature identical. Remember: correctness first — a failed case costs more than any speedup gains.