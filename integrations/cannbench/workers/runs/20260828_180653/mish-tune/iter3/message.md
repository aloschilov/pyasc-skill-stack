The harness evaluated your candidate.py on the NPU. Result:
- score 0.0/100 (compile 0.0/20, accuracy 0.0/30, perf 0.0/50)
- 0/20 cases passed, avg speedup 0.0x

Failed cases:
- level1/mish_1: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | 算子执行期间输出: | /home/l00958488/cann-bench/src/kernel_eval/security/torch_op_guard.py:313: UserWarning: Cannot create tensor with interal format while allow_internel_format=False, tensor will be created with base format. (Triggered internally at ../torch_npu/csrc/aten/common/TensorFactories.cpp:340.)
- level1/mish_2: AI算子执行失败: UB overflow: 253952 bytes are available, 328448 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_3: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_4: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_5: AI算子执行失败: UB overflow: 253952 bytes are available, 328448 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_6: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_7: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_8: AI算子执行失败: UB overflow: 253952 bytes are available, 328448 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_9: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_10: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_11: AI算子执行失败: UB overflow: 253952 bytes are available, 328448 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_12: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_13: AI算子执行失败: UB overflow: 253952 bytes are available, 328448 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_14: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_15: AI算子执行失败: UB overflow: 253952 bytes are available, 328448 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_16: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_17: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_18: AI算子执行失败: UB overflow: 253952 bytes are available, 328448 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_19: AI算子执行失败: UB overflow: 253952 bytes are available, 353024 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple
- level1/mish_20: AI算子执行失败: UB overflow: 253952 bytes are available, 328448 bytes are used. | Traceback (most recent call last): |   File "/home/l00958488/cann-bench/src/kernel_eval/eval/op_runner.py", line 395, in _run_simple

Improve candidate.py accordingly and overwrite it in place. Keep the public callable name/signature identical. Remember: correctness first — a failed case costs more than any speedup gains.