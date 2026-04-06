# asc.language.adv.Matmul.set_tail

#### Matmul.set_tail(tail_m: PlainValue | int = -1, tail_n: PlainValue | int = -1, tail_k: PlainValue | int = -1) → None

在不改变Tiling的情况下，重新设置本次计算的singleCoreM/singleCoreN/singleCoreK，以元素为单位。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetTail(int tailM = -1, int tailN = -1, int tailK = -1)
```

**参数说明**

- tail_m：重新设置的singleCoreM值。
- tail_n：重新设置的singleCoreN值。
- tail_k：重新设置的singleCoreK值。

**调用示例**

```python
asc.adv.register_matmul(pip, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.set_tail(tail_m, tail_n, tail_k) # 如果是尾核，需要调整single_core_m/single_core_n/single_core_k
mm.iterate_all(gm_c)
```
