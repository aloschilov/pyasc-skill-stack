# asc.language.adv.Matmul.set_quant_scalar

#### Matmul.set_quant_scalar(quant_scalar: int) → None

本接口提供对输出矩阵的所有值采用同一系数进行量化或反量化的功能，即整个C矩阵对应一个量化参数，量化参数的shape为[1]。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetQuantScalar(const uint64_t quantScalar)
```

**参数说明**

- quant_scalar：量化或反量化系数。

**约束说明**

- 需与set_dequant_type保持一致。
- 本接口必须在iterate或者iterate_all前调用。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm, tiling)
tmp = 0.1
ans = int.from_bytes(struct.pack('<f', tmp), 'little', signed=True) & 0xFFFFFFFF
mm.set_quant_scalar(ans)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
```
