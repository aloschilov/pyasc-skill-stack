# asc.language.adv.Matmul.set_hf32

#### Matmul.set_hf32(enable_hf32: PlainValue | bool = False, trans_mode: PlainValue | int = 0) → None

在纯Cube模式（只有矩阵计算）下，设置是否使能HF32（矩阵乘计算时可采用的数据类型）模式。使能后，在矩阵乘计算时，
float32数据类型会转换为hf32数据类型，可提升计算性能，但同时也会带来精度损失。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetHF32(bool enableHF32 = false, int32_t transMode = 0)
```

**参数说明**

- enable_hf32：配置是否开启HF32模式，默认值false(不开启)。
- trans_mode：配置在开启HF32模式时，float转换为hf32时所采用的ROUND模式。默认值0。

**约束说明**

本接口仅支持在纯Cube模式下调用

**调用示例**

```python
asc.adv.register_matmul(pip, mm, tiling)  # A/B/C/BIAS类型是float
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.set_hf32(True)
mm.iterate_all(gm_c)
mm.set_hf32(False)
```
