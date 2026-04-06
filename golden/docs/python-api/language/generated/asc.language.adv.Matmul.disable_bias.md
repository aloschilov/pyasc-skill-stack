# asc.language.adv.Matmul.disable_bias

#### Matmul.disable_bias() → None

清除Bias标志位，表示Matmul计算时没有Bias参与。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void DisableBias()
```

**参数说明**

无。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.disable_bias()   # 清除tiling中的Bias标志位
mm.iterate_all(gm_c)
```
