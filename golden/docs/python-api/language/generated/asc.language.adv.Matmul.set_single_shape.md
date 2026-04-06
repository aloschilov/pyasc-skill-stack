# asc.language.adv.Matmul.set_single_shape

#### Matmul.set_single_shape(single_m: int, single_n: int, single_k: int) → None

设置Matmul单核计算的形状singleCoreM、singleCoreN、singleCoreK，单位为元素。
用于运行时修改shape，比如复用Matmul对象来处理尾块。与SetTail接口功能一致，建议使用本接口。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)
```

**参数说明**

- single_m：设置的singleCoreM大小，单位为元素。
- single_n：设置的singleCoreN大小，单位为元素。
- single_k：设置的singleCoreK大小，单位为元素。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.set_single_shape(tail_m, tail_n ,tail_k)     # 如果是尾核，需要调整single_core_m/single_core_n/single_core_k
mm.iterate_all(gm_c)
```
