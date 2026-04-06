# asc.language.adv.Matmul.set_org_shape

#### Matmul.set_org_shape(org_m: int, org_n: int, org_ka: int, org_kb: int | None = None, org_kc: int | None = None) → None

设置Matmul计算原始完整的形状M、N、K，单位为元素个数。用于运行时修改shape，比如复用同一个Matmul对象，从不同的矩阵块取数据计算。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
```

```c++
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0)
```

**参数说明**

- org_m：设置原始完整的形状M大小，单位为元素。
- org_n：设置原始完整的形状N大小，单位为元素。
- org_ka：设置矩阵A原始完整的形状Ka大小，单位为元素。
- org_kb：设置矩阵B原始完整的形状Kb大小，单位为元素。
- org_kc：设置输出C矩阵的N，单位为元素。需要输入B矩阵的N和输出C矩阵的N不一样时可设置，默认为0（即使用B矩阵的N，不进行修改）。

备注：Ascend C第一个函数原型对应的python参数：org_m，org_n，org_ka；Ascend C第二个函数原型对应的python参数：org_m，org_n，org_ka，org_kb，org_kc。

**约束说明**

- 本接口需要在set_tensor_a接口、set_tensor_b接口及set_single_shape接口前调用。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
# 复用mm对象
mm.set_org_shape(org_m, org_n, org_k)
mm.set_tensor_a(gm_a1)
mm.set_tensor_b(gm_b1)
mm.set_bias(gm_bias1)
mm.iterate_all(gm_c1)
```
