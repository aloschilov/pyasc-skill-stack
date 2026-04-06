# asc.language.adv.Matmul.iterate

#### Matmul.iterate(en_partial_sum: bool = False, sync: bool = True, local_c_matrix: BaseTensor | None = None) → MatmulIterator

每调用一次Iterate，会计算出一块baseM \* baseN的C矩阵。

**对应的Ascend C函数原型**

```c++
template <bool sync = true>
__aicore__ inline bool Iterate(bool enPartialSum = false)
```

```c++
template <bool sync = true, typename T>
__aicore__ inline bool Iterate(bool enPartialSum, const LocalTensor<T>& localCmatrix)
```

**参数说明**

- en_partial_sum: 是否将矩阵乘的结果累加于现有的CO1数据，默认值为false。
- sync: 设置同步或者异步模式。
- local_c_matrix: 由用户申请的CO1上的LocalTensor内存，用于存放矩阵乘的计算结果。

**约束说明**

- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- 对于用户自主管理CO1的iterate函数，创建Matmul对象时，必须定义C矩阵的内存逻辑位置为TPosition::CO1、数据排布格式为CubeFormat::NZ、数据类型为float或int32_t。

**调用示例**

```python
# 同步模式样例
while mm.iterate() as count:
    mm.get_tensor_c(tensor=ub_cmatrix)
# 异步模式样例
mm.iterate(sync=False)
# 其他操作
for i in range(single_m // base_m * single_n // base_n):
    mm.get_tensor_c(tensor=ub_cmatrix, sync=False)
```
