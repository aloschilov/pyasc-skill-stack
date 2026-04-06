# asc.language.adv.Matmul.iterate_batch

#### Matmul.iterate_batch(tensor: BaseTensor, batch_a: int, batch_b: int, en_sequential_write: bool, matrix_stride_a: int = 0, matrix_stride_b: int = 0, matrix_stride_c: int = 0, en_partial_sum: bool = False, en_atomic: int = 0, sync: bool = True, wait_iterate_batch: bool | None = None) → None

#### Matmul.iterate_batch(tensor: BaseTensor, en_partial_sum, en_atomic, en_sequential_write: bool, matrix_stride_a: int = 0, matrix_stride_b: int = 0, matrix_stride_c: int = 0, sync: bool = True) → None

该接口提供批量处理Matmul的功能，调用一次iterate_batch，可以计算出多个singleCoreM \* singleCoreN大小的C矩阵。

**对应的Ascend C函数原型**

```c++
template <bool sync = true, bool waitIterateBatch = false>
__aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0)
```

```c++
template <bool sync = true>
__aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0)
```

```c++
__aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)
```

```c++
__aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)
```

**参数说明**

- tensor: C矩阵。类型为GlobalTensor或LocalTensor。
- batch_a: 左矩阵的batch数。
- batch_b: 右矩阵的batch数。
- en_sequential_write: 是否开启连续写模式。
- matrix_stride_a: A矩阵源操作数相邻nd矩阵起始地址间的偏移，单位是元素，默认值是0。
- matrix_stride_b: B矩阵源操作数相邻nd矩阵起始地址间的偏移，单位是元素，默认值是0。
- matrix_stride_c: 该参数预留，开发者无需关注。
- en_partial_sum: 是否将矩阵乘的结果累加于现有的CO1数据，默认值为false。
- en_atomic: 是否开启Atomic操作，默认值为0。
- sync: 设置同步或者异步模式。
- wait_iterate_batch: 是否需要通过wait_iterate_batch接口等待iterate_batch执行结束，仅在异步场景下使用。

**约束说明**

- 该接口只支持Norm模板，即BatchMatmul只支持Norm模板。
- 对于BSNGD、SBNGD、BNGS1S2 Layout格式，输入A、B矩阵按分形对齐后的多Batch数据总和应小于L1 Buffer的大小；对于NORMAL Layout格式没有这种限制，但需通过MatmulConfig配置输入A、B矩阵多Batch数据大小与L1 Buffer的大小关系；
- 对于BSNGD、SBNGD、BNGS1S2 Layout格式，称左矩阵、右矩阵的G轴分别为a_layout_info_g、b_layout_info_g，则a_layout_info_g / batch_a = b_layout_info_g / batch_b；对于NORMAL Layout格式，batch_a、batch_b必须满足倍数关系。
- 如果接口输出到Unified Buffer上，输出C矩阵大小Base_m\*Base_n应小于分配的Unified Buffer内存大小。
- 如果接口输出到Unified Buffer上，且单核计算的N方向大小single_core_n非32字节对齐，C矩阵的CubeFormat仅支持ND_ALIGN格式，输出C矩阵片时，自动将single_core_n方向上的数据补齐至32字节。
- 对于BSNGD、SBNGD Layout格式，输入输出只支持ND格式数据。对于BNGS1S2、NORMAL Layout格式， 输入支持ND/NZ格式数据。
- 对于BSNGD、SBNGD Layout格式，不支持连续写模式。
- 该接口不支持量化模式，即不支持set_quant_scalar、set_quant_vector接口。
- BSNGD场景，不支持一次计算多行SD，需要算子程序中循环计算，即(a_layout_info_n \* a_layout_info_g) / batch_a、(b_layout_info_n \* b_layout_info_g) / batch_b均为整数。
- 异步模式不支持iterate_batch搬运到UB上。
- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- 使用该接口时，A矩阵、B矩阵不支持int4b_t类型的输入，即BatchMatmul不支持int4b_t类型的矩阵输入。

**调用示例**

```python
# 定义matmul type
a_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half, False, asc.LayoutMode.BSNGD)
b_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half, True, asc.LayoutMode.BSNGD)
c_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float, False, asc.LayoutMode.BNGS1S2)
bias_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float)
mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type)
asc.adv.register_matmul(pipe, mm)
mm.init(tiling)
batch_c = batch_a
if batch_b > batch_c:
    batch_c = batch_b
g_lay = tiling.a_layout_info_g
if tiling.b_layout_info > g_lay:
    g_lay = tiling.b_layout_info_g
for_extent = tiling.a_layout_info_b * tiling.a_layout_info_n * g_lay / tiling.batch_num
for i in range(for_extent):
    batch_offset_a = i * tiling.a_layout_info_d * batch_a
    batch_offset_b = i * tiling.b_layout_info_d * batch_b
    mm.set_tensor_a(gm_a[batch_offset_a], is_transpose_a_in)
    mm.set_tensor_b(gm_b[batch_offset_b], is_transpose_b_in)
    idx_c = i * batch_c
    if tiling.c_layout_info_g == 1 and (tiling.b_layout_info_g != 1 or tiling.a_layout_info_g != 1):
        d = tiling.b_layout_info_g
        if tiling.a_layout_info_g > d:
            d = tiling.a_layout_info_g
        idx_c = idx_c // d
    if tiling.is_bias:
        batch_offset_bias = idx_c * tiling.c_layout_info_s2
        mm.ste_bias(gm_bias[batch_offset_bias])
    batch_offset_c = idx_c * tiling.c_layout_info_s2
    if c_type.layout == asc.LayoutMode.BNGS1S2:
        batch_offset_c = idx_c * tiling.c_layout_infos2 * tiling.c_layout_info_s1
    mm.iterate_batch(tensor=gm_c[offsetc], batch_a=batch_a, batch_b=batch_b, en_sequential_write=False)
```
