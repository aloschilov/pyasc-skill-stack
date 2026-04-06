# asc.language.adv.Matmul.iterate_n_batch

#### Matmul.iterate_n_batch(batch_loop: PlainValue | int, batch_a: PlainValue | int, batch_b: PlainValue | int, en_sequential_write: PlainValue | bool, matrix_stride_a: PlainValue | int = 0, matrix_stride_b: PlainValue | int = 0, matrix_stride_c: PlainValue | int = 0, sync: PlainValue | bool = True, wait_iterate_batch: PlainValue | bool = False) → None

调用一次IterateNBatch，会进行N次IterateBatch计算，计算出N个多Batch的singleCoreM \* singleCoreN大小的C矩阵。
在调用该接口前，需将MatmulConfig中的isNBatch参数设为true，使能多Batch输入多Batch输出功能，并调用SetWorkspace接口申请临时空间，
用于缓存计算结果，即IterateNBatch的结果输出至SetWorkspace指定的Global Memory内存中。
对于BSNGD、SBNGD、BNGS1S2的Layout格式，
调用该接口之前需要在tiling中使用SetALayout/SetBLayout/SetCLayout/SetBatchNum设置A/B/C的Layout轴信息和最大BatchNum数；
对于Normal数据格式则需使用SetBatchInfoForNormal设置A/B/C的M/N/K轴信息和A/B矩阵的BatchNum数。
实例化Matmul时，通过MatmulType设置Layout类型，当前支持3种Layout类型：BSNGD、SBNGD、BNGS1S2。

**对应的Ascend C函数原型**

```c++
template <bool sync = true, bool waitIterateBatch = false>
__aicore__ inline void IterateNBatch(const uint32_t batchLoop, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)
```

**参数说明**

- sync：设置同步或者异步模式。
- wait_iterate_batch：是否需要通过WaitIterateBatch接口等待IterateNBatch执行结束，仅在异步场景下使用。
- batch_loop：当前计算的BMM个数。
- batch_a：当前单次BMM调用计算左矩阵的batch数。
- batch_b：当前单次BMM调用计算右矩阵的batch数，brc场景batchA/B不相同。
- en_sequential_write：输出是否连续存放数据。
- matrix_stride_a：A矩阵源操作数相邻nd矩阵起始地址间的偏移，默认值是0。
- matrix_stride_b：B矩阵源操作数相邻nd矩阵起始地址间的偏移，默认值是0。
- matrix_stride_c：该参数预留，开发者无需关注。

**约束说明**

- 单BMM内计算遵循之前的约束条件。
- 对于BSNGD、SBNGD、BNGS1S2 Layout格式，输入A、B矩阵多Batch数据总和应小于L1 Buffer的大小。
- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

**调用示例**

```python
@asc.jit
def kernel_matmul_rpc_batch(a_gm: asc.GlobalAddress, b_gm: asc.GlobalAddress, c_gm: asc.GlobalAddress, bias_gm: asc.GlobalAddress, tiling: asc.adv.TCubeTiling, workspace_gm: asc.GlobalAddress, is_transpose_a_in: int, is_transpose_b_in: int, batch_a: int, batch_b: int):
    # 定义matmul type
    a_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half, False, asc.LayoutMode.BSNGD)
    b_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half, True, asc.LayoutMode.BSNGD)
    c_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float, False, asc.LayoutMode.BNGS1S2)
    bias_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float)
    a_global = asc.GlobalTensor()
    size_a = tiling.a_layout_info_b * tiling.a_layout_info_s * tiling.a_layout_info_n * tiling.a_layout_info_g * tiling.a_layout_info_d * 4
    size_a = tiling.b_layout_info_b * tiling.b_layout_info_s * tiling.b_layout_info_n * tiling.b_layout_info_g * tiling.b_layout_info_d * 4
    size_bias = tiling.c_layout_info_b * tiling.c_layout_info_n * tiling.c_layout_info_g * tiling.c_layout_info_s2 * 8
    a_global = set_global_buffer(a_gm, size_a)
    b_global = set_global_buffer(b_gm, size_b)
    bias_global = set_global_buffer(bias_gm, size_bias)
    tiling.share_mode = 0
    tiling.share_l1_size = 512 * 1024
    tiling.share_l0c_size = 128 * 1024
    tiling.share_ub_size = 0
    offset_a = 0
    offset_b = 0
    offset_c = 0
    offset_bias = 0
    a_global = a_global[offset_a]
    b_global = b_global[offset_b]
    bias_global = bias_global[offset_bias]
    # 创建Matmul实例
    mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type)
    pipe = asc.Tpipe()
    asc.adv.register_matmul(pipe, mm)
    mm.init(tiling)
    mm.set_tensor_a(a_global, is_transpose_a_in)
    mm.set_tensor_b(b_global, is_transpose_b_in)
    g_lay = tiling.a_layout_info_g
    if tiling.b_layout_info > g_lay:
        g_lay = tiling.b_layout_info_g
    for_extent = tiling.a_layout_info_b * tiling.a_layout_info_n * g_lay / tiling.batch_num
    mm.set_workspace(c_global)
    mm.iterate_n_batch(for_extent, batch_a, batch_b, False)
```
