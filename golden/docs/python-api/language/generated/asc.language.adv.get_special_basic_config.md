# asc.language.adv.get_special_basic_config

### asc.language.adv.get_special_basic_config(basic_m: int, basic_n: int, basic_k: int, single_core_m: int, single_core_n: int, single_core_k: int, step_m: int, step_n: int, intrinsics_limit: bool | None = False, batch_loop: bool | None = False, bmm_mode: BatchMode | None = BatchMode.BATCH_LESS_THAN_L1) → MatmulConfig

用于配置SpecialBasicBlock模板的参数，获取自定义SpecialBasicBlock模板。当前为预留接口。

**对应的Ascend C函数原型**

```c++
__aicore__ constexpr MatmulConfig GetSpecialBasicConfig(const uint32_t basicM, const uint32_t basicN,
const uint32_t basicK, const uint32_t singleCoreM, const uint32_t singleCoreN, const uint32_t singleCoreK,
const uint32_t stepM, const uint32_t stepN, const bool intrinsicsLimit = false, const bool batchLoop = false,
const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
```

**参数说明**

- basic_m: 用于设置参数basicM。与TCubeTiling结构体中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。
- basic_n: 用于设置参数basicN。与TCubeTiling结构体中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。
- basic_k: 用于设置参数basicK。与TCubeTiling结构体中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。
- single_core_m: 用于设置参数singleCoreM。单核内M轴shape大小，以元素为单位。
- single_core_n: 用于设置参数singleCoreN。单核内N轴shape大小，以元素为单位。
- single_core_k: 用于设置参数singleCoreK。单核内K轴shape大小，以元素为单位。
- step_m: 用于设置参数stepM。左矩阵在A1中缓存的bufferM方向上baseM的倍数。
- step_n: 用于设置参数stepN。右矩阵在B1中缓存的bufferN方向上baseN的倍数。
- intrinsics_limit: 用于设置参数intrinsicsCheck。
  当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从Global Memory到
  L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API
  内部通过循环执行数据的搬入。参数取值如下：
  - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
  - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
- batch_loop: 用于设置参数isNBatch。
  是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用IterateNBatch实现多
  Batch输入多Batch输出。参数取值如下：
  - False：不使能多Batch（默认值）。
  - True：使能多Batch。
- bmm_mode: 用于设置参数batchMode。
  BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。
  参数取值如下：
  - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size。
  - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size。
  - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。

**返回值说明**

MatmulConfig结构体。
