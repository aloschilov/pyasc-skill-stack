# asc.language.adv.get_mm_config

### asc.language.adv.get_mm_config(\*args, \*\*kwargs) → MatmulConfig

灵活的自定义Matmul模板参数配置。

**对应的Ascend C函数原型**

```c++
template <MatmulConfigMode configMode, typename... ArgTypes>
__aicore__ inline constexpr MatmulConfig GetMMConfig(ArgTypes&&... args)
```

**参数说明**

MatmulShapeParams参数：

- single_core_m: 单核内M轴shape大小，以元素为单位。
- single_core_n: 单核内N轴shape大小，以元素为单位。
- single_core_n: 单核内K轴shape大小，以元素为单位。
- basic_m: Matmul计算时base块M轴长度，以元素为单位。
- basic_n: Matmul计算时base块N轴长度，以元素为单位。
- basic_k: Matmul计算时base块K轴长度，以元素为单位。

MatmulQuantParams参数：

- is_per_tensor: A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否为per tensor。
  - True：per tensor量化。
  - False：per channel量化。
- has_anti_quant_offset: A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否使用offset系数。

MatmulBatchParams参数：

- is_n_batch: 是否多Batch输入多Batch输出。仅对BatchMatmul有效。参数取值如下：
  - False：不使能多Batch（默认值）。
  - True：使能多Batch。
- batch_mode: 用于设置参数BatchMode。
  batchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。
  参数取值如下：
  - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size；
  - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size；
  - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。
- is_bias_batch: 批量多Batch的Matmul场景，即BatchMatmul场景，Bias的大小是否带有Batch轴。参数取值如下：
  - True: Bias带有Batch轴，Bias大小为Batch \* N（默认值）。
  - False: Bias不带Batch轴，Bias大小为N，多Batch计算Matmul时，会复用Bias。
  - MatmulFuncParams参数：
- intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
  - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）；
  - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
- en_vec_nd2_nz: 使能通过vector指令进行ND2NZ。参数取值如下：
  - False：不使能通过vector指令进行ND2NZ（默认值）；
  - True：使能通过vector指令进行ND2NZ。
- enable_l1_cache: 是否使能L1 Buffer缓存Unified Buffer计算块。参数取值如下：
  - True: 使能L1 Buffer缓存Unified Buffer计算块。
  - False: 不使能L1 Buffer缓存Unified Buffer计算块。
- do_mte2_preload: 在MTE2流水间隙较大，且M/N数值较大时可通过该参数开启对应M/N方向的预加载功能，开启后能减小MTE2间隙，提升性能。
  预加载功能仅在MDL模板有效（不支持SpecialMDL模板）。参数取值如下：
  - 0： 不开启（默认值）。
  - 1: 开启M方向preload。
  - 2: 开启N方向preload。
- iterate_order: 用于设置参数iterateOrder。
- schedule_type: 用于设置参数scheduleType。配置Matmul数据搬运模式。参数取值如下：
  - scheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运；
  - scheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运。
- enable_reuse: SetSelfDefineData函数设置的回调函数中的dataPtr是否直接传递计算数据。参数取值如下：
  - False：直接传递计算数据，仅限单个值。
  - True：传递GM上存储的数据地址信息。
- enable_ub_reuse: 是否使能Unified Buffer复用。参数取值如下：
  - False：使能Unified Buffer复用。
  - True：不使能Unified Buffer复用。
- is_partial_output: 是否开启PartialOutput功能。参数取值如下：
  - False：开启PartialOutput功能，一次Iterate的K轴不进行累加计算，Matmul每次计算输出局部baseK的baseM \* baseN大小的矩阵分片。
  - True：不开启PartialOutput功能，一次Iterate的K轴进行累加计算，Matmul每次计算输出SingleCoreK长度的baseM \* baseN大小的矩阵分片。
- is_a2_b2_shared: 是否开启A2和B2的全局管理，即控制所有Matmul对象是否共用A2和B2的double buffer机制。参数取值如下：
  - True：开启。
  - False：关闭（默认值）。
- is_enable_channel_split: 是否使能channel_split功能。参数取值如下：
  - False：默认值，不使能channel_split功能，输出的分形为16\*16。
  - True：使能channel_split功能，输出的分形为16\*8。
- enable_kdim_reorder_load: 是否使能K轴错峰加载数据。
  - False：默认值，关闭K轴错峰加载数据的功能。
  - True：开启K轴错峰加载数据的功能。

**返回值说明**

MatmulConfig结构体。

**调用示例**

```python
# 获取MatmulConfig模板为Norm模板
config_mode = asc.adv.MatmulConfigMode.CONFIG_NORM
# single_core_m、single_core_n、single_core_k、basic_m、basic_n、basic_k
shape_params = asc.adv.MatmulShapeParams(128, 128, 128, 64, 64, 64)
# B矩阵量化时为per channel且不适用offset系数
quant_params = asc.adv.MatmulQuantParams(False, False)
# 不使能多Batch
batch_params = asc.adv.MatmulBatchParams(False)
#不进行芯片指令搬运地址偏移量校验，使能通过vector进行ND2NZ
func_params = asc.adv.MatmulFuncParams(False, True)
mm_config = asc.adv.get_mm_config(shape_params, quant_params, batch_params, func_params, config_mode)
```
