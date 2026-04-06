# asc.language.adv.get_normal_config

### asc.language.adv.get_normal_config(intrinsics_limit: bool | None = False, batch_loop: bool | None = False, is_vec_nd2_nz: bool | None = False, bmm_mode: BatchMode | None = BatchMode.BATCH_LESS_THAN_L1, is_msg_reuse: bool | None = True, iterate_order: IterateOrder | None = IterateOrder.ORDER_M, schedule_type: ScheduleType | None = ScheduleType.INNER_PRODUCT, en_unit_flag: bool | None = True) → MatmulConfig

用于配置Norm模板的参数，获取自定义Norm模板。

**对应的Ascend C函数原型**

```c++
__aicore__ constexpr MatmulConfig GetNormalConfig(const bool intrinsicsLimit = false, const bool batchLoop = false,
const bool isVecND2NZ = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1, const bool isMsgReuse = true,
const IterateOrder iterateOrder = IterateOrder::UNDEF, const ScheduleType scheduleType = ScheduleType::INNER_PRODUCT,
const bool enUnitFlag = true, const bool enableMixDualMaster = false)
```

**参数说明**

- intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
  - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
  - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
- batch_loop: 用于设置参数isNBatch。参数取值如下：
  - False：不使能多Batch（默认值）。
  - True：使能多Batch。
- is_vec_nd2_nz: 用于设置参数enVecND2NZ。参数取值如下：
  - False：不使能通过vector指令进行ND2NZ（默认值）。
  - True：使能通过vector指令进行ND2NZ。
- bmm_mode: 用于设置参数batchMode。参数取值如下：
  - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size。
  - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size。
  - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。
- is_msg_reuse: 用于设置参数enableReuse。参数取值如下：
  - True：直接传递计算数据，仅限单个值。
  - False：传递GM上存储的数据地址信息。
- iterate_order: 用于设置参数iterateOrder。
- schedule_type: 用于设置参数scheduleType。配置Matmul数据搬运模式。参数取值如下：
  - ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运。
  - ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运。
- en_unit_flag: 用于设置参数enUnitFlag。参数取值如下：
  - False：不使能UnitFlag功能。
  - True：使能UnitFlag功能。

**返回值说明**

MatmulConfig结构体。

**调用示例**

```python
mm_cfg = asc.adv.get_normal_config()
mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
```
