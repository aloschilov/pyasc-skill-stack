# asc.language.adv.get_ib_share_norm_config

### asc.language.adv.get_ib_share_norm_config(intrinsics_limit: bool | None = False, batch_loop: bool | None = False, is_vec_nd2_nz: bool | None = False, bmm_mode: BatchMode | None = BatchMode.BATCH_LESS_THAN_L1, is_double_cache: bool | None = False, en_unit_flag: bool | None = True) → MatmulConfig

用于配置IBShare模板的参数，获取自定义IBShare模板。

**对应的Ascend C函数原型**

```c++
__aicore__ constexpr MatmulConfig GetIBShareNormConfig(const bool intrinsicsLimit = false, const bool batchLoop = false,
const bool isVecND2NZ = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1, const bool isDoubleCache = false,
const bool enUnitFlag = true)
```

**参数说明**

- intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
  - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
  - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
- bmm_mode: 用于设置参数batchMode。参数取值如下：
  - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size。
  - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size。
  - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。
- batch_loop: 用于设置参数isNBatch。参数取值如下：
  - False：不使能多Batch（默认值）。
  - True：使能多Batch。
- is_vec_nd2_nz: 用于设置参数enVecND2NZ。参数取值如下：
  - False：不使能通过vector指令进行ND2NZ（默认值）。
  - True：使能通过vector指令进行ND2NZ。
- is_double_cache: 用于设置参数enableDoubleCache。参数取值如下：
  - False：L1 Buffer上同时缓存一块数据（默认值）。
  - True：使能L1 Buffer上同时缓存两块数据。
- en_unit_flag: 用于设置参数enUnitFlag。参数取值如下：
  - False：不使能UnitFlag功能。
  - True：使能UnitFlag功能。

**返回值说明**

MatmulConfig结构体。

**调用示例**

```python
mm_cfg = asc.adv.get_ib_share_norm_config()
mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
```
