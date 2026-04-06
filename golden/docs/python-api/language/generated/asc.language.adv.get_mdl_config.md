# asc.language.adv.get_mdl_config

### asc.language.adv.get_mdl_config(intrinsics_limit: bool | None = False, batch_loop: bool | None = False, do_mte2_preload: int | None = 0, is_vec_nd2_nz: bool | None = False, is_per_tensor: bool | None = False, has_anti_quant_offset: bool | None = False, en_unit_flag: bool | None = False, is_msg_reuse: bool | None = True, enable_ub_reuse: bool | None = True, enable_l1_cache_ub: bool | None = False, enable_mix_dual_master: bool | None = False, enable_kdim_reorder_load: bool | None = False) → MatmulConfig

用于配置MDL模板的参数，获取自定义MDL模板。

**对应的Ascend C函数原型**

```c++
__aicore__ constexpr MatmulConfig GetMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false,
const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false,
bool hasAntiQuantOffset = false, const bool enUnitFlag = false, const bool isMsgReuse = true,
const bool enableUBReuse = true, const bool enableL1CacheUB = false, const bool enableMixDualMaster = false,
const bool enableKdimReorderLoad = false)
```

**参数说明**

- intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
  - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
  - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
- batchLoop: 用于设置参数isNBatch。参数取值如下：
  - False：不使能多Batch（默认值）。
  - True：使能多Batch。
- do_mte2_pre_load: 用于设置参数enVecND2NZ。参数取值如下：
  - False：不使能通过vector指令进行ND2NZ（默认值）。
  - True：使能通过vector指令进行ND2NZ。
- is_vec_nd2_nz: 用于设置参数enVecND2NZ。参数取值如下：
  - False：不使能通过vector指令进行ND2NZ（默认值）。
  - True：使能通过vector指令进行ND2NZ。
- is_per_tensor: 用于设置参数isPerTensor。参数取值如下：
  - True：per tensor量化。
  - False：per channel量化。
- has_anti_quant_offset: 用于设置参数hasAntiQuantOffset。
- en_unit_flag: 用于设置参数enUnitFlag。参数取值如下：
  - False：不使能UnitFlag功能。
  - True：使能UnitFlag功能。
- is_msg_reuse: 用于设置参数enableReuse。参数取值如下：
  - True：直接传递计算数据，仅限单个值。
  - False：传递GM上存储的数据地址信息。
- enable_ub_reuse: 用于设置参数enableUBReuse。参数取值如下：
  - True：使能Unified Buffer复用。
  - False：不使能Unified Buffer复用。
- enable_l1_cache_ub: 用于设置参数enableL1CacheUB。参数取值如下：
  - True：使能L1 Buffer缓存Unified Buffer计算块。
  - False：不使能L1 Buffer缓存Unified Buffer计算块。
- enable_mix_dual_master: 用于设置参数enableMixDualMaster。
- enable_kdim_reorder_load: 用于设置参数enableKdimReorderLoad。

**返回值说明**

MatmulConfig结构体。

**调用示例**

```python
mm_cfg = asc.adv.get_mdl_config()
mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
```
