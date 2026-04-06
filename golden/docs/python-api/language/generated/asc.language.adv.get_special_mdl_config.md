# asc.language.adv.get_special_mdl_config

### asc.language.adv.get_special_mdl_config(intrinsics_limit: bool | None = False, batch_loop: bool | None = False, do_mte2_pre_load: int | None = 0, is_vec_nd2_nz: bool | None = False, is_per_tensor: bool | None = False, has_anti_quant_offset: bool | None = False) → MatmulConfig

用于配置SpecialMDL模板的参数，获取自定义SpecialMDL模板。

**对应的Ascend C函数原型**

```c++
__aicore__ constexpr MatmulConfig GetSpecialMDLConfig(const bool intrinsicsLimit = false, const bool batchLoop = false,
const uint32_t doMTE2Preload = 0, const bool isVecND2NZ = false, bool isPerTensor = false, bool hasAntiQuantOffset = false)
```

**参数说明**

- intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
  - False：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
  - True：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
- do_mte2_pre_load: 用于设置参数enVecND2NZ。参数取值如下：
  - False：不使能通过vector指令进行ND2NZ（默认值）。
  - True：使能通过vector指令进行ND2NZ。
- is_vec_nd2_nz: 用于设置参数enVecND2NZ。参数取值如下：
  - False：不使能通过vector指令进行ND2NZ（默认值）。
  - True：使能通过vector指令进行ND2NZ。
- batch_loop: 用于设置参数isNBatch。参数取值如下：
  - False：不使能多Batch（默认值）。
  - True：使能多Batch。
- is_per_tensor: 用于设置参数isPerTensor。参数取值如下：
  - True：per tensor量化。
  - False：per channel量化。
- has_anti_quant_offset: 用于设置参数hasAntiQuantOffset。

**返回值说明**

MatmulConfig结构体。

**调用示例**

```python
mm_cfg = asc.adv.get_special_mdl_config()
mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
```
