# asc.language.adv.get_basic_config

### asc.language.adv.get_basic_config(basic_m: int, basic_n: int, basic_k: int, intrinsics_limit: bool | None = False, batch_loop: bool | None = False, bmm_mode: BatchMode | None = BatchMode.BATCH_LESS_THAN_L1) → MatmulConfig

用于配置BasicBlock模板的参数，获取自定义BasicBlock模板。

**对应的Ascend C函数原型**

```c++
__aicore__ constexpr MatmulConfig GetBasicConfig(const uint32_t basicM, const uint32_t basicN,
const uint32_t basicK, const bool intrinsicsLimit = false, const bool batchLoop = false,
const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1)
```

**参数说明**

- basic_m: 用于设置参数basicM。与TCubeTiling结构体中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。
- basic_n: 用于设置参数basicN。与TCubeTiling结构体中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。
- basic_k: 用于设置参数basicK。与TCubeTiling结构体中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。
- intrinsics_limit: 用于设置参数intrinsicsCheck。参数取值如下：
  - false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。
  - true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。
- batch_loop: 用于设置参数isNBatch。参数取值如下：
  - false：不使能多Batch（默认值）。
  - true：使能多Batch。
- bmm_mode: 用于设置参数batchMode。参数取值如下：
  - batchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size。
  - batchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size。
  - batchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。

**返回值说明**

MatmulConfig结构体。

**调用示例**

```python
mm_cfg = asc.adv.get_basic_config(128, 256, 64)
mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, mm_cfg)
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
```
