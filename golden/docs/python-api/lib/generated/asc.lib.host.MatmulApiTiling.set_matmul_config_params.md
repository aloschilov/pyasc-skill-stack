# asc.lib.host.MatmulApiTiling.set_matmul_config_params

#### MatmulApiTiling.set_matmul_config_params(\*args, \*\*kwargs)

在计算Tiling时，用于自定义设置MatmulConfig参数。本接口中配置的参数对应的功能在Tiling与Kernel中需要保持一致，
所以本接口中的参数取值，需要与Kernel侧对应的MatmulConfig参数值保持一致。

**对应的Ascend C函数原型**

```c++
void SetMatmulConfigParams(int32_t mmConfigTypeIn = 1, bool enableL1CacheUBIn = false,
                          ScheduleType scheduleTypeIn = ScheduleType::INNER_PRODUCT,
                          MatrixTraverse traverseIn = MatrixTraverse::NOSET, bool enVecND2NZIn = false)
void SetMatmulConfigParams(const MatmulConfigParams& configParams)
```

**参数说明**
- mm_config_type_in：设置Matmul的模板类型，需要与Matmul对象创建的模板一致，当前只支持配置为0或1。
- enable_l1_cache_ub_in：配置是否使能L1缓存UB计算块；参考使能场景：MTE3和MTE2流水串行较多的场景。
- schedule_type_in：配置Matmul数据搬运模式。
- traverse_in：Matmul做矩阵运算的循环迭代顺序，即一次迭代计算出[baseM, baseN]大小的C矩阵分片后，自动偏移到下一次迭代输出的C矩阵位置的偏移顺序。
- en_vec_nd2nz_in：是否使能ND2NZ。
- config_params：config相关参数，类型为MatmulConfigParams。

**返回值说明**

-1表示设置失败； 0表示设置成功。

**约束说明**
- 本接口必须在GetTiling接口前调用。
- 若Matmul对象使用NBuffer33模板策略，即MatmulPolicyNBuffer33MatmulPolicy，则在调用GetTiling接口生成Tiling参数前，必须通过本接口将scheduleTypeIn参数设置为ScheduleType::N_BUFFER_33，以启用NBuffer33模板策略的Tiling生成逻辑。

**调用示例**
```python
import asc.lib.host as host
ascendc_platform = host.get_ascendc_platform()
tiling = host.MatmulApiTiling(ascendc_platform)
tiling.set_a_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_b_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT16)
tiling.set_c_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_bias_type(host.TPosition.GM, host.CubeFormat.ND, host.DataType.DT_FLOAT)
tiling.set_shape(1024, 1024, 1024)
tiling.set_org_shape(1024, 1024, 1024)
tiling.set_bias(True)
tiling.set_buffer_space(-1, -1, -1)
tiling.set_matmul_config_params(0)    # 额外设置
tiling_data = host.TCubeTiling()
ret = tiling.get_tiling(tiling_data)
```
