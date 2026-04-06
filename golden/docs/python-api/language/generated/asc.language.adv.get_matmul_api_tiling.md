# asc.language.adv.get_matmul_api_tiling

### asc.language.adv.get_matmul_api_tiling(mm_cfg: MatmulConfig, l1_size: int, a_type: MatmulType, b_type: MatmulType, c_type: MatmulType, bias_type: MatmulType | None = None) → MatmulApiStaticTiling

本接口用于在编译期间获取常量化的Matmul Tiling参数。

**对应的Ascend C函数原型**

```c++
template<class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr MatmulApiStaticTiling GetMatmulApiTiling(const MatmulConfig& mmCFG, int32_t l1Size = Impl::L1_SIZE)
```

**参数说明**

- mm_cfg：获取的MatmulConfig模板。
- l1_size：可用的L1大小，默认值L1_SIZE。
- a_type：A矩阵类型信息，通过MatmulType来定义。
- b_type：B矩阵类型信息，通过MatmulType来定义。
- c_type：C矩阵类型信息，通过MatmulType来定义。
- bias_type：BIAS矩阵类型信息，通过MatmulType来定义。

**返回值说明**

MatmulApiStaticTiling，常量化Tiling参数。

**约束说明**

- 入参mm_cfg，在调用获取MatmulConfig模板的接口获取时，需要使用常数值指定(base_m, base_n, base_k)或者指定(base_m, base_n, base_k, single_core_m, single_core_n, single_core_k)，并且指定的参数值需要和tiling计算的值保持一致。
- Batch Matmul场景支持全量常量化，但不支持使用空指针替代REGIST_MATMUL_OBJ的入参tiling。

**调用示例**

```python
# 定义Matmul对象
a_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half)
b_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.half)
c_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float)
bias_type = asc.adv.MatmulType(asc.TPosition.GM, asc.CubeFormat.ND, asc.float)
# 这里CFG使用get_normal_config接口获取，并指定已知的singleshape信息和base_m, base_n, base_k,指定的数值跟运行时tiling保持一致
static_tiling = asc.adv.get_api_tiling(mm_cfg, 524288, a_type, b_type, c_type, bias_type)
mm = asc.adv.Matmul(a_type, b_type, c_type, bias_type, static_tiling)
```
