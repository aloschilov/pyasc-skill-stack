# asc.language.adv.Matmul.set_user_def_info

#### Matmul.set_user_def_info(tiling_ptr: GlobalAddress) → None

使能模板参数MatmulCallBackFunc（自定义回调函数）时，设置算子tiling地址，用于回调函数使用，该接口仅需调用一次。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetUserDefInfo(const uint64_t tilingPtr)
```

**参数说明**

- tiling_ptr：设置的算子tiling地址。

**约束说明**

- 若回调函数中需要使用tiling_ptr参数时，必须调用此接口；若回调函数不使用tilingPtr参数，无需调用此接口。
- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

**调用示例**

```python
tiling_ptr = tiling
mm.set_user_def_info(tiling_ptr)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.iterate_all()
```
