# asc.language.adv.Matmul.set_self_define_data

#### Matmul.set_self_define_data(data_ptr: GlobalAddress | PlainValue | int) → None

使能模板参数MatmulCallBackFunc（自定义回调函数）时，设置需要的计算数据或在GM上存储的数据地址等信息，用于回调函数使用。复用同一个Matmul对象时，可以多次调用本接口重新设置对应数据信息。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetSelfDefineData(const uint64_t dataPtr)
```

```c++
__aicore__ inline void SetSelfDefineData(T dataPtr)
```

Ascend 910C 不支持SetSelfDefineData(T dataPtr)接口原型。
Ascend 910B 不支持SetSelfDefineData(T dataPtr)接口原型。

**参数说明**

- data_ptr：设置的算子回调函数需要的计算数据或在GM上存储的数据地址等信息。其中，类型T支持用户自定义基础结构体。

**约束说明**

- 若回调函数中需要使用data_ptr参数时，必须调用此接口；若回调函数不使用data_ptr参数，无需调用此接口。
- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- 本接口必须在set_tensor_a接口、set_tensor_b接口之前调用。

**调用示例**

```python
data_gm_ptr = asc.GlobalTensor()    # 保存有回调函数需使用的计算数据的GM
mm.set_self_define_data(data_gm_ptr)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.iterate_all()
```
