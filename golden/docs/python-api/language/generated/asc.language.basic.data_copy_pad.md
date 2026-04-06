# asc.language.basic.data_copy_pad

### asc.language.basic.data_copy_pad(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), data_copy_params: DataCopyExtParams, pad_params: DataCopyPadExtParams) → None

### asc.language.basic.data_copy_pad(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), data_copy_params: DataCopyExtParams) → None

### asc.language.basic.data_copy_pad(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), data_copy_params: DataCopyExtParams, nd2nz_params: Nd2NzParams) → None

### asc.language.basic.data_copy_pad(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), data_copy_params: DataCopyParams, pad_params: DataCopyPadParams) → None

### asc.language.basic.data_copy_pad(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), data_copy_params: DataCopyParams) → None

### asc.language.basic.data_copy_pad(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), data_copy_params: DataCopyParams, nd2nz_params: Nd2NzParams) → None

DataCopyPad接口提供数据非对齐搬运的功能，其中从Global Memory搬运数据至Local Memory时，可以根据开发者的需要自行填充数据。

**对应的Ascend C函数原型**

通路：Global Memory->Local Memory

```c++
template <typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                    const DataCopyExtParams &dataCopyParams, const DataCopyPadExtParams<T> &padParams)
```

通路：Local Memory->Global Memory

```c++
template <typename T>
__aicore__ inline void DataCopyPad(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                    const DataCopyExtParams &dataCopyParams)
```

通路：Local Memory->Local Memory，实际搬运过程是VECIN/VECOUT->GM->TSCM

```c++
template <typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                    const DataCopyExtParams &dataCopyParams, const Nd2NzParams &nd2nzParams)
```

通路：Global Memory->Local Memory (DataCopyParams版本)

```c++
template<typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                    const DataCopyParams& dataCopyParams, const DataCopyPadParams& padParams)
```

通路：Local Memory->Global Memory (DataCopyParams版本)

```c++
template<typename T>
__aicore__ inline void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
                                    const DataCopyParams& dataCopyParams)
```

通路：Local Memory->Local Memory，实际搬运过程是VECIN/VECOUT->GM->TSCM (DataCopyParams版本)

```c++
template<typename T>
__aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                    const DataCopyParams& dataCopyParams, const Nd2NzParams& nd2nzParams)
```

**参数说明**

- dst: 目的操作数，类型为LocalTensor或GlobalTensor。
  LocalTensor的起始地址需要保证32字节对齐。
  GlobalTensor的起始地址无地址对齐约束。
- src: 源操作数，类型为LocalTensor或GlobalTensor。
  LocalTensor的起始地址需要保证32字节对齐。
  GlobalTensor的起始地址无地址对齐约束。
- dataCopyParams: 搬运参数。
  DataCopyExtParams类型：支持更大的操作数步长等参数取值范围
  DataCopyParams类型：标准搬运参数
- padParams: 从Global Memory搬运数据至Local Memory时，用于控制数据填充过程的参数。
  DataCopyPadExtParams<T>类型：支持泛型填充值
  DataCopyPadParams类型：仅支持uint64_t数据类型且填充值只能为0
- nd2nzParams: 从VECIN/VECOUT->TSCM进行数据搬运时，用于控制数据格式转换的参数。
  Nd2NzParams类型，ndNum仅支持设置为1。

**约束说明**

- leftPadding、rightPadding的字节数均不能超过32Bytes。
- 当数据类型长度为64位时，paddingValue只能设置为0。
- 不同产品型号对函数原型的支持存在差异，请参考官方文档选择产品型号支持的函数原型进行开发。

**调用示例**

GM->VECIN搬运数据并填充：

```python
# 从GM->VECIN搬运，使用DataCopyParams和DataCopyPadParams
src_local = in_queue_src.alloc_tensor(asc.half)
copy_params = asc.DataCopyParams(1, 20 * asc.half.sizeof(), 0, 0)
pad_params = asc.DataCopyPadParams(True, 0, 2, 0)
asc.data_copy_pad(src_local, src_global, copy_params, pad_params)
```
