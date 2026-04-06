# asc.language.basic.dump_acc_chk_point

### asc.language.basic.dump_acc_chk_point(tensor: [LocalTensor](../core.md#asc.language.core.LocalTensor), index: int, count_off: int, dump_size: int) → None

### asc.language.basic.dump_acc_chk_point(tensor: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), index: int, count_off: int, dump_size: int) → None

基于算子工程开发的算子，可以使用该接口 Dump 指定 Tensor 的内容。
同时支持打印自定义的附加信息（仅支持 uint32_t 类型的信息），
例如用于打印当前执行位置、行号等调试信息。
与 dump_tensor 不同的是，该接口支持指定 Tensor 的偏移位置进行 Dump，适用于精细化调试和问题定位。

**对应的 Ascend C 函数原型**

```c++
template <typename T>
__aicore__ inline void DumpAccChkPoint(
    const LocalTensor<T>& tensor,
    uint32_t index,
    uint32_t countOff,
    uint32_t dumpSize);

template <typename T>
__aicore__ inline void DumpAccChkPoint(
    const GlobalTensor<T>& tensor,
    uint32_t index,
    uint32_t countOff,
    uint32_t dumpSize);
```

**参数说明**

- tensor：
  需要 Dump 的 Tensor，支持 LocalTensor 和 GlobalTensor。
- index：
  Dump 检查点索引编号，用于区分不同 Dump 位置。
- count_off：
  自定义附加信息，仅支持 uint32_t 类型，通常用于记录行号、
  步骤编号等调试信息。
- dump_size：
  Dump 的元素个数，从 Tensor 指定偏移位置开始连续 Dump。

**约束说明**

- 该接口主要用于调试和问题定位，建议仅在 Debug 场景下使用。
- 附加信息 count_off 仅支持 uint32_t 类型。
- Dump 行为可能影响性能，不建议在性能敏感路径中频繁调用。

**调用示例**

```python
x_local = asc.LocalTensor(
    dtype=asc.float16,
    pos=asc.TPosition.VECIN,
    addr=0,
    tile_size=512,
)
x_gm = asc.GlobalTensor()
x_gm.set_global_buffer(x)

asc.dump_acc_chk_point(tensor=x_local, index=0, count_off=1, dump_size=5)
asc.dump_acc_chk_point(tensor=x_gm, index=0, count_off=1, dump_size=5)
```
