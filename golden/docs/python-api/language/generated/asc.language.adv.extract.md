# asc.language.adv.extract

### asc.language.adv.extract(dst_value: [LocalTensor](../core.md#asc.language.core.LocalTensor), dst_index: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int) → None

处理Sort的结果数据，输出排序后的value和index。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void Extract(const LocalTensor<T> &dstValue,
                                const LocalTensor<uint32_t> &dstIndex,
                                const LocalTensor<T> &sorted,
                                const int32_t repeatTime)
```

**参数说明**

- dst_value：排序结果的数值部分。
- dst_index：排序结果的索引部分。
- src：源操作数。
- repeat_time：重复迭代次数。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
dst_value_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
dst_index_local = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
sort_tmp_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=256)
asc.extract(dst_value=dst_value_local, dst_index=dst_index_local, src=sort_tmp_local, repeat_time=4)
```
