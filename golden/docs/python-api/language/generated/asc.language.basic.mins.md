# asc.language.basic.mins

### asc.language.basic.mins(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: int | float, count: int, is_set_mask: bool = True) → None

### asc.language.basic.mins(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: int | float, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.mins(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), scalar: int | float, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) → None

源操作数矢量内每个元素与标量相比，如果比标量大，则取标量值，比标量的值小，则取源操作数。

**对应的Ascend C函数原型**

```c++
template <typename T, bool isSetMask = true>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                    const T& scalarValue, const int32_t& calCount)
```

```c++
template <typename T, bool isSetMask = true>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                    const T& scalarValue, uint64_t mask[], const uint8_t repeatTimes,
                                    const UnaryRepeatParams& repeatParams)
```

```c++
template <typename T, bool isSetMask = true>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                    const T& scalarValue, uint64_t mask, const uint8_t repeatTimes,
                                    const UnaryRepeatParams& repeatParams)
```

**参数说明**

- is_set_mask：是否在接口内部设置mask模式和mask值。
- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- scalar：源操作数，数据类型需要与目的操作数中的元素类型保持一致。
- count：参与计算的元素个数。
- mask：用于控制每次迭代内参与计算的元素。
- repeat_times：重复迭代次数。
- params：元素操作控制结构信息。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  mask = 128
  scalar = 2
  # repeat_times = 4，一次迭代计算128个数，共计算512个数
  # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
  # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
  params = asc.UnaryRepeatParams(1, 1, 8, 8)
  asc.mins(dst, src, scalar, mask=mask, repeat_times=4, repeat_params=params)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  mask = [uint64_max, uint64_max]
  scalar = 2
  # repeat_times = 4，一次迭代计算128个数，共计算512个数
  # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
  # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
  params = asc.UnaryRepeatParams(1, 1, 8, 8)
  asc.mins(dst, src, scalar, mask=mask, repeat_times=4, repeat_params=params)
  ```
- tensor前n个数据计算样例
  ```python
  asc.mins(dst, src, scalar, count=512)
  ```
