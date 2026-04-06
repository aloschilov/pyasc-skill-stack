# asc.language.basic.sqrt

### asc.language.basic.sqrt(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int) → None

### asc.language.basic.sqrt(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: int, repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.sqrt(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) → None

按元素做开方。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                    const int32_t& calCount)
```

```c++
template <typename T, bool isSetMask = true>
__aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                    uint64_t mask[], const uint8_t repeatTimes,
                                    const UnaryRepeatParams& repeatParams)
```

```c++
template <typename T, bool isSetMask = true>
__aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                    uint64_t mask, const uint8_t repeatTimes,
                                    const UnaryRepeatParams& repeatParams)
```

**参数说明**

- is_set_mask：是否在接口内部设置mask。
- dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- count: 参与计算的元素个数。
- mask: 用于控制每次迭代内参与计算的元素。
- repeat_times: 重复迭代次数。
- params: 控制操作数地址步长的参数。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  mask = 256 // asc.half.sizeof()
  # repeat_times = 4，一次迭代计算128个数，共计算512个数
  # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
  # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
  params = asc.UnaryRepeatParams(1, 1, 8, 8)
  asc.sqrt(dst, src, mask=mask, repeat_times=4, repeat_params=params)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  mask = [uint64_max, uint64_max]
  # repeat_times = 4，一次迭代计算128个数，共计算512个数
  # dst_blk_stride, src_blk_stride = 1，单次迭代内数据连续读取和写入
  # dst_rep_stride, src_rep_stride = 8，相邻迭代间数据连续读取和写入
  params = asc.UnaryRepeatParams(1, 1, 8, 8)
  asc.sqrt(dst, src, mask=mask, repeat_times=4, repeat_params=params)
  ```
- tensor前n个数据计算样例
  ```python
  asc.sqrt(dst, src, count=512)
  ```
