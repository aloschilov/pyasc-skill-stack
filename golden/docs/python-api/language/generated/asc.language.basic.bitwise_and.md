# asc.language.basic.bitwise_and

### asc.language.basic.bitwise_and(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), count: int, is_set_mask: bool = True) → None

### asc.language.basic.bitwise_and(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: int, repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.bitwise_and(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: List[int], repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

每对elements按位与运算。命名为 bitwise_and 避免与Python关键字重名。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void And(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                    const LocalTensor<T>& src1, const int32_t& count);
```

```c++
template <typename T, bool isSetMask = true>
 __aicore__ inline void And(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                    const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTimes,
                                    const BinaryRepeatParams& repeatParams);
```

```c++
template <typename T, bool isSetMask = true>
__aicore__ inline void And(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                    const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTimes,
                                    const BinaryRepeatParams& repeatParams);
```

**参数说明**

- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src0, src1：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- count：参与计算的元素个数。
- mask：用于控制每次迭代内参与计算的元素。
- repeat_times：重复迭代次数。
- params：控制操作数地址步长的参数。
- is_set_mask: 是否在接口内部设置mask。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。
- 使用整个tensor参与计算接口符号重载时，运算量为目的LocalTensor的总长度。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  mask = 128
  # repeat_times = 4，一次迭代计算128个数，共计算512个数
  # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
  # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.bitwise_and(dst, src0, src1, mask=mask, repeat_times=4, repeat_params=params)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  mask = [uint64_max, uint64_max]
  # repeat_times = 4，一次迭代计算128个数，共计算512个数
  # dst_blk_stride, src0_blk_stride, src1_blk_stride = 1，单次迭代内数据连续读取和写入
  # dst_rep_stride, src0_rep_stride, src1_rep_stride = 8，相邻迭代间数据连续读取和写入
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.bitwise_and(dst, src0, src1, mask=mask, repeat_times=4, repeat_params=params)
  ```
- tensor前n个数据计算样例
  ```python
  asc.bitwise_and(dst, src0, src1, count=512)
  ```
