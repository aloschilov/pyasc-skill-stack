# asc.language.basic.compare

### asc.language.basic.compare(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), cmp_mode: CMPMODE, count: int) → None

### asc.language.basic.compare(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), cmp_mode: CMPMODE, mask: int, repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.compare(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), cmp_mode: CMPMODE, mask: List[int], repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.compare(src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), cmp_mode: CMPMODE, mask: List[int], repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.compare(src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), cmp_mode: CMPMODE, mask: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

逐元素比较两个tensor大小，如果比较后的结果为真，则输出的结果的对应比特位为1，否则为0。可将结果存入寄存器中。

**对应的Ascend C函数原型**

```c++
template <typename T, typename U>
__aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                const LocalTensor<T>& src1, CMPMODE cmpMode, uint32_t count);
```

```c++
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Copmare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask[],
                                uint8_t repeatTimes, const BinaryRepeatParams& repeatParams);
```

```c++
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
                                const LocalTensor<T>& src1, CMPMODE cmpMode, const uint64_t mask,
                                uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
```

```c++
template <typename T, bool isSetMask = true>
__aicore__ inline void Copmare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode,
                                const uint64_t mask[], const BinaryRepeatParams& repeatParams);
```

```c++
template <typename T, bool isSetMask = true>
__aicore__ inline void Compare(const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE cmpMode,
                                const uint64_t mask, const BinaryRepeatParams& repeatParams);
```

**参数说明**

- dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src0, src1: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- cmp_mode: CMPMODE类型，表示比较模式。
  - LT: src0小于（less than）src1
  - GT: src0大于（greater than）src1
  - GE: src0大于或等于（greater than or equal to）src1
  - EQ: src0等于（equal to）src1
  - NE: src0不等于（not equal to）src1
  - LE: src0小于或等于（less than or equal to）src1
- count: 参与计算的元素个数。
- mask: 用于控制每次迭代内参与计算的元素。
- repeat_times: 重复迭代次数。
- repeat_params: 控制操作数地址步长的参数。
- is_set_mask: 是否在接口内部设置mask。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- dst按照小端顺序排序成二进制结果，对应src中相应位置的数据比较结果。
- 使用整个tensor参与计算的运算符重载功能，src0和src1需满足256字节对齐；使用tensor前n个数据参与计算的接口，设置count时，需要保证count个元素所占空间256字节对齐。
- 将结果存入寄存器的接口没有repeat输入，repeat默认为1，即一条指令计算256B的数据。
- 将结果存入寄存器的接口会将结果写入128bit的cmpMask寄存器中，可以用GetCmpMask接口获取寄存器保存的数据。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  mask = 128
  # repeat_times = 1，一次迭代计算128个数
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.compare(dst, src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, repeat_params=params)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  mask = [uint64_max, uint64_max]
  # repeat_times = 1，一次迭代计算128个数
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.compare(dst, src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_times=1, repeat_params=params)
  ```
- tensor前n个数据计算样例
  ```python
  asc.compare(dst, src0, src1, cmp_mode=asc.CMPMODE.LT, count=512)
  ```
- tensor高维切分计算样例-mask连续模式，结果存入寄存器中
  ```python
  mask = 128
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.compare(src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_params=params)
  ```
- tensor高维切分计算样例-mask逐bit模式，结果存入寄存器中
  ```python
  mask = [uint64_max, uint64_max]
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.compare(src0, src1, cmp_mode=asc.CMPMODE.LT, mask=mask, repeat_params=params)
  ```
