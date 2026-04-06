# asc.language.basic.cast

### asc.language.basic.cast(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), round_mode: RoundMode, count: int) → None

### asc.language.basic.cast(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), round_mode: RoundMode, mask: int, repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.cast(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), round_mode: RoundMode, mask: List[int], repeat_times: int, repeat_params: UnaryRepeatParams, is_set_mask: bool = True) → None

根据源操作数和目的操作数Tensor的数据类型进行精度转换。

**对应的Ascend C函数原型**

```c++
template <typename T, typename U>
__aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& round_mode, const uint32_t count)
```

```c++
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& round_mode, const uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

```c++
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& round_mode, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

**参数说明**

- T: 目的操作数数据类型。支持的数据类型根据产品型号不同而有所区别。
- U: 源操作数数据类型。支持的数据类型根据产品型号不同而有所区别。
- is_set_mask: 是否在接口内部设置mask。true表示在接口内部设置mask，false表示在接口外部设置mask。
- dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要32字节对齐。
- src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。起始地址需要32字节对齐。
- round_mode: 精度转换处理模式，类型是RoundMode枚举类，控制精度转换处理模式：
  - CAST_NONE: 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入。
  - CAST_RINT: rint，四舍六入五成双舍入。
  - CAST_FLOOR: floor，向负无穷舍入。
  - CAST_CEIL: ceil，向正无穷舍入。
  - CAST_ROUND: round，四舍五入舍入。
  - CAST_TRUNC: trunc，向零舍入。
  - CAST_ODD: Von Neumann rounding，最近邻奇数舍入。
- count: 参与计算的元素个数。
- mask/mask[]: 用于控制每次迭代内参与计算的元素。支持连续模式和逐bit模式。
- repeat_time: 重复迭代次数，取值范围[0,255]。
- repeat_params: 控制操作数地址步长的参数，UnaryRepeatParams类型。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。特别地，对于长度较小的数据类型转换为长度较大的数据类型时，地址重叠可能会导致结果错误。
- 每个repeat能处理的数据量取决于数据精度、AI处理器型号。
- 当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。
- 当dst或src为int4b_t时，申请Tensor空间时只需申请相同数量的int8_t数据空间的一半。
- 当dst或src为int4b_t时，tensor高维切分计算接口的连续模式的mask与tensor前n个数据计算接口的count必须为偶数；对于逐bit模式，对应同一字节的相邻两个比特位的数值必须一致。

**调用示例**

- tensor高维切分计算样例-mask连续模式
  ```python
  mask = 256 // asc.int32.sizeof()
  params = asc.UnaryRepeatParams(1, 1, 8, 4)
  asc.cast(dst, src, asc.RoundMode.CAST_CEIL, mask=mask, repeat_times=8, params=params)
  ```
- tensor高维切分计算样例-mask逐bit模式
  ```python
  mask = [0, 0xFFFFFFFFFFFFFFFF]
  params = asc.UnaryRepeatParams(1, 1, 8, 4)
  asc.cast(dst, src, asc.RoundMode.CAST_CEIL, mask=mask, repeat_times=8, params=params)
  ```
- tensor前n个数据计算样例
  ```python
  asc.cast(dst, src, asc.RoundMode.CAST_CEIL, count=512)
  ```
