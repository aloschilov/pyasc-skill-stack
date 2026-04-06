# asc.language.basic.select

### asc.language.basic.select(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mask: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: float, sel_mode: SelMode, count: int) → None

### asc.language.basic.select(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mask: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mode: SelMode, count: int) → None

### asc.language.basic.select(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mask: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: float, sel_mode: SelMode, mask: List[int], repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.select(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mask: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: float, sel_mode: SelMode, mask: int, repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.select(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mask: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_times: int, repeat_params: BinaryRepeatParams) → None

### asc.language.basic.select(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mask: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mode: SelMode, mask: List[int], repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.select(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mask: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), sel_mode: SelMode, mask: int, repeat_times: int, repeat_params: BinaryRepeatParams, is_set_mask: bool = True) → None

### asc.language.basic.select(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_times: int, repeat_params: BinaryRepeatParams, sel_mode: SelMode) → None

给定两个源操作数src0和src1，根据sel_mask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。

**对应的Ascend C函数原型**

```c++
template <typename T, typename U>
__aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                               const LocalTensor<T>& src0, T src1, SELMODE selMode, uint32_t count)
```

```c++
template <typename T, typename U>
__aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                               const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                               SELMODE selMode, uint32_t count)
```

```c++
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                               const LocalTensor<T>& src0, T src1, SELMODE selMode, uint64_t mask[],
                               uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

```c++
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                               const LocalTensor<T>& src0, T src1, SELMODE selMode, uint64_t mask,
                               uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

```c++
template <typename T, typename U>
__aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                               const LocalTensor<T>& src0, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

```c++
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                               const LocalTensor<T>& src0, const LocalTensor<T>& src1, SELMODE selMode, uint64_t mask[],
                               uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

```c++
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask,
                               const LocalTensor<T>& src0, const LocalTensor<T>& src1, SELMODE selMode, uint64_t mask,
                               uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

```c++
template <typename T, SELMODE selMode>
__aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                               uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
```

**参数说明**

- dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- sel_mask: 选取mask，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src1: 源操作数。
  - 当selMode为模式0或模式2时：类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
  - 当selMode为模式1时，类型为T，标量数据类型。
- sel_mode: SELMODE类型，表示指令模式。
  - VSEL_CMPMASK_SPR: 模式0，根据selMask在两个tensor中选取元素，selMask中有效数据的个数存在限制，具体取决于源操作数的数据类型。
  - VSEL_TENSOR_SCALAR_MODE: 模式1，根据selMask在1个tensor和1个scalar标量中选取元素，selMask无有效数据限制。
  - VSEL_TENSOR_TENSOR_MODE: 模式2，根据selMask在两个tensor中选取元素，selMask无有效数据限制。
- count: 参与计算的元素个数。
- mask: 用于控制每次迭代内参与计算的元素。
- repeat_times: 重复迭代次数。
- repeat_params: 控制操作数地址步长的参数。
- is_set_mask: 是否在接口内部设置mask。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。
- 对于模式1和模式2，使用时需要预留8K的Unified Buffer空间，作为接口的临时数据存放区。

**调用示例**

- tensor 前 n 个数据计算样例（模式 0 / 模式 2）
  ```python
  asc.select(
      z_local,
      y_local,
      x_local,
      p_local,
      sel_mode=asc.SelMode.VSEL_CMPMASK_SPR,
      count=512
  )
  ```
- tensor 前 n 个数据计算样例（模式 1）
  ```python
  asc.select(
      z_local,
      y_local,
      x_local,
      0.0,
      sel_mode=asc.SelMode.VSEL_TENSOR_SCALAR_MODE,
      count=512
  )
  ```
- tensor 高维切分计算样例 —— 标量 + mask 逐 bit（模式 1）
  ```python
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.select(
      z_local,
      y_local,
      x_local,
      0.0,
      sel_mode=asc.SelMode.VSEL_TENSOR_SCALAR_MODE,
      mask=mask,
      repeat_times=1,
      repeat_params=params
  )
  ```
- tensor 高维切分计算样例 —— 标量 + mask 连续（模式 1）
  ```python
  mask = 512
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.select(
      z_local,
      y_local,
      x_local,
      0.0,
      sel_mode=asc.SelMode.VSEL_TENSOR_SCALAR_MODE,
      mask=mask,
      repeat_times=1,
      repeat_params=params
  )
  ```
- tensor 高维切分计算样例 —— 不传入 mask（需配合寄存器 mask 使用，模式 1）
  ```python
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.select(
      z_local,
      y_local,
      x_local,
      repeat_times=1,
      repeat_params=params
  )
  ```
- tensor 高维切分计算样例 —— Tensor + mask 逐 bit（模式 0 / 模式 2）
  ```python
  uint64_max = 2**64 - 1
  mask = [uint64_max, uint64_max]
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.select(
      z_local,
      y_local,
      x_local,
      p_local,
      sel_mode=asc.SelMode.VSEL_CMPMASK_SPR,
      mask=mask,
      repeat_times=1,
      repeat_params=params
  )
  ```
- tensor 高维切分计算样例 —— Tensor + mask 连续（模式 0 / 模式 2）
  ```python
  mask = 512
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.select(
      z_local,
      y_local,
      x_local,
      p_local,
      sel_mode=asc.SelMode.VSEL_CMPMASK_SPR,
      mask=mask,
      repeat_times=1,
      repeat_params=params
  )
  ```
- tensor 高维切分计算样例 —— 寄存器版本（无 selMask，模式 0 / 模式 2）
  ```python
  params = asc.BinaryRepeatParams(1, 1, 1, 8, 8, 8)
  asc.select(
      z_local,
      x_local,
      p_local,
      repeat_times=1,
      repeat_params=params,
      sel_mode=asc.SelMode.VSEL_CMPMASK_SPR
  )
  ```
