# asc.language.adv.Matmul.get_tensor_c

#### Matmul.get_tensor_c(tensor: BaseTensor, en_atomic: int = 0, en_sequential_write: bool = False, sync: bool = True, optional_tensor: BaseTensor | None = None) → None

#### Matmul.get_tensor_c(en_atomic: int = 0, en_sequential_write: bool = False, sync: bool = True) → [GlobalTensor](../core.md#asc.language.core.GlobalTensor)

本接口和iterate接口配合使用，用于在调用iterate完成迭代计算后，
根据MatmulConfig参数中的ScheduleType取值获取一块或两块baseM \* baseN大小的矩阵分片。

**对应的Ascend C函数原型**

```c++
template <bool sync = true>
__aicore__ inline void GetTensorC(const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0, bool enSequentialWrite = false)
```

```c++
template <bool sync = true>
__aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false)
```

```c++
template <bool sync = true>
__aicore__ inline void GetTensorC(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0, bool enSequentialWrite = false)
```

```c++
template <bool sync = true>
__aicore__ inline GlobalTensor<DstT> GetTensorC(uint8_t enAtomic = 0, bool enSequentialWrite = false)
```

**参数说明**

- tensor: 取出C矩阵到VECIN/GM。
- en_atomic: 是否开启Atomic操作，默认值为0。
- en_sequential_write: 是否开启连续写模式，默认值false。
- sync: 设置同步或者异步模式。
- optional_tensor: 取出C矩阵到VECIN，此参数使能时，tensor类型必须为GlobalTensor。

**约束说明**

- 传入的C矩阵地址空间大小需要保证不小于base_m \* base_n。
- 异步场景时，需要使用一块临时空间来缓存iterate计算结果，调用get_tensor_c时会在该临时空间中获取C的矩阵分片。临时空间通过set_workspace接口进行设置。set_workspace接口需要在iterate接口之前调用。
- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

**调用示例**

- 获取C矩阵，输出至VECIN
  ```python
  # 同步模式样例
  while mm.iterate() as count:
  mm.get_tensor_c(tensor=ub_cmatrix)
  # 异步模式样例
  mm.iterate(sync=False)
  # 其他操作
  for i in range(single_m // base_m * single_n // base_n):
      mm.get_tensor_c(tensor=ub_cmatrix, sync=False)
  ```
- 获取C矩阵，输出至GM，同步模式样例
  ```python
  while mm.iterate() as count:
      mm.get_tensor_c(tensor=gm)
  ```
- 获取C矩阵，同时输出至GM和VECIN，同步模式样例
  ```python
  while mm.iterate() as count:
      mm.get_tensor_c(tensor=gm, optional_tensor=ub_cmatrix)
  ```
- 获取API接口返回的GM上的C矩阵，手动拷贝至UB，异步模式样例
  ```python
  # base_m * base_n = 128 * 256
  mm.set_tensor_a(gm_a)
  mm.set_tensor_b(gm_b)
  mm.set_tail(single_m, single_n, single_k)
  mm.iterate(sync=False)
  for i in range(single_m // base_m * single_n // base_n):
      global = mm.get_tensor_c(sync=False)
      for j in range(4):
          local = que.alloc_tensor(dtype=asc.half)
          asc.data_copy(local, global[64* 128 * i:], count=64 * 128)
  ```
