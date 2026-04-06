# asc.language.adv.Matmul.iterate_all

#### Matmul.iterate_all(tensor: BaseTensor, en_atomic: int = 0, sync: bool = True, en_sequential_write: bool | None = None, wait_iterate_all: bool | None = None, fake_msg: bool | None = None) → None

调用一次iterate_all，会计算出singleCoreM \* singleCoreN大小的C矩阵。

**对应的Ascend C函数原型**

```c++
template <bool sync = true>
__aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, bool enSequentialWrite = false, bool waitIterateAll = false, bool fakeMsg = false)
```

```c++
template <bool sync = true>
__aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, uint8_t enAtomic = 0)
```

**参数说明**

- tensor: C矩阵，类型为GlobalTensor或LocalTensor。
- en_atomic: 是否开启Atomic操作，默认值为0。
- sync: 设置同步或者异步模式。
- en_sequential_write: 是否开启连续写模式，仅支持输出到Global Memory场景。
- wait_iterate_all: 是否需要通过wait_iterate_all接口等待iterate_all执行结束，仅支持异步输出到Global Memory场景。
- fake_msg: 仅在IBShare场景和IntraBlockPartSum场景使用，仅在支持输出到Global Memory场景。

**约束说明**

- 传入的C矩阵地址空间大小需要保证不小于single_core_m \* single_core_n个元素。

**调用示例**

```python
asc.adv.register_matmul(pipe, workspace, mm, tiling)
mm.set_tensor_a(gm_a)
mm.set_tensor_b(gm_b)
mm.set_bias(gm_bias)
mm.iterate_all(gm_c)
```
