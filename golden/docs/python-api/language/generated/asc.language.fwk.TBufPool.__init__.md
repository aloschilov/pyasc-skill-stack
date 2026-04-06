# asc.language.fwk.TBufPool._\_init_\_

#### TBufPool.\_\_init_\_(pos: TPosition | None, buf_id_size: int) → None

#### TBufPool.\_\_init_\_(handle: Value) → None

创建TBufPool对象时，初始化数据成员。

**对应的Ascend C函数原型**

```c++
template <TPosition pos, uint32_t bufIDSize = defaultBufIDSize>
__aicore__ inline TBufPool();
```

**参数说明**

- pos：TBufPool逻辑位置，可以为VECIN、VECOUT、VECCALC、A1、B1、C1。
- buf_id_size：TBufPool可分配Buffer数量，默认为4，不超过16。
  对于非共享模式的资源分配，在本TBufPool上再次申请TBufPool时，申请的buf_id_size不能超过原TBufPool剩余可用的Buffer数量；
  对于共享模式的资源分配，在本TBufPool上再次申请TBufPool时，申请的buf_id_size不能超过原TBufPool设置的Buffer数量。

**约束说明**

无。
