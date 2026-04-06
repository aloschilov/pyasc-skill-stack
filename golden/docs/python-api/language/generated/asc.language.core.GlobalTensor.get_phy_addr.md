# asc.language.core.GlobalTensor.get_phy_addr

#### GlobalTensor.get_phy_addr() → GlobalAddress

#### GlobalTensor.get_phy_addr(offset: int) → GlobalAddress

获取全局数据的地址。

**对应的Ascend C函数原型**

```c++
__aicore__ inline const __gm__ PrimType* GetPhyAddr() const
```

```c++
__aicore__ inline __gm__ PrimType* GetPhyAddr(const uint64_t offset) const
```

**参数说明**

- offset：偏移的元素个数，用于指定数据的位置。

**返回值说明**

全局数据的地址。
