# asc.language.core.LocalTensor.get_phy_addr

#### LocalTensor.get_phy_addr() → int

#### LocalTensor.get_phy_addr(offset: int) → int

返回LocalTensor的地址或指定偏移量后的地址。

**对应的Ascend C函数原型**

```c++
__aicore__ inline uint64_t GetPhyAddr() const
__aicore__ inline uint64_t GetPhyAddr(const uint32_t offset) const
```

**参数说明**

- offset：偏移量。

**返回值说明**

LocalTensor的地址或指定偏移量后的地址。

**调用示例**

```python
real_addr = input_local.get_phy_addr()
```
