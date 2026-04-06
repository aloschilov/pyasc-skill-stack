# asc.language.core.LocalMemAllocator.get_cur_addr

#### LocalMemAllocator.get_cur_addr() → int

返回当前物理位置空闲的起始地址。

**对应的Ascend C函数原型**

```c++
__aicore__ inline uint32_t GetCurAddr() const
```

**参数说明**

无。

**返回值说明**

当前物理位置空闲的起始地址，范围为[0，物理内存最大值)。

**调用示例**

```python
allocator = asc.LocalMemAllocator()
# 默认的物理位置为UB，由于从0地址开始分配，下面的打印结果为0
addr = allocator.get_cur_addr()
asc.printf("current addr is %u\n", addr)
```
