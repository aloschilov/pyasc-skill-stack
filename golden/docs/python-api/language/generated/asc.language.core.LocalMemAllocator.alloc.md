# asc.language.core.LocalMemAllocator.alloc

#### LocalMemAllocator.alloc(pos: TPosition, data_type: DataType, tile_size: ConstExpr[int]) → [LocalTensor](../core.md#asc.language.core.LocalTensor)

#### LocalMemAllocator.alloc(pos: TPosition, data_type: DataType, tile_size: int) → [LocalTensor](../core.md#asc.language.core.LocalTensor)

根据用户指定的逻辑位置、数据类型、数据长度返回对应的 LocalTensor 对象。

**对应的Ascend C函数原型**

```c++
// 原型1：tileSize为模板参数
// 当tileSize为常量时，建议使用此接口，以获得更优的性能
template <TPosition pos, class DataType, uint32_t tileSize>
__aicore__ inline LocalTensor<DataType> Alloc()

// 原型2：tileSize为接口入参
// 当tileSize为动态参数时使用此接口
template <TPosition pos, class DataType>
LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize)
```

**参数说明**

- pos：TPosition 位置，需要符合 LocalMemAllocator 中指定的 Hardware 物理位置。
- data_type：LocalTensor 的数据类型，只支持基础数据类型，当前不支持 TensorTrait 类型。
- tile_size：LocalTensor 的元素个数，其数量不应超过当前物理位置剩余的内存空间。

**返回值说明**

根据用户输入构造的 LocalTensor 对象。

**约束说明**

tile_size 不应超过当前物理位置剩余的内存空间。剩余的内存空间可以通过物理内存最大值与当前可用内存地址（get_cur_addr 返回值）的差值来计算。

**调用示例**

```python
allocator = asc.LocalMemAllocator()

# 用户指定逻辑位置 VECIN，float 类型，Tensor 中有 1024 个元素
tensor1 = allocator.alloc(asc.TPosition.VECIN, float, 1024)

# 用户指定逻辑位置 VECIN，float 类型，Tensor 中有 tileLength 个元素
tile_length = 512
tensor2 = allocator.alloc(asc.TPosition.VECIN, float, tile_length)
```
