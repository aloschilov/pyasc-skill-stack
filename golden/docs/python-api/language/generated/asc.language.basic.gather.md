# asc.language.basic.gather

### asc.language.basic.gather(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), src_offset: [LocalTensor](../core.md#asc.language.core.LocalTensor), src_base: int, mask: int, repeat_times: int, dst_rep_stride: int) → None

### asc.language.basic.gather(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), src_offset: [LocalTensor](../core.md#asc.language.core.LocalTensor), src_base: int, mask: List[int], repeat_times: int, dst_rep_stride: int) → None

### asc.language.basic.gather(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), src_offset: [LocalTensor](../core.md#asc.language.core.LocalTensor), src_base: int, count: int) → None

给定输入的张量和一个地址偏移张量，本接口根据偏移地址将输入张量按元素收集到结果张量中。

**对应的Ascend C函数原型**

- tensor前n个数据计算
  ```c++
  template <typename T>
  __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr, const uint32_t count)
  ```
- tensor高维切分计算
  - mask逐bit模式
    ```c++
    template <typename T>
    __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                        const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr,
                        const uint64_t mask[], const uint8_t repeatTime, const uint16_t dstRepStride)
    ```
  - mask连续模式
    ```c++
    template <typename T>
    __aicore__ inline void Gather(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                        const LocalTensor<uint32_t>& srcOffset, const uint32_t srcBaseAddr,
                        const uint64_t mask, const uint8_t repeatTime, const uint16_t dstRepStride)
    ```

**参数说明**

- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- src: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- src_offset：每个元素在src中对应的地址偏移。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
  该偏移量相对于src的起始基地址而言。单位为Bytes。取值要求如下：
  - 取值应保证src元素类型位宽对齐。
  - 偏移地址后不能超出UB大小数据的范围。
  - 地址偏移的取值范围：不能超出uint32_t的范围。
- src_base：src的起始基地址，用于指定Gather操作中源操作数的起始位置，单位为Bytes。取值应保证src元素类型位宽对齐，否则会导致非预期行为。
- count：执行处理的数据个数。
- mask：
  控制每次迭代内参与计算的元素。
  - **逐bit模式**：mask为数组形式。数组长度和数组元素的取值范围和操作数的数据类型有关。可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。
    - 操作数 16 位：数组长度 2，mask[0], mask[1] ∈ [0, 2⁶⁴-1]，且不能同时为 0
    - 操作数 32 位：数组长度 1，mask[0] ∈ (0, 2⁶⁴-1]
    - 操作数 64 位：数组长度 1，mask[0] ∈ (0, 2³²-1]
    - 例如：mask = [8, 0]，表示仅第 4 个元素参与计算
  - **连续模式**：mask为整数形式。表示前面连续多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。
    - 操作数 16 位：mask ∈ [1, 128]
    - 操作数 32 位：mask ∈ [1, 64]
    - 操作数 64 位：mask ∈ [1, 32]
- repeat_time：指令迭代次数，每次迭代完成8个datablock（32Bytes）的数据收集，数据范围：repeat_time∈[0,255]。
- dst_rep_stride：相邻迭代间的地址步长，单位是datablock（32Bytes）。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 操作数地址重叠约束请参考 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址重叠约束”。

**调用示例**

```python
z_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=512)
src_offset = asc.LocalTensor(dtype=asc.uint32, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
x_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
asc.gather(z_local, x_local, src_offset, src_base=0, count=512)
asc.gather(z_local, x_local, src_offset, src_base=0, mask=512, repeat_times=1, dst_rep_stride=8)
uint64_max = 2**64 - 1
mask_bits = [uint64_max, uint64_max]
asc.gather(z_local, x_local, src_offset, src_base=0, mask=mask_bits, repeat_times=1, dst_rep_stride=8)
```
