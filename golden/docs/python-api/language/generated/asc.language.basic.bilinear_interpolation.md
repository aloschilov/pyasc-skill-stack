# asc.language.basic.bilinear_interpolation

### asc.language.basic.bilinear_interpolation(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0_offset: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: int, h_repeat: int, repeat_mode: bool, dst_blk_stride: int, v_r_offset: int, v_repeat: int, shared_tmp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

### asc.language.basic.bilinear_interpolation(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0_offset: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), mask: List[int], h_repeat: int, repeat_mode: bool, dst_blk_stride: int, v_r_offset: int, v_repeat: int, shared_tmp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

分为水平迭代和垂直迭代。
每个水平迭代顺序地从src0_offset读取8个偏移值，表示src0的偏移，每个偏移值指向src0的一个data_block的起始地址，如果repeat_mode=false，从src1中取一个值，
与src0中8个data_block中每个值进行乘操作；如果repeat_mode=true，从src1中取8个值，按顺序与src0中8个data_block中的值进行乘操作，
最后当前迭代的dst结果与前一个dst结果按data_block进行累加，存入目的地址，在同一个水平迭代内dst地址不变。
然后进行垂直迭代，垂直迭代的dst起始地址为上一轮垂直迭代的dst起始地址加上v_r_offset，本轮垂直迭代占用dst空间为dst起始地址之后的8个data_block，每轮垂直迭代进行h_repeat次水平迭代。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void BilinearInterpolation(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
                    const LocalTensor<uint32_t> &src0Offset, const LocalTensor<T> &src1, uint64_t mask[],
                    uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset,
                    uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer)
```

```c++
template <typename T>
__aicore__ inline void BilinearInterpolation(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
                    const LocalTensor<uint32_t> &src0Offset, const LocalTensor<T> &src1, uint64_t mask,
                    uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset,
                    uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer)
```

**参数说明**

- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src0, src1：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- count：参与计算的元素个数。
- mask：用于控制每次迭代内参与计算的元素。
- repeat_times：重复迭代次数。
- params：控制操作数地址步长的参数。

**调用示例**

- 接口样例-mask连续模式
  ```python
  mask = 128;         # mask连续模式
  hRepeat = 2;        # 水平迭代2次
  repeatMode = false; # 迭代模式
  dstBlkStride = 1;   # 单次迭代内数据连续写入
  vROffset = 128;     # 相邻迭代间数据连续写入
  vRepeat = 2;        # 垂直迭代2次
  asc.bilinear_interpolation(dst_local, src0_local, src0_offset_local, src1_local, mask, hRepeat, repeatMode,
  dstBlkStride, vROffset, vRepeat, tmpLocal)
  ```
- 接口样例-mask逐bit模式
  ```python
  mask = [uint64_max, uint64_max];         # mask逐bit模式
  hRepeat = 2;        # 水平迭代2次
  repeatMode = false; # 迭代模式
  dstBlkStride = 1;   # 单次迭代内数据连续写入
  vROffset = 128;     # 相邻迭代间数据连续写入
  vRepeat = 2;        # 垂直迭代2次
  asc.bilinear_interpolation(dst_local, src0_local, src0_offset_local, src1_local, mask, hRepeat, repeatMode,
  dstBlkStride, vROffset, vRepeat, tmpLocal)
  ```
