# asc.language.basic.brcb

### asc.language.basic.brcb(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_times: int, repeat_params: BrcbRepeatParams) → None

给定一个输入张量，每一次取输入张量中的8个数填充到结果张量的8个datablock（32Bytes）中去，每个数对应一个datablock。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void Brcb(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                            const uint8_t repeatTime, const BrcbRepeatParams& repeatParams)
```

**参数说明**

- dst：目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。
- src0：源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要32字节对齐。数据类型和dst保持一致。
- repeat_time：指令迭代次数，每次迭代完成8个datablock的数据收集，数据范围：repeat_time∈[0,255]。
- repeat_params：用于控制指令迭代的相关参数。

**约束说明**

- 不支持src0与dst为同一块内存地址。

**调用示例**

```python
brcb_params = asc.BrcbRepeatParams(1, 8)
asc.brcb(x_buf, y_buf, 2, brcb_params)
```
