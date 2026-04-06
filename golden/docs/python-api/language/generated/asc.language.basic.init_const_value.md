# asc.language.basic.init_const_value

### asc.language.basic.init_const_value(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), init_const_value_params: InitConstValueParams) → None

将特定TPosition的LocalTensor初始化为某一具体数值。

**对应的Ascend C函数原型**

```c++
template <typename T, typename U = PrimT<T>, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void InitConstValue(const LocalTensor<T> &dst, const InitConstValueParams<U> &initConstValueParams)
```

**参数说明**

- dst：目的操作数，类型为 LocalTensor，支持的 TPosition 为 A1/A2/B1/B2。如果TPosition为A1/B1，起始地址需要满足32B对齐；如果TPosition为A2/B2，起始地址需要满足512B对齐。
- init_const_value_params：初始化相关参数，类型为InitConstValueParams。仅支持配置迭代次数（repeat_times）和初始化值（init_value）场景下，其他参数配置无效。每次迭代处理固定数据量（512字节），迭代间无间隔。支持配置所有参数场景下，支持配置迭代次数（repeat_times）、初始化值（init_value）、每个迭代处理的数据块个数（block_num）和迭代间间隔（dst_gap）。
  - repeat_times：迭代次数。默认值为0。
    - 仅支持配置迭代次数（repeat_times）和初始化值（init_value）场景下，repeat_times∈[0, 255]。
    - 支持配置所有参数场景下，repeat_times∈[0, 32767]。
  - block_num：每次迭代初始化的数据块个数，取值范围：block_num∈[0, 32767] 。默认值为0。
    - dst的位置为A1/B1时，每一个block（数据块）大小是32B；
    - dst的位置为A2/B2时，每一个block（数据块）大小是512B。
  - dst_gap：目的操作数前一个迭代结束地址到后一个迭代起始地址之间的距离。取值范围：dst_gap∈[0, 32767] 。默认值为0。
    - dst的位置为A1/B1时，单位是32B；
    - dst的位置为A2/B2时，单位是512B。
  - init_value：初始化的value值，支持的数据类型与dst保持一致。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
import asc
dst = asc.LocalTensor(dtype=asc.float32, pos=asc.TPosition.A1, addr=0, tile_size=128)
params = asc.InitConstValueParams(repeat_times=1, block_num=2, dst_gap=0, init_value=2.2)
asc.init_const_value(dst=dst, init_const_value_params=params)
```
