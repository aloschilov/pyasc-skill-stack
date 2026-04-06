# asc.language.basic.load_data_with_sparse

### asc.language.basic.load_data_with_sparse(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), idx: [LocalTensor](../core.md#asc.language.core.LocalTensor), load_data_params: LoadData2DParams) → None

用于搬运存放在B1里的512B的稠密权重矩阵到B2里，同时读取128B的索引矩阵用于稠密矩阵的稀疏化。
索引矩阵的数据类型为int2，需要拼成int8的数据类型，再传入接口。
索引矩阵在一个int8的地址中的排布是逆序排布的，例如：索引矩阵1 2 0 1 0 2 1 0，
在地址中的排布为1 0 2 1 0 1 2 0，其中1 0 2 1（对应索引矩阵前四位1 2 0 1）为一个int8，0 1 2 0（对应索引矩阵后四位0 2 1 0）为一个int8。

**对应的Ascend C函数原型**

```c++
template <typename T = int8_t, typename U = uint8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int8_t>::value, bool>::type = true, typename Std::enable_if<Std::is_same<PrimT<U>, uint8_t>::value, bool>::type = true>
__aicore__ inline void LoadDataWithSparse(const LocalTensor<T> &dst, const LocalTensor<T> &src, const LocalTensor<U> &idx, const LoadData2dParams &loadDataParam)
```

**参数说明**

- dst：输出，目的操作数，类型为LocalTensor，支持的TPosition为B2，LocalTensor的起始地址需要512字节对齐。支持的数据类型为int8_t。数据连续排列顺序要求为小N大Z格式。
- src：输入，源操作数，类型为LocalTensor，支持的TPosition为B1，LocalTensor的起始地址需要32字节对齐。支持的数据类型为int8_t。
- idx：输入，源操作数，类型为LocalTensor，支持的TPosition为B1，LocalTensor的起始地址需要32字节对齐。支持的数据类型为int8_t。
- load_data_param：输入，LoadData参数结构体，LoadData2DParams类型。
  - start_index：分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：start_index∈[0, 65535] 。单位：512B。默认为0。
  - repeat_times：迭代次数，每个迭代可以处理512B数据。取值范围：repeat_times∈[1, 255]。
  - src_stride：相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。
  - sid：预留参数，配置为0即可。
  - dst_gap：相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dst_gap∈[0, 65535]。默认为0。
  - if_transpose：是否启用转置功能，对每个分形矩阵进行转置，默认为false（注意：只有A1->A2和B1->B2通路才能使能转置，使能转置功能时，源操作数、目的操作数仅支持uint16_t/int16_t/half数据类型）:
    - true：启用
    - false：不启用
  - addr_mode：预留参数，配置为0即可。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- repeat_times=0表示不执行。
- 每次迭代中的start_index不能小于零。
- 不支持转置功能。

**调用示例**

```python
import asc
dst = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.B2, addr=0, tile_size=512)
src = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.B1, addr=0, tile_size=512)
idx = asc.LocalTensor(dtype=asc.uint8, pos=asc.TPosition.B1, addr=0, tile_size=512)
params = asc.LoadData2DParams(repeat_times=1, src_stride=0, if_transpose=False)
asc.load_data_with_sparse(dst=dst, src=src, idx=idx, load_data_param=params)
```
