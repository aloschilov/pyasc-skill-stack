# asc.language.basic.fixpipe

### asc.language.basic.fixpipe(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: FixpipeParamsV220, config: FixpipeConfig = FixpipeConfig.cfg_row_major) → None

### asc.language.basic.fixpipe(dst: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), cbuf_workspace: [LocalTensor](../core.md#asc.language.core.LocalTensor), params: FixpipeParamsV220, config: FixpipeConfig = FixpipeConfig.cfg_row_major) → None

矩阵计算完成后，对结果进行处理，例如对计算结果进行量化操作，并把数据从CO1搬迁到Global Memory中。

**对应的Ascend C函数原型**

通路CO1->GM，不使能tensor量化功能：

```c++
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsV220& intriParams)
```

通路CO1->GM，使能tensor量化功能：

```c++
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t,
          typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
                                const LocalTensor<S>& cbufWorkspace, const FixpipeParamsV220& intriParams)
```

**参数说明**

- dst: 目的操作数，类型为GlobalTensor。数据格式为NZ或ND格式。经过Fixpipe处理，在量化操作之后，会将矩阵计算中多申请的数据删除。
- src: 源操作数，支持的TPosition为CO1，为Mmad接口计算的结果，类型为LocalTensor。支持的数据类型为float/int32_t，数据格式为NZ格式。起始地址需要满足64B对齐。
- cbufWorkspace: 量化参数，类型为LocalTensor<uint64_t>，支持的TPosition为A1。仅当quantPre为VDEQF16/VQF322B8_PRE/VREQ8时支持。
- intriParams: Fixpipe搬运参数，类型为FixpipeParamsV220，包含以下参数：
  - nSize: 源NZ矩阵在N方向上的大小。
    - 不使能NZ2ND功能：取值范围：nSize∈[1, 4095]。若使能channelSplit功能，nSize必须为8的倍数，若不使能channelSplit功能，nSize必须为16的倍数。
    - 使能NZ2ND功能：取值范围：nSize∈[1, 4095]。
  - mSize: 源NZ矩阵在M方向上的大小。
    - 不使能NZ2ND功能：取值范围：mSize∈[1, 65535]。
    - 使能NZ2ND功能：取值范围：mSize∈[1, 8192]。
  - srcStride: 源NZ矩阵中相邻Z排布的起始地址偏移，取值范围：srcStride∈[0, 65535]，单位：C0_Size(16\*sizeof(T)，T为src的数据类型)。
  - dstStride: 目的矩阵的地址步长。
    - 不使能NZ2ND功能：目的NZ矩阵中相邻Z排布的起始地址偏移，取值不为0，单位：datablock(32Bytes)。
    - 使能NZ2ND功能：目的ND矩阵每一行中的元素个数，取值不为0，单位：element。
  - quantPre: 量化模式，类型为QuantMode_t，默认值为QuantMode_t::NoQuant（不使能量化功能）。支持以下取值：
    - NoQuant：不使能量化功能
    - F322F16：float量化成half，量化结果支持INF_NAN模式
    - F322BF16：float量化成bfloat16_t，量化结果支持INF_NAN模式
    - DEQF16：int32_t量化成half，scalar量化，量化结果不支持INF_NAN模式
    - VDEQF16：int32_t量化成half，tensor量化，量化结果不支持INF_NAN模式
    - QF322B8_PRE：float量化成uint8_t/int8_t，scalar量化
    - VQF322B8_PRE：float量化成uint8_t/int8_t，tensor量化
    - REQ8：int32_t量化成uint8_t/int8_t，scalar量化
    - VREQ8：int32_t量化成uint8_t/int8_t，tensor量化
  - deqScalar: scalar量化参数，表示单个scale值，quantPre量化模式为scalar量化时需要设置该参数。支持的数据类型为uint64_t。
  - ndNum: 源NZ矩阵的数目，也就是传输ND矩阵的数目，取值范围：ndNum∈[1, 65535]。
  - srcNdStride: 不同NZ矩阵起始地址之间的间隔，取值范围：srcNdStride∈[1, 512]，单位：1024B。当ndNum配置为1时，srcNdStride配置为0即可，不生效。
  - dstNdStride: 目的相邻ND矩阵起始地址之间的偏移，取值范围：dstNdstride∈[1, 65535]，单位：element。当ndNum配置为1时，dstNdStride配置为0即可，不生效。
  - reluEn: 是否使能relu的开关，false：不使能relu功能；true：使能relu功能。
  - unitFlag: Mmad指令和Fixpipe指令细粒度的并行控制。
    - 0：保留值
    - 2：使能unitFlag，硬件执行完指令之后，不会设置寄存器
    - 3：使能unitFlag，硬件执行完指令之后，会将unitFlag关闭
  - isChannelSplit: 是否使能通道拆分的功能。默认为false，不使能该功能。仅在src和dst都为float时才能使能通道拆分，且不能同时使能ChannelSplit和NZ2ND功能。

**约束说明**

- ndNum=0 表示不执行，此指令将不被执行并报warning。
- 对于量化输入为float32数据类型的说明：标准的IEEE 754 float32格式为：1bit符号位，8bits指数位，23bits尾数位；当前AI处理器支持的float32格式为：1bit符号位，8bits指数位，10bits尾数位。如果用户提供的是标准的IEEE 754 float32输入，API内部会处理成处理器支持的float32格式进行计算，此时如果golden数据生成过程中使用的是标准的IEEE 754 float32数据，则可能引入精度不匹配问题，需要修正golden数据的生成，将量化参数的23bits尾数位的低13bits数据位清零再参与量化计算。

**调用示例**

- 通路CO1->GM，不使能tensor量化功能，使能F322F16量化
  ```python
  dst_gm = asc.GlobalTensor()
  dst_gm.set_global_buffer(x)
  src_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
  fixpipe_params = asc.FixpipeParamsV220(
    n_size=16, m_size=16, src_stride=32, dst_stride=32,
    quant_pre=asc.QuantModes.NoQuant, deq_scalar=0,
    nd_num=1, src_nd_stride=0, dst_nd_stride=0,
    relu_en=False, unit_flag=0, is_channel_split=False
  )
  asc.fixpipe(dst_gm, src_local, fixpipe_params)
  ```
- 通路CO1->GM，使能tensor量化功能（VDEQF16）
  ```python
  dst_gm = asc.GlobalTensor()
  dst_gm.set_global_buffer(x)
  src_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=512)
  workspace_local = asc.LocalTensor(dtype=asc.uint64, pos=asc.TPosition.VECIN, addr=512, tile_size=1024)
  fixpipe_params = asc.FixpipeParamsV220(
    n_size=16, m_size=32, src_stride=32, dst_stride=16,
    quant_pre=asc.QuantModes.VDEQF16, deq_scalar=0,
    nd_num=1, src_nd_stride=2, dst_nd_stride=512,
    relu_en=False, unit_flag=0, is_channel_split=False
  )
  asc.fixpipe(c_gm, c_local, cbuf_workspace, fixpipe_params)
  ```
