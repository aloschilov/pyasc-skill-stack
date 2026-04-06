# asc.language.basic.mmad_with_sparse

### asc.language.basic.mmad_with_sparse(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), fm: [LocalTensor](../core.md#asc.language.core.LocalTensor), filter: [LocalTensor](../core.md#asc.language.core.LocalTensor), mmad_params: MmadParams) → None

完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵， 右矩阵B为稠密矩阵 。
对于矩阵A，在mmad_with_sparse计算时完成稠密化；
对于矩阵B，在计算执行前的输入数据准备时自行完成稠密化（按照下文中介绍的稠密算法进行稠密化），
所以输入本接口的B矩阵为稠密矩阵。B稠密矩阵需要通过调用load_data_with_sparse载入，同时加载索引矩阵，
索引矩阵在矩阵B稠密化的过程中生成，再用于A矩阵的稠密化。

**对应的Ascend C函数原型**

```c++
template <typename T = int32_t, typename U = int8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int32_t>::value, bool>::type = true, typename Std::enable_if<Std::is_same<PrimT<U>, int8_t>::value, bool>::type = true>
__aicore__ inline void MmadWithSparse(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<U>& filter, const MmadParams& mmadParams)
```

**参数说明**

- dst：输出，目的操作数，结果矩阵，类型为LocalTensor，支持的TPosition为CO1。起始地址需要256个元素（1024字节）对齐。
- fm：输入，源操作数，左矩阵A，类型为LocalTensor，支持的TPosition为A2。LocalTensor的起始地址需要512字节对齐。
- filter：输入，源操作数，右矩阵B，类型为LocalTensor，支持的TPosition为B2。LocalTensor的起始地址需要512字节对齐。
- mmad_params：输入，矩阵乘相关参数，类型为MmadParams。
  - m：左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。
  - n：右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。
  - k：左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。
  - cmatrix_init_val：配置C矩阵初始值是否为0。默认值true。
    - true：C矩阵初始值为0；
    - false：C矩阵初始值通过cmatrix_source参数进行配置。
  - cmatrix_source：配置C矩阵初始值是否来源于C2（存放Bias的硬件缓存区）。默认值为false。
    - false：来源于CO1；
    - true：来源于C2。
  - is_bias：该参数废弃，新开发内容不要使用该参数。如果需要累加初始矩阵，请使用带bias的接口来实现；也可以通过cmatrix_init_val和cmatrix_source参数配置C矩阵的初始值来源来实现。推荐使用带bias的接口，相比于配置cmatrix_init_val和cmatrix_source参数更加简单方便。配置是否需要累加初始矩阵，默认值为false，取值说明如下：
    - false：矩阵乘，无需累加初始矩阵，C = A \* B。
    - true：矩阵乘加，需要累加初始矩阵，C += A \* B。
  - fm_offset：预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。
  - en_ssparse：预留参数。
  - en_winograd_a：预留参数。
  - en_winograd_b：预留参数。
  - unit_flag：预留参数。
  - k_direction_align：预留参数。

**约束说明**

- 原始稀疏矩阵B每4个元素中应保证最多2个非零元素，如果存在3个或更多非零元素，则仅使用前2个非零元素。
- 当M、K、N中的任意一个值为0时，该指令不会被执行。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
import asc
dst = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.CO1, addr=0, tile_size=400)
fm = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.A2, addr=0, tile_size=400)
filter = asc.LocalTensor(dtype=asc.int8, pos=asc.TPosition.B2, addr=0, tile_size=400)
mmad_params = asc.MmadParams(
    m=20,
    n=20,
    k=20,
    is_bias=False,
    fm_offset=0,
    en_ssparse=False,
    en_winograd_a=False,
    en_winograd_b=False
)
asc.mmad_with_sparse(dst, fm, filter, mmad_params)
```
