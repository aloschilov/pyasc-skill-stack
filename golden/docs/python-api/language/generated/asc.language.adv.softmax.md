# asc.language.adv.softmax

### asc.language.adv.softmax(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), sum: [LocalTensor](../core.md#asc.language.core.LocalTensor), max: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), tiling: SoftmaxTiling, temp_buffer: [LocalTensor](../core.md#asc.language.core.LocalTensor) | None = None, reuse_source: bool = False, basic_block: bool = False, data_format_nz: bool = False) → None

将输入tensor[m0, m1, …mt, n]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作[m, n]。
为方便理解，通过Python脚本实现的方式，表达其计算公式（以输入为ND格式为例）如下，其中src是源操作数（输入），dst、sum、max为目的操作数（输出）。

```python
def softmax(src):
    # 基于last轴进行rowmax（按行取最大值）处理
    max = np.max(src, axis=-1, keepdims=True)
    sub = src - max
    exp = np.exp(sub)
    # 基于last轴进行rowsum（按行求和）处理
    sum = np.sum(exp, axis=-1, keepdims=True)
    dst = exp / sum
    return dst, max, sum
```

**对应的Ascend C函数原型**

- 接口框架申请临时空间
  - LocalTensor的数据类型相同
    ```c++
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor,
                                  const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                  const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```
  - LocalTensor的数据类型不同
    ```c++
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor,
                                  const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor,
                                  const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```
  - 不带sumTensor和maxTensor参数
    ```c++
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```
- 通过sharedTmpBuffer入参传入临时空间
  - LocalTensor的数据类型相同
    ```c++
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor,
                                  const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
                                  const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
                                  const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```
  - LocalTensor的数据类型不同
    ```c++
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor,
                                  const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor,
                                  const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
                                  const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```
  - 不带sumTensor和maxTensor参数
    ```c++
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                  const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
                                  const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```

**参数说明**

- dst：目的操作数。
- sum：目的操作数。
- max：目的操作数。
- src：源操作数。
- tiling：SoftMax计算所需Tiling信息。
- tmp_buffer：临时空间。
- reuse_source：该参数预留，传入默认值false即可。
- basic_block：src和dst的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。
- data_format_nz：当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。

**约束说明**

- src和dst的Tensor空间可以复用。
- sum和max为输出，并且last轴长度必须固定32Byte，非last轴大小需要和src以及dst保持一致。
- sum和max的数据类型需要保持一致。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 不支持tmp_buffer与源操作数和目的操作数地址重叠。

开发者需要对GM上的原始输入(ori_src_M, ori_src_K)在M或K方向补齐数据到(src_M, src_K)，补齐的数据会参与部分运算，
在输入输出复用的场景下，API的计算结果会覆盖src中补齐的原始数据，在输入输出不复用的场景下，
API的计算结果会覆盖dst中对应src补齐位置的数据。

**调用示例**

```python
src_local = in_queue_src.deque(T)
sum_temp_local = sum_queue.alloc_tensor(T)
max_temp_local = max_queue.alloc_tensor(T)
dst_local = out_queue_dst.alloc_tensor(T)

src_shape = asc.SoftMaxShapeInfo(height, width, height, width);
asc.adv.softmax(dst_local, sum_temp_local, max_temp_local, srcLocal, tiling, src_shape);

out_queue_dst.EnQue(dstLocal)
max_queue.free_tensor(max_temp_local)
sum_queue.free_tensor(sum_temp_local)
in_queue_src.free_tensor(src_local)
```
