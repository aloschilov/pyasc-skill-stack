# asc.language.basic.load_image_to_local

### asc.language.basic.load_image_to_local(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), load_data_params: LoadImageToLocalParams) → None

将图像数据从GM搬运到A1/B1。 搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。
图像预处理的相关参数通过set_aipp_functions进行配置。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void LoadImageToLocal(const LocalTensor<T>& dst, const LoadImageToLocalParams& loadDataParams)
```

**参数说明**

- dst：输出，目的操作数，类型为LocalTensor，支持的TPosition为A1/B1。LocalTensor的起始地址需要保证32字节对齐。不同产品支持的数据类型不同：
  - Atlas A3 训练/推理系列产品：int8_t/half
  - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：int8_t/half
  - Atlas 200I/500 A2 推理产品：uint8_t/int8_t/half
- load_data_params：输入，LoadData参数结构体，类型为LoadImageToLocalParams，包含以下子参数：
  - horiz_size：输入，从源图中加载图片的水平宽度，单位为像素，取值范围：horiz_size∈[2, 4095]。
  - vert_size：输入，从源图中加载图片的垂直高度，单位为像素，取值范围：vert_size∈[2, 4095]。
  - horiz_start_pos：输入，加载图片在源图片上的水平起始地址，单位为像素，取值范围：horiz_start_pos∈[0, 4095]，默认为0。注意：当输入图片为YUV420SP、XRGB8888、RGB888和YUV400格式时，该参数需要是偶数。
  - vert_start_pos：输入，加载图片在源图片上的垂直起始地址，单位为像素，取值范围：vert_start_pos∈[0, 4095]，默认为0。注意：当输入图片为YUV420SP格式时，该参数需要是偶数。
  - src_horiz_size：输入，源图像水平宽度，单位为像素，取值范围：src_horiz_size∈[2, 4095]。注意：当输入图片为YUV420SP格式时，该参数需要是偶数。
  - top_pad_size：输入，目的图像顶部填充的像素数，取值范围：top_pad_size∈[0, 32]，默认为0。进行数据填充时使用，需要先调用SetAippFunctions(ISASI)通过AippPaddingParams配置填充的数值，再通过topPadSize、botPadSize、leftPadSize、rightPadSize配置填充的大小范围。
  - bot_pad_size：输入，目的图像底部填充的像素数，取值范围：bot_pad_size∈[0, 32]，默认为0。
  - left_pad_size：输入，目的图像左边填充的像素数，取值范围：left_pad_size∈[0, 32]，默认为0。
  - right_pad_size：输入，目的图像右边填充的像素数，取值范围：right_pad_size∈[0, 32]，默认为0。
  - sid：输入，预留参数，为后续功能保留，开发者暂时无需关注，使用默认值即可。

**返回值说明**

无

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。
- 加载到dst的图片的大小加padding的大小必须小于等于L1的大小。
- 对于XRGB输入格式的数据，芯片在处理的时候会默认丢弃掉第四个通道的数据，所以需要在set_aipp_functions接口里设置好通道交换的参数后输出RGB格式的数据。

**调用示例**

```python
dst = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.A1, addr=0, tile_size=128)
load_data_params = asc.LoadImageToLocalParams(2, 2, 0, 0, 2, 0, 0, 0, 0)
asc.load_image_to_local(dst, load_data_params)
```
