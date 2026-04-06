# asc.language.basic.set_aipp_functions

### asc.language.basic.set_aipp_functions(src0: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), input_format: AippInputFormat, config: AippParams) → None

### asc.language.basic.set_aipp_functions(src0: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), src1: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), input_format: AippInputFormat, config: AippParams) → None

设置图片预处理（AIPP，AI core pre-process）相关参数。和load_image_to_local(ISASI)接口配合使用。
设置后，调用load_image_to_local(ISASI)接口可在搬运过程中完成图像预处理操作。

**对应的Ascend C函数原型**

输入图片格式为YUV400、RGB888、XRGB8888:

```c++
template<typename T, typename U>
void SetAippFunctions(const GlobalTensor<T>& src0, AippInputFormat format, AippParams<U> config)
```

输入图片格式为YUV420 Semi-Planar:

```c++
template<typename T, typename U>
void SetAippFunctions(const GlobalTensor<T>& src0, const GlobalTensor<T>& src1, AippInputFormat format, AippParams<U> config)
```

**参数说明**

- src0：源图片在Global Memory上的矩阵
- src1：源图片格式为YUV420SP时，表示UV维度在Global Memory上的矩阵
- input_format：源图片的图片格式
- config：图片预处理的相关参数，类型为AippParams

**约束说明**

- src0、src1在Global Memory上的地址对齐要求如下：
  - YUV420SP：src0必须2Bytes对齐，src1必须2Bytes对齐
  - XRGB8888：src0必须4Bytes对齐
  - RGB888：src0无对齐要求
  - YUV400：src0无对齐要求

**调用示例**

```python
swap_settings = asc.AippSwapParams(is_swap_rb=True)
cpad_settings = asc.AippChannelPaddingParams(c_padding_mode=0, c_padding_value=-1)

aipp_config_int8 = asc.AippParams(
    dtype=asc.int8,
    swap_params=swap_settings,
    c_padding_params=cpad_settings
)

asc.set_aipp_functions(rgb_gm, asc.AippInputFormat.RGB888_U8, aipp_config_int8)
```
