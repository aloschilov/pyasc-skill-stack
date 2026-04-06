# asc.language.basic.set_load_data_repeat

### asc.language.basic.set_load_data_repeat(param: LoadDataRepeatParam) → None

用于设置 load_3d_v2 接口的 repeat 参数。设置 repeat 参数后，可以通过调用一次 load_3d_v2 接口完成多个迭代的数据搬运。

**对应的 Ascend C 函数原型**

```c++
__aicore__ inline void SetLoadDataRepeat(const LoadDataRepeatParam& repeatParams)
```

**参数说明**

- repeatParams
  : 设置load_3d_v2接口的repeat参数，类型为LoadDataRepeatParam。
- repeatParams
  : height/width方向上的迭代次数，取值范围：repeatTime ∈[0, 255] 。默认值为1
- repeatStride
  : height/width方向上的前一个迭代与后一个迭代起始地址的距离，取值范围：n∈[0, 65535]，默认值为0。
    repeatMode为0，repeatStride的单位为16个元素。
    repeatMode为1，repeatStride的单位和具体型号有关。
- repeatMode
  : 控制repeat迭代的方向，取值范围：k∈[0, 1] 。默认值为0。
    0：迭代沿height方向；
    1：迭代沿width方向。

**约束说明**
: 无

**调用示例**

```python
import asc
static_param = asc.LoadDataRepeatParam(
    repeatTime=4,
    repeatStride=8,
    repeatMode=0
)
asc.set_load_data_repeat(static_param)
```
