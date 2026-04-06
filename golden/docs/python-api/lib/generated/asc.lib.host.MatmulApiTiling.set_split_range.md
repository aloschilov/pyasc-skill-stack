# asc.lib.host.MatmulApiTiling.set_split_range

#### MatmulApiTiling.set_split_range(self: libhost.MatmulApiTilingBase, max_base_m: SupportsInt = -1, max_base_n: SupportsInt = -1, max_base_k: SupportsInt = -1, min_base_m: SupportsInt = -1, min_base_n: SupportsInt = -1, min_base_k: SupportsInt = -1) → int

设置baseM/baseN/baseK的最大值和最小值。 目前Tiling暂时不支持该功能。

**对应的Ascend C函数原型**

```c++
int32_t SetSplitRange(int32_t maxBaseM = -1, int32_t maxBaseN = -1, int32_t maxBaseK = -1,
                      int32_t minBaseM = -1, int32_t minBaseN = -1, int32_t minBaseK = -1)
```

**参数说明**

- max_base_m：设置最大的baseM值，默认值为-1。-1表示不设置指定的baseM最大值，该值由Tiling函数自行计算。
- max_base_n：设置最大的baseN值，默认值为-1。-1表示不设置指定的baseN最大值，该值由Tiling函数自行计算。
- max_base_k：设置最大的baseK值，默认值为-1。-1表示不设置指定的baseK最大值，该值由Tiling函数自行计算。
- min_base_m：设置最小的baseM值，默认值为-1。-1表示不设置指定的baseM最小值，该值由Tiling函数自行计算。
- min_base_n：设置最小的baseN值，默认值为-1。-1表示不设置指定的baseN最小值，该值由Tiling函数自行计算。
- min_base_k：设置最小的baseK值，默认值为-1。-1表示不设置指定的baseK最小值，该值由Tiling函数自行计算。

**返回值说明**

-1表示设置失败； 0表示设置成功。

**约束说明**

若base_m/base_n/base_k不满足C0_size对齐，计算Tiling时会将该值对齐到C0_size。提示，half/bfloat16_t数据类型的C0_size为16，float数据类型的C0_size为8，int8_t数据类型的C0_size为32，int4b_t数据类型的C0_size为64。
