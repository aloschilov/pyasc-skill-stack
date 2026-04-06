# asc.lib.host.MatmulApiTiling.set_double_buffer

#### MatmulApiTiling.set_double_buffer(self: libhost.MatmulApiTilingBase, a: bool, b: bool, c: bool, bias: bool, trans_nd2nz: bool = True, trans_nz2nd: bool = True) → int

设置A/B/C/Bias是否使能double buffer功能，以及是否需要做ND2NZ或者NZ2ND的转换，主要用于Tiling函数内部调优。

**对应的Ascend C函数原型**

```c++
int32_t SetDoubleBuffer(bool a, bool b, bool c, bool bias, bool transND2NZ = true, bool transNZ2ND = true)
```

**参数说明**

- dequant_type：设置量化或反量化时的模式。

**返回值说明**

-1表示设置失败； 0表示设置成功。
