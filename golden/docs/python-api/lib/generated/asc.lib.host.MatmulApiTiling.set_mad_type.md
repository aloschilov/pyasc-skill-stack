# asc.lib.host.MatmulApiTiling.set_mad_type

#### MatmulApiTiling.set_mad_type(self: libhost.MatmulApiTilingBase, mad_type: libhost.MatrixMadType) → int

设置是否使能HF32模式。当前版本暂不支持。

**对应的Ascend C函数原型**

```c++
int32_t SetMadType(MatrixMadType madType)
```

**参数说明**

- mad_type：设置Matmul模式。

**返回值说明**

-1表示设置失败； 0表示设置成功。
