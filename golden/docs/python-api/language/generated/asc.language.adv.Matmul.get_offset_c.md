# asc.language.adv.Matmul.get_offset_c

#### Matmul.get_offset_c() → MatrixOffset

预留接口，为后续功能做预留。
获取本次计算时当前分片在整个C矩阵中的位置。

**对应的Ascend C函数原型**

```c++
__aicore__ inline MatrixOffset GetOffsetC()
```

**参数说明**

无。

**MatrixOffset结构体如下：**

```c++
struct MatrixOffset {
    int32_t offset;
    int32_t row, col;
    int32_t height, width;
};
```
