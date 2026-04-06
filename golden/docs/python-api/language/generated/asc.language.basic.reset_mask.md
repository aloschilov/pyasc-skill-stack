# asc.language.basic.reset_mask

### asc.language.basic.reset_mask() → None

恢复mask的值为默认值（全1），表示矢量计算中每次迭代内的所有元素都将参与运算。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void ResetMask()
```

**参数说明**

无。

**约束说明**

无。

**调用示例**

```python
len = 128
asc.set_mask_count()
asc.set_vector_mask(len, dtype=asc.float16, mode=asc.MaskMode.COUNTER)
asc.reset_mask()
```
