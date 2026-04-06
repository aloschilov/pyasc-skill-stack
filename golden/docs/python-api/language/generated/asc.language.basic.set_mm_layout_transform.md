# asc.language.basic.set_mm_layout_transform

### asc.language.basic.set_mm_layout_transform(mm_layout_mode: bool) → None

调用该接口后，可设置 Mmad 的 M/N 方向优先顺序，
控制矩阵乘加计算时先按 N 再按 M 方向还是先按 M 再按 N 方向。

**对应的 Ascend C 函数原型**

```c++
__aicore__ inline void SetMMLayoutTransform(bool mmLayoutMode);
```

**参数说明**

- mm_layout_mode：
  Mmad M/N 方向控制参数，bool 类型。
  - True：代表 CUBE 将首先通过 N 方向，然后通过 M 方向产生结果。
  - False：代表 CUBE 将首先通过 M 方向，然后通过 N 方向产生结果。

**约束说明**

无。

**调用示例**

```python
asc.set_mm_layout_transform(True)
asc.set_mm_layout_transform(False)
```
