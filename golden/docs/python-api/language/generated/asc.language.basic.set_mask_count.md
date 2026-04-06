# asc.language.basic.set_mask_count

### asc.language.basic.set_mask_count() → None

设置掩码模式为 Counter 模式。在该模式下，
矢量计算时不需要开发者显式指定迭代次数和处理非对齐尾块，只需调用 set_mask_count 即可自动推断。

**对应的 Ascend C 函数原型**

```c++
__aicore__ inline void SetMaskCount();
```

**参数说明**

- 无

**约束说明**

- 设置为 Counter 模式后，建议在矢量计算完成后调用 set_mask_norm 恢复 Normal 模式。

**调用示例**

```python
asc.set_mask_count()
```
