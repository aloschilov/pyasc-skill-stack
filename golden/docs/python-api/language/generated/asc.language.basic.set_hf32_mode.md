# asc.language.basic.set_hf32_mode

### asc.language.basic.set_hf32_mode(hf32_mode: bool) → None

调用该接口后，可设置 Mmad 计算是否开启 HF32 模式。
开启 HF32 模式后，L0A/L0B 中的 FP32 数据在参与矩阵乘法计算之前将被舍入为 HF32 精度。

**对应的 Ascend C 函数原型**

```c++
__aicore__ inline void SetHF32Mode(bool hf32Mode);
```

**参数说明**

- hf32_mode：
  Mmad HF32 模式控制参数，bool 类型。
  - True：L0A/L0B 中的 FP32 数据将在矩阵乘法之前被舍入为 HF32。
  - False：执行常规的 FP32 矩阵乘法计算。

**约束说明**

无。

**调用示例**

```python
asc.set_hf32_mode(True)
asc.set_hf32_mode(False)
```
