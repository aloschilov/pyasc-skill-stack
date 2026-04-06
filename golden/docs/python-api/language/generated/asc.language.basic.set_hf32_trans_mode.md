# asc.language.basic.set_hf32_trans_mode

### asc.language.basic.set_hf32_trans_mode(trans_mode: bool) → None

调用该接口后，可设置 Mmad 的 HF32 取整模式，仅在 HF32 模式开启时有效。
在 HF32 模式下，将按照给定模式对 FP32 数据进行舍入。

**对应的 Ascend C 函数原型**

```c++
__aicore__ inline void SetHF32TransMode(bool hf32TransMode);
```

**参数说明**

- hf32_trans_mode：
  Mmad HF32 取整模式控制参数，bool 类型。
  - True：FP32 将以向零靠近的方式舍入为 HF32。
  - False：FP32 将以最接近偶数的方式舍入为 HF32。

**约束说明**

无。

**调用示例**

```python
asc.set_hf32_trans_mode(True)
asc.set_hf32_trans_mode(False)
```
