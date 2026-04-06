# asc.language.basic.init_soc_state

### asc.language.basic.init_soc_state() → None

在由于AI Core上存在一些全局状态，如原子累加状态、Mask模式等，在实际运行中，这些值可以被前序执行的算子修改而导致计算出现不符合预期的行为，在静态Tensor编程的场景中用户必须在Kernel入口处调用此函数来初始化AI Core状态 。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void InitSocState()
```

**参数说明**

无。

**约束说明**

不调用该接口，在部分场景下可能导致计算结果出现精度错误或者卡死等问题。

**调用示例**

```python
asc.init_soc_state()
```
