# asc.language.basic.trap

### asc.language.basic.trap() → None

在Kernel侧调用，NPU模式下会中断AI Core的运行，CPU模式下等同于assert。可用于Kernel侧异常场景的调试。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void Trap()
```

**参数说明**

无。

**调用示例**

```python
asc.trap()
```
