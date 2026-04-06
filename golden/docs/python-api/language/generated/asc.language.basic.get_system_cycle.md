# asc.language.basic.get_system_cycle

### asc.language.basic.get_system_cycle() → int

获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = (cycle数/50) us 。

**对应的Ascend C函数原型**

```c++
__aicore__ inline int64_t GetSystemCycle()
```

**参数说明**

无。

**约束说明**

该接口是PIPE_S流水，若需要测试其他流水的指令时间，需要在调用该接口前通过pipe_barrier插入对应流水的同步

**调用示例**

```python
cycle = asc.get_system_cycle()
```
