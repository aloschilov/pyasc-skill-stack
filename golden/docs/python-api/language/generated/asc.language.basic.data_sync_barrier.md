# asc.language.basic.data_sync_barrier

### asc.language.basic.data_sync_barrier(arg0: MemDsbT) → None

用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束。

**对应的Ascend C函数原型**

```c++
template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier()
```

**参数说明**

- arg0: 模板参数，表示需要等待的内存位置，类型为MemDsbT，可取值为：
  - ALL，等待所有内存访问指令。
  - DDR，等待GM访问指令。
  - UB，等待UB访问指令。
  - SEQ，预留参数，暂未启用，为后续的功能扩展做保留。

**约束说明**

无。

**调用示例**

```python
asc.mmad(...)
asc.data_sync_barrier(asc.MemDsbT.ALL)
asc.fixpipe(...)
```
