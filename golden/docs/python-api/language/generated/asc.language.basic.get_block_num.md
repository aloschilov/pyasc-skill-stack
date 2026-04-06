# asc.language.basic.get_block_num

### asc.language.basic.get_block_num() → int

获取当前任务配置的核数，用于代码内部的多核逻辑控制等。

**对应的Ascend C函数原型**

```c++
__aicore__ inline int64_t GetBlockNum()
```

**参数说明**

无。

**调用示例**

```python
loop_size = total_size // asc.get_block_num()
```
