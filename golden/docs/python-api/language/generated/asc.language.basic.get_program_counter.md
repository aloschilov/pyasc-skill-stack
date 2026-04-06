# asc.language.basic.get_program_counter

### asc.language.basic.get_program_counter() → int

获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。

**对应的Ascend C函数原型**

```c++
__aicore__ inline int64_t GetProgramCounter()
```

**参数说明**

无。

**调用示例**

```python
pc = asc.get_program_counter()
```
