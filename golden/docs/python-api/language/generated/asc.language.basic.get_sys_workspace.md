# asc.language.basic.get_sys_workspace

### asc.language.basic.get_sys_workspace() → GlobalAddress

获取系统workspace指针。

**对应的Ascend C函数原型**

```c++
__aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()
```

**参数说明**

无。

**调用示例**

```python
workspace = asc.get_sys_workspace()
```
