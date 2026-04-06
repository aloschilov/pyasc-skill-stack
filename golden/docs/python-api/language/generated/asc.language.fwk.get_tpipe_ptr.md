# asc.language.fwk.get_tpipe_ptr

### asc.language.fwk.get_tpipe_ptr() → [TPipe](../fwk.md#asc.language.fwk.TPipe)

创建TPipe对象时，对象初始化会设置全局唯一的TPipe指针。本接口用于获取该指针，获取该指针后，可进行TPipe相关的操作。

**对应的Ascend C函数原型**

```c++
__aicore__ inline AscendC::TPipe* GetTPipePtr()
```

**调用示例**

```python
pipe = asc.Tpipe()
x_gm.set_global_buffer(x, 2048)
in_queue_x = asc.TQue(asc.TPosition.VECIN, 2)
get_tpipe_ptr.init_buffer(in_queue_x, 2, 128 * asc.half.sizeof())
```