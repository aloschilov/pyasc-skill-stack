# asc.language.fwk.TPipe.release_event_id

#### TPipe.release_event_id(id: int, event: HardEvent = HardEvent.V_S) → None

用于释放HardEvent（硬件类型同步事件）的TEventID，通常与AllocEventID搭配使用。

**对应的Ascend C函数原型**

```c++
template <HardEvent evt>
__aicore__ inline void ReleaseEventID(TEventID id)
```

**参数说明**

- event：HardEvent硬件同步类型。
- id：TEventID类型，调用alloc_event_id申请获得的TEventID。

**约束说明**

alloc_event_id、release_event_id需成对出现，release_event_id传入的TEventID需由对应的alloc_event_id申请而来。

**调用示例**

```python
event_id = asc.get_tpipe_ptr().alloc_event_id(asc.HardEvent.V_S)
asc.set_flag(asc.HardEvent.V_S, event_id)
...
asc.wait_flag(asc.HardEvent.V_S, event_id)
asc.get_tpipe_ptr().release_event_id(event_id, asc.HardEvent.V_S)
```
