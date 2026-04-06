# asc.language.fwk.TPipe.alloc_event_id

#### TPipe.alloc_event_id(event: HardEvent = HardEvent.V_S) → int

用于申请HardEvent（硬件类型同步事件）的TEventID，必须与ReleaseEventID搭配使用，调用该接口后，会占用申请的TEventID，直至调用ReleaseEventID释放。

**对应的Ascend C函数原型**

```c++
template <HardEvent evt>
__aicore__ inline TEventID TPipe::AllocEventID()
```

**参数说明**

- event：HardEvent硬件同步类型。

**返回值说明**

TEventID

**约束说明**

TEventID有数量限制，使用结束后应该立刻调用release_event_id释放，防止TEventID耗尽。

**调用示例**

```python
event_id = asc.get_tpipe_ptr().alloc_event_id(asc.HardEvent.V_S)
asc.set_flag(asc.HardEvent.V_S, event_id)
...
asc.wait_flag(asc.HardEvent.V_S, event_id)
asc.get_tpipe_ptr().release_event_id(event_id, asc.HardEvent.V_S)
```
