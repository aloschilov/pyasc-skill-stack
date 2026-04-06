# asc.language.fwk.TPipe.fetch_event_id

#### TPipe.fetch_event_id(event: HardEvent = HardEvent.V_S) → int

根据HardEvent（硬件类型的同步事件）获取相应可用的TEventID，此接口不会申请TEventID，仅提供可用的TEventID。

**对应的Ascend C函数原型**

```c++
template <HardEvent evt>
__aicore__ inline TEventID TPipe::FetchEventID()
__aicore__ inline TEventID TPipe::FetchEventID(HardEvent evt)
```

**参数说明**

- event：HardEvent硬件同步类型。

**返回值说明**

TEventID。

**约束说明**

相比于alloc_event_id，fetch_event_id适用于临时使用ID的场景，获取ID后，不会对ID进行占用。在一些复杂的使用场景下，需要开发者自行保证使用正确。
比如相同流水连续调用set_flag/wait_flag，如果两次传入的ID都是使用fetch_event_id获取的，因为两者ID相同会出现程序卡死等未定义行为，这时推荐用户使用alloc_event_id。

**调用示例**

```python
event_id = asc.get_tpipe_ptr().fetch_event_id(asc.HardEvent.V_S)
asc.set_flag(asc.HardEvent.V_S, event_id)
asc.wait_flag(asc.HardEvent.V_S, event_id)
```
