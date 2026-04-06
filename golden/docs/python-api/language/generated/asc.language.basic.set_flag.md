# asc.language.basic.set_flag

### asc.language.basic.set_flag(event: HardEvent, event_id: int = 0) → None

同一核内不同流水线之间的同步指令，具有数据依赖的不同流水指令之间需要插此同步。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetFlag(TEventID id)
__aicore__ inline void WaitFlag(TEventID id)
```

**参数说明**

- id: 事件ID，由用户自己指定。

**约束说明**

- set_flag/wait_flag必须成对出现。
- 禁止用户在使用set_flag和wait_flag时，自行指定event_id，容易与框架同步事件冲突，导致卡死问题。event_id需要通过alloc_event_id或者fetch_event_id来获取。

**调用示例**

如data_copy需要等待set_value执行完成后才能执行，需要插入PIPE_S到PIPE_MTE3的同步。

```python
dst = asc.GlobalTensor()
src = asc.LocalTensor()
src.set_value(0, 0)
data_size = 512
event_id = global_pipe.fetch_event_id(event=asc.HardEvent.S_MTE3)
asc.set_flag(event=asc.HardEvent.S_MTE3, event_id=event_id)
asc.wait_flag(event=asc.HardEvent.S_MTE3, event_id=event_id)
asc.data_copy(dst, src, data_size)
```
