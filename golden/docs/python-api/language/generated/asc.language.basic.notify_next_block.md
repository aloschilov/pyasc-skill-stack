# asc.language.basic.notify_next_block

### asc.language.basic.notify_next_block(gm_workspace: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), ub_workspace: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

多核同步接口，通过写入 Global Memory 中的标志位，通知下一个 AI Core 当前核的操作已完成。

此接口通常与 wait_pre_block 配对使用。当前核调用此函数后，等待此核的下一个核将能够通过 wait_pre_block 检测到状态变化，从而继续执行。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void NotifyNextBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
```

**参数说明**

- gm_workspace (asc.GlobalTensor): Global Memory 上的临时工作空间。
  : - 用于核间通信的共享内存区域。通过向此空间写入一个特定的标志位，来通知下一个核。
    - 类型必须为 GlobalTensor<int32_t>。
- ub_workspace (asc.LocalTensor): UB 上的临时工作空间。
  : - 用于在 AI Core 内部操作 gm_workspace 的暂存区。
    - 类型必须为 LocalTensor<int32_t>。

**返回值**

无。

**约束说明**

- 需要保证每个核调用该接口的次数相同。
- gm_workspace申请的空间最少要求为：blockNum \* 32Bytes；ub_workspace申请的空间最少要求为：blockNum \* 32 + 32Bytes；其中blockNum为调用的核数，可调用get_block_num获取。
- 分离模式下，使用该接口进行多核同步时，仅对AIV核生效，wait_pre_block和notify_next_block之间仅支持插入矢量计算相关指令，对矩阵计算相关指令不生效。
- 使用该接口进行多核控制时，算子调用时指定的逻辑blockNum必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。

**调用示例**

```python
asc.notify_next_block(gm_workspace, ub_workspace)
```
