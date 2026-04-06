# asc.language.basic.wait_pre_block

### asc.language.basic.wait_pre_block(gm_workspace: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), ub_workspace: [LocalTensor](../core.md#asc.language.core.LocalTensor)) → None

多核同步接口，通过读取 Global Memory 中的标志位，等待上一个 AI Core 完成操作。

此接口通常与 notify_next_block 配对使用，形成核间同步的“握手”机制。
在使用前，需要确保已通过 init_determine_compute_workspace 接口初始化了核间同步所需的共享内存。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void WaitPreBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace)
```

**参数说明**

- gm_workspace (asc.GlobalTensor): Global Memory 上的临时工作空间。
  : - 用于核间通信的共享内存区域。通过读取此空间中的值，来判断上一个核是否已完成操作。
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
asc.wait_pre_block(gm_workspace, ub_workspace)
```
