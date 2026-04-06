# asc.language.basic.ib_set

### asc.language.basic.ib_set(gm_workspace: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), ub_workspace: [LocalTensor](../core.md#asc.language.core.LocalTensor), block_idx: int, event_id: int, is_aiv_only: bool = True) → None

当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。
调用ib_set设置某一个核的标志位，与ib_wait成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。

**对应的Ascend C函数原型**

```c++
template <bool isAIVOnly = true>
__aicore__ inline void IBSet(
    const GlobalTensor<int32_t>& gmWorkspace,
    const LocalTensor<int32_t>&  ubWorkspace,
    int32_t blockIdx,
    int32_t eventID)
```

**参数说明**

- gmWorkspace: 外部存储核状态的公共缓存，类型为GlobalTensor。
- ubWorkspace: 存储当前核状态的公共缓存。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- blockIdx: 表示等待核的idx号，取值范围：[0, 核数-1]。
- eventID: 用来控制当前核的set、wait事件。
- isAIVOnly:控制是否为AIVOnly模式，默认为true。

**约束说明**

- gmWorkspace申请的空间最少要求为：核数 \* 32Bytes \* eventID_max + blockIdx_max \* 32Bytes + 32Bytes。（eventID_max和blockIdx_max分别指eventID、blockIdx的最大值 ）；
- 注意：如果是AIVOnly模式，核数 = GetBlockNum()；如果是MIX模式，核数 = GetBlockNum() \* 2；
- ubWorkspace申请的空间最少要求为：32Bytes；
- gmWorkspace缓存的值需要初始化为0。
- 使用该接口进行多核控制时，算子调用时指定的逻辑blockNum必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。

**调用示例**

```python
gm = asc.GlobalTensor()
gm.set_global_buffer(x)
ub = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
asc.ib_set(gm, ub, block_idx=0, event_id=0)
```
