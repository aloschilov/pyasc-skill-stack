# asc.language.basic.sync_all

### asc.language.basic.sync_all(is_aiv_only: bool = True) → None

### asc.language.basic.sync_all(gm_workspace: [GlobalTensor](../core.md#asc.language.core.GlobalTensor), ub_workspace: [LocalTensor](../core.md#asc.language.core.LocalTensor), used_cores: int = 0, is_aiv_only: bool = True) → None

当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。
目前多核同步分为硬同步和软同步，硬件同步是利用硬件自带的全核同步指令由硬件保证多核同步，软件同步是使用软件算法模拟实现。

**对应的Ascend C函数原型**

```c++
// 软同步
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(
    const GlobalTensor<int32_t>& gmWorkspace,
    const LocalTensor<int32_t>&  ubWorkspace,
    const int32_t usedCores = 0)
```

```c++
// 硬同步
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll()
```

**参数说明**

- gmWorkspace: gmWorkspace为用户定义的全局Global空间，作为所有核共用的缓存，用于保存每个核的状态标记，类型为GlobalTensor，支持的数据类型为int32_t。
- ubWorkspace: ubWorkspace为用户定义的局部Local空间，每个核单独自用，用于标记当前核的状态。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，支持的数据类型为int32_t。
- usedCores: 指定多少个核之间的同步，传入数值不能超过算子调用时指定的逻辑blockNum。此参数为默认参数，不传此参数表示全核软同步。
- isAIVOnly: 控制SyncAll作用于纯Vector算子或融合（Cube和Vector融合）算子。可选值：
  - true（默认值）：纯Vector算子的全核同步，仅执行Vector核的全核同步。
  - false：融合算子的全核同步，先分别完成Vector核和Cube核的全核同步，再执行两者之间的同步（软同步接口不支持此功能）。

**约束说明**

- gmWorkspace缓存申请的空间大小要求大于等于核数\*32Bytes，并且缓存的值需要初始化为0。目前常见的有两种初始化方式：
  - 通过在host侧进行初始化操作，确保传入该接口时，gmWorkspace缓存已经初始化为0；
  - 在kernel侧初始化的时候对gmWorkspace缓存初始化，需要注意的是，每个核上都需要初始化全部的gmWorkspace缓存空间。
- ubWorkspace申请的空间大小要求大于等于核数\*32Bytes。
- 使用该接口进行多核控制时，算子调用时指定的逻辑blockNum必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。
- 在分离模式下，建议使用硬同步接口而非软同步接口。软同步接口仅适用于纯Vector场景，且性能较低。使用硬同步接口时，需根据场景设置Kernel类型：
  - 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
  - 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。

**调用示例**

- 软同步
  > ```python
  > gm = asc.GlobalTensor()
  > gm.set_global_buffer(x)
  > ub = asc.LocalTensor(dtype=asc.int32, pos=asc.TPosition.VECIN, addr=0, tile_size=32)
  > asc.sync_all(gm, ub, used_cores=0)
  > ```
- 硬同步
  > ```python
  > asc.sync_all()
  > ```
