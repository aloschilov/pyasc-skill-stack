# asc.language.basic.sort32

### asc.language.basic.sort32(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src0: [LocalTensor](../core.md#asc.language.core.LocalTensor), src1: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int) → None

排序函数，一次迭代可以完成32个数的排序。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void Sort32(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                             const LocalTensor<uint32_t>& src1, const int32_t repeatTime)
```

**参数说明**
- dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src1: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- repeat_time: 重复迭代次数。

**约束说明**

- 当存在score[i]与score[j]相同时，如果i>j，则score[j]将首先被选出来，排在前面。
- 每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
# repeat_time = 4, 对128个数分成4组进行排序，每次完成1组32个数的排序
asc.sort32(dst_local, src_local0, src_local1, 4)
```
