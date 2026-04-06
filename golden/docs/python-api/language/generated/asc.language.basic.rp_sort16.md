# asc.language.basic.rp_sort16

### asc.language.basic.rp_sort16(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int) → None

根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void RpSort16(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime)
```

**参数说明**

- dst: 目的操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- src0: 源操作数。类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。
- repeat_time: 重复迭代次数。

**约束说明**

- 用户需保证src和dst中存储的Region Proposal数目大于实际所需数据，否则会存在tensor越界错误。
- 当存在proposal[i]与proposal[j]的score值相同时，如果i>j，则proposal[j]将首先被选出来，排在前面。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
# repeat_time = 2, 对2个Region Proposal进行排序
asc.rp_sort16(dst_local, src_local, 2)
```
