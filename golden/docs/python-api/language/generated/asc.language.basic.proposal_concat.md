# asc.language.basic.proposal_concat

### asc.language.basic.proposal_concat(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int, mode_number: int) → None

将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposals的对应位置里。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void ProposalConcat(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber)
```

**参数说明**

- dst：目的操作数。
- src：源操作数。数据类型需要与dst保持一致。
- repeat_time：重复迭代次数。每次迭代完成16个元素合入到16个Region Proposals里，下次迭代跳至相邻的下一组16个Region Proposals和下一组16个元素。取值范围：repeatTime∈[0,255]。
- mode_number：合入位置参数，取值范围：modeNumber∈[0, 5]
  - 0：合入x1
  - 1：合入y1
  - 2：合入x2
  - 3：合入y2
  - 4：合入score
  - 5：合入label

**约束说明**

- 用户需保证dst中存储的proposal数目大于等于实际所需数目，否则会存在tensor越界错误。
- 用户需保证src中存储的元素大于等于实际所需数目，否则会存在tensor越界错误。
- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
asc.proposal_concat(dst, src, repeat_time=2, mode_number=4)
```
