# asc.language.basic.proposal_extract

### asc.language.basic.proposal_extract(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int, mode_number: int) → None

与proposal_concat功能相反，从Region Proposals内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void ProposalExtract(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber)
```

**参数说明**

- dst：目的操作数。
- src：源操作数，数据类型需与dst一致。
- repeat_time：重复迭代次数。每次迭代处理16个Region Proposals的元素抽取并重排，下次迭代跳至相邻的下一组16个Region Proposals。取值范围：repeatTime∈[0,255]。
- mode_number：抽取位置参数，取值范围：modeNumber∈[0,5]
  - 0：抽取x1
  - 1：抽取y1
  - 2：抽取x2
  - 3：抽取y2
  - 4：抽取score
  - 5：抽取label

**约束说明**

- 用户需保证src中存储的proposal数量不小于实际所需数量，否则可能发生tensor越界。
- 用户需保证dst中可容纳的元素数量不小于实际抽取数量。
- 操作数地址需满足通用对齐约束（32字节对齐）。

**调用示例**

```python
asc.proposal_extract(dst, src, repeat_time=2, mode_number=4)
```
