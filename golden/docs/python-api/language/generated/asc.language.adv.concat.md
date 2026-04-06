# asc.language.adv.concat

### asc.language.adv.concat(dst: [LocalTensor](../core.md#asc.language.core.LocalTensor), src: [LocalTensor](../core.md#asc.language.core.LocalTensor), tmp: [LocalTensor](../core.md#asc.language.core.LocalTensor), repeat_time: int) → None

对数据进行预处理，将要排序的源操作数src一一对应的合入目标数据concat中，数据预处理完后，可以进行Sort。

**对应的Ascend C函数原型**

```c++
template <typename T>
__aicore__ inline void Concat(LocalTensor<T> &concat, const LocalTensor<T> &src,
                                const LocalTensor<T> &tmp, const int32_t repeatTime)
```

**参数说明**

- dst：目的操作数。
- src：源操作数。
- tmp：输入，临时空间，用于接口内部复杂计算的中间存储。数据类型与src一致。
- repeat_time：输入，重复迭代次数，每次迭代处理16个元素，下次迭代跳至相邻的下一组16个元素。取值范围：[0, 255]。

**约束说明**

- 操作数地址对齐要求请参见 [《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi) 中的“通用说明和约束-通用地址对齐约束”。

**调用示例**

```python
concat_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECOUT, addr=0, tile_size=128)
value_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=128)
concat_tmp_local = asc.LocalTensor(dtype=asc.float16, pos=asc.TPosition.VECIN, addr=0, tile_size=256)
asc.concat(dst=concat_local, src=value_local, tmp=concat_tmp_local, repeat_time=8)
```
