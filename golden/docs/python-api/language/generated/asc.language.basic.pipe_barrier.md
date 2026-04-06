# asc.language.basic.pipe_barrier

### asc.language.basic.pipe_barrier(pipe: PipeID) → None

阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步。

**对应的Ascend C函数原型**

```c++
template <pipe_t pipe>
__aicore__ inline void PipeBarrier()
```

**参数说明**

- pipe: 模板参数，表示阻塞的流水类别。

**约束说明**

Scalar流水之间的同步由硬件自动保证，调用pipe_barrier(PIPE_S)会引发硬件错误。

**调用示例**

```python
asc.add(dst0, src0, src1, 512)
asc.pipe_barrier(asc.PipeID.PIPE_V)
asc.mul(dst1, dst0, src2, 512)
```
