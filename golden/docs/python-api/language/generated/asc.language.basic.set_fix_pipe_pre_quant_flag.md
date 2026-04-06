# asc.language.basic.set_fix_pipe_pre_quant_flag

### asc.language.basic.set_fix_pipe_pre_quant_flag(config: int) → None

DataCopy（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中标量量化参数。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetFixpipePreQuantFlag(uint64_t config)
```

**参数说明**

- config: 量化过程中使用到的标量量化参数，类型为uint64_t。

**调用示例**

```python
deq_scalar = 11
asc.set_fix_pipe_pre_quant_flag(deq_scalar)
```
