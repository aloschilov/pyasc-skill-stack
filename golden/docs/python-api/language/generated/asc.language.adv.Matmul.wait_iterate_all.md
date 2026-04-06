# asc.language.adv.Matmul.wait_iterate_all

#### Matmul.wait_iterate_all() → None

等待iterate_all异步接口返回，支持连续输出到Global Memory。

**对应的Ascend C函数原型**

```c++
__aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)
```

**参数说明**

无。

**约束说明**

- 配套iterate_all异步接口使用。
- 仅支持连续输出至Global Memory。

**调用示例**

```python
mm = asc.adv.Matmul(a_type, b_type, c_type, bais_type)
mm.set_tensor_a(gm_a[offset_a:])
mm.set_tensor_b(gm_b[offset_b:])
if tiling.is_bias:
    mm.set_bias(gm_bias[offset_bias])
mm.iterate_all(tensor=gm_c[offset_c], en_atomic=0, sync=False, en_sequential_write=False, wait_iterate_all=True)
mm.wait_iterate_all()
```
