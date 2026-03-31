# pyasc Tensor Operations

## Element-wise binary operations

All binary operations follow the same pattern:

```python
asc.{op}(dst_local[offset:], src1_local[offset:], src2_local[offset:], count)
```

| Operation | API | Description |
|-----------|-----|-------------|
| Addition | `asc.add(dst, src1, src2, count)` | dst = src1 + src2 |
| Subtraction | `asc.sub(dst, src1, src2, count)` | dst = src1 - src2 |
| Multiplication | `asc.mul(dst, src1, src2, count)` | dst = src1 * src2 |
| Division | `asc.div(dst, src1, src2, count)` | dst = src1 / src2 |

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dst` | `LocalTensor[offset:]` | Destination tensor (VECOUT position) |
| `src1` | `LocalTensor[offset:]` | First source tensor (VECIN position) |
| `src2` | `LocalTensor[offset:]` | Second source tensor (VECIN position) |
| `count` | `int` | Number of elements to process |

### Advanced parameters

Some operations support additional parameters:

```python
asc.add(dst, src1, src2, mask=mask, repeat_time=n,
        repeat_params=asc.BinaryRepeatParams(dst_stride, src1_stride, src2_stride,
                                              dst_rep_stride, src1_rep_stride, src2_rep_stride))
```

## Tensor types

### GlobalTensor

Represents a tensor in global (GM) memory:

```python
x_gm = asc.GlobalTensor()
x_gm.set_global_buffer(global_address + offset, length)
```

### LocalTensor

Represents a tensor in local (UB) memory:

```python
x_local = asc.LocalTensor(dtype, position, byte_offset, element_count)
```

| Parameter | Description |
|-----------|-------------|
| `dtype` | Data type (from GlobalAddress or explicit like `asc.float32`) |
| `position` | `asc.TPosition.VECIN`, `VECOUT`, `VECCALC`, etc. |
| `byte_offset` | Starting byte offset in the local memory partition |
| `element_count` | Number of elements |

### Tensor slicing

```python
x_local[offset:]       # Returns new tensor view starting at element offset
x_gm[offset:]          # Same for global tensors
```
