# pyasc API Quick Reference

## Kernel declaration

```python
@asc.jit
def kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, n: int):
    ...
```

## Kernel launch

```python
kernel[core_num, rt.current_stream()](x_tensor, y_tensor, z_tensor, block_length)
```

## Global tensor setup

```python
x_gm = asc.GlobalTensor()
x_gm.set_global_buffer(x + offset, length)
```

## Local tensor allocation

```python
x_local = asc.LocalTensor(dtype, asc.TPosition.VECIN, byte_offset, element_count)
```

## Data copy

```python
asc.data_copy(dst_local[offset:], src_gm[offset:], count)  # GM -> UB
asc.data_copy(dst_gm[offset:], src_local[offset:], count)  # UB -> GM
```

## Vector operations

```python
asc.add(dst, src1, src2, count)
asc.sub(dst, src1, src2, count)
asc.mul(dst, src1, src2, count)
asc.div(dst, src1, src2, count)
```

## Sync primitives

```python
asc.set_flag(asc.HardEvent.MTE2_V, buf_id)    # copy-in done
asc.wait_flag(asc.HardEvent.MTE2_V, buf_id)
asc.set_flag(asc.HardEvent.V_MTE3, buf_id)    # compute done
asc.wait_flag(asc.HardEvent.V_MTE3, buf_id)
asc.set_flag(asc.HardEvent.MTE3_MTE2, buf_id) # copy-out done
asc.wait_flag(asc.HardEvent.MTE3_MTE2, buf_id)
```

## Block index

```python
block_idx = asc.get_block_idx()
offset = block_idx * block_length
```

## Configuration

```python
import asc.runtime.config as config
config.set_platform(config.Backend.Model, None)  # simulator
config.set_platform(config.Backend.NPU, config.Platform("Ascend910B"))  # hardware
```

## JIT compile options

```python
@asc.jit(always_compile=True)       # force recompile
@asc.jit(auto_sync=True)            # auto sync insertion
@asc.jit(opt_level=0)               # no optimization
@asc.jit(matmul_cube_only=True)     # pure cube mode
```
