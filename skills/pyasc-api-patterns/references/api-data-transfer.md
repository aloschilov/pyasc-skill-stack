# pyasc Data Transfer

## `asc.data_copy`

Copies data between global memory (GM) and local memory (UB).

### GM -> UB (load)

```python
asc.data_copy(dst_local[local_offset:], src_gm[gm_offset:], element_count)
```

### UB -> GM (store)

```python
asc.data_copy(dst_gm[gm_offset:], src_local[local_offset:], element_count)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dst` | `LocalTensor` or `GlobalTensor` (sliced) | Destination |
| `src` | `GlobalTensor` or `LocalTensor` (sliced) | Source |
| `count` | `int` | Number of elements to copy |

## Data flow pattern

The standard data flow for a compute kernel:

```
GM (input) --[data_copy]--> UB (VECIN)
                               |
                          [compute: add/mul/etc]
                               |
UB (VECOUT) --[data_copy]--> GM (output)
```

## Tiling and buffering

To process large tensors:

1. Divide total work across cores using `asc.get_block_idx()`
2. Within each core, process in tiles
3. Optionally use double buffering (BUFFER_NUM=2) for pipeline overlap

```python
for i in range(TILE_NUM * BUFFER_NUM):
    buf_id = i % BUFFER_NUM
    asc.data_copy(x_local[buf_id * tile_len:], x_gm[i * tile_len:], tile_len)
    # sync + compute + sync
    asc.data_copy(z_gm[i * tile_len:], z_local[buf_id * tile_len:], tile_len)
```
