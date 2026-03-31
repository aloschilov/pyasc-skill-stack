# pyasc Synchronization and Pipeline

## Pipeline stages

Ascend NPU has separate hardware units for different operations:

| Unit | Purpose | Pipeline label |
|------|---------|---------------|
| MTE2 | Memory Transfer Engine 2 (GM -> UB) | Copy-in |
| V | Vector compute unit | Compute |
| MTE3 | Memory Transfer Engine 3 (UB -> GM) | Copy-out |

## Sync events

| Event | Meaning | When to use |
|-------|---------|-------------|
| `asc.HardEvent.MTE2_V` | Copy-in complete, compute may start | After `data_copy(local, gm)`, before compute |
| `asc.HardEvent.V_MTE3` | Compute complete, copy-out may start | After compute, before `data_copy(gm, local)` |
| `asc.HardEvent.MTE3_MTE2` | Copy-out complete, next copy-in may start | After `data_copy(gm, local)`, before next iteration's copy-in |

## Manual sync pattern

```python
for i in range(TILE_NUM * BUFFER_NUM):
    buf_id = i % BUFFER_NUM

    # Stage 1: Copy in
    asc.data_copy(x_local[buf_id * tile_len:], x_gm[i * tile_len:], tile_len)
    asc.data_copy(y_local[buf_id * tile_len:], y_gm[i * tile_len:], tile_len)

    # Sync: copy-in done -> compute may start
    asc.set_flag(asc.HardEvent.MTE2_V, buf_id)
    asc.wait_flag(asc.HardEvent.MTE2_V, buf_id)

    # Stage 2: Compute
    asc.add(z_local[buf_id * tile_len:], x_local[buf_id * tile_len:], y_local[buf_id * tile_len:], tile_len)

    # Sync: compute done -> copy-out may start
    asc.set_flag(asc.HardEvent.V_MTE3, buf_id)
    asc.wait_flag(asc.HardEvent.V_MTE3, buf_id)

    # Stage 3: Copy out
    asc.data_copy(z_gm[i * tile_len:], z_local[buf_id * tile_len:], tile_len)

    # Sync: copy-out done -> next copy-in may start
    asc.set_flag(asc.HardEvent.MTE3_MTE2, buf_id)
    asc.wait_flag(asc.HardEvent.MTE3_MTE2, buf_id)
```

## Auto sync alternative

Instead of manual sync, use the compiler's auto-sync:

```python
@asc.jit(auto_sync=True)
def kernel(...):
    # No set_flag/wait_flag needed; compiler inserts them
    asc.data_copy(x_local, x_gm, n)
    asc.add(z_local, x_local, y_local, n)
    asc.data_copy(z_gm, z_local, n)
```

To inspect inserted syncs: `@asc.jit(auto_sync=True, auto_sync_log="sync.log")`

## Double buffering

With `BUFFER_NUM=2`, two buffers alternate to overlap copy and compute:

- Even iterations use buffer 0
- Odd iterations use buffer 1
- `buf_id = i % BUFFER_NUM`

This allows the copy-in of tile N+1 to overlap with the compute of tile N.
