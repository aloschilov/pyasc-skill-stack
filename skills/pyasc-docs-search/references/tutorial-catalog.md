# pyasc Tutorial Catalog

## Available tutorials

| # | Name | Path | Description | Key APIs |
|---|------|------|-------------|----------|
| 01 | add | `~/workspace/pyasc/python/tutorials/01_add/` | Manual sync vector add | `asc.add`, `data_copy`, `set_flag`/`wait_flag` |
| 02 | add_framework | `~/workspace/pyasc/python/tutorials/02_add_framework/` | Framework-managed sync add | TPipe, TQue, framework sync |
| 03 | matmul_mix | `~/workspace/pyasc/python/tutorials/03_matmul_mix/` | MIX mode matmul (cube + vector) | Matmul APIs, mixed compute |
| 04 | matmul_cube_only | `~/workspace/pyasc/python/tutorials/04_matmul_cube_only/` | Pure cube matmul | Matmul APIs, cube_only mode |
| 05 | matmul_leakyrelu | `~/workspace/pyasc/python/tutorials/05_matmul_leakyrelu/` | Matmul + LeakyReLU fusion | Matmul + vector fused op |

## Recommended starting point

For first-time kernel development, start with **01_add** (manual sync). It demonstrates:
- `@asc.jit` kernel decoration
- `GlobalTensor` and `LocalTensor` setup
- `data_copy` for GM <-> UB transfers
- `asc.add` for vector computation
- Manual `set_flag`/`wait_flag` synchronization
- `np.testing.assert_allclose` verification (numpy only)
- Multi-core launch with `kernel[cores, stream](...)`

## Complexity progression

```
01_add (basic, manual sync)
  -> 02_add_framework (framework sync)
    -> 03_matmul_mix (cube + vector)
      -> 04_matmul_cube_only (pure cube)
        -> 05_matmul_leakyrelu (fused op)
```
