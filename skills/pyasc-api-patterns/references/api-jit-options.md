# pyasc asc2 JIT Compile Options

## Decorator syntax

```python
@asc2.jit(always_compile=True)        # standard for development
@asc2.jit                              # defaults (uses cache)
```

## Compile parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `always_compile` | `bool` | `False` | Force recompilation, bypass cache |
| `opt_level` | `int` (0-3) | Compiler default | Bisheng optimization level |
| `matmul_cube_only` | `bool` | `False` | Pure cube mode (matrix compute only) |

Note: `insert_sync=True` and `run_asc2_passes=True` are defaults for `@asc2.jit`.
Do not disable them unless debugging a specific issue.

## Launch syntax (asc2)

```python
kernel[core_num](arg1, arg2, ...)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `core_num` | `int` | Yes | Number of cores to use |

**asc2 does NOT use a stream argument.** The v1 syntax `kernel[core_num, stream](...)` must not be used.

## JIT cache behavior

- **Cache location**: `${PYASC_HOME}/.pyasc/cache` (or `${PYASC_CACHE_DIR}`)
- **Cache key**: compile options + kernel parameters + global variables + source code
- **Force rebuild**: Set `always_compile=True` or delete cache directory

## Environment variables

| Variable | Purpose |
|----------|---------|
| `PYASC_HOME` | Cache root directory (default: user home) |
| `PYASC_CACHE_DIR` | Specific cache directory |
| `PYASC_DUMP_PATH` | Save generated ASC-IR and Ascend C code for inspection |

## Kernel vs device function

| Aspect | Kernel function | Device function |
|--------|----------------|-----------------|
| Called from | Host: `kernel[cores](...)` | Other `@asc2.jit` functions |
| Compile options | Effective | Ignored |
| `return` | Not allowed | Allowed (top-level only) |
