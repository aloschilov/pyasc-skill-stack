# pyasc JIT Compile Options

## Decorator syntax

```python
@asc.jit                              # defaults
@asc.jit(always_compile=True)         # with options
```

## Compile parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kernel_type` | `asc.runtime.config.KernelType` | Auto-detected | Kernel type classification |
| `opt_level` | `int` (0-3) | Compiler default | Bisheng optimization level |
| `auto_sync` | `bool` | `False` | Let compiler insert sync automatically |
| `auto_sync_log` | `str` | None | File path to log auto-sync insertions |
| `matmul_cube_only` | `bool` | `False` | Pure cube mode (matrix compute only) |
| `always_compile` | `bool` | `False` | Force recompilation, bypass cache |

## Runtime parameters (launch syntax)

```python
kernel[core_num, stream](arg1, arg2, ...)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `core_num` | `int` | Yes | Number of cores to use (must not exceed hardware) |
| `stream` | Stream object | No | Execution stream from `rt.current_stream()` |

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
| Called from | Host: `kernel[cores, stream](...)` | Other `@asc.jit` functions |
| Compile options | Effective | Ignored |
| `return` | Not allowed | Allowed (top-level only) |
