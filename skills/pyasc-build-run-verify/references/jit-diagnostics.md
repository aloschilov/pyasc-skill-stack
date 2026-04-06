# pyasc JIT Diagnostics Guide

## Debugging compilation issues

### Step 1: Enable dump output

```bash
export PYASC_DUMP_PATH=/tmp/pyasc_dump
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
python3.10 kernel.py -r Model -v Ascend910B1
```

Check `$PYASC_DUMP_PATH` for:
- Generated ASC-IR (MLIR format)
- Generated Ascend C code
- Compilation logs

### Step 2: Force recompile

```python
@asc.jit(always_compile=True)
def kernel(...):
    ...
```

This bypasses the JIT cache and forces a fresh compilation.

### Step 3: Reduce optimization

```python
@asc.jit(opt_level=0)
def kernel(...):
    ...
```

Lower optimization may produce clearer error messages.

### Step 4: Enable auto-sync logging

```python
@asc.jit(auto_sync=True, auto_sync_log="sync.log")
def kernel(...):
    ...
```

Review `sync.log` to see what sync instructions the compiler inserts.

## Common error patterns

### Syntax errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Unsupported AST node` | Used unsupported Python syntax | Check pyasc-syntax-constraints |
| `Cannot resolve type` | Wrong parameter type | Use supported types or ConstExpr |
| `Undefined variable` | Variable only defined in one if-branch | Initialize before if, or use ConstExpr |

### Runtime errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: core_num` | Invalid core count | Use count <= hardware cores |
| `ImportError: asc` | pyasc not installed | pip install pyasc |
| `CANN not found` | Missing CANN toolkit | source set_env.sh |

### Verification errors

| Error | Cause | Fix |
|-------|-------|-----|
| `AssertionError: allclose` | Output mismatch | Check computation logic, sync order |
| Large diff values | Precision issue | Check dtype, consider float32 |
| All zeros output | Missing data copy or sync | Verify data_copy and sync flags |

## Cache management

```bash
# Clear JIT cache
rm -rf ${PYASC_HOME:-$HOME}/.pyasc/cache

# Set custom cache location
export PYASC_CACHE_DIR=/tmp/pyasc_cache
```
