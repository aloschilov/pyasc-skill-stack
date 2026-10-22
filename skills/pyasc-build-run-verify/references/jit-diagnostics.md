# pyasc JIT Diagnostics Guide

## Debugging compilation issues

### Step 1: Enable dump output

```bash
export PYASC_DUMP_PATH=/tmp/pyasc_dump
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
python3.10 kernel.py -r Model -v Ascend950PR_9599
```

Check `$PYASC_DUMP_PATH` for:
- Generated ASC-IR (MLIR format)
- Generated Ascend C code
- Compilation logs

### Step 2: Force recompile

```python
@asctile.jit(always_compile=True)
def kernel(...):
    ...
```

This bypasses the JIT cache and forces a fresh compilation.

### Step 3: Reduce optimization

```python
@asctile.jit(opt_level=0)
def kernel(...):
    ...
```

Lower optimization may produce clearer error messages.

### Step 4: Enable auto-sync logging

```python
@asctile.jit(insert_sync=True)
def kernel(...):
    ...
```

Inspect `PYASC_DUMP_PATH` output to see the inserted synchronization.

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
| `ImportError: asc` or `asctile` | Source/runtime mismatch or pyasc not installed | Install the wheel built from the selected v2 commit |
| `CANN not found` | Missing CANN toolkit | source set_env.sh |

### Verification errors

| Error | Cause | Fix |
|-------|-------|-----|
| `AssertionError: allclose` | Output mismatch | Check computation logic, sync order |
| Large diff values | Precision issue | Check dtype, consider float32 |
| All zeros output | Missing data copy or sync | Verify data_copy and sync flags |

## Cache management

```bash
# Isolate an experiment without deleting shared/user caches.
PYASC_PROBE_DIR=$(mktemp -d /tmp/pyasc-jit-probe.XXXXXX)
export PYASC_CACHE_DIR="$PYASC_PROBE_DIR/cache"
export PYASC_DUMP_PATH="$PYASC_PROBE_DIR/dump"
```

Run each option/import-order probe in a fresh process. Preserve the directory
with its provenance; forced recompilation does not reset in-process keyword
discovery. For timing, separate compile/verify_sync/debug work from warmed
execution and record the exact simulator library family and clock units.
