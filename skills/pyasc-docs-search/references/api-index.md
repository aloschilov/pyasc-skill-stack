# pyasc API Index

## Core Types (`asc.language.core`)

| API | Description |
|-----|-------------|
| `asc.GlobalTensor` | Global memory tensor handle |
| `asc.LocalTensor` | Local (UB) memory tensor |
| `asc.GlobalAddress` | Global memory address (kernel parameter type) |
| `asc.TPosition` | Tensor logical position enum (VECIN, VECOUT, etc.) |
| `asc.HardEvent` | Pipeline synchronization event enum |
| `asc.ConstExpr[T]` | Compile-time constant parameter wrapper |
| `asc.get_block_idx()` | Get current core's block index |
| `asc.BinaryRepeatParams` | Repeat parameters for binary ops |

## Basic Vector Operations (`asc.language.basic`)

| API | Description |
|-----|-------------|
| `asc.add(dst, src1, src2, count)` | Element-wise addition |
| `asc.sub(dst, src1, src2, count)` | Element-wise subtraction |
| `asc.mul(dst, src1, src2, count)` | Element-wise multiplication |
| `asc.div(dst, src1, src2, count)` | Element-wise division |
| `asc.data_copy(dst, src, count)` | Data copy between memory levels |

## Sync Primitives (`asc.language.core`)

| API | Description |
|-----|-------------|
| `asc.set_flag(event, flag_id)` | Signal pipeline event |
| `asc.wait_flag(event, flag_id)` | Wait for pipeline event |

## Advanced APIs (`asc.language.adv`)

| API | Description |
|-----|-------------|
| Matmul APIs | Matrix multiplication (see tutorials 03-05) |

## Framework APIs (`asc.language.fwk`)

| API | Description |
|-----|-------------|
| TPipe, TQue | Framework-managed pipeline and queues |

## Runtime and Configuration

| API | Description |
|-----|-------------|
| `asc.jit` | JIT compilation decorator |
| `asc.runtime.config.Backend` | Execution backend (Model, NPU) |
| `asc.runtime.config.Platform` | Target platform |
| `asc.runtime.config.set_platform(backend, platform)` | Set execution platform |
| `asc.lib.runtime.current_stream()` | Get current execution stream |

## Documentation paths

| Resource | Path |
|----------|------|
| Full API docs | `~/workspace/pyasc/docs/python-api/` |
| Basic API source | `~/workspace/pyasc/python/asc/language/basic/` |
| Core API source | `~/workspace/pyasc/python/asc/language/core/` |
| Advanced API source | `~/workspace/pyasc/python/asc/language/adv/` |
| Framework API source | `~/workspace/pyasc/python/asc/language/fwk/` |
