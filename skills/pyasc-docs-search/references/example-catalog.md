# pyasc Example and Test Catalog

## Test examples by category

### Unit tests (`~/workspace/pyasc/python/test/unit/`)

| Category | Path | Coverage |
|----------|------|----------|
| Codegen | `unit/codegen/` | AST visitor, code generation |
| Basic APIs | `unit/language/basic/` | Vector ops, data copy |
| Core APIs | `unit/language/core/` | Types, enums, sync |
| Advanced APIs | `unit/language/adv/` | Matmul, advanced ops |
| Framework APIs | `unit/language/fwk/` | TPipe, TQue |
| Host lib | `unit/lib/host/` | Host-side helpers |
| Runtime | `unit/runtime/` | Config, compilation |

### Kernel tests (`~/workspace/pyasc/python/test/kernels/`)

End-to-end kernel tests that compile and run on Model or NPU backend.

### Generalization tests (`~/workspace/pyasc/python/test/generalization/`)

| Category | Path |
|----------|------|
| Basic API generalization | `generalization/basic/` |
| Advanced API generalization | `generalization/adv/` |

### Backend tests (`~/workspace/pyasc/test/`)

| Category | Path |
|----------|------|
| Dialect tests | `test/Dialect/` |
| Target tests | `test/Target/AscendC/` |
| Tool tests | `test/tools/` |

## Useful test patterns

- **Mock launcher**: `python/test/unit/` uses `mock_launcher_run` for testing without hardware
- **Pytest**: Standard pytest runner: `pytest ./python/test/unit/language/basic/test_vector_binary.py`
- **Build + run**: `bash test/build_llt.sh --run_python_ut`
