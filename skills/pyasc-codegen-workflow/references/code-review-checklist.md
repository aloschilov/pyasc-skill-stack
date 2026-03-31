# pyasc Code Review Checklist

## Mandatory checks for every kernel

### 1. Syntax compliance
- [ ] All constructs inside `@asc.jit` are in the supported set
- [ ] No `print`, `break`, `continue`, `lambda`, `yield`
- [ ] No `try/except`, `raise`, `with open`
- [ ] No `import` or `from...import` inside JIT
- [ ] No nested functions or `global`/`nonlocal`
- [ ] No class methods as kernels
- [ ] No `async def` or `async with`

### 2. `@asc.jit` correctness
- [ ] Kernel function decorated with `@asc.jit`
- [ ] Kernel function does not return a value
- [ ] Device helper functions may return (top-level only)
- [ ] Compile options (if any) are valid

### 3. Type correctness
- [ ] Kernel parameters use supported types (GlobalAddress, int, float, etc.)
- [ ] `asc.ConstExpr[T]` used for compile-time parameters
- [ ] Variables initialized before conditional use

### 4. Sync flags
- [ ] `set_flag`/`wait_flag` present between pipeline stages
- [ ] Correct events: MTE2_V (copy-in -> compute), V_MTE3 (compute -> copy-out), MTE3_MTE2 (copy-out -> next copy-in)
- [ ] Buffer ID matches between set_flag and wait_flag

### 5. Output verification
- [ ] Verification present (torch.allclose, numpy, or similar)
- [ ] Tolerance is reasonable (e.g., atol=1e-5 for float32)
- [ ] Reference computation is correct

### 6. Code quality
- [ ] Implementation matches design document
- [ ] No unnecessary hardcoded values
- [ ] Reasonable variable names
- [ ] Launch function properly configured (core_num, stream)
