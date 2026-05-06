# pyasc Kernel Development Quick Start (asc2 API)

## Prerequisites

1. Python 3.10.x (pyasc is installed under this version)
2. pyasc with asc2 support (`pip install pyasc`)
3. CANN Toolkit (`source $HOME/Ascend/cann/set_env.sh`)
4. numpy (`pip install numpy`)
5. pytest >= 7.0 (`pip3.10 install "pytest>=7.0"`)

See [docs/cann-setup.md](../../docs/cann-setup.md) for detailed CANN environment setup.

## Verify environment

```bash
bash skills/pyasc-env-check/scripts/check_env.sh
```

## Develop a new kernel

### Step 1: Initialize project

```bash
bash skills/pyasc-codegen-workflow/scripts/init_kernel_project.sh my_kernel
bash skills/pyasc-codegen-workflow/scripts/verify_environment.sh my_kernel
```

### Step 2: Design

1. Review the golden asc2 kernel: `golden/kernels/abs_f16.py`
2. Review the external asc2 reference: `~/workspace/pyasc/python/test/kernels/asc2/test_vadd.py`
3. Select APIs from `skills/pyasc-api-patterns/` (asc2 tensor/load/store pattern)
4. Check syntax constraints in `skills/pyasc-syntax-constraints/`
5. Write `kernels/my_kernel/docs/design.md`

### Step 3: Implement

1. Copy `skills/pyasc-codegen-workflow/templates/kernel-template.py` to `kernels/my_kernel/kernel.py`
2. Implement the kernel function with `@asc2.jit(always_compile=True)`
3. Pattern: `asc2.tensor` -> `asc2.load` -> compute -> `asc2.store`
4. Add output verification with `np.testing.assert_allclose`

### Step 4: Verify

```bash
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
python3.10 kernels/my_kernel/kernel.py -r Model -v Ascend950PR_9599
```

Or via pytest:

```bash
pytest kernels/my_kernel/kernel.py --backend Model --platform Ascend950PR_9599
```

## Example: abs_f16 (asc2)

See the golden reference at `golden/kernels/abs_f16.py`:

```bash
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
python3.10 golden/kernels/abs_f16.py -r Model -v Ascend950PR_9599
```
