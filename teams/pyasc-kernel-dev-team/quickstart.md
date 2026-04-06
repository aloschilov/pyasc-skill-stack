# pyasc Kernel Development Quick Start

## Prerequisites

1. Python 3.10.x (pyasc and torch are installed under this version)
2. pyasc >= 1.1.1 (`pip install pyasc`)
3. CANN Toolkit (`source $HOME/Ascend/cann/set_env.sh`)
4. numpy < 2.0 (`pip install "numpy<2"`)
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

1. Review relevant tutorials in `golden/tutorials/`
2. Select APIs from `skills/pyasc-api-patterns/`
3. Check syntax constraints in `skills/pyasc-syntax-constraints/`
4. Write `kernels/my_kernel/docs/design.md`

### Step 3: Implement

1. Copy `skills/pyasc-codegen-workflow/templates/kernel-template.py` to `kernels/my_kernel/kernel.py`
2. Implement the kernel function with `@asc.jit`
3. Add output verification with `torch.allclose`

### Step 4: Verify

```bash
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
python3.10 kernels/my_kernel/kernel.py -r Model -v Ascend910B1
```

## Example: vector add

See the completed example at `kernels/add/`:

```bash
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
python3.10 kernels/add/kernel.py -r Model -v Ascend910B1
```
