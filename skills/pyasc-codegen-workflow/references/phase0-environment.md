# Phase 0: Environment Preparation

## Purpose

Ensure the development environment is ready and the kernel project directory is properly initialized before any design or implementation work begins.

## Steps

### Step 1: Initialize kernel project

Run the initialization script to create the project directory:

```bash
bash <skills_path>/pyasc-codegen-workflow/scripts/init_kernel_project.sh {kernel_name}
```

This creates:
- `kernels/{kernel_name}/` — project root
- `kernels/{kernel_name}/docs/` — for design.md and environment.json
- `kernels/{kernel_name}/test/` — for test data
- `kernels/{kernel_name}/README.md` — project description

### Step 2: Verify environment

Run the environment verification script:

```bash
bash <skills_path>/pyasc-codegen-workflow/scripts/verify_environment.sh {kernel_name}
```

This checks and saves to `environment.json`:
- Python version (must be 3.9-3.12)
- pyasc installation status
- numpy version (must be < 2.0)
- torch and torch_npu availability
- CANN toolkit status
- NPU hardware availability
- Model backend availability

### Step 3: Review environment.json

Read `kernels/{kernel_name}/docs/environment.json` and confirm:
- pyasc is installed
- At least one backend (Model or NPU) is available
- No critical errors

## CP-0 Exit Conditions

- [ ] Project directory `kernels/{kernel_name}/` exists
- [ ] Subdirectories `docs/` and `test/` exist
- [ ] `environment.json` is saved and contains valid data
- [ ] No blocking environment issues (pyasc installed, backend available)

## Common issues

| Issue | Fix |
|-------|-----|
| pyasc not installed | `pip install pyasc` or build from source |
| numpy >= 2.0 | `pip install "numpy<2"` |
| CANN not found | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| No backend available | Install CANN toolkit for Model backend |
