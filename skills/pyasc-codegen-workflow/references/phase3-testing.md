# Phase 3: Verification and Delivery

## Purpose

Final verification of the kernel and preparation for delivery.

## Prerequisites

- Phase 2 completed with acceptance score >= 8.5
- kernel.py runs and verification passes

## Process

### Step 1: Verification

Run the kernel with available backends:

```bash
# Model backend (always available with CANN)
python kernel.py -r Model

# NPU backend (if hardware available)
python kernel.py -r NPU -v Ascend910B
```

### Step 2: Record results

Document verification results:

| Backend | Status | Output |
|---------|--------|--------|
| Model | Pass/Fail | Verification output |
| NPU | Pass/Fail/N/A | Verification output |

### Step 3: Limitation statement

If NPU hardware is unavailable:
- State: "Verified on Model backend only. NPU verification pending hardware availability."
- This is acceptable for the first vertical slice.

### Step 4: Delivery

Provide:
- `kernel.py` — complete, verified kernel implementation
- `docs/design.md` — design document
- `docs/environment.json` — environment snapshot
- Verification results summary

## CP-3 Exit Conditions

- [ ] Kernel runs on at least one backend
- [ ] Output verification passes
- [ ] Limitations stated explicitly if runtime unavailable
- [ ] All deliverables present
