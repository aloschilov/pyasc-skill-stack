You are implementing ONE operator for the CANN Bench benchmark (https://gitcode.com/cann/cann-bench) as a pyasc asctile JIT kernel module targeting an Ascend 950PR NPU. You have NO NPU access — the official harness evaluates your file on real hardware after you finish. Score per operator = 0.2 compile + 0.3 accuracy (all {{N_CASES}} cases must pass a relative-error check against golden.py) + 0.5 performance (profiler-measured kernel time vs an aclnn baseline).

# Task

Operator: **{{OP_NAME}}**. Write the file `candidate.py` in the current working directory: a complete Python module whose public callable `{{CALLABLE}}` implements the schema below. Nothing else — no tests, no other files.

# Operator specification

## desc.md

{{DESC}}

## proto.yaml

```yaml
{{PROTO}}
```

## golden.py — reference semantics ONLY (you must NOT compute with torch like this; reimplement the math in asctile kernels)

```python
{{GOLDEN}}
```

## Evaluation cases your module must handle (shapes, dtypes, value ranges, attrs)

{{CASES_SUMMARY}}

# Reference module — sigmoid.py from this submission (structure to copy; it scores 100% accuracy on this harness)

```python
{{REFERENCE_MODULE}}
```

{{CONSTRAINTS}}

# Operator-specific guidance

{{GUIDANCE}}

# Deliverable

Write ONLY `candidate.py` (complete, self-contained module). Public callable named exactly `{{CALLABLE}}` with the exact schema signature including attr defaults. Think hard about numerical stability across the full value ranges listed above, and about the UB budget for your op chain before picking TILE.

IMPORTANT — no local execution: pyasc/asc/asctile/torch_npu are NOT installed on this machine and there is no NPU here. Do NOT run, import, compile-check, or test your code, and do not install packages — any such attempt wastes your entire time budget. Reason statically, write `candidate.py`, optionally run `python3 -m py_compile candidate.py` (syntax only), then STOP and reply DONE.
