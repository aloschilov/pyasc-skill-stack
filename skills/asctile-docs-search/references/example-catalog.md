# Shipped asctile examples

From the skill-stack root, use `reference/asctile/target/`:

- `test_vadd.py`: elementwise addition and tail shapes
- `test_reduce_sum.py`, `test_reduce_max.py`: reductions
- `test_softmax.py`, `test_rms_norm.py`, `test_layer_norm.py`: normalization
- `test_transpose.py`, `test_broadcast.py`, `test_gather.py`: movement and indexing
- `matmul_v3/`: matrix multiplication variants

These are upstream-authored references. `reference/asctile/PIN.json` records
their revision, original location and hashes. Consult the actual parametrized
cases rather than assuming every example is qualified for all shapes.

The starter is `skills/asctile-codegen-workflow/templates/kernel-template.py`.
It is a development template, not a correctness oracle. Existing generated
kernels and historical campaign results must not replace the upstream reference.
