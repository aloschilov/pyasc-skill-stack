# Current pyasc v2 tutorial catalog

Use `python/tutorials/asctile/` for high-level tile examples. Current notable
resources include fused softmax and the operator examples under `examples/`:

- `examples/06_gelu/gelu.py`
- `examples/07_swiglu/swiglu.py`
- `examples/08_rmsnorm/rmsnorm.py`

The numbered `examples/01_add` through `05_matmul_leakyrelu` demonstrate the
lower-level `asc` API and manual/framework synchronization. They remain useful
for hardware concepts, but their `GlobalTensor`, `data_copy`, queues, stream
launch, and sync syntax are not the current `asctile` kernel interface.
