# Current pyasc v2 example catalog

Relative to the selected source root:

- `python/test/asctile/target/`: handwritten production-oriented target tests
- `python/test/asctile/kernels/`: focused end-to-end AscTile kernels
- `python/test/asctile/unit/`: API-level behavior where present
- `python/tutorials/asctile/`: current high-level tutorials
- `examples/06_gelu`, `examples/07_swiglu`, `examples/08_rmsnorm`: runnable
  operator examples
- `python/test/kernels/`: lower-level `asc` API tests; do not copy them as
  AscTile syntax without translating the programming model

Choose the nearest example from the exact selected commit. Record a different
commit as historical evidence rather than treating it as an API contract.
