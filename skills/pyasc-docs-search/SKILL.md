---
name: pyasc-docs-search
description: Local-first index for current pyasc v2 APIs, target tests, examples, docs, and CANN operator references. Use when verifying an API or finding an implementation reference.
---

# Current pyasc v2 resource routing

For a pinned campaign, its exact source snapshot outranks every checkout,
cached example, and web page. For the current CANNBench work use
the source snapshot verified at `0a631f70968c3cb7c33ce45330a85768dd5a6f06`.

## Search order

1. Operator `proto.yaml`, `cases.yaml`, and `golden.py`.
2. `python/asctile/__init__.py` for the exported tile API.
3. `python/asctile/language/` and `python/asctile/runtime/` for signatures and
   validation behavior.
4. `python/test/asctile/target/` for handwritten production-shape patterns.
5. `python/test/asctile/kernels/` and `python/tutorials/asctile/` for smaller
   examples.
6. `docs/python-api/`, `docs/programming-guide/`, and
   `docs/python_syntax_support.md`.
7. CANN `ops-*` source for semantic and tiling reference.

Use `rg` against the exact source. Useful searches:

```bash
rg -n "def copy_in|def copy_out|def global_tensor" python/asctile
rg -n "@asctile.jit|asctile.softmax|asctile.rms_norm" python/test/asctile
rg -n "reuse_alloc|vf_fusion|insert_sync" python/asctile/runtime
```

## Current package distinction

- `asctile`: high-level tile API used by current v2 kernels.
- `asc`: shared dtypes, addresses, runtime configuration, and lower-level API.

Do not translate a current kernel to `asc2`; that package was renamed away.
Do not translate `global_tensor/copy_in/copy_out` back to
`tensor/load/store`; those names are from older snapshots.

## CANN references

- `https://gitcode.com/cann/ops-math` — elementwise, reductions, movement
- `https://gitcode.com/cann/ops-nn` — normalization, matmul, neural-network ops
- `https://gitcode.com/cann/ops-transformer` — transformer-specific ops
- `https://gitcode.com/cann/ops-cv` — vision ops
- `https://gitee.com/ascend/samples` — AscendC examples

Use the CANN implementation to understand semantics, tiling, buffering, and
hardware assumptions. The deliverable remains pyasc/asctile code.

## Evidence rule

When an API or workaround matters, cite the exact source file and commit.
Treat behavior copied from a different v2 snapshot as a hypothesis until the
pinned compile/runtime gate confirms it.

## References

- [API index](references/api-index.md)
- [Example catalog](references/example-catalog.md)
- [Tutorial catalog](references/tutorial-catalog.md)
