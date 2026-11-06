---
name: asctile-docs-search
description: Find pinned PyASC/asctile API documentation, reference kernels and task semantics. Use when checking an API or choosing a correctness reference.
---

# PyASC/asctile resource routing

Start with the skill-stack checkout that supplied this skill. Resolve paths
below from that checkout, including when skills are installed through symlinks.
Do not search unrelated workspaces for an apparently matching runtime.

## Source authority

For the repository workflow, `reference/images/ENV.json` selects the runtime
and `reference/asctile/PIN.json` identifies the matching upstream source and
resolved target-suite path. Read these files rather than assuming a branch,
source layout or historical campaign commit.

An explicitly selected campaign can use another source/runtime pair. Record
that choice and keep its evidence separate; it does not replace the repository
pin for ordinary kernel development.

## Search order

1. The user's operator contract: signature, shapes, dtypes and tolerances.
2. `asctile-api-patterns` and `asctile-syntax-constraints` for the supported surface.
3. `reference/asctile/target/` for upstream reference kernels and test semantics.
4. The matching runtime/source when an exact signature remains uncertain.
5. CANN operator sources when additional semantic or tiling context is needed.

Useful local searches, from the skill-stack root:

```bash
rg -n 'copy_in|copy_out|global_tensor' skills/asctile-api-patterns reference/asctile/target
rg -n 'reduce_sum|softmax|rms_norm' reference/asctile/target
```

A source checkout is optional for ordinary authoring. If needed, obtain the
revision named by the pin and discover its paths there; old `python/asctile`
paths are not stable across upstream revisions. If the runtime is in Docker,
inspect it there instead of attempting imports on the host.

## Package and evidence rules

The published runtime uses `from asc.experimental import asctile`. Older source
builds may expose top-level `asctile`. Match the chosen runtime; do not translate
new tile kernels to the obsolete `asc2` or `tensor/load/store` interface.
Generated kernels are test subjects, never correctness references. Cite the
source path and revision for consequential API decisions; compilation or
simulator results establish only the tested behavior.

## Focused references

- [API index](references/api-index.md): operations to look up.
- [Example catalog](references/example-catalog.md): shipped reference kernels.
- [Tutorial catalog](references/tutorial-catalog.md): optional upstream tutorials.
