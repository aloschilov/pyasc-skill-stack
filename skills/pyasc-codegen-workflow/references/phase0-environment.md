# Phase 0: pin the environment

Record the selected `origin/v2` commit, source path, wheel hash/ABI/architecture,
Python and CANN versions, compiler path, platform, backend, and worker models.
Verify `asc` and `asctile` resolve to the matching runtime: the published
wheel provides `from asc.experimental import asctile`, while older source
builds provide a top-level `import asctile`. For the CANNBench campaign use
the vendored exact-v2 QEMU gate; do not substitute a repository package build
from another revision.

Exit only when source and runtime provenance match and at least compile-only
validation is available. Record camodel/NPU absence as a limitation.
