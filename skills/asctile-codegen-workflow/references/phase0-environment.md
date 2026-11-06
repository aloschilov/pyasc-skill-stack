# Phase 0: identify the selected runtime

For the repository workflow, read `reference/images/ENV.json` and
`reference/asctile/PIN.json`. Record the image digest, matching source revision,
Python/CANN versions, backend and platform. Reuse this runtime; no additional
source fetch or host installation is needed for ordinary authoring.

If a task explicitly selects another runtime, record its source revision and
wheel hash/ABI/architecture. The published wheel provides
`from asc.experimental import asctile`; older source builds may provide a
top-level `import asctile`. Do not mix evidence between these runtimes.

When verification is delegated to Docker, keep authoring on the host and use
the repository qualification command afterward. A missing host import is not
a reason to search unrelated checkouts. Record unavailable verification as
blocked; compile-only evidence does not establish numerical or NPU correctness.
