# Delivery consistency for the skill stack

Use when reviewing repository changes, skill guidance or a ready-for-review PR.

- Compare the requested outcome with the final code, skill instructions, current
  documentation and acceptance evidence. Resolve contradictions before calling
  the work complete. A future issue may describe missing scope; it cannot count
  as completed acceptance.
- Describe current behavior in product documentation. Avoid PR numbers, queue
  updates, repair chronology and local transport details in user-facing guides.
  Put investigation history in external evidence or the review discussion.
- Keep qualification summaries small: tested revision and scope, result,
  limitations, durable archive link and checksum. Preserve original evidence;
  never rewrite an old run to imply it tested a newer profile or implementation.
- Treat an operator recipe as a bounded example, not generic coverage. Shared
  orchestration must select and validate its task contract explicitly. Adding
  another recipe requires its own numerical probe and qualification.
- Before pruning agent/team scaffolding, map design, implementation, review,
  verification, repair and tuning handoffs to maintained skills or tools. Keep
  usable loops available. If automation is absent, say so and track that gap
  with concrete issue acceptance criteria; do not silently remove the phase.
- Report `passed`, `failed`, `blocked` and `not-run` separately. An acceptance gate
  requiring complete coverage cannot be green when required cases are blocked.
- Inspect the final PR diff and current-head CI, not only previous successful
  runs. Keep logs, candidates and generated evidence outside checkout; commit
  checks, small regression fixtures and concise qualification references.
