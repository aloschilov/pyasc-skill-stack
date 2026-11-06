# Optional upstream tutorials

The shipped template and pinned target suite are the first resources for
ordinary kernel development; an additional source checkout is not required.

If a tutorial is needed, inspect the source revision named by
`reference/asctile/PIN.json`. Discover tutorial/example paths in that revision
with `git ls-tree -r --name-only <revision>`; paths such as
`python/tutorials/asctile/` were used historically and may have moved.

Check the programming model before copying code. Lower-level `asc` examples
using queues, `data_copy` and explicit synchronization are not direct asctile
syntax examples. Prefer examples using `@asctile.jit`, `global_tensor`,
`copy_in` and `copy_out`, and retain the source revision in the design record.
