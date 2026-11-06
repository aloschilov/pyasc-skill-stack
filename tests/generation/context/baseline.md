# PyASC baseline notes — paired-context-v2

These authored API notes are a separate experimental intervention. They are
not the retired AGENTS.md, a skill bundle, a reference kernel or a task solution.
The experiment records their exact bytes; no historical P4 equivalence is claimed.

The public device API is `asc.experimental.asctile`. Device functions use
`@asctile.jit`; the host entry point launches one device kernel with square-bracket
block count syntax. The harness provides tensors and preallocates the output.

Device tensors and host tensors are distinct. `asctile.global_tensor` describes
global memory; `asctile.copy_in` loads a device tile and `asctile.copy_out` writes
it back. Tile shapes and memory shapes must agree. Arithmetic, reduction and
broadcasting operate on device tensors inside the compiled function. Check the
result rank and shape against the output descriptor after a reduction. Do not
assume a scalar or vector automatically has the memory shape of its output.

Multiple blocks must partition work explicitly. Partial tiles require a valid
memory shape; a padded compute tile is not permission to access unallocated memory.
Constants and shapes used by the compiler must be expressible in the device DSL.
The Model simulator verifies values, not hardware performance.

Do not install packages, change runtime configuration, inspect the evaluator or
copy another candidate. The supplied task and common ABI contract take precedence.
