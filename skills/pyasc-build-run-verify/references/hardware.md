# Hardware evaluation

Use a supported PyASC/CANN environment for direct execution (`host` means the
current interpreter, including inside a container), or submit a packaged
operator to CANNBench. A simulator pass is not hardware evidence.

For an external one-operator smoke, follow `integrations/cannbench/README.md`.
Retain the exact candidate, runtime and archive hashes, task contract, service
job identity, runner hardware, complete correctness results and timings outside
checkout. Report the tested scope and failed cases; do not infer generation
quality or a performance improvement from one successful hardware smoke.
