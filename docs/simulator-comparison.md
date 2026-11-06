# Explicit simulator timing comparison

This optional research command restores the bounded comparison in
[#21](https://github.com/aloschilov/pyasc-skill-stack/issues/21). It selects one
official Ascend C Add sample: float32 `[8,2048]`, eight blocks. It does not restore
the retired global capabilities catalog or generate candidates.

## Recorded initial experiment

At `b3bca27`, the fixed repository candidate completed the experiment in
**62.00 seconds**, on native ARM64 Docker with the pinned CANN 9.2.0/PyASC
`d0f64a6` image. Each variant had one warm-up and five measured launches:
**18 Model launches, 54 numerical checks and three rejected suppressed-launch
controls**. All **176 runtime/compiler hashes**, including the loaded runtime
wrapper, matched between variants.

| Variant | Compile seconds | Median launch ticks |
| --- | ---: | ---: |
| Official Ascend C sample | 0.930 | 3612 |
| PyASC, fusion disabled | 0.906 | 3476 |
| Same PyASC, fusion enabled | 0.890 | 3476 |

The reference/candidate median ratio is **1.0391**. Fusion produced the **same
binary bytes**, with median ratio **1.0**; this sample demonstrates no fusion
benefit. Five samples do not establish statistical significance, generalized
optimization benefit or NPU performance. This qualifies the fixed comparison
protocol, not other operators, inputs or architectures.

[Durable experiment archive](https://github.com/aloschilov/pyasc-skill-stack/releases/download/qualification-timing-b3bca27/qualification-timing-b3bca27.tar.gz)
contains both attempts, original reference/license, compiler commands, binaries,
per-launch samples, input/runtime hashes and 199 indexed files.
SHA256: `da7e7b2f84d93ccb0e380704db457989302d622928eccc28e53b8994f303c795`.
The initial successful attempt predates the enforced loaded-wrapper check and
is retained separately from the accepted second attempt.

## Inputs and protocol

The canonical source is `cann/asc-devkit` commit
`a63d5261c7c8077a0081e867f43a0e884fec066b` (v9.2.0),
`examples/01_simd_cpp_api/00_introduction/01_add/add_tpipe_tque/add_tpipe_tque.asc`.
The executor checks its exact bytes, retains its license and original source, and
extracts the unchanged device kernel and copyright notice. Only the NPU host
`main` and its unused ACL/data-file includes are excluded. This is the official
sample algorithm, not the optimized ACLNN Add implementation or an NPU baseline.

Use a native Linux Docker daemon and the already pulled image pinned by
`reference/images/ENV.json`. Keep the reference checkout and output outside this
checkout. No source is downloaded implicitly:

```bash
git clone https://gitcode.com/cann/asc-devkit.git /outside/asc-devkit
git -C /outside/asc-devkit checkout a63d5261c7c8077a0081e867f43a0e884fec066b
python3 scripts/simulator_comparison.py run \
  --reference /outside/asc-devkit \
  --candidate tests/fixtures/simulator-timing/candidate.py \
  --output /outside/timing-001 --warmup 1 --samples 5
```

An explicit candidate must export a PyASC JIT function `candidate(x, y, out)` for
that specification. The shipped candidate is handwritten. Its unchanged bytes
are compiled twice, with `vf_fusion=False` and `True`; requested and effective
options must differ only in that field. The canonical Ascend C sample retains
its explicit queue synchronization and is compiled without automatic sync.
All compiler commands, binaries, options and input/runtime hashes are retained.

Each of the three variants runs in a fresh Model process inside the same pinned,
network-disabled container. Compilation is timed separately. The binary is
registered once; one warm-up launch precedes five measured launches by default.
Each sample reads `current_tick()` immediately before kernel launch and after
synchronization. Host/device copies and NumPy checks are outside this interval.
Every launch checks output and preservation of both inputs. A final suppressed
launch must reach the launch boundary and fail the output check. Static-only,
missing, blocked, nonpositive or unpaired results fail qualification.

The output reports per-launch deltas, medians and paired descriptive ratios.
Warm-up values are retained but excluded. Ticks are simulator measurements, not
NPU speedup, statistical significance or a fixed acceptance threshold. The
model's whole-process `Total tick` log is not used as a sample. Runtime payloads
must match across variants; do not combine different architectures or runs.
The 360-second container deadline removes only that invocation's container and
retains failure evidence. Each attempt requires a new output directory.

There is no schedule, model API, hardware submission or implicit runtime build.
Keep artifacts outside Git; preserve an accepted run in a durable archive with
checksum before documenting its result here.
