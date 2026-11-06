# Explicit source-runtime build

The normal skill and CI paths continue to consume their existing upstream image
digests. This optional path for [#26](https://github.com/aloschilov/pyasc-skill-stack/issues/26)
builds a separate development runtime; it does not replace a qualified runtime.
Initial implementation has offline tests. A clean build and simulator result must
be retained before this path is called qualified.

## Ownership and inputs

[compiler-team-ru/asctile](https://github.com/compiler-team-ru/asctile/tree/f5f95882a7a6c912d7a74cf6204fb9934d31e7a9)
owns the wheel compiler recipe. `scripts/source_runtime.py` invokes its unchanged
`ci/Dockerfile` and `ci/build.sh` from that exact commit. It does not copy the
compiler build logic, restore overlays, dispatch upstream publication or upload
to PyPI. `compiler-team-ru/pyasc-images` separately owns CANN environments and
images consuming published wheels.

The supported build target is **native Linux x86_64, CPython 3.12**. Other hosts,
ABIs and emulated architectures fail before the build. Inputs are:

| Input | Selected value |
| --- | --- |
| PyASC source | `d0f64a63c432761aa57648e67be7582780a17561` by default; explicit full commit override |
| Upstream recipe | `f5f95882a7a6c912d7a74cf6204fb9934d31e7a9` |
| manylinux_2_28 x86_64 base | `quay.io/pypa/manylinux_2_28_x86_64@sha256:076b19ef6e698f28649fa817c49d14443201e01da85240e8e52f93e8daa49bff` |
| LLVM | 19.1.7, `cd708029e0b2869e80abe31ddb175f7c35361f90`; upstream checksum-pinned export |
| Build dependencies | Upstream `ci/requirements-build.lock`; actual compiler/RPM/pip inventory in build passport |
| Runtime dependencies | numpy 1.26.4, pybind11 2.10.3, typing_extensions 4.16.0; downloaded wheel hashes retained |
| Qualification environment | `ghcr.io/compiler-team-ru/pyasc-env@sha256:8cf07d34df92d2433449099f7d78d20faa44b68769a89eec88bc5352943ab940` |
| Qualification ABI/platform | Ubuntu 22.04, CPython 3.12.15, CANN 9.2.0, Model Ascend950PR_9599 |

The environment digest is the **wheel-free** layer from the public
[upstream matrix](https://github.com/compiler-team-ru/pyasc-images/releases/tag/matrix-1.1.1.dev20261105121928-d162e39cbacc2dc9).
The matrix's prebuilt PyASC wheel is not substituted for the source build.
The wheel itself contains neither CANN nor a simulator. A simulator image adds
those components; a development image additionally supplies tools such as Torch
and pytest. Neither image is a CANNBench submission wheel.

## Build and qualify

Use Docker and Git on a dedicated native host, with output outside every checkout:

```bash
python3 scripts/source_runtime.py build \
  --root /outside-checkout/source-runtime-001 \
  --source d0f64a63c432761aa57648e67be7582780a17561
python3 scripts/source_runtime.py qualify --root /outside-checkout/source-runtime-001
```

Each root must be new. Build timeout is 30 minutes for the toolchain image and
60 minutes for the wheel. Qualification pulls its pinned environment explicitly,
then runs in a fresh network-disabled container with a 10-minute deadline. Only
the invocation's unique container is removed; there is no global Docker prune.
Build failures retain logs and inputs; a new attempt uses a new root.

Alternatively manually dispatch **Source runtime qualification** on the desired
branch. PRs run only offline tests. Its build and simulator jobs use separate
fresh hosted x86_64 runners and retain artifacts for 90 days. There is no cron,
model generation, NPU call, CANNBench submission or automatic publication.

The build retains the upstream wheel, source export, build passport, dependency
wheels and a file hash inventory. Upstream performs ABI/license checks and two
clean import installations. The stack then checks every input wheel's RECORD and
assembles an explicit `pyasc_runtime` wheel containing modern
`asc.experimental.asctile`, numpy, pybind11, typing_extensions and licenses.
It does not create a legacy `asctile` alias or adapt arbitrary kernels.

Qualification exercises the existing CANNBench packager **offline** with a small
fixed add fixture, installs that exact submission wheel into a fresh venv, and
compares all installed runtime payload bytes against the source-bundle manifest
and submission wheel. Separate fresh-cache runs check compilation, Model
execution and numerical rejection when launch is suppressed. The scope is
float32 add[128], one block; it is not official CANNBench task acceptance,
performance measurement or NPU qualification. The existing generated CI's
historical runtime pin stays unchanged.

Reproduction means verified source/configuration equivalence and recorded output
hashes. Wheel versions contain the build timestamp and the native toolchain may
produce different bytes; bit-reproducibility is not claimed. Preserve a selected
successful archive in a durable Release with checksum; ordinary Actions artifacts
alone do not satisfy long-term qualification retention.

## Recovery

If an old digest or binary disappears, first retrieve the unchanged archived
qualification and verify its checksum. If unavailable, retain its status as
unavailable. A source build creates a new runtime identity, even at the same
source revision. Re-run import, compilation and numerical checks, retain the new
archive and explicitly bind any later hardware submission to it. Never overwrite
the old digest, label, candidate or receipt to make a newer binary appear already
qualified. Unsupported alternate revisions fail normally; they do not fall back
to a different source or API.
