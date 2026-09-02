# CANN Bench integration

Runs the official [CANN Bench](https://gitcode.com/cann/cann-bench) harness
against pyasc asc2 kernels, producing scores comparable with the
[cannbench.com leaderboard](https://cannbench.com/leaderboard).

## Layout

| Path | Role |
|---|---|
| `tasks/<op>/` | Vendored cann-bench task specs (`proto.yaml`, `desc.md`, `cases.yaml`, `cases.csv`, `golden.py`) used as generation inputs. The current set covers all eight L1 ops plus the L2 `rms_norm` expansion task. |
| `submission/` | Submit-ready cann-bench source dir: `build.sh` + `setup.py` + `cann_bench/` package. `bash build.sh` produces `dist/cann_bench-*.whl` (rule `EVAL-CANN-003`). |
| `submission/cann_bench/<op>.py` | One public callable per operator, name/signature matching the op's `proto.yaml` schema. All numerical work happens in `@asc2.jit` kernels. |
| `submission/cann_bench/_pyasc_runtime.py` | One-time `config.set_platform(Backend.NPU, <SoC>)`; override SoC with `CANN_BENCH_PYASC_PLATFORM`. |
| `setup-site-mcp.sh` / `site-mcp.sh` | Install and launch the official CANNBench MCP for local OpenCode use. Credentials stay in gitignored `.secrets/cannbench.env`. |

## How the interop works

The harness (`src/kernel_eval/eval/op_runner.py` in cann-bench) calls each
submitted callable with **NPU-resident torch tensors** under anti-cheat
guards (`TorchOpGuard` blocks torch builtin math; `DeviceResidencyGuard`
blocks bulk NPU-to-CPU outflow). pyasc's launcher accepts NPU torch tensors
zero-copy: `TorchNpuTensorArgument.copy_to_device()` returns
`tensor.data_ptr()` (see `asc/runtime/memory_handle.py` in pyasc). So each
wrapper is just:

```python
out = torch.empty_like(x)          # torch-owned NPU buffer
kernel[cores](x, out, size, ...)   # zero-copy launch on data_ptr()s
return out
```

Performance is measured by `torch_npu.profiler` kernel_details (device-side
CANN profiling), which captures pyasc-launched kernels like any other device
task.

## Kernel design (shared by all four pilot ops)

- Pattern A (1-D flatten) with a **grid-stride tile loop** so any element
  count works, and `real_shape` loads/stores for tail tiles (no host
  padding, no extra device kernels in the wrapper).
- `TILE = 2048` for sigmoid/exp, 1024 for mish/gelu-tanh, 512 for
  gelu-erf (UB budget under static allocation grows with the op chain),
  up to 72 cores (C310/950PR), `unroll_factor=2` double-buffering.
- f16/bf16 promoted to f32 in-kernel; **cancellation-free math forms**
  for the f32 negative tails (required by wide-range cases like
  [-88, 88] under the 1.2e-4 relative-error threshold):
  - mish: exact identity `tanh(softplus(x)) = (1+2w)/(1+2w+2w²)` for
    `x >= 0`, `(w²+2w)/(w²+2w+2)` for `x < 0`, `w = e^-|x|` — the naive
    `log(1+e^x)` flushes to 0 for `x < -16`.
  - gelu tanh mode: `1 + tanh(u) = 2·sigmoid(2u)` with the stable
    sigmoid `e^min(s,0) / (1 + e^-|s|)`.
  - gelu erf mode: `1 + erf(v)` via the Numerical Recipes erfc fit
    `erfc(z) = t·e^(-z²+P(t))`, `t = 1/(1+z/2)` (rel. err < 1.2e-7).
  - sigmoid/exp rely on IEEE saturation to hit the correct limits at ±inf.

## Historical local NPU evaluation

The original pilot used a PR-comment tunnel to a 950PR box. That transport is
retired and must not be used. The paths below are retained only as historical
reproduction metadata:

- cann-bench clone: `/home/l00958488/cann-bench` (950PR baselines shipped in
  `tasks/metadata/950pr.json`)
- eval venv: `/home/l00958488/cbvenv` (torch + torch_npu 2.10.0)
- pyasc: `/home/l00958488/pyasc-fork1` (built cp310 `libpyasc`)
- env wrapper: `/home/l00958488/cbenv.sh` (CANN 9.1.0 + venv + PYTHONPATH)

```bash
# on the remote box
bash cbenv.sh bash /home/l00958488/submission/build.sh
cd /home/l00958488/cann-bench
bash /home/l00958488/cbenv.sh ./scripts/run_evaluation.sh \
    --source-dir /home/l00958488/submission \
    --task-dir tasks/level1/sigmoid --operator Sigmoid \
    --device-id 0 --no-perf     # correctness first, then drop --no-perf
```

Reports land in `reports/` on the box; archived copies live in
`evidence/cannbench/` here.

> **Perf-stage prerequisite:** the harness's own `cann_bench_utils`
> C++/AscendC clone kernel (anti-cheat input rotation) must be built for
> the box's SoC. The default/auto-detect fallback is `ascend910b`, which
> crashes a 950PR with device error 507035 during perf collection while
> correctness still passes. Fix: `bash src/cann_bench_utils/build.sh
> --soc=ascend950 --clean` + reinstall the wheel into the eval venv.

## Results (2026-08-29, 950PR, full scoring, 8/8 L1 operators)

Phase 2 (`workers/` — opencode GLM-5.2 workers with definite prompts and a
serialized private CANNBench eval queue) tuned the 4 pilot ops and generated
the other 4 L1 ops. All accepted kernels pass 20/20 cases.

| Operator | Score | Perf | Avg speedup | Notes |
|---|---:|---:|---:|---|
| Sigmoid | 71.86 | 21.86 | 0.86x | tuned (pilot 68.65) |
| Exp | 71.39 | 21.39 | 0.85x | tuned (pilot 69.43) |
| Mish | 61.89 | 11.89 | 0.43x | tuned (pilot 61.13) |
| Gelu | 57.79 | 7.79 | 0.30x | tuned (pilot 57.19) |
| MaskedScale | 83.68 | 33.68 | 1.63x | generated |
| SwiGlu | 64.46 | 14.46 | 0.59x | generated |
| ForeachAddcdivScalar | 82.62 | 32.62 | 1.28x | generated |
| ForeachNorm | 67.46 | 17.46 | 0.75x | generated |

Leaderboard comparison (EasyAsc/PYPTO on the same hardware class) and
campaign findings: `evidence/cannbench/comparison.md`. Worker pipeline
usage: `workers/README.md`.

### Private site validation with pyasc v2

The self-contained submission vendors `compiler-team/pyasc` branch `v2` at
commit `ac1222a48c8914d3f81297c7570d1a84f0f26778`. It builds a
`cp312-cp312-linux_x86_64` CANNBench wheel offline, including `asc`, `asc2`,
the native `libpyasc` extension, and pybind11's runtime build helper.

Private CANNBench job `job_5a902df0e8c1` (`sub_a395400f9c50`) succeeded on
950PR with 20/20 MaskedScale cases, score **82.6074**, geometric-mean speedup
1.5539×, and no anti-cheat failures. The runner used Python 3.12.13 and
`Ascend950PR_957c`; platform selection is intentionally auto-detected so the
same submission also works on other 950PR revisions. Full job data and stage
logs are archived in `evidence/cannbench/site_prebuilt_job_5a902df0e8c1.json`
and `evidence/cannbench/site_prebuilt_job_5a902df0e8c1_logs.json`.

### Full official private run (2026-09-02)

Job `job_cd51d6c2ca67` (`sub_d13af9d97a79`) evaluated the complete package on
950PR: all eight L1 operators plus the L2 RMSNorm expansion. It passed 178/180
cases with no anti-cheat failures and an aggregate geometric-mean speedup of
0.5772×.

| Operator | Cases | Score | Avg speedup |
|---|---:|---:|---:|
| Exp | 20/20 | 68.72 | 0.745× |
| ForeachAddcdivScalar | 20/20 | 79.76 | 1.181× |
| ForeachNorm | 20/20 | 62.52 | 0.498× |
| Gelu | 20/20 | 56.37 | 0.238× |
| MaskedScale | 20/20 | 82.38 | 1.546× |
| Mish | 20/20 | 60.60 | 0.379× |
| Sigmoid | 20/20 | 68.92 | 0.752× |
| SwiGlu | 18/20 | 55.81 | 0.459× |
| RmsNorm | 20/20 | 57.79 | 0.338× |

RmsNorm is accepted into the canonical package. SwiGlu remains the only
official correctness gap: case 9 hits a missing CANN 9.1 `Slice` binary in
the host-side narrow/contiguous fallback, and case 12 reports mismatched NaN
positions for the same degenerate-layout path. Full job data, logs, and the
compact per-operator result are archived as
`evidence/cannbench/site_full_job_cd51d6c2ca67.json`,
`evidence/cannbench/site_full_job_cd51d6c2ca67_logs.json`, and
`evidence/cannbench/full_official_9op_summary.json`.

A targeted kernel-only fallback rerun, job `job_0f5fdee4443a`
(`sub_37a5f6fb2734`), terminated with subprocess failures on all 20 cases.
The site report exposes only `rc=1`, without stderr or a traceback; local
pyasc/CaModel compilation succeeds, so the exact runner-side failure cannot
be diagnosed from the retained report. The rejected candidate and complete
site evidence are archived under
`workers/runs/20260901_full_official/eval2_swi_glu_fix/` and
`evidence/cannbench/swi_glu_fix_failed_summary.json`. The canonical SwiGlu
was restored to the version validated at 18/20 above. No submission credits
remain for another official run in the current quota window.

## CANNBench site MCP

The project-level `opencode.json` registers `cann-bench-site`. Local setup:

```bash
bash integrations/cannbench/setup-site-mcp.sh
# Create .secrets/cannbench.env (mode 0600) with BENCHSITE_API_TOKEN,
# then verify:
opencode mcp list
```

The local ARM host pins `mcp<2`: MCP 2.x currently pulls a cryptography wheel
that raises `Illegal instruction` on this CPU. The CANNBench adapter supports
MCP >=1.2, and 1.29.x connects successfully.

For a fresh x86_64 pyasc runtime wheel on this ARM host, run
`bash integrations/cannbench/build-runtime-qemu.sh`. The script verifies or
installs the Docker `amd64` binfmt handler, builds under CPython 3.12 through
Buildx/QEMU, verifies the wheel, and refreshes its local SHA-256 manifest.
The build is pinned to the exact LLVM 20 snapshot used by pyasc v2:
`86b69c31642e98f8357df62c09d118ad1da4e16a`
(`llvm-86b69c31-ubuntu-x64.tar.gz`). Its local ARM reference installation is
`/home/aloschilov/workspace/llvm`; QEMU consumes the matching x86_64 artifact,
not a Debian MLIR package. Four compile jobs are used deliberately so emulated
compilation does not starve site uploads.

The internal runtime wheel is stripped and ZIP-LZMA-compressed for transport;
`merge_wheels.py` expands it and emits a normal standards-compatible final
wheel. Private worker submissions stream multipart data with a one-hour upload
limit because the official adapter's fixed 120-second in-memory upload retries
cannot carry the self-contained runtime on a constrained link.
