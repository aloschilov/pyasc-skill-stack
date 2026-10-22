"""Worker campaign driver: opencode/GLM-5.2 candidates -> private site evals.

Work items are either ``tune`` (improve perf of an existing operator module,
accept only if correctness stays 100% and the harness score improves) or
``generate`` (write a new operator module, accept when all cases pass).

Usage:
    python3 driver.py --items sigmoid:tune,gelu:tune --iterations 3 --workers 3
    python3 driver.py --items masked_scale:generate --dry-run   # prompt only

Each iteration: build/resume worker session -> static-check candidate.py ->
stage an immutable private submission -> full-perf site run -> accept locally
or leave the incumbent unchanged -> feed the result digest back to the worker.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import prompts
from contracts import parameter_names, public_signature
from evalqueue import EvalQueue, RemoteError

WORKERS_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKERS_DIR.parent.parent.parent
SUBMISSION_PKG = WORKERS_DIR.parent / "submission" / "cann_bench"
EVIDENCE_DIR = REPO_ROOT / "evidence" / "cannbench"
SCRATCH_ROOT = WORKERS_DIR / ".scratch"
RUNS_DIR = SCRATCH_ROOT / "cannbench" / "runs"
SKILLS_ROOT = REPO_ROOT / "skills"

DEFAULT_MODELS = ("gateway/glm-5.3", "gateway/qwen3.8-max")
WORKER_TIMEOUT_S = 900
REQUIRED_SKILLS = (
    "pyasc-cannbench-kernel",
    "pyasc-syntax-constraints",
    "pyasc-code-review",
    "pyasc-build-run-verify",
)

PHASE_SKILLS = {
    "design": (
        "pyasc-cannbench-kernel",
        "pyasc-syntax-constraints",
    ),
    "implement": (
        "pyasc-cannbench-kernel",
        "pyasc-syntax-constraints",
    ),
    "review": (
        "pyasc-cannbench-kernel",
        "pyasc-code-review",
        "pyasc-build-run-verify",
    ),
    "repair": (
        "pyasc-cannbench-kernel",
    ),
}

PHASE_PROMPTS = {
    "design": """\
This is the DESIGN phase of a provenance-gated kernel workflow. Before doing
anything else, invoke the OpenCode `skill` tool for each exact skill:
{skills}. Read task.md, apply those skills, and write a concise design.md that
covers the algorithm, pinned-v2 APIs, all 20 cases, tiling, tails, UB budget,
numerical risks, anti-cheat constraints, and the local validation ladder. Do
not write candidate.py yet. End with DESIGN_DONE.
Use at most six tool calls, including the relevant skill reference and the
design file write. Do not inspect current submission modules, old run
artifacts, or the full pyasc source; task.md and the loaded skill references
are the allowed implementation context.
""",
    "implement": """\
This is the IMPLEMENTATION phase of a provenance-gated kernel workflow.
Before writing code, invoke the OpenCode `skill` tool for each exact skill:
{skills}. Read task.md and design.md, apply those skills, and write the complete
submission module to candidate.py. Perform only a Python syntax check; do not
build or run pyasc locally. Create no other files. End with IMPLEMENT_DONE.
Do not inspect current submission modules or old generated candidates.
Use at most six tool calls: two skill calls, task/design reads, candidate write,
and syntax check. The pinned API contract is in task.md. Do not search API
sources or any other checkout; write a concrete candidate for the measured
compiler gate to verify.
Budget discipline: write candidate.py as your FIRST artifact-producing action
after the reads; keep analysis inside code comments and prioritize completing
the file over explanation so the output budget cannot be exhausted before the
file exists.
""",
    "review": """\
This is the independent REVIEW phase of a provenance-gated kernel workflow.
Before reviewing, invoke the OpenCode `skill` tool for each exact skill:
{skills}. Read candidate.py and only the target-operator section of task.md.
Apply the skill, review the exact public contract, host/JIT boundary, tails,
dtypes, special values, and UB budget; fix candidate.py in place if needed,
then run only `python3 -m py_compile candidate.py`. Do not replace numerical
work with torch operations and create no other files. End with REVIEW_DONE.
For local campaigns, read compile_report.json: it records the measured compiler
result for this exact implementation. Repair any reported failure. The driver
will compile the reviewed candidate again before qualification.
Use at most ten tool calls, including the three skills, task/candidate/report reads,
any repair write, and the syntax check.
Do not inspect current submission modules or old generated candidates.
""",
    "repair": """\
This is a measured-feedback REPAIR phase. Before changing code, invoke the
OpenCode `skill` tool for each exact skill: {skills}. Read task.md, design.md,
candidate.py, and compile_feedback.md. Apply the skill to the measured local
failure for this operator, repair candidate.py in place without copying an
earlier repository implementation, and run only `python3 -m py_compile
candidate.py`. Create no other files. End with REPAIR_DONE.
Use at most seven tool calls and stop as soon as the measured failure is fixed.
    """,
}
WORKFLOW_PHASES = ("design", "implement", "review")

GUIDANCE = {
    "rms_norm": """\
- Normalize independently over the last dimension: y = x * gamma *
  rsqrt(mean(x*x) + epsilon). Flatten all leading dimensions into rows.
- Reuse the proven full-row/split-D patterns from the skill-stack RMSNorm
  goldens. Accumulate sum-of-squares in float32, including f16/bf16 inputs.
- D ranges from 2 to 8192 and includes non-aligned prime tails. A full-row
  tile is preferred when it fits UB; otherwise stream D in aligned chunks
  and keep a loop-carried scalar accumulator before a second write pass.
- Distribute rows grid-stride across cores. gamma is shared across rows and
  has the same dtype as x. epsilon is a runtime float argument.
- Special-value cases include inf, nan, all-zero rows, and fp16 boundary
  values. Do not clamp or replace IEEE values; match the golden propagation.
- Host code may reshape/view metadata but must not call torch math or the
  native torch RMSNorm operator.""",
    "masked_scale": """\
- y = x * mask * scale, elementwise, identical shapes. x dtype is one of
  f16/bf16/f32 and mask dtype INDEPENDENTLY one of int8/uint8/f16/bf16/f32
  (see the cases table). The pyasc JIT specializes per launched-tensor dtype
  automatically, so ONE kernel source covers all combinations.
- float-dtype masks: mf = m.to(asc.float32); y = xf * mf * scale with scale
  as a runtime float kernel argument. Output dtype/shape = x dtype/shape.
- int8/uint8 masks — MEASURED FAILURE to avoid: any vector op directly on an
  int8 tile (including `.to(asc.float32)` and `m * 1.0`) fails with
  "RuntimeError: 'input' dtype must be one of int16, int32, int64, float16,
  bfloat16, float32, got int8". The hardware DOES support int8 -> float16
  conversion: first `mh = asc2.cast(m, asc.float16)` (copy_in of the int8
  tile itself is fine), then `mf = mh.to(asc.float32)`. Never touch the int8
  tile with arithmetic.
- torch has no uint8 route into the kernel: on the host reinterpret WITHOUT
  copying via `mask = mask.view(torch.int8)`. Mask bytes can be up to 255
  (case 10), which reinterprets as negative int8 — fix in-kernel after the
  cast: `mf = asc2.where(mf < 0.0, mf + 256.0, mf)`. Apply the +256 fixup
  ONLY in the uint8-origin variant (pass a flag or use a separate kernel);
  int8 masks are genuinely signed and must NOT be fixed up.
- Keep the where destination 256-byte rule in mind for the fixup tile.
- MEASURED pyasc CODEGEN BUG to avoid: a kernel instantiation whose tiles
  include BOTH float16 and bfloat16 types fails to compile with
  "redefinition of 'c0_f16' with a different type: 'bfloat16_t' vs 'half'"
  whenever anything beyond a bare cast happens at 16 bits (mask dtype f16
  with x bf16 — cases 3, 6, 16 are the risky combos). VERIFIED-SAFE rule:
  the ONLY operation allowed on a 16-bit tile is a single cast toward f32
  (`m.to(asc.float32)`, or for int8 the hop
  `asc2.cast(m, asc.float16).to(asc.float32)`). No comparisons, no where,
  no arithmetic, no asc2.full at f16/bf16 anywhere in the kernel; do ALL
  logic (including the uint8 +256 fixup) in f32 and cast exactly once on
  copy_out to x's dtype.""",
    "swi_glu": """\
- input splits into x0 / x1 halves along attr dim (cases use dim 0, 1, 2 and
  -1; that dim is always even). output = silu(x0) * x1 where
  silu(v) = v * sigmoid(v); use the stable sigmoid form.
- Zero-copy strided access (do NOT call .contiguous()/.narrow() to
  materialize x0/x1 — extra device copies destroy the perf score): on the
  host compute outer = prod(shape[:dim]), C = shape[dim],
  inner = prod(shape[dim+1:]), half_cols = (C // 2) * inner. The contiguous
  input viewed as 2-D [outer, C * inner] has x0 in columns [0, half_cols)
  and x1 in columns [half_cols, 2 * half_cols) of every row. Kernel: 2-D
  global tensors (runtime row/col counts are fine), loop over (row,
  col-chunk) with [1, TILE] tiles and real_shape tails; output is
  [outer, half_cols].
- Distribute work over rows AND column-chunks so small-outer cases still use
  many cores (e.g. grid-stride over row * num_col_tiles + col_tile).
- MEASURED FAILURE to avoid: 2-D copy tiles require the LAST dimension to be
  32-byte aligned. Case 12 is [1000003, 2] bf16 with dim=1 -> half_cols = 1
  element = 2 bytes -> "RuntimeError: Last dimension of tensor must be
  aligned by 32 bytes, got 1 x 2 bytes". Add a host-side fallback for
  degenerate layouts: when `half_cols * input.element_size() < 32`, split
  with metadata ops (which the anti-cheat explicitly allows):
  `x0 = input.narrow(dim, 0, C // 2).contiguous()` and
  `x1 = input.narrow(dim, C // 2, C // 2).contiguous()`, then run a plain
  1-D elementwise silu-mul kernel over x0/x1. The two extra copy kernels
  cost perf only on that one case; all aligned cases must keep the
  zero-copy 2-D path.""",
    "foreach_addcdiv_scalar": """\
- Signature: foreach_addcdiv_scalar(x1: List[Tensor], x2, x3, scalar: float)
  -> List[Tensor]. Elementwise per list entry: y_i = x1_i + (x2_i / x3_i) *
  scalar; shapes match within each triple, lists are short.
- Host loops over the list, one kernel launch per tensor triple,
  empty_like output each.
- scalar values include 0.0, +-0.5, +-1.0, 1.5, 2.0, inf and nan — pass it
  as a runtime float kernel argument; IEEE propagation through the hardware
  ops matches the golden. NO host special-casing.
- Compute in f32 internally even for f16/bf16 (division is
  precision-sensitive).""",
    "foreach_norm": """\
- Signature: foreach_norm(x: List[Tensor], scalar: float) -> List[Tensor];
  per entry a FULL reduction to a 0-dim tensor: y = (sum |v|^p)^(1/p) with
  p = scalar in {-1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, inf}.
- Choose the reduction form on the HOST per p (it is a Python float):
  p == 1 -> sum(abs(v)); p == 2 -> sqrt(sum(v*v)); p == inf -> max(abs(v));
  otherwise -> S = sum(exp(log(abs(v)) * p)) and result = exp(log(S) / p).
  log(0) = -inf propagates correctly through exp — do not clamp.
- Parallel reduction (both building blocks VERIFIED on this hardware):
  grid-stride kernel where each core keeps a loop-carried accumulator (seed
  with asc2.reduce_sum(asc2.full([1, 64], 0.0, dtype=asc.float32)), then
  acc = acc + asc2.reduce_sum(...) in a PLAIN asc2.range loop — do NOT pass
  gm_barrier, it does not exist on this build) and finally
  asc2.atomic_add(asc2.full([8], acc, dtype=asc.float32), out_gm, [0]) to
  combine across cores (host zero-seeds via torch.zeros — tensor creation
  is allowed). For p == inf there is no atomic max for the final combine
  from all cores — either use asc2.atomic_max (exists, same signature) or
  per-core slots + a second single-core pass. Apply the final power in a
  second tiny single-core kernel reading the accumulated sum. Host returns
  out8[0] (a 0-dim view of the result buffer) — matches torch.norm's
  output shape/dtype.
- Accumulate in f32; cast the final scalar back to the input dtype.""",
}

EXTRA_LEVERS = {
    "sigmoid": """\
5. sigmoid's chain is short and TILE is already 2048; check whether
   TILE=4096 still fits UB (temps * 4 * TILE * 2 <= ~250000), and whether a
   narrow-tile variant for small cases (see table) recovers parallelism.""",
    "exp": """\
5. exp is the shortest possible chain (one asc2.exp); the score is bounded
   by DMA. Try TILE=4096/8192 within UB, and a small-shape narrow-tile
   variant.""",
    "mish": """\
5. mish currently runs TILE=1024 with ~14 temporaries. Reusing repeated
   subexpressions (w * 2.0 appears twice) and dropping temps may admit
   TILE=2048. The where-blend needs both branch quotients — consider a
   formulation computing numerator/denominator via one where each.""",
    "gelu": """\
5. The erf-mode kernel's 9-step Horner chain forces TILE=512 (each Horner
   step is a fresh tile temporary under static allocation). Reducing
   temporaries (e.g. fewer polynomial steps that still meet ~1e-5 relative
   accuracy, or restructuring so p is reused in place) buys TILE=1024+.
   The tanh-mode kernel is short and could run TILE=2048 independently —
   the two kernels need not share a tile size.""",
}

# module name == op dir name == callable name for all L1 ops in this campaign
ALL_OPS = ["sigmoid", "exp", "mish", "gelu",
           "masked_scale", "swi_glu", "foreach_addcdiv_scalar", "foreach_norm",
           "rms_norm"]

INIT_HEADER = '''"""pyasc asc2 submission package for CANN Bench.

Each module exposes one public callable whose name and signature match the
operator's ``proto.yaml`` schema. All numerical work happens in pyasc asc2
kernels launched directly with torch_npu-owned tensors. The pinned pyasc v2
runtime resolves their device buffers without host-side pointer extraction.
"""

'''


@dataclass
class WorkItem:
    op: str
    kind: str  # "tune" | "generate"
    accepted_score: float | None = None
    status: str = "pending"
    history: list = field(default_factory=list)
    task_dir: Path | None = None
    contract: dict | None = None

    @property
    def name(self) -> str:
        return f"{self.op}-{self.kind}"


@dataclass(frozen=True)
class WorkerResult:
    returncode: int
    session_id: str | None
    loaded_skills: tuple[str, ...]
    skill_dirs: tuple[tuple[str, str], ...]
    text_output: str
    # True when the worker's final step finished with reason=length, i.e. the
    # model exhausted its output budget (reasoning tokens included) before
    # producing the required artifact. Lets run_phase adapt the retry message
    # instead of reissuing the identical prompt deterministically.
    finish_length: bool = False

    def missing_skills(self, required: tuple[str, ...]) -> tuple[str, ...]:
        loaded = set(self.loaded_skills)
        return tuple(name for name in required if name not in loaded)

    def skill_gate_passed(self, required: tuple[str, ...]) -> bool:
        return self.returncode == 0 and not self.missing_skills(required)


# ---------------------------------------------------------------- static check

# Whitelisted host tensor factories: allocation and index/metadata creation
# only - never arithmetic. arange/ones/full mirror the FakeTorch gate's
# supported factories (added for the measured unique/top_k index gap); using
# them for host-side math remains banned via the math checks below.
ALLOWED_TORCH_CALLS = {"empty", "empty_like", "zeros", "zeros_like", "tensor",
                       "arange", "ones", "full"}

GATE_IMAGE = "pyasc-cannbench-local-eval:0a631f70-v2"

GATE_INFRA_SIGNATURES = (
    "emitted non-JSON output",
    "exec format error",
    "Cannot connect to the Docker daemon",
    "docker: Error response from daemon",
    "compile gate image build failed",
    "image not found",
)


class GateInfrastructureError(RuntimeError):
    """Compile-gate infrastructure is broken; abort the scope instead of
    consuming candidate iteration/repair budget on invalid evidence."""


def _gate_infra_failure(report: dict) -> str | None:
    """Detect infrastructure (not candidate) failure in a gate report."""
    if not isinstance(report, dict):
        return None
    text = " ".join(str(report.get(k) or "")
                    for k in ("fatal_error", "stderr_tail", "stdout_tail", "error"))
    for signature in GATE_INFRA_SIGNATURES:
        if signature in text:
            return signature
    return None


def _gate_preflight() -> str | None:
    """Cheap fast-fail check of gate infrastructure before any phase budget.

    Returns None on success, otherwise a short reason. Verifies the pinned
    gate image is inspectable and its entrypoint actually executes on this
    host (catches e.g. lost qemu binfmt registration after a reboot).
    """
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", GATE_IMAGE],
            capture_output=True, text=True, timeout=30)
        if inspect.returncode != 0:
            return f"gate image {GATE_IMAGE} not inspectable"
        probe = subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/amd64",
             "--entrypoint", "python3", GATE_IMAGE, "-c", "print('gate-exec-ok')"],
            capture_output=True, text=True, timeout=180)
        if probe.returncode != 0 or "gate-exec-ok" not in probe.stdout:
            return ("gate container exec failed: "
                    + (probe.stderr or probe.stdout)[-200:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"gate preflight error: {exc}"
    return None
BANNED_TORCH_SUBMODULES = {"nn", "ops", "functional", "linalg", "special",
                           "fft", "cuda", "npu"}
EXPECTED_PARAMETERS = {
    "sigmoid": ("x",),
    "exp": ("x", "base", "scale", "shift"),
    "mish": ("x",),
    "gelu": ("x", "approximate"),
    "masked_scale": ("x", "mask", "scale"),
    "swi_glu": ("input", "dim"),
    "foreach_addcdiv_scalar": ("x1", "x2", "x3", "scalar"),
    "foreach_norm": ("x", "scalar"),
    "rms_norm": ("x", "gamma", "epsilon"),
}


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def static_check(path: Path, callable_name: str, task_dir: Path | None = None) -> list[str]:
    problems = []
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    callable_node = next((
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == callable_name
    ), None)
    if callable_node is None:
        problems.append(
            f"missing top-level public callable def {callable_name}(...)")
    else:
        actual = tuple(arg.arg for arg in callable_node.args.posonlyargs + callable_node.args.args + callable_node.args.kwonlyargs)
        expected = parameter_names(task_dir, callable_name) if task_dir else EXPECTED_PARAMETERS[callable_name]
        if actual != expected:
            problems.append(
                f"public signature parameters {actual!r} != {expected!r}"
            )
        if callable_node.args.vararg or callable_node.args.kwarg:
            problems.append("public callable must not use *args or **kwargs")
        if task_dir:
            expected_args = public_signature(task_dir, callable_name)
            def defaults(arguments):
                positional = arguments.posonlyargs + arguments.args
                values = dict(zip((arg.arg for arg in positional[-len(arguments.defaults):]), arguments.defaults)) if arguments.defaults else {}
                values.update((arg.arg, value) for arg, value in zip(arguments.kwonlyargs, arguments.kw_defaults) if value is not None)
                return {key: ast.dump(value) for key, value in values.items()}
            expected_defaults = defaults(expected_args)
            actual_defaults = defaults(callable_node.args)
            if actual_defaults != expected_defaults:
                problems.append("Public callable defaults differ from official golden")

    has_jit = any(
        isinstance(n, ast.FunctionDef) and any(
            (_dotted(d) or _dotted(getattr(d, "func", None) or d)) in ("asc2.jit", "asctile.jit", "asc.jit")
            for d in n.decorator_list)
        for n in ast.walk(tree))
    if not has_jit:
        problems.append("no public pyasc JIT kernel found")

    if "ensure_npu_platform" not in source:
        problems.append("missing ensure_npu_platform import/call "
                        "(from ._pyasc_runtime import ensure_npu_platform)")
    if "import torch_npu" in source:
        problems.append("do not import torch_npu (not needed, confuses the "
                        "harness environment)")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = _dotted(node)
            if name and name.startswith("torch."):
                sub = name.split(".")[1]
                if sub in BANNED_TORCH_SUBMODULES:
                    problems.append(f"banned torch usage: {name}")
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "data_ptr":
                problems.append(
                    "banned Tensor.data_ptr(): pass the Tensor itself to the "
                    "pyasc v2 JIT launch so dtype specialization is preserved"
                )
            if name and name.startswith("torch.") and name.count(".") == 1:
                fn = name.split(".")[1]
                if fn not in ALLOWED_TORCH_CALLS:
                    problems.append(
                        f"banned torch call: {name}() — torch is allowed only "
                        f"for {sorted(ALLOWED_TORCH_CALLS)}; all math must be "
                        f"asc2 kernels")
    for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        decorators = [_dotted(getattr(dec, "func", dec)) for dec in function.decorator_list]
        if any(name in ("asctile.jit", "asc.jit", "asc2.jit") for name in decorators):
            continue
        # Tensor allocation/view assembly is host metadata; copying values into
        # output tensors belongs in pyasc kernels and must not bypass codegen.
        allocated = set()
        for node in ast.walk(function):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
                call = _dotted(node.value.func)
                if call in {"torch.empty", "torch.empty_like", "torch.zeros", "torch.zeros_like"}:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    allocated.update(target.id for target in targets if isinstance(target, ast.Name))
        for node in ast.walk(function):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "copy_"):
                problems.append("banned host Tensor.copy_(): write outputs inside pyasc JIT kernels")
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                            and target.value.id in allocated):
                        problems.append("banned host Tensor indexed assignment: " + target.value.id)
    return sorted(set(problems))


# ------------------------------------------------------------------- workers

def _finished_on_length(stdout: str) -> bool:
    """Detect a worker run whose final step hit the output-length limit."""
    last_reason = None
    for raw in stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") in ("step_finish", "step-finish"):
            part = event.get("part", {})
            if isinstance(part, dict) and "reason" in part:
                last_reason = part.get("reason")
    return last_reason == "length"


def _parse_worker_trace(
    stdout: str,
) -> tuple[str | None, tuple[str, ...], tuple[tuple[str, str], ...], str]:
    """Extract skill calls whose source resolves into this checkout."""
    session_id = None
    loaded = []
    skill_dirs = []
    text_parts = []
    for raw in stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        session_id = event.get("sessionID") or session_id
        part = event.get("part") or {}
        state = part.get("state") or {}
        if event.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        if (
            event.get("type") == "tool_use"
            and part.get("tool") == "skill"
            and state.get("status") == "completed"
        ):
            name = (state.get("input") or {}).get("name")
            directory = str((state.get("metadata") or {}).get("dir") or "")
            expected = (SKILLS_ROOT / str(name)).resolve()
            source_matches = bool(directory) and Path(directory).resolve() == expected
            if isinstance(name, str):
                skill_dirs.append((name, directory))
            if source_matches and name not in loaded:
                loaded.append(name)
    return session_id, tuple(loaded), tuple(skill_dirs), "\n".join(text_parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _skill_source_manifest(name: str) -> dict[str, str]:
    root = SKILLS_ROOT / name
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def worker_timeout_seconds() -> int:
    """Allow scoped generation budgets while keeping every worker bounded."""
    raw = os.environ.get("CANNBENCH_WORKER_TIMEOUT_SECONDS", str(WORKER_TIMEOUT_S))
    if not raw.isascii() or not raw.isdigit() or not 1 <= int(raw) <= 7200:
        raise ValueError("CANNBENCH_WORKER_TIMEOUT_SECONDS must be an integer in 1..7200")
    return int(raw)


def run_worker(scratch: Path, message: str, log_path: Path,
               model: str, *, session_id: str | None = None,
               timeout_s: int | None = None) -> WorkerResult:
    if timeout_s is None:
        timeout_s = worker_timeout_seconds()
    cmd = [
        "opencode", "run", "--pure", "--format", "json",
        "--dir", str(scratch), "-m", model,
    ]
    if session_id:
        cmd.extend(["--session", session_id])
    cmd.append(message)
    # Pin skill discovery to this checkout. External skill catalogs are
    # disabled so similarly named user skills cannot satisfy provenance.
    config = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT", "{}"))
    config["skills"] = {"paths": [str(SKILLS_ROOT)]}
    config.setdefault("mcp", {})["cann-bench-site"] = {"enabled": False}
    config["permission"] = {
        "read": "allow", "glob": "allow", "grep": "allow", "skill": "allow",
        # OpenCode matches edits against project-relative paths. A scoped
        # transfer may inherit the enclosing VM checkout as its project root.
        "edit": {"*": "deny", **{
            "*" + str(scratch.relative_to(REPO_ROOT) / name): "allow"
            for name in ("candidate.py", "design.md")
        }},
        "bash": {"*": "deny", "python3 -m py_compile candidate.py": "allow"},
        "external_directory": {"*": "deny", str(REPO_ROOT / "*"): "allow"},
        "task": "deny", "webfetch": "deny",
    }
    env = {
        **os.environ,
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
    }
    if os.environ.get("CANNBENCH_OPENCODE_STATE_ROOT"):
        # Independent concurrent workers must not contend on OpenCode's SQLite
        # state. Keep each worker's sessions across phases outside artifacts.
        state = Path(os.environ["CANNBENCH_OPENCODE_STATE_ROOT"]) / hashlib.sha256(str(scratch).encode()).hexdigest()
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        env["XDG_DATA_HOME"] = str(state)
    proc = subprocess.Popen(cmd, cwd=scratch, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        log_path.write_text(
            f"rc={proc.returncode}\n=== STDOUT ===\n{stdout}\n"
            f"=== STDERR ===\n{stderr}\n")
        session_id, loaded, skill_dirs, text_output = _parse_worker_trace(stdout)
        return WorkerResult(
            proc.returncode, session_id, loaded, skill_dirs, text_output,
            _finished_on_length(stdout),
        )
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc, grace_seconds=3.0)
        stdout, stderr = proc.communicate(timeout=5.0)
        log_path.write_text(
            f"worker timed out after {timeout_s}s\n"
            f"=== PARTIAL STDOUT ===\n{stdout}\n=== STDERR ===\n{stderr}\n")
        session_id, loaded, skill_dirs, text_output = _parse_worker_trace(stdout)
        return WorkerResult(124, session_id, loaded, skill_dirs, text_output,
                            _finished_on_length(stdout))


def run_phase(scratch: Path, phase: str, iter_dir: Path,
              models: tuple[str, ...], start_index: int,
              attempts: int, *, readonly_review: bool = False) -> tuple[WorkerResult, str] | None:
    """Run one isolated workflow phase and enforce its native skill calls."""
    required = PHASE_SKILLS[phase]
    message = PHASE_PROMPTS[phase].format(skills=", ".join(required))
    if phase == "review" and readonly_review:
        message = (
            "Independent read-only REVIEW. Invoke the OpenCode skill tool for "
            "each exact skill: " + ", ".join(required) + ". Read task.md, candidate.py "
            "and compile_report.json if present. Check the public contract, dtypes, "
            "tails, special values, host/JIT boundary and measured failures. "
            "Do not edit candidate.py or any other file. Describe every necessary "
            "repair in your response for the implementation model. Emit exactly one verdict "
            "on its own line: REVIEW_PASS when no repair is needed, or REVIEW_REPAIR "
            "when a repair is necessary. End with REVIEW_DONE."
        )
    expected = {
        "design": scratch / "design.md",
        "implement": scratch / "candidate.py",
        "review": scratch / "candidate.py",
        "repair": scratch / "candidate.py",
    }[phase]
    marker = f"{phase.upper()}_DONE"

    attempt_message = message
    for attempt in range(1, attempts + 1):
        model = models[(start_index + attempt - 1) % len(models)]
        result = run_worker(
            scratch,
            attempt_message,
            iter_dir / f"{phase}.attempt{attempt}.log",
            model,
        )
        missing = result.missing_skills(required)
        artifact_exists = expected.exists()
        marker_seen = marker in result.text_output
        gate_passed = (
            result.skill_gate_passed(required)
            and artifact_exists
            and marker_seen
        )
        trace = {
            "phase": phase,
            "model": model,
            "session_id": result.session_id,
            "required_skills": list(required),
            "loaded_skills": list(result.loaded_skills),
            "observed_skill_dirs": dict(result.skill_dirs),
            "missing_skills": list(missing),
            "artifact": str(expected.relative_to(scratch)),
            "artifact_exists": artifact_exists,
            "completion_marker": marker,
            "completion_marker_seen": marker_seen,
            "finish_length": result.finish_length,
            "skill_gate_passed": gate_passed,
            "returncode": result.returncode,
        }
        (iter_dir / f"{phase}.skill-trace.attempt{attempt}.json").write_text(
            json.dumps(trace, indent=2), encoding="utf-8"
        )
        if gate_passed:
            return result, model
        if phase in ("design", "implement") and expected.exists():
            rejected = iter_dir / f"{phase}.rejected.attempt{attempt}{expected.suffix}"
            shutil.copy2(expected, rejected)
            expected.unlink()
        if attempt < attempts:
            if result.finish_length and not artifact_exists:
                # The model burned its output budget (reasoning included)
                # before writing the artifact; reissuing the identical prompt
                # deterministically re-exhausts (measured: top_k/unique iter2,
                # both attempts). Adapt the retry instead.
                print(f"[{phase}] attempt {attempt} exhausted its output "
                      "budget before producing the artifact; retrying with "
                      "budget discipline", flush=True)
                attempt_message = (
                    message
                    + "\n\nOUTPUT BUDGET DISCIPLINE: the previous attempt "
                    "exhausted its output budget (finish reason=length) "
                    "before producing "
                    + expected.name + ". Write the COMPLETE " + expected.name
                    + " file FIRST in this attempt, before any explanation; "
                    "keep analysis inside code comments, then emit " + marker
                    + "."
                )
            else:
                print(
                    f"[{phase}] attempt {attempt} rejected before evaluation; "
                    f"artifact={artifact_exists}, missing_skills={list(missing)}; "
                    "retrying",
                    flush=True,
                )
            time.sleep(10)
    return None


# ------------------------------------------------------------------ package

def deployed_modules() -> list[str]:
    mods = []
    for p in sorted(SUBMISSION_PKG.glob("*.py")):
        if p.stem not in ("__init__", "_pyasc_runtime"):
            mods.append(p.stem)
    return mods


def render_init(modules: list[str]) -> str:
    body = "".join(f"from .{m} import {m}\n" for m in sorted(modules))
    exports = ", ".join(f'"{m}"' for m in sorted(modules))
    return INIT_HEADER + body + f"\n__all__ = [{exports}]\n"


# ---------------------------------------------------------------- item loop

def incumbent_score(op: str) -> float | None:
    report_path = EVIDENCE_DIR / f"{op}_final_eval.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    return round(report["operators"][0]["score"], 2)


def decide_accept(item: WorkItem, digest: dict) -> bool:
    if digest.get("hard_failure"):
        return False
    if digest["passed"] != digest["total"]:
        return False
    if item.kind == "generate":
        return True
    baseline = item.accepted_score
    return baseline is None or digest["score"] > baseline + 0.05


def _terminate_process_group(proc: subprocess.Popen, grace_seconds: float = 15.0) -> None:
    """Terminate a subprocess and every descendant (SIGTERM, then SIGKILL)."""
    import signal
    import time as _time

    # All callers launch with start_new_session=True. The group can outlive
    # its leader, so retain its known ID even after the leader exits.
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = _time.monotonic() + grace_seconds
    while _time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        _time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _gate_timeout_seconds() -> tuple[float, str | None]:
    """Validate PYASC_CANN_GATE_TIMEOUT_SECONDS; fall back with a warning."""
    raw = os.environ.get("PYASC_CANN_GATE_TIMEOUT_SECONDS", "")
    if not raw:
        return 900.0, None
    try:
        value = float(raw)
    except ValueError:
        return 900.0, f"invalid PYASC_CANN_GATE_TIMEOUT_SECONDS={raw!r}"
    if value <= 0:
        return 900.0, f"non-positive PYASC_CANN_GATE_TIMEOUT_SECONDS={raw!r}"
    return value, None


def _gate_container_name(op: str, candidate_arg: str) -> str:
    digest = hashlib.sha256(candidate_arg.encode("utf-8")).hexdigest()[:12]
    safe_op = "".join(
        character if character.isalnum() or character in "_.-" else "-"
        for character in op)[:32]
    return f"pyasc-cannbench-gate-{safe_op}-{digest}"


def _planned_case_count(task_dir: Path | None) -> int:
    """Official planned case count (all 20 must be retained on timeout)."""
    if task_dir is not None:
        try:
            text = (task_dir / "cases.yaml").read_text(encoding="utf-8")
            count = text.count("case_id:")
            if count > 0:
                return count
        except OSError:
            pass
    return 20


def _docker_container_state(name: str) -> str:
    """running | present | absent | unknown | daemon_unavailable.

    Docker transport/daemon errors are NOT proof of absence.
    """
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--type", "container", "--format",
             "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return "daemon_unavailable"
    except subprocess.SubprocessError:
        return "unknown"
    if proc.returncode == 0:
        return "running" if proc.stdout.strip() == "true" else "present"
    message = (proc.stderr or "") + (proc.stdout or "")
    for marker in ("No such object", "No such container", "no such"):
        if marker in message:
            return "absent"
    for marker in ("Cannot connect to the Docker daemon",
                   "Is the docker daemon running", "docker daemon"):
        if marker in message:
            return "daemon_unavailable"
    return "unknown"


def _reconcile_container(name: str, budget_seconds: float = 90.0) -> str:
    """Reconcile a container to a confirmed state before the lock is released."""
    deadline = time.monotonic() + budget_seconds
    consecutive_absent = 0
    while True:
        state = _docker_container_state(name)
        if state == "absent":
            consecutive_absent += 1
            if consecutive_absent >= 2:
                return "absent"
        elif state in ("running", "present"):
            consecutive_absent = 0
            try:
                subprocess.run(["docker", "rm", "-f", name],
                               capture_output=True, text=True, timeout=60)
            except subprocess.SubprocessError:
                pass
        elif state == "daemon_unavailable":
            # A disconnected daemon may leave containers running (live restore).
            consecutive_absent = 0
        else:  # unknown: a healthy daemon error is not proof of absence
            consecutive_absent = 0
        if time.monotonic() >= deadline:
            return "unreconciled"
        time.sleep(0.5)


def parse_gate_report(stdout: str) -> dict:
    """Retain compiler diagnostics when the wrapper appends a lifecycle failure.

    Accept only a single JSON object, or a compiler report followed by one
    explicit failed timeout report. A completed compiler report never overrides
    an unsuccessful container lifecycle.
    """
    decoder = json.JSONDecoder()
    documents = []
    remaining = stdout.strip()
    while remaining:
        document, end = decoder.raw_decode(remaining)
        if not isinstance(document, dict):
            raise ValueError("gate report must be an object")
        documents.append(document)
        remaining = remaining[end:].lstrip()
    if len(documents) == 1:
        return documents[0]
    if len(documents) != 2:
        raise ValueError("unexpected gate report sequence")
    completed, lifecycle = documents
    if (lifecycle.get("status") != "failed"
            or lifecycle.get("gate_timed_out") is not True
            or not completed.get("operator")
            or completed.get("operator") != lifecycle.get("operator")
            or not isinstance(completed.get("case_results"), list)
            or any(not isinstance(case, dict) for case in completed["case_results"])):
        raise ValueError("unbound gate lifecycle report")
    report = dict(lifecycle)
    report["completed_compile_report"] = completed
    report["compile_passed"] = 0
    report["dispatch_passed"] = 0
    report["case_results"] = [case for case in completed["case_results"]
                              if case.get("compile") != "passed"
                              or case.get("dispatch") != "passed"]
    return report


def _communicate_gate(proc, acquired_marker, execution_seconds):
    """Start execution budget only after the wrapper acquires shared locks."""
    execution_deadline = None
    while True:
        now = time.monotonic()
        if execution_deadline is None and acquired_marker.exists():
            execution_deadline = now + execution_seconds
        # Waiting for another serialized job is not candidate execution.
        remaining = execution_deadline - now if execution_deadline is not None else 1.0
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, execution_seconds)
        try:
            return proc.communicate(timeout=min(1.0, remaining))
        except subprocess.TimeoutExpired:
            continue


def compile_overlay_arguments() -> list[str]:
    """Bind a scoped source overlay to the path mounted inside the gate."""
    raw = os.environ.get('CANNBENCH_SOURCE_OVERLAY')
    expected = os.environ.get('CANNBENCH_SOURCE_OVERLAY_SHA256')
    if bool(raw) != bool(expected):
        raise ValueError('Compiler source overlay path and SHA256 must be paired')
    if not raw:
        return []
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    relative = path.relative_to(REPO_ROOT.resolve())
    if _sha256(path / 'MANIFEST.json') != expected:
        raise ValueError('Compiler source overlay manifest hash mismatch')
    return ['--source-overlay', str(relative), '--source-overlay-sha256', expected]


def local_evaluate(op: str, candidate: Path, iter_dir: Path, task_dir: Path | None = None) -> dict:
    try:
        candidate_arg = candidate.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        return {"hard_failure": True, "error": str(exc), "status": "failed"}
    command = [
        str(WORKERS_DIR / "run_local_compile_gate.sh"),
        "--candidate", str(candidate_arg), "--op", op,
    ]
    if task_dir:
        relative = task_dir.resolve().relative_to(REPO_ROOT)
        command.extend(["--cases", str(relative / "cases.yaml"), "--proto", str(relative / "proto.yaml")])
    try:
        overlay_arguments = compile_overlay_arguments()
        if overlay_arguments and task_dir is None:
            raise ValueError('Compiler source overlay requires an explicit official task directory')
        command.extend(overlay_arguments)
    except (ValueError, OSError) as exc:
        return {"hard_failure": True, "error": str(exc), "status": "failed"}
    # The gate script bounds the container itself; this is the driver-side
    # backstop. Before releasing the lock we reconcile the exact container to a
    # confirmed state; if that is impossible we retain the lock (fail closed)
    # rather than kill the wrapper under a possibly-live container.
    gate_timeout, timeout_warning = _gate_timeout_seconds()
    build_raw = os.environ.get("PYASC_CANN_GATE_BUILD_TIMEOUT_SECONDS", "1800")
    build_timeout = int(build_raw) if build_raw.isascii() and build_raw.isdigit() else 1800
    build_timeout = build_timeout if build_timeout > 0 else 1800
    backstop = gate_timeout + build_timeout + 180.0
    container_name = _gate_container_name(op, str(candidate_arg))
    planned_cases = _planned_case_count(task_dir)
    child_env = dict(os.environ)
    child_env["PYASC_CANN_GATE_TIMEOUT_SECONDS"] = str(int(gate_timeout))
    child_env["PYASC_CANN_GATE_CONTAINER"] = container_name
    acquired_marker = (iter_dir / "gate-lock-acquired").resolve()
    acquired_marker.unlink(missing_ok=True)
    child_env["PYASC_CANN_GATE_LOCK_MARKER"] = str(acquired_marker)
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True, env=child_env,
    )
    timed_out = False
    reconcile_result = None
    stdout = stderr = ""
    try:
        stdout, stderr = _communicate_gate(proc, acquired_marker, backstop)
    except subprocess.TimeoutExpired:
        reconcile_result = _reconcile_container(container_name)
        if reconcile_result == "unreconciled":
            report = {
                "status": "failed",
                "fatal_error": (
                    "compile gate backstop: container not confirmed stopped; "
                    "execution lock retained"
                ),
                "gate_timed_out": True,
                "gate_timeout_seconds": gate_timeout,
                "container": container_name,
                "container_reconcile": reconcile_result,
                "lock_retained": True,
                "cases": planned_cases,
                "planned_cases": planned_cases,
                "case_results": [],
            }
            if timeout_warning:
                report["gate_timeout_warning"] = timeout_warning
            (iter_dir / "local_compile.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return {
                "hard_failure": True,
                "status": "failed",
                "passed": 0,
                "total": planned_cases,
                "fatal_error": report["fatal_error"],
                "lock_retained": True,
                "evidence": "verified-local-compile",
                "limitations": [
                    "gate container unreconciled",
                    "execution lock retained (fail closed)",
                ],
            }
        _terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        timed_out = True
    try:
        report = parse_gate_report(stdout)
    except ValueError:
        report = {
            "status": "failed",
            "fatal_error": "local compile gate emitted non-JSON output",
            "stdout_tail": stdout[-4000:],
        }
    report["returncode"] = proc.returncode
    report["gate_timeout_seconds"] = gate_timeout
    report["container"] = container_name
    if timeout_warning:
        report["gate_timeout_warning"] = timeout_warning
    if timed_out:
        report["gate_timed_out"] = True
        report["container_reconcile"] = reconcile_result
        report["status"] = "failed"
        report.setdefault("fatal_error", "compile gate timeout (driver backstop)")
    report.setdefault("planned_cases", planned_cases)
    if not report.get("cases"):
        report["cases"] = planned_cases
    if stderr:
        report["stderr_tail"] = stderr[-8000:]
    if not timed_out:
        # Timeout reconciliations stay ambiguous (slow candidate compile or
        # stuck container) and keep their existing classification; only a
        # non-timeout gate report can be declared infrastructure failure.
        infra = _gate_infra_failure(report)
        if infra is not None:
            (iter_dir / "local_compile.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise GateInfrastructureError(
                f"compile gate infrastructure failure ({infra}); aborting scope "
                "without consuming candidate iteration/repair budget")
    (iter_dir / "local_compile.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "hard_failure": proc.returncode != 0 or report.get("status") != "passed",
        "status": report.get("status", "failed"),
        "passed": report.get("compile_passed", 0),
        "total": report.get("cases", 20),
        "dispatch_passed": report.get("dispatch_passed", 0),
        "unique_specializations": report.get("unique_specializations", 0),
        "unique_specializations_passed": report.get(
            "unique_specializations_passed", 0
        ),
        "fatal_error": report.get("fatal_error"),
        "failure_details": [
            case for case in report.get("case_results", [])
            if (
                case.get("dispatch") != "passed"
                or case.get("compile") != "passed"
            )
        ],
        "evidence": "verified-local-compile",
        "limitations": [
            "does not execute numerical code",
            "does not measure NPU performance",
        ],
    }


def compact_local_feedback(digest: dict, limit: int = 6) -> dict:
    """Deduplicate repeated per-case failures before the next model prompt."""
    signatures = []
    affected_cases = []
    for case in digest.get("failure_details", []):
        affected_cases.append(case.get("case_id"))
        messages = list(case.get("compile_errors", []))
        if case.get("error"):
            messages.append(case["error"])
        for message in messages:
            first_line = str(message).splitlines()[0]
            if first_line not in signatures:
                signatures.append(first_line)
    return {
        "status": digest.get("status"),
        "passed": digest.get("passed"),
        "total": digest.get("total"),
        "dispatch_passed": digest.get("dispatch_passed"),
        "failure_signatures": signatures[:limit],
        "affected_cases": affected_cases,
        "instruction": (
            "Repair the candidate against these measured pinned-v2 failures; "
            "do not remove or shrink any CANNBench case."
        ),
    }


def _iteration_order(path: Path) -> int:
    digits = path.name.removeprefix("iter")
    return int(digits) if digits.isdigit() else 0


def load_resume_records(resume_root: Path, catalog: dict, skills_root: Path) -> dict:
    """Bind native iterations of an INTERRUPTED campaign snapshot.

    The source run was cancelled before it could write ``summary.json``, so
    the reuse binding anchors on three immutable facts instead: the pause
    manifest (every recorded file is re-hashed and must match), the source
    run's catalog manifest (must equal the current official catalog), and the
    per-iteration native records validated by ``generation_gate`` against the
    current task contract and the unchanged skill trees. The source snapshot
    is never mutated; replay happens on copies only.

    Returns ``{op: [record, ...]}`` where each list orders a recorded-passing
    iteration first (newest first), then recorded-failing iterations (newest
    first) so a candidate that newly passes the current gate (e.g. after a
    compiler fix) is found by replay in order, and failures feed the fresh
    generation with measured feedback.
    """
    sys.path.insert(0, str(WORKERS_DIR.parent / "ci"))
    from generation_gate import validate_iteration_record
    resume_root = resume_root.resolve()
    pause_file = resume_root / "PAUSE-SNAPSHOT.json"
    if not pause_file.is_file() or pause_file.is_symlink():
        raise ValueError("Resume source lacks a regular PAUSE-SNAPSHOT.json")
    pause = json.loads(pause_file.read_text())
    if pause.get("source_changed_during_copy") != []:
        raise ValueError("Pause snapshot was not captured from an unchanged source")
    if not pause.get("source_run_id"):
        raise ValueError("Pause snapshot lacks its source run identity")
    file_hashes = pause.get("file_sha256")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("Pause manifest records no file hashes")
    for relative, expected in file_hashes.items():
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts
                or path.as_posix() != relative or relative == "PAUSE-SNAPSHOT.json"):
            raise ValueError("Unsafe pause snapshot path: " + relative)
        target = resume_root / path
        if (any(part.is_symlink() for part in (target, *target.parents))
                or not target.resolve().is_relative_to(resume_root)):
            raise ValueError("Symlink or escaping pause snapshot path: " + relative)
        if not target.is_file() or _sha256(target) != expected:
            raise ValueError("Pause snapshot file changed on disk: " + relative)
    # Glob discovery must never consume unlisted native evidence. This is an
    # immutable backup: all files except its own hash manifest are bound.
    actual_files = set()
    for path in resume_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink in pause snapshot: " + str(path))
        if path.is_file() and path != pause_file:
            actual_files.add(path.relative_to(resume_root).as_posix())
    if actual_files != set(file_hashes):
        raise ValueError("Pause snapshot contains unbound or missing files")
    generation = resume_root / "generation"
    manifest = generation / "catalog-manifest.json"
    if "generation/catalog-manifest.json" not in file_hashes:
        raise ValueError("Pause snapshot does not bind its catalog manifest")
    recorded_catalog = json.loads(manifest.read_text())
    # Re-fetching identical official bytes changes only acquisition time.
    # Preserve exact equality for scope, case identities and task hashes.
    source_contract = {k: v for k, v in recorded_catalog.items() if k != "fetched_at"}
    current_contract = {k: v for k, v in catalog.items() if k != "fetched_at"}
    if source_contract != current_contract:
        raise ValueError("Resume source does not cover the exact complete official catalog")
    binding = {
        "source_run_id": pause["source_run_id"],
        "snapshot_manifest_sha256": _sha256(pause_file),
        "source_catalog_manifest_sha256": _sha256(manifest),
        "catalog_sha256": catalog["catalog_sha256"],
        "source_file_count": len(file_hashes),
    }
    records = {}
    for spec in catalog["operators"]:
        name = spec["function_name"]
        item_dir = generation / (name + "-generate")
        iterations = sorted(item_dir.glob("iter*"), key=_iteration_order,
                            reverse=True) if item_dir.is_dir() else []
        valid = []
        for iteration in iterations:
            try:
                record = validate_iteration_record(
                    spec, iteration, skills_root, allow_failed_compile=True)
            except (ValueError, KeyError, json.JSONDecodeError, OSError):
                continue
            report = json.loads((iteration / "local_compile.json").read_text())
            record["resume_binding"] = dict(binding)
            record["recorded_status"] = report.get("status")
            record["recorded_passed"] = report.get("compile_passed")
            valid.append(record)
        if valid:
            # Stable sort: recorded passes stay newest-first before failures.
            records[name] = sorted(
                valid,
                key=lambda r: not (
                    r["recorded_status"] == "passed"
                    and r["recorded_passed"] == spec["case_count"]
                ),
            )
    return records


def load_reuse_records(reuse_root: Path, catalog: dict, skills_root: Path) -> dict:
    """Bind recorded native candidates of a previous full-catalog campaign.

    The source campaign stays immutable; only read-only validation happens
    here. The parent summary must cover the exact complete requested official
    operator set and catalog, so a historical subset or a hand-authored tree
    is rejected. A compiler-only failed candidate is acceptable for replay;
    static-contract failures (no recorded compiler evidence) never are.
    """
    sys.path.insert(0, str(WORKERS_DIR.parent / "ci"))
    from generation_gate import validate_iteration_record
    summary_file = reuse_root / "summary.json"
    if not summary_file.is_file():
        raise ValueError("Reuse source lacks a campaign summary")
    summary = json.loads(summary_file.read_text())
    names = {op["function_name"] for op in catalog["operators"]}
    requested = summary.get("requested_operators") or []
    if (set(requested) != names or len(requested) != len(names)
            or summary.get("catalog_sha256") != catalog["catalog_sha256"]):
        raise ValueError("Reuse source does not cover the complete official catalog")
    records = {}
    for spec in catalog["operators"]:
        name = spec["function_name"]
        item_dir = reuse_root / (name + "-generate")
        iterations = sorted(item_dir.glob("iter*"), key=_iteration_order, reverse=True) if item_dir.is_dir() else []
        for iteration in iterations:
            try:
                records[name] = validate_iteration_record(
                    spec, iteration, skills_root, allow_failed_compile=True)
                break
            except (ValueError, KeyError, json.JSONDecodeError, OSError):
                continue
    return records


def requalify_recorded_iteration(item: WorkItem, record: dict, run_root: Path,
                                 *, attempt: int = 1) -> bool:
    """Replay a validated recorded candidate under the current compile gate.

    The source campaign is never mutated. If the unchanged candidate passes
    every required case, its native evidence is copied into a NEW iteration of
    the new run with an explicit derivation record (original provenance SHA
    retained, validation updated to the replay report hash) and the candidate
    is qualified locally. Otherwise False is returned so the existing
    fresh-generation path runs. Models and sessions remain the original native
    generation; the replay is never relabeled as fresh model generation.
    """
    stage = run_root / item.name / (
        "reuse-attempt" if attempt == 1 else f"reuse-attempt-{attempt}")
    stage.mkdir(parents=True, exist_ok=True)
    candidate = stage / "candidate.py"
    shutil.copy2(record["iteration"] / "candidate.py", candidate)
    problems = static_check(candidate, item.op, item.task_dir)
    if problems:
        (stage / "static_check.json").write_text(json.dumps(problems, indent=2) + "\n")
        item.history.append({"iter": 0, "result": "reuse-static-fail", "problems": problems})
        return False
    print(f"[{item.name}] reuse: replaying the recorded candidate under the "
          "current local compile gate", flush=True)
    digest = local_evaluate(item.op, candidate, stage, item.task_dir)
    (stage / "digest.json").write_text(json.dumps(digest, indent=2) + "\n")
    required = item.contract["case_count"] if item.contract else digest.get("total")
    if (digest.get("hard_failure") or digest.get("passed") != required
            or digest.get("total") != required
            or _sha256(candidate) != record["candidate_sha256"]):
        item.history.append({
            "iter": 0,
            "result": "reuse-replay-fail",
            "passed": f"{digest.get('passed')}/{digest.get('total')}",
        })
        return False
    iter_dir = run_root / item.name / "iter1"
    iter_dir.mkdir(parents=True, exist_ok=True)
    # Copy the recorded native evidence unchanged; the replay compiler report
    # and the derived provenance replace only the replayed artifacts.
    for path in record["iteration"].iterdir():
        if path.is_file() and path.name not in ("provenance.json", "local_compile.json", "digest.json"):
            shutil.copy2(path, iter_dir / path.name)
    shutil.copy2(stage / "local_compile.json", iter_dir / "local_compile.json")
    # Retain original bytes, including failed compiler reports, so derivation
    # hashes remain independently checkable after the parent artifact expires.
    shutil.copytree(record["iteration"], iter_dir / "original-record")
    original = record["provenance"]
    provenance = json.loads(json.dumps(original))
    provenance["validation"] = {
        **(original.get("validation") or {}),
        "report_sha256": _sha256(iter_dir / "local_compile.json"),
        "passed": digest.get("passed"),
        "total": digest.get("total"),
        "replay": True,
    }
    provenance["reuse"] = {
        "mode": "requalified-replay",
        "source_iteration": str(record["iteration"]),
        "original_provenance_sha256": record["provenance_sha256"],
        "original_candidate_sha256": record["candidate_sha256"],
        "original_compiler_report_sha256": record["compiler_report_sha256"],
        "original_validation": original.get("validation"),
        "note": ("Candidate unchanged from the recorded native generation; "
                 "replayed under the current local compile gate. Models and "
                 "sessions remain the original native generation."),
    }
    if record.get("resume_binding"):
        provenance["reuse"]["interrupted_snapshot"] = record["resume_binding"]
    (iter_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    package_dir = run_root / "locally_qualified" / "cann_bench"
    provenance_dir = run_root / "locally_qualified" / "provenance"
    package_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, package_dir / f"{item.op}.py")
    shutil.copy2(iter_dir / "provenance.json", provenance_dir / f"{item.op}.json")
    item.history.append({
        "iter": 1,
        "result": "locally-qualified-reuse",
        "passed": f"{digest.get('passed')}/{digest.get('total')}",
        "models": original["models"],
    })
    item.status = (
        f"locally qualified {digest['passed']}/{digest['total']} "
        "(requalified unchanged candidate; numerics/performance unverified)"
    )
    return True


def process_item(item: WorkItem, queue: EvalQueue | None, run_root: Path,
                 iterations: int, dry_run: bool, evaluation: str,
                 models: tuple[str, ...], phase_attempts: int,
                 reuse_record: dict | None = None,
                 implementation_model: str | None = None,
                 repair_attempts: int = 0) -> WorkItem:
    if implementation_model is not None and implementation_model not in models:
        raise ValueError('implementation model must be in models')
    if repair_attempts < 0:
        raise ValueError('repair attempts must be non-negative')
    item_dir = run_root / item.name
    item_dir.mkdir(parents=True, exist_ok=True)
    # Keep scratch inside the checkout so project permissions and skill paths
    # are stable, while .gitignore prevents generated files from being staged.
    scratch = SCRATCH_ROOT / run_root.name / item.name
    scratch.mkdir(parents=True, exist_ok=True)
    item.accepted_score = incumbent_score(item.op)

    if item.kind == "tune":
        message = prompts.build_tuning_prompt(
            item.op, item.op, item.op,
            extra_levers=EXTRA_LEVERS.get(item.op, ""))
    else:
        overlay_arguments = compile_overlay_arguments()
        overlay_context = {}
        if overlay_arguments:
            overlay_context = {'source_overlay': REPO_ROOT / overlay_arguments[1],
                               'source_overlay_sha256': overlay_arguments[3]}
        message = prompts.build_generation_prompt(
            item.op, item.op, item.op, guidance=GUIDANCE.get(item.op, "-"), task_dir=item.task_dir,
            **overlay_context)
    base_message = message
    (item_dir / "prompt.md").write_text(base_message)

    if dry_run:
        item.status = "dry-run: prompt written"
        return item

    replay_records = []
    if reuse_record is not None:
        # A single completed-campaign record or an ordered interrupted-run
        # record list (recorded passes first, then newest failures).
        replay_records = (list(reuse_record) if isinstance(reuse_record, list)
                          else [reuse_record])
    if replay_records and item.kind == "generate" and evaluation == "local":
        for attempt, record in enumerate(replay_records, start=1):
            if implementation_model is not None:
                original = record["provenance"]
                implementation_sha = original.get("implementation_sha256") or original.get("reviewed_compile", {}).get("candidate_sha256")
                if (original.get("models", {}).get("implement") != implementation_model
                        or (models == (implementation_model,) and any(original.get("models", {}).get(p) != implementation_model for p in WORKFLOW_PHASES))
                        or not implementation_sha
                        or implementation_sha != original.get("candidate_sha256")):
                    item.history.append({"iter": 0, "result": "reuse-model-policy-rejected",
                                         "attempt": attempt})
                    continue
            if requalify_recorded_iteration(item, record, run_root,
                                            attempt=attempt):
                return item
        print(f"[{item.name}] {len(replay_records)} recorded candidate "
              "replay(s) failed; generating fresh candidates", flush=True)
        replay = item_dir / "reuse-attempt"
        if (replay / "digest.json").is_file():
            message = "Recorded candidate replay failed under the current gate. " + json.dumps(
                compact_local_feedback(json.loads((replay / "digest.json").read_text())))
        elif (replay / "static_check.json").is_file():
            message = "Recorded candidate violates the current static contract. " + (replay / "static_check.json").read_text()

    canonical_module = SUBMISSION_PKG / f"{item.op}.py"

    for it in range(1, iterations + 1):
        iter_dir = item_dir / f"iter{it}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        task_message = (
            base_message
            if message == base_message
            else base_message + "\n\n# Evaluator feedback from the previous iteration\n\n" + message
        )
        (iter_dir / "message.md").write_text(task_message)
        print(f"[{item.name}] iter {it}: three-phase worker running",
              flush=True)

        candidate = scratch / "candidate.py"
        design = scratch / "design.md"
        candidate.unlink(missing_ok=True)  # never re-submit a stale file
        design.unlink(missing_ok=True)
        (scratch / "task.md").write_text(task_message, encoding="utf-8")

        # Keep one coherent implementation model across an iteration and use
        # the next model for independent review. A failed next iteration
        # rotates roles, while per-phase attempts still fail over immediately.
        primary_index = (it - 1) % len(models)
        reviewer_index = (primary_index + 1) % len(models)
        phase_model_indexes = {
            "design": primary_index,
            "implement": primary_index,
            "review": reviewer_index,
        }
        phase_results = {}
        actual_phase_models = {}
        reviewed_compile = None
        implementation_sha = None
        for phase in WORKFLOW_PHASES:
            start_index = phase_model_indexes[phase]
            if phase == "review" and "implement" in actual_phase_models:
                if evaluation == "local":
                    print(f"[{item.name}] iter {it}: compiler evidence before review", flush=True)
                    before_review = local_evaluate(item.op, candidate, iter_dir, item.task_dir)
                    pre_report = iter_dir / "pre_review_compile.json"
                    shutil.copy2(iter_dir / "local_compile.json", pre_report)
                    reviewed_compile = {"candidate_sha256": _sha256(candidate),
                                        "report_sha256": _sha256(pre_report)}
                    (scratch / "compile_report.json").write_text(json.dumps({
                        **reviewed_compile, "measured_feedback": compact_local_feedback(before_review),
                        "full_report": json.loads(pre_report.read_text()),
                    }, indent=2) + "\n")
                implementation_index = models.index(
                    actual_phase_models["implement"]
                )
                start_index = (implementation_index + 1) % len(models)
            phase_models = tuple(m for m in models if m != actual_phase_models.get("implement")) if phase == "review" else models
            phase_start = 0 if phase == "review" else start_index
            if implementation_model is not None and (phase in ("design", "implement") or models == (implementation_model,)):
                phase_models = (implementation_model,)
                phase_start = 0
            if phase == "review":
                implementation_sha = _sha256(candidate)
            model = phase_models[phase_start]
            print(
                f"[{item.name}] iter {it}: {phase} phase with {model}",
                flush=True,
            )
            phase_result = run_phase(
                scratch, phase, iter_dir,
                phase_models, phase_start, phase_attempts,
                **({"readonly_review": True} if implementation_model is not None and phase == "review" else {})
            )
            if phase_result is None:
                break
            result, actual_model = phase_result
            phase_results[phase] = result
            actual_phase_models[phase] = actual_model

        if len(phase_results) != len(WORKFLOW_PHASES) or not candidate.exists():
            completed = list(phase_results)
            message = (
                "The previous three-phase workflow did not pass its mandatory "
                "skill provenance gates. Start over from the full original "
                "operator task and produce a fresh candidate."
            )
            item.history.append({
                "iter": it,
                "result": "skill-gate-fail",
                "completed_phases": completed,
            })
            continue
        shutil.copy(candidate, iter_dir / "candidate.py")
        shutil.copy(design, iter_dir / "design.md")
        loaded_skills = []
        observed_dirs = {}
        sessions = {}
        phase_model_record = {}
        for phase, result in phase_results.items():
            sessions[phase] = result.session_id
            phase_model_record[phase] = actual_phase_models[phase]
            for name in result.loaded_skills:
                if name not in loaded_skills:
                    loaded_skills.append(name)
            observed_dirs.update(dict(result.skill_dirs))
        provenance = {
            "generator": "opencode",
            "opencode_version": subprocess.run(
                ["opencode", "--version"], capture_output=True, text=True,
                check=False,
            ).stdout.strip(),
            "models": phase_model_record,
            "sessions": sessions,
            "workflow_phases": list(WORKFLOW_PHASES),
            "required_skills": list(REQUIRED_SKILLS),
            "loaded_skills": loaded_skills,
            "observed_skill_dirs": observed_dirs,
            "skill_gate_passed": set(REQUIRED_SKILLS).issubset(loaded_skills),
            "skill_sources": {
                name: {
                    "path": str(SKILLS_ROOT / name),
                    "files": _skill_source_manifest(name),
                } for name in REQUIRED_SKILLS
            },
            "prompt_sha256": _sha256(scratch / "task.md"),
            "design_sha256": _sha256(design),
            "candidate_sha256": _sha256(candidate),
            "task_contract": item.contract,
            "evaluation_mode": evaluation,
            "reviewed_compile": reviewed_compile,
            "implementation_sha256": implementation_sha,
            "required_implementation_model": implementation_model,
            "readonly_review": implementation_model is not None,
        }
        (iter_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

        repairs_left = repair_attempts
        repair_round = 0
        while True:
            problems = static_check(candidate, item.op, item.task_dir)
            if implementation_model is not None:
                if actual_phase_models["implement"] != implementation_model:
                    problems.append("Implementation model policy violated")
                if implementation_sha != _sha256(candidate):
                    problems.append("Independent reviewer changed candidate; implementation model must repair")
                verdicts = [line.strip() for line in phase_results["review"].text_output.splitlines()
                            if line.strip() in ("REVIEW_PASS", "REVIEW_REPAIR")]
                if verdicts != ["REVIEW_PASS"]:
                    problems.append("Independent review did not provide a single REVIEW_PASS verdict")
            if not provenance["skill_gate_passed"]:
                problems.append("Incomplete native skill provenance")
            if actual_phase_models["implement"] == actual_phase_models["review"] and models != (implementation_model,):
                problems.append("Review must use a different model")
            if not problems:
                break
            if repairs_left and candidate.is_file():
                # Targeted repair within the iteration: keep the design and
                # patch the candidate against the measured findings instead
                # of regenerating from scratch (which churns defect classes).
                repairs_left -= 1
                repair_round += 1
                repair_dir = iter_dir / f"repair{repair_round}"
                repair_dir.mkdir(parents=True, exist_ok=True)
                print(f"[{item.name}] iter {it}: targeted repair attempt "
                      f"{repair_round}/{repair_attempts}: {problems}", flush=True)
                candidate_text = (
                    "\n\nFailing candidate.py (repair this structure; "
                    "preserve what works):\n```python\n"
                    + candidate.read_text(encoding="utf-8") + "\n```\n"
                )
                repair_message = (
                    "Targeted repair of an existing candidate. Apply ONLY "
                    "the repairs required by the findings below, with a "
                    "minimal diff: preserve the architecture, every behavior "
                    "that already works, all official cases, and the full "
                    "contract. Do not redesign. Do not remove or shrink any "
                    "case. Overwrite candidate.py with the repaired source.\n\n"
                    "Static checks failed on candidate.py:\n- "
                    + "\n- ".join(problems)
                    + "\n\nIndependent review feedback:\n"
                    + phase_results["review"].text_output
                    + candidate_text)
                (scratch / "task.md").write_text(repair_message,
                                                 encoding="utf-8")
                item.history.append({
                    "iter": it, "attempt": f"repair{repair_round}",
                    "result": "repair-requested", "problems": problems})
                impl_models = ((implementation_model,)
                               if implementation_model is not None else models)
                repair_impl = run_phase(scratch, "implement", repair_dir,
                                        impl_models, 0, phase_attempts)
                if repair_impl is None or not candidate.is_file():
                    item.history.append({
                        "iter": it, "attempt": f"repair{repair_round}",
                        "result": "repair-implement-failed"})
                    break
                result, actual_model = repair_impl
                phase_results["implement"] = result
                actual_phase_models["implement"] = actual_model
                sessions["implement"] = result.session_id
                phase_model_record["implement"] = actual_model
                for name in result.loaded_skills:
                    if name not in loaded_skills:
                        loaded_skills.append(name)
                if evaluation == "local":
                    print(f"[{item.name}] iter {it} repair {repair_round}: "
                          "compiler evidence before review", flush=True)
                    before_review = local_evaluate(
                        item.op, candidate, repair_dir, item.task_dir)
                    pre_report = repair_dir / "pre_review_compile.json"
                    shutil.copy2(repair_dir / "local_compile.json", pre_report)
                    reviewed_compile = {
                        "candidate_sha256": _sha256(candidate),
                        "report_sha256": _sha256(pre_report)}
                    (scratch / "compile_report.json").write_text(json.dumps({
                        **reviewed_compile,
                        "measured_feedback": compact_local_feedback(before_review),
                        "full_report": json.loads(pre_report.read_text()),
                    }, indent=2) + "\n")
                implementation_sha = _sha256(candidate)
                if implementation_model is not None and models == (implementation_model,):
                    review_models = (implementation_model,)
                else:
                    review_models = tuple(
                        m for m in models
                        if m != actual_phase_models["implement"])
                repair_review = run_phase(
                    scratch, "review", repair_dir, review_models, 0,
                    phase_attempts,
                    **({"readonly_review": True}
                       if implementation_model is not None else {}))
                if repair_review is None:
                    item.history.append({
                        "iter": it, "attempt": f"repair{repair_round}",
                        "result": "repair-review-failed"})
                    break
                result, actual_model = repair_review
                phase_results["review"] = result
                actual_phase_models["review"] = actual_model
                sessions["review"] = result.session_id
                phase_model_record["review"] = actual_model
                for name in result.loaded_skills:
                    if name not in loaded_skills:
                        loaded_skills.append(name)
                shutil.copy(candidate, iter_dir / "candidate.py")
                shutil.copy(candidate, repair_dir / "candidate.py")
                provenance.update({
                    "candidate_sha256": _sha256(candidate),
                    "implementation_sha256": implementation_sha,
                    "reviewed_compile": reviewed_compile,
                    "sessions": sessions,
                    "models": phase_model_record,
                    "loaded_skills": loaded_skills,
                    "skill_gate_passed": set(REQUIRED_SKILLS).issubset(loaded_skills),
                })
                provenance.setdefault("repair_attempts", []).append({
                    "attempt": repair_round,
                    "problems": problems,
                    "candidate_sha256": _sha256(candidate),
                    "implement_model": actual_phase_models["implement"],
                    "review_model": actual_phase_models["review"],
                })
                (iter_dir / "provenance.json").write_text(
                    json.dumps(provenance, indent=2), encoding="utf-8")
                item.history.append({
                    "iter": it, "attempt": f"repair{repair_round}",
                    "result": "repair-applied",
                    "candidate_sha256": _sha256(candidate)})
                continue
            break
        if problems:
            print(f"[{item.name}] iter {it}: static check failed: "
                  f"{problems}", flush=True)
            candidate_text = ""
            if candidate.is_file():
                candidate_text = (
                    "\n\nFailing candidate.py (repair this structure; "
                    "preserve what works):\n```python\n"
                    + candidate.read_text(encoding="utf-8") + "\n```\n"
                )
            message = ("Static checks failed on candidate.py:\n- "
                       + "\n- ".join(problems)
                       + "\n\nFix these and overwrite candidate.py."
                       + "\nIndependent review feedback:\n" + phase_results["review"].text_output
                       + candidate_text)
            item.history.append(
                {"iter": it, "result": "static-fail", "problems": problems,
                 "repair_attempts_used": repair_round})
            continue

        if evaluation == "local":
            print(
                f"[{item.name}] iter {it}: exact-v2 local compile gate",
                flush=True,
            )
            digest = local_evaluate(item.op, candidate, iter_dir, item.task_dir)
            (iter_dir / "digest.json").write_text(
                json.dumps(digest, indent=2) + "\n", encoding="utf-8"
            )
            provenance["validation"] = {
                "label": "verified-local-compile",
                "pyasc_commit": "0a631f70968c3cb7c33ce45330a85768dd5a6f06",
                "report_sha256": _sha256(iter_dir / "local_compile.json"),
                "passed": digest.get("passed"),
                "total": digest.get("total"),
                "limitations": digest.get("limitations", []),
            }
            (iter_dir / "provenance.json").write_text(
                json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
            )
            accepted = not digest["hard_failure"]
            item.history.append({
                "iter": it,
                "result": "locally-qualified" if accepted else "local-fail",
                "passed": f"{digest.get('passed')}/{digest.get('total')}",
                "models": phase_model_record,
            })
            if accepted:
                package_dir = run_root / "locally_qualified" / "cann_bench"
                provenance_dir = run_root / "locally_qualified" / "provenance"
                package_dir.mkdir(parents=True, exist_ok=True)
                provenance_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, package_dir / f"{item.op}.py")
                shutil.copy2(
                    iter_dir / "provenance.json",
                    provenance_dir / f"{item.op}.json",
                )
                item.status = (
                    f"locally qualified {digest['passed']}/{digest['total']} "
                    "(numerics/performance unverified)"
                )
                return item
            message = json.dumps(compact_local_feedback(digest), indent=2)
            (scratch / "compile_feedback.md").write_text(
                "# Exact-v2 local compile feedback\n\n```json\n"
                + message + "\n```\n",
                encoding="utf-8",
            )
            continue

        if queue is None:
            raise RuntimeError("remote evaluation requested without EvalQueue")
        with queue.lock:
            print(f"[{item.name}] iter {it}: private evaluation on CANNBench",
                  flush=True)
            mods = deployed_modules()
            is_new = item.op not in mods
            staged_init = render_init(mods + [item.op] if is_new else mods)
            try:
                digest = queue.evaluate(item.op, candidate, staged_init, iter_dir)
            except RemoteError as exc:
                digest = {"hard_failure": True, "log_tail": str(exc)}

            accept = decide_accept(item, digest)
            if accept:
                shutil.copy(candidate, canonical_module)
                if is_new:
                    (SUBMISSION_PKG / "__init__.py").write_text(
                        render_init(mods + [item.op]))
                if (iter_dir / "report.json").exists():
                    shutil.copy(iter_dir / "report.json",
                                EVIDENCE_DIR / f"{item.op}_final_eval.json")
                item.accepted_score = digest.get("score")
            else:
                # Site submissions are immutable and private. Rejection only
                # means the local canonical package is left unchanged.
                pass

        (iter_dir / "digest.json").write_text(json.dumps(digest, indent=1))
        item.history.append(
            {"iter": it, "result": "accept" if accept else "reject",
             "score": digest.get("score"),
             "passed": f"{digest.get('passed')}/{digest.get('total')}"})
        print(f"[{item.name}] iter {it}: "
              f"{'ACCEPTED' if accept else 'rejected'} "
              f"score={digest.get('score')} "
              f"passed={digest.get('passed')}/{digest.get('total')}",
              flush=True)

        if item.kind == "generate" and accept:
            item.status = f"generated, score {digest['score']}"
            return item

        if it < iterations:
            digest["instruction"] = (
                "Your candidate was ACCEPTED and is now the incumbent — "
                "improve on it further (perf only, keep correctness)."
                if accept else None)
            message = prompts.build_feedback(digest)

    if evaluation == "local":
        item.status = "not locally qualified; see iteration evidence"
    else:
        item.status = f"done, best score {item.accepted_score}"
    return item


def campaign_output_root(output_dir: Path | None = None, timestamp: str | None = None) -> Path:
    """Choose an ignored scratch directory unless the caller pins an output path."""
    if output_dir is not None:
        return output_dir.resolve()
    stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / stamp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True,
                        help="comma list of <op>:<tune|generate>, or all")
    parser.add_argument("--catalog", type=Path, help="Verified official manifest; required for full-catalog generation")
    parser.add_argument("--output-dir", type=Path, help="New campaign directory for exact CI artifact binding")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help="comma-separated working OpenCode models; review uses the next model",
    )
    parser.add_argument(
        "--evaluation", choices=("local", "remote"), default="local",
        help="local is credit-free compile/lowering only; remote consumes credits",
    )
    parser.add_argument("--implementation-model",
                        help="pin design/implementation to this model; require non-mutating independent review")
    parser.add_argument("--phase-attempts", type=int, default=2)
    parser.add_argument("--repair-attempts", type=int, default=0,
                        help="targeted repairs per iteration before falling "
                             "back to from-scratch regeneration (default 0: "
                             "disabled, legacy behavior)")
    parser.add_argument("--skip-gate-preflight", action="store_true",
                        help="skip the compile-gate infrastructure preflight "
                             "(image inspect + exec probe) before the scope")
    parser.add_argument("--reuse-generation", type=Path,
                        help="Previous campaign run directory; validated recorded "
                             "native candidates are requalified before fresh generation")
    parser.add_argument("--resume-interrupted", type=Path,
                        help="Interrupted campaign snapshot (PAUSE-SNAPSHOT.json "
                             "root); validated recorded native iterations are "
                             "replayed in order before fresh generation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.reuse_generation is not None:
        if not args.catalog:
            parser.error("--reuse-generation requires --catalog (full official scope)")
        if args.evaluation == "remote":
            parser.error("--reuse-generation replays candidates under local evaluation only")
    if args.resume_interrupted is not None:
        if args.reuse_generation is not None:
            parser.error("--reuse-generation and --resume-interrupted are mutually exclusive")
        if not args.catalog:
            parser.error("--resume-interrupted requires --catalog (full official scope)")
        if args.evaluation == "remote":
            parser.error("--resume-interrupted replays candidates under local evaluation only")

    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if args.implementation_model and args.implementation_model not in models:
        parser.error('--implementation-model must be in --models')
    if len(set(models)) < 2 and models != (args.implementation_model,):
        parser.error("use one pinned --implementation-model or at least two distinct models")
    if args.workers < 1 or args.iterations < 1:
        parser.error("workers and iterations must be positive")
    if args.phase_attempts < 1:
        parser.error("--phase-attempts must be at least 1")
    if args.repair_attempts < 0:
        parser.error("--repair-attempts must be non-negative")

    items = []
    catalog = None
    catalog_ops = {}
    if args.catalog:
        sys.path.insert(0, str(WORKERS_DIR.parent / "ci"))
        from validate_catalog import validate
        catalog = validate(args.catalog.parent)
        catalog_ops = {op["function_name"]: op for op in catalog["operators"]}
        if args.evaluation == "remote":
            parser.error("Full-catalog remote submission requires the durable campaign submission stage")
    if args.items.strip() == "all" and catalog is None:
        parser.error("--items all requires --catalog; a historical subset is not the full benchmark")
    item_specs = (
        [f"{op}:generate" for op in catalog_ops]
        if args.items.strip() == "all"
        else args.items.split(",")
    )
    for spec in item_specs:
        op, _, kind = spec.strip().partition(":")
        if op not in (catalog_ops or ALL_OPS) or kind not in ("tune", "generate"):
            parser.error(f"bad item {spec!r}")
        if catalog and kind != "generate":
            parser.error("Authoritative catalog campaigns generate fresh candidates")
        if any(item.op == op for item in items):
            parser.error("Duplicate work item: " + op)
        items.append(WorkItem(op=op, kind=kind,
            task_dir=args.catalog.parent / "tasks" / op if catalog else None,
            contract=catalog_ops.get(op)))

    run_root = campaign_output_root(args.output_dir, time.strftime("%Y%m%d_%H%M%S"))
    run_root.mkdir(parents=True)
    if catalog:
        (run_root / "catalog-manifest.json").write_text(json.dumps(catalog, indent=2) + "\n")
    reuse_records = {}
    if args.reuse_generation is not None:
        try:
            reuse_records = load_reuse_records(
                args.reuse_generation.resolve(), catalog, SKILLS_ROOT)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parser.error("Reuse source rejected: " + str(exc))
        print(f"reuse: {len(reuse_records)} recorded candidates requalifiable",
              flush=True)
    if args.resume_interrupted is not None:
        try:
            resume_records = load_resume_records(
                args.resume_interrupted.resolve(), catalog, SKILLS_ROOT)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parser.error("Interrupted snapshot rejected: " + str(exc))
        reuse_records = resume_records
        # Durable binding of the immutable interrupted source to this run.
        # The source lacks a completed summary by definition; this record is
        # the run's evidence of what was bound, never a fabricated summary.
        binding = {
            "mode": "resume-interrupted",
            "source_snapshot": str(args.resume_interrupted.resolve()),
            "requested_operators": [item.op for item in items],
            "catalog_sha256": catalog["catalog_sha256"],
            "records": {
                op: [
                    {
                        "iteration": str(record["iteration"]),
                        "candidate_sha256": record["candidate_sha256"],
                        "provenance_sha256": record["provenance_sha256"],
                        "compiler_report_sha256": record["compiler_report_sha256"],
                        "recorded_status": record.get("recorded_status"),
                        "recorded_passed": record.get("recorded_passed"),
                        "resume_binding": record.get("resume_binding"),
                    }
                    for record in op_records
                ]
                for op, op_records in sorted(resume_records.items())
            },
        }
        (run_root / "resume-binding.json").write_text(
            json.dumps(binding, indent=2) + "\n", encoding="utf-8")
        print(f"resume-interrupted: {len(resume_records)} operators with "
              "validated native records replay before fresh generation",
              flush=True)
    print(f"run dir: {run_root}", flush=True)

    if (args.evaluation == "local" and not args.dry_run
            and not args.skip_gate_preflight):
        preflight_error = _gate_preflight()
        if preflight_error is not None:
            print(f"[driver] compile gate infrastructure preflight FAILED: "
                  f"{preflight_error}. Refusing to spend phase budget; fix "
                  "the gate infrastructure and resume.", flush=True)
            return 2

    queue = EvalQueue() if args.evaluation == "remote" else None
    infra_abort = None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                process_item, item, queue, run_root, args.iterations,
                args.dry_run, args.evaluation, models, args.phase_attempts,
                reuse_records.get(item.op), args.implementation_model,
                args.repair_attempts,
            ) for item in items
        ]
        results = []
        for item, future in zip(items, futures):
            try:
                results.append(future.result())
            except GateInfrastructureError as exc:
                # Infrastructure failure must not consume iteration budgets.
                # Stop starting new operators; in-flight ones settle, then
                # the scope exits with an explicit abort marker for resume.
                infra_abort = str(exc)
                item.status = "infra-abort: " + str(exc)
                results.append(item)
                for pending in futures:
                    pending.cancel()
            except Exception as exc:
                item.status = "failed: " + type(exc).__name__ + ": " + str(exc)
                results.append(item)

    print("\n=== campaign summary ===")
    for item in results:
        print(f"{item.name}: {item.status}")
        for h in item.history:
            print(f"  {h}")
    summary = {
        "evaluation": args.evaluation,
        "models": list(models),
        "resumed_interrupted_snapshot": (
            str(args.resume_interrupted.resolve())
            if args.resume_interrupted is not None else None),
        "requested_operators": [item.op for item in items],
        "required_operators": catalog["required_operators"] if catalog else len(items),
        "required_cases": catalog["required_cases"] if catalog else None,
        "catalog_sha256": catalog["catalog_sha256"] if catalog else None,
        "locally_qualified_operators": [
            item.op for item in results
            if item.status.startswith("locally qualified")
        ],
        "items": [
            {"item": item.name, "status": item.status, "history": item.history}
            for item in results
        ],
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if infra_abort is not None:
        (run_root / "INFRA-ABORT.json").write_text(
            json.dumps({"reason": infra_abort,
                        "note": "scope stopped at boundary to avoid consuming "
                                "candidate iteration/repair budget on invalid "
                                "infrastructure evidence; fix the gate "
                                "infrastructure and resume with "
                                "--resume-interrupted",
                        "completed_items": [
                            {"item": item.name, "status": item.status,
                             "history": item.history}
                            for item in results]},
                       indent=2) + "\n", encoding="utf-8")
        print(f"[driver] INFRA-ABORT: {infra_abort}; scope stopped "
              "(see INFRA-ABORT.json)", flush=True)
        return 2
    if args.evaluation == "local" and summary["locally_qualified_operators"]:
        package_dir = run_root / "locally_qualified" / "cann_bench"
        shutil.copy2(SUBMISSION_PKG / "_pyasc_runtime.py", package_dir)
        (package_dir / "__init__.py").write_text(
            render_init(summary["locally_qualified_operators"]),
            encoding="utf-8",
        )
        qualification = {
            "status": (
                "complete" if len(summary["locally_qualified_operators"])
                == len(items) else "partial"
            ),
            "evidence": "verified-local-compile",
            "operators": summary["locally_qualified_operators"],
            "case_counts": {item.op: item.contract["case_count"] if item.contract else 20 for item in items},
            "limitations": [
                "generated kernels were not executed numerically",
                "performance was not measured",
                "canonical submission modules were not overwritten",
            ],
        }
        (run_root / "locally_qualified" / "QUALIFICATION.json").write_text(
            json.dumps(qualification, indent=2) + "\n", encoding="utf-8"
        )
    complete = len(summary["locally_qualified_operators"]) == len(items)
    return 0 if args.evaluation == "remote" or args.dry_run or complete else 1


if __name__ == "__main__":
    sys.exit(main())
