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
from evalqueue import EvalQueue, RemoteError

WORKERS_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKERS_DIR.parent.parent.parent
SUBMISSION_PKG = WORKERS_DIR.parent / "submission" / "cann_bench"
EVIDENCE_DIR = REPO_ROOT / "evidence" / "cannbench"
RUNS_DIR = WORKERS_DIR / "runs"
SCRATCH_ROOT = WORKERS_DIR / ".scratch"
SKILLS_ROOT = REPO_ROOT / "skills"

DEFAULT_MODELS = ("dashscope/glm-5.2", "dashscope/qwen3.7-max")
WORKER_TIMEOUT_S = 360
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
Use at most four tool calls. Do not inspect current submission modules, old run
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
""",
    "review": """\
This is the independent REVIEW phase of a provenance-gated kernel workflow.
Before reviewing, invoke the OpenCode `skill` tool for each exact skill:
{skills}. Read candidate.py and only the target-operator section of task.md.
Apply the skill, review the exact public contract, host/JIT boundary, tails,
dtypes, special values, and UB budget; fix candidate.py in place if needed,
then run only `python3 -m py_compile candidate.py`. Do not replace numerical
work with torch operations and create no other files. End with REVIEW_DONE.
Use at most five tool calls.
Do not inspect current submission modules or old generated candidates.
""",
    "repair": """\
This is a measured-feedback REPAIR phase. Before changing code, invoke the
OpenCode `skill` tool for each exact skill: {skills}. Read task.md, design.md,
candidate.py, and compile_feedback.md. Apply the skill to the measured local
failure for this operator, repair candidate.py in place without copying an
earlier repository implementation, and run only `python3 -m py_compile
candidate.py`. Create no other files. End with REPAIR_DONE.
Use at most five tool calls and stop as soon as the measured failure is fixed.
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

    def missing_skills(self, required: tuple[str, ...]) -> tuple[str, ...]:
        loaded = set(self.loaded_skills)
        return tuple(name for name in required if name not in loaded)

    def skill_gate_passed(self, required: tuple[str, ...]) -> bool:
        return self.returncode == 0 and not self.missing_skills(required)


# ---------------------------------------------------------------- static check

ALLOWED_TORCH_CALLS = {"empty", "empty_like", "zeros", "zeros_like", "tensor"}
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


def static_check(path: Path, callable_name: str) -> list[str]:
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
        actual = tuple(arg.arg for arg in callable_node.args.args)
        expected = EXPECTED_PARAMETERS[callable_name]
        if actual != expected:
            problems.append(
                f"public signature parameters {actual!r} != {expected!r}"
            )
        if callable_node.args.vararg or callable_node.args.kwarg:
            problems.append("public callable must not use *args or **kwargs")

    has_jit = any(
        isinstance(n, ast.FunctionDef) and any(
            (_dotted(d) or _dotted(getattr(d, "func", None) or d)) == "asc2.jit"
            for d in n.decorator_list)
        for n in ast.walk(tree))
    if not has_jit:
        problems.append("no @asc2.jit kernel found")

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
    return sorted(set(problems))


# ------------------------------------------------------------------- workers

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


def run_worker(scratch: Path, message: str, log_path: Path,
               model: str, *, session_id: str | None = None,
               timeout_s: int = WORKER_TIMEOUT_S) -> WorkerResult:
    cmd = [
        "opencode", "run", "--pure", "--format", "json",
        "--dir", str(scratch), "-m", model,
    ]
    if session_id:
        cmd.extend(["--session", session_id])
    cmd.append(message)
    # Pin skill discovery to this checkout. External skill catalogs are
    # disabled so similarly named user skills cannot satisfy provenance.
    config = {
        "skills": {"paths": [str(SKILLS_ROOT)]},
        "mcp": {"cann-bench-site": {"enabled": False}},
    }
    env = {
        **os.environ,
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
    }
    try:
        proc = subprocess.run(cmd, cwd=scratch, env=env, capture_output=True,
                              text=True, timeout=timeout_s)
        log_path.write_text(
            f"rc={proc.returncode}\n=== STDOUT ===\n{proc.stdout}\n"
            f"=== STDERR ===\n{proc.stderr}\n")
        session_id, loaded, skill_dirs, text_output = _parse_worker_trace(
            proc.stdout
        )
        return WorkerResult(
            proc.returncode, session_id, loaded, skill_dirs, text_output
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        log_path.write_text(
            f"worker timed out after {timeout_s}s\n"
            f"=== PARTIAL STDOUT ===\n{out[-8000:]}\n")
        session_id, loaded, skill_dirs, text_output = _parse_worker_trace(out)
        return WorkerResult(124, session_id, loaded, skill_dirs, text_output)


def run_phase(scratch: Path, phase: str, iter_dir: Path,
              models: tuple[str, ...], start_index: int,
              attempts: int) -> tuple[WorkerResult, str] | None:
    """Run one isolated workflow phase and enforce its native skill calls."""
    required = PHASE_SKILLS[phase]
    message = PHASE_PROMPTS[phase].format(skills=", ".join(required))
    expected = {
        "design": scratch / "design.md",
        "implement": scratch / "candidate.py",
        "review": scratch / "candidate.py",
        "repair": scratch / "candidate.py",
    }[phase]
    marker = f"{phase.upper()}_DONE"

    for attempt in range(1, attempts + 1):
        model = models[(start_index + attempt - 1) % len(models)]
        result = run_worker(
            scratch,
            message,
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


def local_evaluate(op: str, candidate: Path, iter_dir: Path) -> dict:
    try:
        candidate_arg = candidate.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        return {"hard_failure": True, "error": str(exc), "status": "failed"}
    command = [
        str(WORKERS_DIR / "run_local_compile_gate.sh"),
        "--candidate", str(candidate_arg), "--op", op,
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = {
            "status": "failed",
            "fatal_error": "local compile gate emitted non-JSON output",
            "stdout_tail": proc.stdout[-4000:],
        }
    report["returncode"] = proc.returncode
    if proc.stderr:
        report["stderr_tail"] = proc.stderr[-8000:]
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


def process_item(item: WorkItem, queue: EvalQueue | None, run_root: Path,
                 iterations: int, dry_run: bool, evaluation: str,
                 models: tuple[str, ...], phase_attempts: int) -> WorkItem:
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
        message = prompts.build_generation_prompt(
            item.op, item.op, item.op, guidance=GUIDANCE.get(item.op, "-"))
    base_message = message
    (item_dir / "prompt.md").write_text(base_message)

    if dry_run:
        item.status = "dry-run: prompt written"
        return item

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
        for phase in WORKFLOW_PHASES:
            start_index = phase_model_indexes[phase]
            if phase == "review" and "implement" in actual_phase_models:
                implementation_index = models.index(
                    actual_phase_models["implement"]
                )
                start_index = (implementation_index + 1) % len(models)
            model = models[start_index]
            print(
                f"[{item.name}] iter {it}: {phase} phase with {model}",
                flush=True,
            )
            phase_result = run_phase(
                scratch, phase, iter_dir, models, start_index, phase_attempts
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
            "evaluation_mode": evaluation,
        }
        (iter_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )

        problems = static_check(candidate, item.op)
        if problems:
            print(f"[{item.name}] iter {it}: static check failed: "
                  f"{problems}", flush=True)
            message = ("Static checks failed on candidate.py:\n- "
                       + "\n- ".join(problems)
                       + "\n\nFix these and overwrite candidate.py.")
            item.history.append(
                {"iter": it, "result": "static-fail", "problems": problems})
            continue

        if evaluation == "local":
            print(
                f"[{item.name}] iter {it}: exact-v2 local compile gate",
                flush=True,
            )
            digest = local_evaluate(item.op, candidate, iter_dir)
            (iter_dir / "digest.json").write_text(
                json.dumps(digest, indent=2) + "\n", encoding="utf-8"
            )
            provenance["validation"] = {
                "label": "verified-local-compile",
                "pyasc_commit": "ac1222a48c8914d3f81297c7570d1a84f0f26778",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True,
                        help="comma list of <op>:<tune|generate>, or all")
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
    parser.add_argument("--phase-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if not models:
        parser.error("at least one --models entry is required")
    if args.phase_attempts < 1:
        parser.error("--phase-attempts must be at least 1")

    items = []
    item_specs = (
        [f"{op}:generate" for op in ALL_OPS]
        if args.items.strip() == "all"
        else args.items.split(",")
    )
    for spec in item_specs:
        op, _, kind = spec.strip().partition(":")
        if op not in ALL_OPS or kind not in ("tune", "generate"):
            parser.error(f"bad item {spec!r}")
        items.append(WorkItem(op=op, kind=kind))

    run_root = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True)
    print(f"run dir: {run_root}", flush=True)

    queue = EvalQueue() if args.evaluation == "remote" else None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                process_item, item, queue, run_root, args.iterations,
                args.dry_run, args.evaluation, models, args.phase_attempts,
            ) for item in items
        ]
        results = [f.result() for f in futures]

    print("\n=== campaign summary ===")
    for item in results:
        print(f"{item.name}: {item.status}")
        for h in item.history:
            print(f"  {h}")
    summary = {
        "evaluation": args.evaluation,
        "models": list(models),
        "requested_operators": [item.op for item in items],
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
            "cases_per_operator": 20,
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
