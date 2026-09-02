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
SUBMISSION_PKG = WORKERS_DIR.parent / "submission" / "cann_bench"
EVIDENCE_DIR = WORKERS_DIR.parent.parent.parent / "evidence" / "cannbench"
RUNS_DIR = WORKERS_DIR / "runs"

OPENCODE_MODEL = "dashscope/glm-5.2"
WORKER_TIMEOUT_S = 1800

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
kernels launched directly on torch_npu-owned device buffers (zero-copy via
``Tensor.data_ptr()``).
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


# ---------------------------------------------------------------- static check

ALLOWED_TORCH_CALLS = {"empty", "empty_like", "zeros", "zeros_like", "tensor"}
BANNED_TORCH_SUBMODULES = {"nn", "ops", "functional", "linalg", "special",
                           "fft", "cuda", "npu"}


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

    has_callable = any(
        isinstance(n, ast.FunctionDef) and n.name == callable_name
        for n in tree.body)
    if not has_callable:
        problems.append(
            f"missing top-level public callable def {callable_name}(...)")

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
            if name and name.startswith("torch.") and name.count(".") == 1:
                fn = name.split(".")[1]
                if fn not in ALLOWED_TORCH_CALLS:
                    problems.append(
                        f"banned torch call: {name}() — torch is allowed only "
                        f"for {sorted(ALLOWED_TORCH_CALLS)}; all math must be "
                        f"asc2 kernels")
    return sorted(set(problems))


# ------------------------------------------------------------------- workers

def run_worker(scratch: Path, message: str, resume: bool,
               log_path: Path) -> bool:
    cmd = ["opencode", "run", "--pure", "-m", OPENCODE_MODEL]
    if resume:
        cmd.append("-c")
    cmd.append(message)
    # opencode resolves its project dir from $PWD, not the process cwd
    env = {**os.environ, "PWD": str(scratch)}
    try:
        proc = subprocess.run(cmd, cwd=scratch, env=env, capture_output=True,
                              text=True, timeout=WORKER_TIMEOUT_S)
        log_path.write_text(
            f"rc={proc.returncode}\n=== STDOUT ===\n{proc.stdout}\n"
            f"=== STDERR ===\n{proc.stderr}\n")
        return proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        log_path.write_text(
            f"worker timed out after {WORKER_TIMEOUT_S}s\n"
            f"=== PARTIAL STDOUT ===\n{out[-8000:]}\n")
        return False


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


def process_item(item: WorkItem, queue: EvalQueue, run_root: Path,
                 iterations: int, dry_run: bool) -> WorkItem:
    item_dir = run_root / item.name
    item_dir.mkdir(parents=True, exist_ok=True)
    # scratch must live OUTSIDE the git repo: opencode resolves relative
    # paths against the enclosing project root, not its cwd
    scratch = Path("/tmp/cbworkers") / run_root.name / item.name
    scratch.mkdir(parents=True, exist_ok=True)
    item.accepted_score = incumbent_score(item.op)

    if item.kind == "tune":
        message = prompts.build_tuning_prompt(
            item.op, item.op, item.op,
            extra_levers=EXTRA_LEVERS.get(item.op, ""))
    else:
        message = prompts.build_generation_prompt(
            item.op, item.op, item.op, guidance=GUIDANCE.get(item.op, "-"))
    (item_dir / "prompt.md").write_text(message)

    if dry_run:
        item.status = "dry-run: prompt written"
        return item

    canonical_module = SUBMISSION_PKG / f"{item.op}.py"
    resume = False

    for it in range(1, iterations + 1):
        iter_dir = item_dir / f"iter{it}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "message.md").write_text(message)
        print(f"[{item.name}] iter {it}: worker running "
              f"({'resume' if resume else 'fresh'})", flush=True)

        candidate = scratch / "candidate.py"
        candidate.unlink(missing_ok=True)  # never re-submit a stale file
        # DashScope rate-limits under concurrency and opencode exits silently
        # (rc=0, no output) when throttled — retry within the iteration
        for attempt in range(3):
            run_worker(scratch, message, resume,
                       iter_dir / f"worker.attempt{attempt + 1}.log")
            if candidate.exists():
                break
            if attempt < 2:
                print(f"[{item.name}] iter {it}: no output from worker "
                      f"(attempt {attempt + 1}), retrying", flush=True)
                time.sleep(60)
            else:
                print(f"[{item.name}] iter {it}: no output from worker "
                      "after 3 attempts", flush=True)
        resume = True
        if not candidate.exists():
            message = ("You did not write candidate.py in the working "
                       "directory. Write the complete module to candidate.py "
                       "now, exactly as specified.")
            item.history.append({"iter": it, "result": "no candidate"})
            continue
        shutil.copy(candidate, iter_dir / "candidate.py")

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

    item.status = f"done, best score {item.accepted_score}"
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True,
                        help="comma list of <op>:<tune|generate>")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = []
    for spec in args.items.split(","):
        op, _, kind = spec.strip().partition(":")
        if op not in ALL_OPS or kind not in ("tune", "generate"):
            parser.error(f"bad item {spec!r}")
        items.append(WorkItem(op=op, kind=kind))

    run_root = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True)
    print(f"run dir: {run_root}", flush=True)

    queue = EvalQueue()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_item, item, queue, run_root,
                               args.iterations, args.dry_run)
                   for item in items]
        results = [f.result() for f in futures]

    print("\n=== campaign summary ===")
    for item in results:
        print(f"{item.name}: {item.status}")
        for h in item.history:
            print(f"  {h}")
    (run_root / "summary.json").write_text(json.dumps(
        [{"item": i.name, "status": i.status, "history": i.history}
         for i in results], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
