#!/usr/bin/env python3
"""Adapt current-v2 upstream target kernels without changing their bodies."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
TASKS = REPO_ROOT / "integrations/cannbench/tasks"
SOURCE_ROOT = ROOT / "pyasc-v2-source/python/test/asctile/target"
RUNTIME_COMMIT = "030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d"
OPS = ("gelu", "rms_norm", "softmax", "transpose")
KERNEL_DEFS = {
    "gelu": ("gelu",),
    "rms_norm": (
        "calculate_square_reduce_sum", "compute_rstd_newton_raphson",
        "compute_y", "rms_norm_kernel",
    ),
    "softmax": ("softmax_fused", "softmax_small_row"),
    "transpose": (
        "transpose_block", "transpose_column", "transpose_line",
        "transpose_nlast_axis", "transpose_nlast_axis_fat",
        "transpose_one_axis", "simple_copy", "transpose_2_axis",
        "simplify_shape",
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def definition_segments(source: str, names: tuple[str, ...]) -> dict[str, str]:
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            starts = [node.lineno, *(d.lineno for d in node.decorator_list)]
            found[node.name] = "".join(lines[min(starts) - 1:node.end_lineno])
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f"missing upstream definitions: {sorted(missing)}")
    return found


def run_adapter(op: str, work: Path) -> dict:
    config = {"skills": {"paths": []}, "mcp": {"cann-bench-site": {"enabled": False}}}
    env = {
        **os.environ,
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
    }
    prompt = f"""You are writing ONLY the host transport adapter for the handwritten pyasc target arm.
Read upstream_target.py, desc.md, proto.yaml, and cases.yaml. Write wrapper.py containing the exact public callable `{op}` required by proto.yaml plus optional plain-Python host helper functions. Do not add imports, decorators, or any new JIT/device function. The final module already contains the upstream @asctile.jit definitions listed below, unchanged:
{', '.join(KERNEL_DEFS[op])}.
Use only those kernels. Call ensure_npu_platform(), allocate outputs with torch.empty/empty_like/zeros, derive launch sizes from tensor metadata, and launch the target kernels. Do not call torch numerical operations or data_ptr. Do not pretend the narrower handwritten target implements semantics it does not have: adapt signatures, shapes, tails and launches, but preserve its algorithm. Current runtime is v2 `{RUNTIME_COMMIT}` and imports asctile. End with ADAPTER_DONE."""
    wrapper = work / "wrapper.py"
    manual = work / "wrapper.manual.py"
    if manual.is_file():
        shutil.copy2(manual, wrapper)
        prior_attempts = []
        for path in sorted(work.glob("adapter.attempt*.json")):
            prior_attempts.append(json.loads(path.read_text(encoding="utf-8")))
        record = {
            "attempt": 0,
            "model": "deterministic-manual-transport",
            "session_id": None,
            "returncode": 0,
            "timed_out": False,
            "skill_call_count": 0,
            "completion_marker_seen": True,
            "wrapper_exists": True,
            "accepted": True,
            "attempts": prior_attempts,
            "reason": "model adapters failed; deterministic host transport does not alter upstream JIT bodies",
        }
        (work / "adapter.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record
    attempts = []
    accepted_record = None
    # Qwen is used first here because GLM produced no adapter before the
    # original 900-second timeout for both initial operators.  That failed
    # batch is retained in orchestration notes; target JIT bodies are not
    # model-generated in either case.
    models = ("dashscope/qwen3.7-max", "dashscope/glm-5.2", "dashscope/qwen3.7-max")
    prior = [path for path in (work / "adapter.attempt1.json", work / "adapter.attempt2.json")
             if path.is_file()]
    if len(prior) == 2 and all(not json.loads(path.read_text(encoding="utf-8")).get("accepted")
                               for path in prior):
        models = ("dashscope/qwen3.7-max",)
        start_attempt = 3
    else:
        start_attempt = 1
    for attempt, model in enumerate(models, start_attempt):
        wrapper.unlink(missing_ok=True)
        cmd = [
            "opencode", "run", "--pure", "--format", "json", "--dir", str(work),
            "-m", model, prompt,
        ]
        try:
            proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True,
                                  text=True, timeout=600, check=False)
            stdout, stderr, returncode, timed_out = (
                proc.stdout, proc.stderr, proc.returncode, False)
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = exc.stdout or "", exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            returncode, timed_out = 124, True
        (work / f"adapter.attempt{attempt}.events.jsonl").write_text(
            stdout, encoding="utf-8")
        (work / f"adapter.attempt{attempt}.stderr.txt").write_text(
            stderr, encoding="utf-8")
        skill_calls = []
        event_text = []
        session = None
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            session = event.get("sessionID") or session
            part = event.get("part") or {}
            if event.get("type") == "text":
                event_text.append(str(part.get("text") or ""))
            if event.get("type") == "tool_use" and part.get("tool") == "skill":
                skill_calls.append(part.get("state"))
        accepted = (
            returncode == 0 and wrapper.is_file()
            and "ADAPTER_DONE" in "\n".join(event_text) and not skill_calls
        )
        record = {
            "attempt": attempt,
            "model": model,
            "session_id": session,
            "returncode": returncode,
            "timed_out": timed_out,
            "skill_call_count": len(skill_calls),
            "completion_marker_seen": "ADAPTER_DONE" in "\n".join(event_text),
            "wrapper_exists": wrapper.is_file(),
            "accepted": accepted,
        }
        attempts.append(record)
        (work / f"adapter.attempt{attempt}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if accepted:
            accepted_record = record
            break
    if accepted_record is None:
        raise RuntimeError(f"adapter worker failed for {op}: {attempts}")
    record = {**accepted_record, "attempts": attempts}
    (work / "adapter.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tree = ast.parse(wrapper.read_text(encoding="utf-8"))
    forbidden = [
        getattr(node, "name", type(node).__name__) for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
        or isinstance(node, ast.FunctionDef) and node.decorator_list
    ]
    public = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == op]
    if forbidden or len(public) != 1:
        raise RuntimeError(f"invalid adapter shape for {op}: forbidden={forbidden}, public={len(public)}")
    return record


def prepare_one(op: str) -> dict:
    candidate = ROOT / "handwritten/candidates" / f"{op}.py"
    existing = ROOT / "handwritten" / op / "provenance.json"
    if candidate.is_file() and existing.is_file():
        prior = json.loads(existing.read_text(encoding="utf-8"))
        if (prior.get("source_commit") == RUNTIME_COMMIT
                and prior.get("candidate_sha256") == sha256_bytes(candidate.read_bytes())):
            return {"operator": op, "candidate_sha256": prior["candidate_sha256"],
                    "resumed": True}
    source_path = SOURCE_ROOT / f"test_{op}.py"
    source = source_path.read_text(encoding="utf-8")
    segments = definition_segments(source, KERNEL_DEFS[op])
    work = ROOT / "handwritten" / op
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, work / "upstream_target.py")
    for name in ("desc.md", "proto.yaml", "cases.yaml", "golden.py"):
        shutil.copy2(TASKS / op / name, work / name)
    adapter = run_adapter(op, work)
    definitions = []
    hashes = {}
    for name in KERNEL_DEFS[op]:
        segment = segments[name]
        hashes[name] = sha256_bytes(segment.encode())
        if op == "gelu" and name == "gelu":
            segment = segment.replace("def gelu(", "def _target_gelu_kernel(", 1)
        definitions.append(segment.rstrip())
    header = """# Source-derived from compiler-team/pyasc v2; see PROVENANCE.json.\nimport math\nimport torch\nimport asctile\n\nfrom ._pyasc_runtime import ensure_npu_platform\n\n"""
    candidate_text = header + "\n\n".join(definitions) + "\n\n" + (work / "wrapper.py").read_text(encoding="utf-8")
    if op == "gelu":
        candidate_text = candidate_text.replace("gelu[", "_target_gelu_kernel[")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(candidate_text, encoding="utf-8")
    compile_proc = subprocess.run(
        ["python3", "-m", "py_compile", str(candidate)],
        capture_output=True, text=True, check=False,
    )
    if compile_proc.returncode:
        raise RuntimeError(f"assembled {op} does not parse: {compile_proc.stderr}")
    provenance = {
        "arm": "handwritten_target",
        "operator": op,
        "source_repository": "https://gitcode.com/compiler-team/pyasc",
        "source_ref": "v2",
        "source_commit": RUNTIME_COMMIT,
        "source_path": f"python/test/asctile/target/test_{op}.py",
        "copied_definitions": list(KERNEL_DEFS[op]),
        "source_definition_sha256": hashes,
        "adapter": adapter,
        "candidate_sha256": sha256_bytes(candidate.read_bytes()),
        "adapter_scope": "host signature/allocation/launch transport only; upstream JIT definition text retained",
    }
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"operator": op, "candidate_sha256": provenance["candidate_sha256"]}


def main() -> int:
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(prepare_one, op): op for op in OPS}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    (ROOT / "handwritten-summary.json").write_text(
        json.dumps(sorted(results, key=lambda x: x["operator"]), indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
