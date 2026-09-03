#!/usr/bin/env python3
"""Generate controlled no-skill and skill-guided four-operator arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
WORKERS = REPO_ROOT / "integrations/cannbench/workers"
SKILLS = REPO_ROOT / "skills"
OPS = ("gelu", "rms_norm", "softmax", "transpose")
RUNTIME_COMMIT = "030e9b2c0ce44cbc5f9523e03e131f4a23c23a2d"
MODELS = {
    "design": "dashscope/glm-5.2",
    "implement": "dashscope/glm-5.2",
    "review": "dashscope/qwen3.7-max",
}
PHASE_SKILLS = {
    "design": ("pyasc-cannbench-kernel", "pyasc-syntax-constraints"),
    "implement": ("pyasc-cannbench-kernel", "pyasc-syntax-constraints"),
    "review": (
        "pyasc-cannbench-kernel",
        "pyasc-code-review",
        "pyasc-build-run-verify",
    ),
}

GUIDANCE = {
    "gelu": """Use current asctile APIs. Cover approximate='none' and 'tanh' separately. The upstream handwritten target is deliberately not available to generated arms. Compute in f32 for half inputs, preserve NaN/Inf positions, and use real_shape tails.""",
    "rms_norm": """Normalize independently over the last dimension. Cover D=2..8192 and all listed f16/bf16/f32 routes. Accumulate squares in f32 and do not emit the optional rstd output from the upstream target; CANNBench returns y only.""",
    "softmax": """Normalize dim, including negative values. Treat x as [outer, axis_size, inner]. Do not use torch.permute/softmax. Use a full-row path when inner==1 and an asctile local-transpose path otherwise. Cover axis_size through 8193 and preserve special values.""",
    "transpose": """Cover every listed rank-2..5 permutation and f16/bf16/f32/int8/int16/int32/int64 dtype. All data movement must happen in asctile kernels. Collapse adjacent dimensions when valid and tile so both input and transposed output physical last dimensions are 32-byte aligned.""",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_events(stdout: str) -> dict:
    session_id = None
    text = []
    skill_calls = []
    for raw in stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        session_id = event.get("sessionID") or session_id
        part = event.get("part") or {}
        state = part.get("state") or {}
        if event.get("type") == "text" and isinstance(part.get("text"), str):
            text.append(part["text"])
        if event.get("type") == "tool_use" and part.get("tool") == "skill":
            skill_calls.append({
                "status": state.get("status"),
                "name": (state.get("input") or {}).get("name"),
                "directory": (state.get("metadata") or {}).get("dir"),
            })
    return {"session_id": session_id, "text": "\n".join(text), "skill_calls": skill_calls}


def phase_prompt(arm: str, phase: str) -> str:
    required = PHASE_SKILLS[phase]
    skill_instruction = (
        "Before any other action, invoke the OpenCode skill tool for every exact skill: "
        + ", ".join(required)
        + ". Apply them to the task."
        if arm == "with_skills"
        else "Do not invoke the skill tool and do not read any SKILL.md. Work only from task.md and artifacts from earlier phases."
    )
    if phase == "design":
        action = "Read task.md and write design.md. Cover every case, dispatch, tiling, tails, dtypes, UB, numerical behavior, and anti-cheat constraints. Do not write candidate.py."
    elif phase == "implement":
        action = "Read task.md and design.md. Write the complete module to candidate.py. Run only python3 -m py_compile candidate.py. Create no other files."
    else:
        action = "Read task.md, design.md, and candidate.py. Independently review all 20 cases and fix candidate.py in place. Run only python3 -m py_compile candidate.py. Create no other files."
    return (
        f"{skill_instruction}\n{action}\n"
        f"The runtime is compiler-team/pyasc v2 commit {RUNTIME_COMMIT}; it exports asctile, not asc2. "
        f"End with {phase.upper()}_DONE."
    )


def run_phase(arm: str, op_root: Path, phase: str) -> dict:
    work = op_root / "work"
    evidence = op_root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    config = {
        "skills": {"paths": [str(SKILLS)] if arm == "with_skills" else []},
        "mcp": {"cann-bench-site": {"enabled": False}},
    }
    env = {
        **os.environ,
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
    }
    preferred = MODELS[phase]
    fallback = "dashscope/qwen3.7-max" if preferred.endswith("glm-5.2") else "dashscope/glm-5.2"
    expected = work / ("design.md" if phase == "design" else "candidate.py")
    required = set(PHASE_SKILLS[phase]) if arm == "with_skills" else set()
    # Resume an accepted phase instead of recreating a proven artifact when a
    # later phase was the one that failed or timed out.
    if expected.is_file():
        for record_path in sorted(evidence.glob(f"{phase}.attempt*.json"), reverse=True):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("accepted"):
                return record
    # The configured Alibaba international endpoint is not authenticated on
    # this host; recovery intentionally retries the known-working DashScope
    # Qwen worker after the first long request has been released.
    recovery = "dashscope/qwen3.7-max"
    models = (preferred, fallback, recovery)
    # A resumed campaign must not spend another 30 minutes repeating two
    # already-recorded provider failures for the same phase.  Continue with a
    # third available OpenCode worker while retaining the earlier evidence.
    prior = [read for read in (evidence / f"{phase}.attempt1.json",
                               evidence / f"{phase}.attempt2.json")
             if read.is_file()]
    if len(prior) == 2 and all(not json.loads(path.read_text(encoding="utf-8")).get("accepted")
                               for path in prior):
        models = (recovery,)
        start_attempt = 3
    else:
        start_attempt = 1
    for attempt, model in enumerate(models, start=start_attempt):
        if phase in {"design", "implement"} and not (attempt >= 3 and expected.is_file()):
            expected.unlink(missing_ok=True)
        prompt = phase_prompt(arm, phase)
        if attempt >= 3:
            prompt = ("RECOVERY RUN: prior provider calls timed out. Do not browse unrelated files, "
                      "do not narrate, and write the required artifact immediately after the required "
                      "skill calls (if any).\n" + prompt)
        cmd = [
            "opencode", "run", "--pure", "--format", "json",
            "--dir", str(work), "-m", model, prompt,
        ]
        try:
            # Full L3 task/design payloads can consume most of a 15-minute
            # window before the worker reaches its first write.  Keep the
            # provenance gate, but allow enough time to materialize and
            # syntax-check the candidate instead of discarding useful runs.
            proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True,
                                  text=True, timeout=1800, check=False)
            stdout, stderr, returncode, timed_out = proc.stdout, proc.stderr, proc.returncode, False
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = exc.stdout or "", exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            returncode, timed_out = 124, True
        (evidence / f"{phase}.attempt{attempt}.events.jsonl").write_text(stdout, encoding="utf-8")
        (evidence / f"{phase}.attempt{attempt}.stderr.txt").write_text(stderr, encoding="utf-8")
        parsed = parse_events(stdout)
        completed = {
            call["name"] for call in parsed["skill_calls"]
            if call["status"] == "completed"
            and call["directory"]
            and Path(call["directory"]).resolve() == (SKILLS / str(call["name"])).resolve()
        }
        marker = f"{phase.upper()}_DONE"
        accepted = (
            returncode == 0 and expected.is_file() and marker in parsed["text"]
            and required.issubset(completed)
            and (arm == "with_skills" or not parsed["skill_calls"])
        )
        record = {
            "phase": phase,
            "attempt": attempt,
            "model": model,
            "returncode": returncode,
            "timed_out": timed_out,
            "session_id": parsed["session_id"],
            "skill_calls": parsed["skill_calls"],
            "required_skills": sorted(required),
            "verified_loaded_skills": sorted(completed),
            "completion_marker_seen": marker in parsed["text"],
            "artifact_exists": expected.is_file(),
            "accepted": accepted,
        }
        (evidence / f"{phase}.attempt{attempt}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if accepted:
            return record
    raise RuntimeError(f"{arm}/{op_root.name}/{phase} failed provenance gate")


def generate_one(arm: str, op: str) -> dict:
    import sys
    sys.path.insert(0, str(WORKERS))
    import prompts

    op_root = ROOT / "generated" / arm / op
    published = ROOT / arm / "candidates" / f"{op}.py"
    provenance_path = op_root / "provenance.json"
    if published.is_file() and provenance_path.is_file():
        prior = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (prior.get("runtime_commit") == RUNTIME_COMMIT
                and prior.get("candidate_sha256") == sha256(published)
                and all(phase.get("accepted") for phase in prior.get("phases", []))):
            return {"arm": arm, "operator": op,
                    "candidate_sha256": prior["candidate_sha256"], "resumed": True}
    work = op_root / "work"
    work.mkdir(parents=True, exist_ok=True)
    prompt = prompts.build_generation_prompt(op, op, op, GUIDANCE[op])
    prompt = (
        f"# Runtime pin\n\nUse compiler-team/pyasc v2 commit `{RUNTIME_COMMIT}`. "
        "This snapshot exports `asctile`; importing `asc2` is invalid.\n\n" + prompt
    )
    (work / "task.md").write_text(prompt, encoding="utf-8")
    records = [run_phase(arm, op_root, phase) for phase in ("design", "implement", "review")]
    candidate = work / "candidate.py"
    if "import asc2" in candidate.read_text(encoding="utf-8"):
        raise RuntimeError(f"{arm}/{op} emitted legacy asc2 import")
    out = ROOT / arm / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, out / f"{op}.py")
    provenance = {
        "arm": arm,
        "operator": op,
        "runtime_commit": RUNTIME_COMMIT,
        "models_requested": MODELS,
        "phases": records,
        "task_sha256": sha256(work / "task.md"),
        "design_sha256": sha256(work / "design.md"),
        "candidate_sha256": sha256(candidate),
        "skills_root": str(SKILLS) if arm == "with_skills" else None,
    }
    (op_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"arm": arm, "operator": op, "candidate_sha256": provenance["candidate_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="no_skills,with_skills")
    parser.add_argument("--ops", default=",".join(OPS))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    arms = [v.strip() for v in args.arms.split(",") if v.strip()]
    ops = [v.strip() for v in args.ops.split(",") if v.strip()]
    unknown = (set(arms) - {"no_skills", "with_skills"}) | (set(ops) - set(OPS))
    if unknown:
        raise SystemExit(f"unsupported values: {sorted(unknown)}")
    work_items = [(arm, op) for arm in arms for op in ops]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate_one, arm, op): (arm, op) for arm, op in work_items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    (ROOT / "generation-summary.json").write_text(
        json.dumps(sorted(results, key=lambda x: (x["arm"], x["operator"])), indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
