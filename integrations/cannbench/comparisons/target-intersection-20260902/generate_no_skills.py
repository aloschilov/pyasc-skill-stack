#!/usr/bin/env python3
"""Generate the no-skills arm of the pyasc target/CANNBench comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
SOURCE_RUN = REPO_ROOT / "integrations/cannbench/workers/runs/20260902_140812"
MODELS = {
    "design": "dashscope/qwen3.7-max",
    "implement": "dashscope/qwen3.7-max",
    "review": "dashscope/glm-5.2",
}
PROMPTS = {
    "design": """Read task.md. Do not invoke the skill tool and do not read any SKILL.md. Design a complete pyasc v2 CANNBench solution covering all 20 cases. Write design.md and end your response with DESIGN_DONE. Do not write candidate.py yet. Use at most four tool calls.""",
    "implement": """Read task.md and design.md. Do not invoke the skill tool and do not read any SKILL.md. Implement the complete solution as candidate.py, run only python3 -m py_compile candidate.py, and end your response with IMPLEMENT_DONE. Create no other files. Use at most five tool calls.""",
    "review": """Read task.md and candidate.py. Do not invoke the skill tool and do not read any SKILL.md. Independently review all 20 cases, the public contract, numerical stability, tails, dtype handling, UB use, and anti-cheat rules. Fix candidate.py in place, run only python3 -m py_compile candidate.py, and end your response with REVIEW_DONE. Create no other files. Use at most five tool calls.""",
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
        if event.get("type") == "text" and isinstance(part.get("text"), str):
            text.append(part["text"])
        if event.get("type") == "tool_use" and part.get("tool") == "skill":
            state = part.get("state") or {}
            skill_calls.append({
                "status": state.get("status"),
                "input": state.get("input"),
            })
    return {
        "session_id": session_id,
        "text": "\n".join(text),
        "skill_calls": skill_calls,
    }


def run_phase(work: Path, phase: str, evidence: Path) -> dict:
    config = {
        "skills": {"paths": []},
        "mcp": {"cann-bench-site": {"enabled": False}},
    }
    env = {
        **os.environ,
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
    }
    command = [
        "opencode", "run", "--pure", "--format", "json",
        "--dir", str(work), "-m", MODELS[phase], PROMPTS[phase],
    ]
    try:
        completed = subprocess.run(
            command, cwd=work, env=env, capture_output=True, text=True,
            timeout=600, check=False)
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        returncode = 124
        timed_out = True
    (evidence / f"{phase}.events.jsonl").write_text(
        stdout, encoding="utf-8")
    (evidence / f"{phase}.stderr.txt").write_text(
        stderr, encoding="utf-8")
    parsed = parse_events(stdout)
    marker = f"{phase.upper()}_DONE"
    record = {
        "phase": phase,
        "model": MODELS[phase],
        "returncode": returncode,
        "timed_out": timed_out,
        "session_id": parsed["session_id"],
        "skill_calls": parsed["skill_calls"],
        "skill_call_count": len(parsed["skill_calls"]),
        "completion_marker": marker,
        "completion_marker_seen": marker in parsed["text"],
    }
    (evidence / f"{phase}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"{phase} worker failed with rc={returncode}")
    if parsed["skill_calls"]:
        raise RuntimeError(f"{phase} invoked skill tool in no-skills arm")
    if not record["completion_marker_seen"]:
        raise RuntimeError(f"{phase} did not emit {marker}")
    return record


def source_prompt(op: str) -> Path:
    directory = "gelu-generate" if op == "gelu" else "foreach_addcdiv_scalar-generate"
    return SOURCE_RUN / directory / "prompt.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ops", default="gelu,foreach_addcdiv_scalar",
        help="comma-separated comparison operators")
    parser.add_argument(
        "--accept-incomplete-review", action="store_true",
        help="retain a valid implementation when an independent review emitted no marker")
    args = parser.parse_args()
    ops = [value.strip() for value in args.ops.split(",") if value.strip()]
    output = ROOT / "no_skills"
    for op in ops:
        op_root = output / op
        work = op_root / "work"
        evidence = op_root / "evidence"
        candidates = output / "candidates"
        work.mkdir(parents=True, exist_ok=True)
        evidence.mkdir(parents=True, exist_ok=True)
        candidates.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_prompt(op), work / "task.md")
        phase_records = []
        for phase in ("design", "implement", "review"):
            required = work / ("design.md" if phase == "design" else "candidate.py")
            record_path = evidence / f"{phase}.json"
            if record_path.is_file() and required.is_file():
                previous = json.loads(record_path.read_text(encoding="utf-8"))
                reusable = (
                    previous.get("returncode") == 0
                    and previous.get("skill_call_count") == 0
                    and previous.get("completion_marker_seen") is True
                )
                if reusable:
                    phase_records.append(previous)
                    continue
                if (
                    phase == "review"
                    and args.accept_incomplete_review
                    and previous.get("returncode") == 0
                    and previous.get("skill_call_count") == 0
                ):
                    previous["accepted"] = False
                    previous["reason"] = "review returned without completion marker or edits"
                    phase_records.append(previous)
                    continue
            phase_records.append(run_phase(work, phase, evidence))
            if not required.is_file():
                raise RuntimeError(f"{phase} did not create {required.name}")
        candidate = work / "candidate.py"
        shutil.copy2(candidate, candidates / f"{op}.py")
        provenance = {
            "variant": "opencode-no-skills",
            "operator": op,
            "models": MODELS,
            "phases": phase_records,
            "skill_call_count": sum(v["skill_call_count"] for v in phase_records),
            "task_sha256": sha256(work / "task.md"),
            "design_sha256": sha256(work / "design.md"),
            "candidate_sha256": sha256(candidate),
            "runtime_commit": "ac1222a48c8914d3f81297c7570d1a84f0f26778",
        }
        (op_root / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "operator": op,
            "candidate": str(candidates / f"{op}.py"),
            "candidate_sha256": provenance["candidate_sha256"],
            "skill_call_count": provenance["skill_call_count"],
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
