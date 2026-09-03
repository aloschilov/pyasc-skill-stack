#!/usr/bin/env python3
"""Finish a generated CANNBench candidate with measured, skill-gated repairs.

This intentionally never reads or overwrites the canonical submission module.
It starts from a model-created seed, records exact-v2 compile feedback, asks one
OpenCode worker to repair it using repository skills, and asks a different
model to review it before producing a locally-qualified artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import driver


def write_feedback(path: Path, digest: dict, static_problems: list[str]) -> None:
    feedback = driver.compact_local_feedback(digest)
    feedback["static_problems"] = static_problems
    path.write_text(
        "# Measured pinned-pyasc-v2 feedback\n\n"
        "This evidence checks all 20 CANNBench dispatch/compile routes. It does "
        "not establish numerical correctness or performance.\n\n```json\n"
        + json.dumps(feedback, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )


def evaluate(op: str, candidate: Path, root: Path, label: str) -> dict:
    target = root / label
    target.mkdir(parents=True, exist_ok=True)
    digest = driver.local_evaluate(op, candidate, target)
    (target / "digest.json").write_text(
        json.dumps(digest, indent=2) + "\n", encoding="utf-8"
    )
    return digest


def record_phase(records: list[dict], phase: str, cycle: int,
                 result: driver.WorkerResult, model: str, candidate: Path) -> None:
    records.append({
        "phase": phase,
        "cycle": cycle,
        "model": model,
        "session_id": result.session_id,
        "loaded_skills": list(result.loaded_skills),
        "observed_skill_dirs": dict(result.skill_dirs),
        "candidate_sha256": driver._sha256(candidate) if candidate.exists() else None,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=driver.ALL_OPS)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--seed-candidate", required=True, type=Path)
    parser.add_argument("--seed-design", type=Path)
    parser.add_argument("--seed-model", default="dashscope/qwen3.7-max")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--models", default="dashscope/qwen3.7-max,dashscope/glm-5.2"
    )
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--repair-cycles", type=int, default=3)
    args = parser.parse_args()

    models = tuple(v.strip() for v in args.models.split(",") if v.strip())
    if len(models) < 2:
        parser.error("at least two models are required for independent review")

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    scratch = driver.SCRATCH_ROOT / (
        f"staged_{time.strftime('%Y%m%d_%H%M%S')}_{args.op}"
    )
    scratch.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.task, scratch / "task.md")
    shutil.copy2(args.seed_candidate, scratch / "candidate.py")
    seed_hash = driver._sha256(scratch / "candidate.py")
    phase_records: list[dict] = []

    if args.seed_design:
        shutil.copy2(args.seed_design, scratch / "design.md")
    else:
        design_result = driver.run_phase(
            scratch, "design", root, models, 0, args.attempts
        )
        if design_result is None:
            print("fresh design did not pass the skill gate", file=sys.stderr)
            return 2
        result, model = design_result
        record_phase(phase_records, "design", 0, result, model,
                     scratch / "candidate.py")

    candidate = scratch / "candidate.py"
    digest = evaluate(args.op, candidate, root, "preflight")
    static_problems = driver.static_check(candidate, args.op)

    author_model: str | None = args.seed_model
    for cycle in range(1, args.repair_cycles + 1):
        if not digest.get("hard_failure") and not static_problems:
            break
        write_feedback(
            scratch / "compile_feedback.md", digest, static_problems
        )
        phase_dir = root / f"repair{cycle}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        phase_result = driver.run_phase(
            scratch, "repair", phase_dir, models, cycle - 1, args.attempts
        )
        if phase_result is None:
            print(f"repair cycle {cycle} did not pass the skill gate", file=sys.stderr)
            return 3
        result, author_model = phase_result
        record_phase(phase_records, "repair", cycle, result, author_model, candidate)
        shutil.copy2(candidate, phase_dir / "candidate.py")
        static_problems = driver.static_check(candidate, args.op)
        digest = evaluate(args.op, candidate, phase_dir, "evaluation")

    if digest.get("hard_failure") or static_problems:
        print("candidate did not pass exact-v2 pre-review qualification", file=sys.stderr)
        return 4

    # A passing seed still needs a native skill invocation in this finishing
    # run. Treat its source model as model[0], then force model[1] to review.
    author_model = author_model or args.seed_model
    reviewer_start = (models.index(author_model) + 1) % len(models)
    review_dir = root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_result = driver.run_phase(
        scratch, "review", review_dir, models, reviewer_start, 1
    )
    if review_result is None:
        print("independent review did not pass the skill gate", file=sys.stderr)
        return 5
    result, reviewer_model = review_result
    if reviewer_model == author_model:
        print("review was not independent", file=sys.stderr)
        return 6
    record_phase(phase_records, "review", 1, result, reviewer_model, candidate)
    shutil.copy2(candidate, review_dir / "candidate.py")
    static_problems = driver.static_check(candidate, args.op)
    digest = evaluate(args.op, candidate, review_dir, "evaluation")
    if digest.get("hard_failure") or static_problems:
        write_feedback(scratch / "compile_feedback.md", digest, static_problems)
        print("reviewed candidate failed exact-v2 qualification", file=sys.stderr)
        return 7

    qualified = root / "qualified"
    qualified.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, qualified / f"{args.op}.py")
    shutil.copy2(scratch / "design.md", qualified / "design.md")
    shutil.copy2(review_dir / "evaluation" / "local_compile.json",
                 qualified / "local_compile.json")
    used_skills = sorted({
        skill for phase in phase_records
        for skill in phase["loaded_skills"]
    })
    provenance = {
        "schema_version": 1,
        "operator": args.op,
        "generator": "opencode",
        "opencode_version": subprocess.run(
            ["opencode", "--version"], capture_output=True, text=True,
            check=False,
        ).stdout.strip(),
        "seed": {
            "path": str(args.seed_candidate.resolve()),
            "sha256": seed_hash,
            "model": args.seed_model,
        },
        "phases": phase_records,
        "required_skills": used_skills,
        "skill_sources": {
            name: {
                "path": str(driver.SKILLS_ROOT / name),
                "files": driver._skill_source_manifest(name),
            }
            for name in used_skills
        },
        "candidate_sha256": driver._sha256(qualified / f"{args.op}.py"),
        "validation": {
            "label": "verified-local-compile",
            "pyasc_commit": "ac1222a48c8914d3f81297c7570d1a84f0f26778",
            "passed": digest.get("passed"),
            "total": digest.get("total"),
            "report_sha256": driver._sha256(qualified / "local_compile.json"),
            "limitations": digest.get("limitations", []),
        },
    }
    (qualified / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "locally-qualified",
        "operator": args.op,
        "passed": digest.get("passed"),
        "total": digest.get("total"),
        "author_model": author_model,
        "reviewer_model": reviewer_model,
        "artifact": str(qualified / f"{args.op}.py"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
