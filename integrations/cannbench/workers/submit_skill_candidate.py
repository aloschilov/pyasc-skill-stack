#!/usr/bin/env python3
"""Submit a provenance-gated worker candidate and accept only a clean 20/20."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from driver import (
    EVIDENCE_DIR,
    REQUIRED_SKILLS,
    SUBMISSION_PKG,
    deployed_modules,
    render_init,
    static_check,
)
from evalqueue import EvalQueue


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(iter_dir: Path) -> tuple[Path, Path, dict]:
    candidate = iter_dir / "candidate.py"
    provenance_path = iter_dir / "provenance.json"
    validation_path = iter_dir / "local_validation.json"
    for path in (candidate, provenance_path, validation_path):
        if not path.is_file():
            raise RuntimeError(f"required artifact is missing: {path}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    actual_hash = sha256(candidate)
    if provenance.get("candidate_sha256") != actual_hash:
        raise RuntimeError("candidate hash does not match provenance.json")
    if validation.get("candidate_sha256") != actual_hash:
        raise RuntimeError("candidate hash does not match local_validation.json")
    if not provenance.get("skill_gate_passed"):
        raise RuntimeError("skill provenance gate did not pass")
    if not set(REQUIRED_SKILLS).issubset(
        provenance.get("loaded_skills") or []
    ):
        raise RuntimeError("provenance does not contain every required skill")
    lowering = validation.get("llvm_lowering") or {}
    if lowering.get("passed_cases") != lowering.get("total_cases"):
        raise RuntimeError("not every case dispatch passed local lowering")
    problems = static_check(candidate, "rms_norm")
    if problems:
        raise RuntimeError(f"static candidate gate failed: {problems}")
    return candidate, provenance_path, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("iter_dir", type=Path)
    args = parser.parse_args()
    iter_dir = args.iter_dir.resolve()
    candidate, provenance_path, provenance = validate_run(iter_dir)

    eval_dir = iter_dir / "official_eval"
    eval_dir.mkdir(exist_ok=True)
    candidate_hash = provenance["candidate_sha256"]
    tag = f"glm52-pyasc-skills-rmsnorm-{candidate_hash[:12]}"
    modules = deployed_modules()
    init_source = render_init(modules)
    digest = EvalQueue().evaluate(
        "rms_norm",
        candidate,
        init_source,
        eval_dir,
        tag=tag,
        provenance_path=provenance_path,
    )
    (eval_dir / "digest.json").write_text(
        json.dumps(digest, indent=2), encoding="utf-8"
    )

    accepted = (
        not digest.get("hard_failure")
        and digest.get("passed") == digest.get("total") == 20
        and digest.get("anti_cheat_failed") == 0
    )
    if not accepted:
        print(json.dumps({"accepted": False, "digest": digest}, indent=2))
        return 2

    shutil.copy2(candidate, SUBMISSION_PKG / "rms_norm.py")
    report = eval_dir / "report.json"
    if report.exists():
        shutil.copy2(report, EVIDENCE_DIR / "rms_norm_skill_generated_eval.json")
        shutil.copy2(report, EVIDENCE_DIR / "rms_norm_final_eval.json")
    official = {
        "accepted": True,
        "candidate_sha256": candidate_hash,
        "model": provenance["model"],
        "skills": provenance["loaded_skills"],
        "tag": tag,
        "digest": digest,
    }
    (iter_dir / "official_evaluation.json").write_text(
        json.dumps(official, indent=2), encoding="utf-8"
    )
    print(json.dumps(official, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
