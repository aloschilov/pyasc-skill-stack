"""Bind full-catalog generated sources to native skill and compiler evidence."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_generation(catalog, run_root: Path, skills_root: Path):
    summary = json.loads((run_root / "summary.json").read_text())
    names = {op["function_name"] for op in catalog["operators"]}
    if (set(summary["requested_operators"]) != names or len(summary["requested_operators"]) != len(names)
            or set(summary["locally_qualified_operators"]) != names
            or summary.get("catalog_sha256") != catalog["catalog_sha256"]):
        raise ValueError("Generation does not qualify the complete official catalog")
    qualified = run_root / "locally_qualified"
    records = {}
    for spec in catalog["operators"]:
        name = spec["function_name"]
        candidate = qualified / "cann_bench" / (name + ".py")
        provenance_file = qualified / "provenance" / (name + ".json")
        provenance = json.loads(provenance_file.read_text())
        if (provenance.get("generator") != "opencode" or provenance.get("evaluation_mode") != "local"
                or provenance.get("skill_gate_passed") is not True
                or provenance.get("candidate_sha256") != sha256(candidate)
                or provenance.get("task_contract") != spec):
            raise ValueError("Candidate or task provenance mismatch: " + name)
        models = provenance["models"]
        if models.get("implement") == models.get("review") or not models.get("review"):
            raise ValueError("Missing independent model review: " + name)
        required = {"pyasc-cannbench-kernel", "pyasc-syntax-constraints", "pyasc-code-review", "pyasc-build-run-verify"}
        if not required <= set(provenance["loaded_skills"]):
            raise ValueError("Missing native skill provenance: " + name)
        for skill in required:
            root = skills_root / skill
            actual = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}
            if provenance["skill_sources"][skill]["files"] != actual:
                raise ValueError("Skill source changed since generation: " + skill)
        validation = provenance["validation"]
        if validation.get("passed") != spec["case_count"] or validation.get("total") != spec["case_count"]:
            raise ValueError("Incomplete compiler coverage: " + name)
        iteration = next((p.parent for p in (run_root / (name + "-generate")).glob("iter*/provenance.json")
                          if sha256(p) == sha256(provenance_file)), None)
        if iteration is None or sha256(iteration / "local_compile.json") != validation["report_sha256"]:
            raise ValueError("Missing exact compiler report: " + name)
        reviewed = provenance.get("reviewed_compile") or {}
        if reviewed.get("report_sha256") != sha256(iteration / "pre_review_compile.json"):
            raise ValueError("Missing measured compiler evidence before review: " + name)
        report = json.loads((iteration / "local_compile.json").read_text())
        if report.get("status") != "passed" or report.get("compile_passed") != spec["case_count"]:
            raise ValueError("Compiler report rejected: " + name)
        for phase in ("design", "implement", "review"):
            traces = [json.loads(p.read_text()) for p in iteration.glob(phase + ".skill-trace.attempt*.json")]
            if not any(t.get("skill_gate_passed") is True and t.get("session_id") == provenance["sessions"][phase]
                       and t.get("model") == models[phase] for t in traces):
                raise ValueError("Missing exact native phase trace: " + name + "/" + phase)
        records[name] = {"candidate_sha256": sha256(candidate), "provenance_sha256": sha256(provenance_file),
                         "compiler_report_sha256": validation["report_sha256"], "models": models}
    return records
