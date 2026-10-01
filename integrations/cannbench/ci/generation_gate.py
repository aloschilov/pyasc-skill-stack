"""Bind full-catalog generated sources to native skill and compiler evidence."""
import hashlib
import json
from pathlib import Path

REQUIRED_SKILLS = {
    "pyasc-cannbench-kernel",
    "pyasc-syntax-constraints",
    "pyasc-code-review",
    "pyasc-build-run-verify",
}
WORKFLOW_PHASES = ("design", "implement", "review")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_iteration_record(spec, iteration: Path, skills_root: Path,
                              *, allow_failed_compile: bool = False) -> dict:
    """Validate one recorded native generation iteration for one operator.

    Binds the candidate hash, task contract, all four required skill source
    manifests, distinct implement/review models, the exact native phase
    traces/sessions and the recorded compiler/pre-review report hashes.
    With ``allow_failed_compile`` a recorded FAILED compiler result is also
    acceptable for controlled replay; a missing native phase, changed
    source/task/skill or missing report is never acceptable.
    """
    candidate = iteration / "candidate.py"
    provenance_file = iteration / "provenance.json"
    if not candidate.is_file() or not provenance_file.is_file():
        raise ValueError("Missing recorded candidate or provenance: " + str(iteration))
    provenance = json.loads(provenance_file.read_text())
    if (provenance.get("generator") != "opencode"
            or provenance.get("evaluation_mode") != "local"
            or provenance.get("skill_gate_passed") is not True
            or provenance.get("candidate_sha256") != sha256(candidate)
            or provenance.get("task_contract") != spec):
        raise ValueError("Candidate or task provenance mismatch: " + iteration.name)
    models = provenance["models"]
    if models.get("implement") == models.get("review") or not models.get("review"):
        raise ValueError("Missing independent model review: " + iteration.name)
    if not REQUIRED_SKILLS <= set(provenance["loaded_skills"]):
        raise ValueError("Missing native skill provenance: " + iteration.name)
    for skill in REQUIRED_SKILLS:
        root = skills_root / skill
        actual = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}
        if provenance["skill_sources"][skill]["files"] != actual:
            raise ValueError("Skill source changed since generation: " + skill)
    validation = provenance.get("validation") or {}
    if not validation.get("report_sha256"):
        raise ValueError("Missing recorded compiler report binding: " + iteration.name)
    if (not (iteration / "local_compile.json").is_file()
            or sha256(iteration / "local_compile.json") != validation["report_sha256"]):
        raise ValueError("Missing exact compiler report: " + iteration.name)
    reviewed = provenance.get("reviewed_compile") or {}
    if (not reviewed.get("report_sha256")
            or not (iteration / "pre_review_compile.json").is_file()
            or sha256(iteration / "pre_review_compile.json") != reviewed["report_sha256"]):
        raise ValueError("Missing measured compiler evidence before review: " + iteration.name)
    report = json.loads((iteration / "local_compile.json").read_text())
    if not allow_failed_compile:
        if (validation.get("passed") != spec["case_count"]
                or validation.get("total") != spec["case_count"]):
            raise ValueError("Incomplete compiler coverage: " + iteration.name)
        if report.get("status") != "passed" or report.get("compile_passed") != spec["case_count"]:
            raise ValueError("Compiler report rejected: " + iteration.name)
    elif not isinstance(report, dict) or "status" not in report:
        raise ValueError("Unusable recorded compiler report: " + iteration.name)
    for phase in WORKFLOW_PHASES:
        traces = [json.loads(p.read_text()) for p in iteration.glob(phase + ".skill-trace.attempt*.json")]
        if not any(t.get("skill_gate_passed") is True and t.get("session_id") == provenance["sessions"][phase]
                   and t.get("model") == models[phase] for t in traces):
            raise ValueError("Missing exact native phase trace: " + phase)
    if provenance.get("reuse"):
        reuse = provenance["reuse"]
        original = validate_iteration_record(spec, iteration / "original-record", skills_root,
                                             allow_failed_compile=True)
        if (reuse.get("mode") != "requalified-replay"
                or reuse.get("original_provenance_sha256") != original["provenance_sha256"]
                or reuse.get("original_candidate_sha256") != original["candidate_sha256"]
                or reuse.get("original_compiler_report_sha256") != original["compiler_report_sha256"]
                or reuse.get("original_validation") != original["provenance"].get("validation")
                or original["candidate_sha256"] != sha256(candidate)
                or original["models"] != models
                or original["provenance"]["sessions"] != provenance["sessions"]):
            raise ValueError("Original native derivation mismatch: " + iteration.name)
    return {"iteration": iteration, "provenance": provenance,
            "candidate_sha256": sha256(candidate),
            "provenance_sha256": sha256(provenance_file),
            "compiler_report_sha256": validation["report_sha256"],
            "models": models}


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
        if provenance.get("candidate_sha256") != sha256(candidate):
            raise ValueError("Candidate or task provenance mismatch: " + name)
        iteration = next((p.parent for p in (run_root / (name + "-generate")).glob("iter*/provenance.json")
                          if sha256(p) == sha256(provenance_file)), None)
        if iteration is None:
            raise ValueError("Missing exact iteration evidence: " + name)
        record = validate_iteration_record(spec, iteration, skills_root)
        records[name] = {"candidate_sha256": record["candidate_sha256"],
                         "provenance_sha256": record["provenance_sha256"],
                         "compiler_report_sha256": record["compiler_report_sha256"],
                         "models": record["models"]}
    return records
