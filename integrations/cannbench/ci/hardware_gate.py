#!/usr/bin/env python3
"""Gate every official CANNBench case using exact hardware job results."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def geometric_mean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values)) if values else None


def evaluate(catalog: dict, payloads: list[dict], hardware="950pr") -> dict:
    errors = []
    specs = catalog["operators"]
    names = [s["operator"] for s in specs]
    ids = [c for s in specs for c in s["case_ids"]]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        errors.append("Duplicate operator or case identity in catalog")
    if catalog.get("required_operators", len(names)) != len(names) or catalog.get("required_cases", len(ids)) != len(ids):
        errors.append("Catalog denominator mismatch")
    for spec in specs:
        if len(spec["case_ids"]) != spec["case_count"]:
            errors.append("Catalog case count mismatch: " + spec["operator"])
    by_name = {s["operator"]: s for s in specs}
    candidates = defaultdict(list)
    case_rows = defaultdict(list)
    invalid_operators = set()
    observed_jobs = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            errors.append("Hardware job payload must be an object")
            continue
        job = payload.get("job", payload)
        if not isinstance(job, dict):
            errors.append("Hardware job must be an object")
            continue
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id.startswith("job_"):
            errors.append("Missing hardware job identity")
            continue
        if job_id in observed_jobs:
            errors.append("Duplicate job: " + job_id)
        observed_jobs.add(job_id)
        benchmark = job.get("benchmark") or {}
        if not isinstance(benchmark, dict):
            errors.append("Malformed benchmark identity: " + job_id)
            continue
        slug = job.get("benchmark_slug", benchmark.get("slug"))
        version = job.get("benchmark_version", benchmark.get("version"))
        if slug != catalog["benchmark_slug"] or version != catalog["benchmark_version"]:
            errors.append("Wrong benchmark/version: " + job_id)
            continue
        if job.get("target_hardware") != hardware or job.get("runner_hardware") != hardware:
            errors.append("Wrong or missing hardware: " + job_id)
            continue
        job_errors_before = len(errors)
        if job.get("status") != "succeeded":
            errors.append("Hardware job is not succeeded: " + job_id)
        result = job.get("results") or {}
        if not isinstance(result, dict):
            errors.append("Malformed hardware results: " + job_id)
            continue
        if result.get("result_stage") != "final" or result.get("contains_performance") is not True:
            errors.append("Missing final hardware performance result: " + job_id)
            continue
        summary = result.get("summary") or {}
        if not isinstance(summary, dict):
            errors.append("Malformed hardware summary: " + job_id)
            continue
        anti_cheat = [container["anti_cheat_failed_cases"] for container in (result, summary)
                      if "anti_cheat_failed_cases" in container]
        if not anti_cheat or any(type(v) is not int or v != 0 for v in anti_cheat):
            errors.append("Missing or failing anti-cheat evidence: " + job_id)
        operators = result.get("operators") or []
        if not isinstance(operators, list) or any(not isinstance(op, dict) for op in operators):
            errors.append("Malformed operator results: " + job_id)
            continue
        reported_cases = sum(op.get("total_cases", 0) for op in operators if type(op.get("total_cases")) is int)
        reported_passed = sum(op.get("passed_cases", 0) for op in operators if type(op.get("passed_cases")) is int)
        for key, count in (("total_cases", reported_cases), ("passed_cases", reported_passed)):
            if key in summary and (type(summary[key]) is not int or summary[key] != count):
                errors.append("Summary counts disagree with operator results: " + job_id)
        job_invalid = len(errors) != job_errors_before
        for operator in operators:
            operator_errors_before = len(errors)
            name = operator.get("operator")
            if not isinstance(name, str):
                errors.append("Malformed operator identity: " + job_id)
                continue
            if name not in by_name:
                errors.append("Unexpected operator: " + str(name))
                continue
            candidates[name].append((job_id, operator))
            spec = by_name[name]
            cases = operator.get("cases") or []
            if not isinstance(cases, list) or any(not isinstance(c, dict) or not isinstance(c.get("case_id"), str) for c in cases):
                errors.append("Malformed case results: " + name)
                invalid_operators.add(name)
                continue
            expected = set(spec["case_ids"])
            if operator.get("rel_path") != spec["rel_path"]:
                errors.append("Wrong task path: " + name)
            actual_ids = [c.get("case_id") for c in cases]
            if set(actual_ids) != expected or len(actual_ids) != len(expected):
                errors.append("Incomplete, duplicate or unexpected case set: " + name)
            success_count = sum(c.get("status") == "success" for c in cases)
            if (type(operator.get("total_cases")) is not int or type(operator.get("passed_cases")) is not int
                    or operator.get("total_cases") != len(cases) or operator.get("passed_cases") != success_count):
                errors.append("Reported counts disagree with exact cases: " + name)
            if type(operator.get("anti_cheat_failed_cases", 0)) is not int or operator.get("anti_cheat_failed_cases", 0) != 0:
                errors.append("Operator anti-cheat failure: " + name)
            if job_invalid or len(errors) != operator_errors_before:
                invalid_operators.add(name)
            for case in cases:
                case_id = case.get("case_id")
                if case_id not in expected:
                    continue
                if case.get("anti_cheat_passed", True) is not True or case.get("anti_cheat_failed", False) is not False:
                    errors.append("Case anti-cheat failure: " + case_id)
                    invalid_operators.add(name)
                case_rows[case_id].append((job_id, case))
    output = []
    all_ratios = []
    for spec in specs:
        name = spec["operator"]
        duplicates = len(candidates[name]) > 1
        if duplicates:
            errors.append("Duplicate operator measurements; no cherry-picking: " + name)
        rows = []
        ratios = []
        for case_id in spec["case_ids"]:
            records = case_rows[case_id]
            row = {"case_id": case_id, "status": "missing", "elapsed_us": None,
                   "baseline_perf_us": None, "speedup": None, "job_id": None, "error": None}
            if duplicates or len(records) > 1:
                row.update(status="invalid", error="Ambiguous duplicate measurement")
            elif len(records) == 1:
                job_id, case = records[0]
                row.update(status=case.get("status", "invalid"), job_id=job_id,
                           elapsed_us=case.get("elapsed_us"), baseline_perf_us=case.get("baseline_perf_us"),
                           error=case.get("error_msg"))
                if row["status"] == "success":
                    if positive_number(row["elapsed_us"]) and positive_number(row["baseline_perf_us"]):
                        ratio = row["baseline_perf_us"] / row["elapsed_us"]
                        if positive_number(ratio):
                            row["speedup"] = ratio
                            ratios.append(ratio)
                        else:
                            row["error"] = "Invalid timing ratio"
                    else:
                        row["error"] = "Missing or invalid positive finite hardware timings"
                    if row["speedup"] is None:
                        errors.append("Invalid hardware timing: " + case_id)
                else:
                    errors.append("Correctness failure: " + case_id)
            else:
                row["error"] = "No official hardware result"
            # Nonfinite raw values are retained as strings so published JSON
            # remains valid while the gate fails.
            for key in ("elapsed_us", "baseline_perf_us"):
                if isinstance(row[key], float) and not math.isfinite(row[key]):
                    row[key] = repr(row[key])
            rows.append(row)
        correct = sum(r["status"] == "success" for r in rows)
        measured = len(ratios)
        if not candidates[name]:
            errors.append("Missing operator hardware evidence: " + name)
        elif correct != spec["case_count"] or measured != spec["case_count"]:
            errors.append("Incomplete correctness/performance coverage: " + name)
        status = "missing" if not candidates[name] else "passed" if correct == measured == spec["case_count"] and not duplicates and name not in invalid_operators else "failed"
        output.append({"operator": name, "function_name": spec["function_name"], "required_cases": spec["case_count"],
                       "correct_cases": correct, "measured_cases": measured, "status": status,
                       "geometric_mean_speedup": geometric_mean(ratios) if measured == spec["case_count"] else None,
                       "cases": rows})
        all_ratios.extend(ratios)
    required = len(ids)
    correct = sum(op["correct_cases"] for op in output)
    measured = sum(op["measured_cases"] for op in output)
    return {"schema_version": 1, "source": "cannbench-hardware", "generated_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_slug": catalog["benchmark_slug"], "benchmark_version": catalog["benchmark_version"],
            "hardware": hardware, "required_operators": len(specs), "required_cases": required,
            "correct_cases": correct, "measured_cases": measured, "gate_passed": not errors and correct == measured == required and required > 0,
            "true_geometric_mean_speedup": geometric_mean(all_ratios) if measured == required and required > 0 else None,
            "errors": errors, "operators": output,
            "generation_provenance": "Hardware results alone do not establish skill-generated acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", default="950pr")
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    payloads = []
    malformed = []
    sources = {}
    for path in sorted(args.jobs_dir.glob("*.json")):
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            payloads.append(json.loads(path.read_text()))
        except (ValueError, TypeError):
            malformed.append("Malformed hardware job JSON: " + path.name)
    document = evaluate(catalog, payloads, args.hardware)
    document["errors"].extend(malformed)
    document["gate_passed"] = document["gate_passed"] and not malformed
    document["job_file_sha256"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Hardware correctness {document['correct_cases']}/{document['required_cases']}; measured {document['measured_cases']}/{document['required_cases']}; gate={document['gate_passed']}")
    return 0 if document["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
