#!/usr/bin/env python3
"""Validate the integrity and complete denominator of a catalog snapshot."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(root: Path) -> dict:
    root = root.resolve()
    document = json.loads((root / "manifest.json").read_text())
    operators = document["operators"]
    names = [o["operator"] for o in operators]
    functions = [o["function_name"] for o in operators]
    ids = [case for op in operators for case in op["case_ids"]]
    if len(names) != len(set(names)) or len(functions) != len(set(functions)) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate operator/function/case in catalog")
    if len(operators) != document["required_operators"] or len(ids) != document["required_cases"]:
        raise ValueError("Catalog denominator mismatch")
    raw = root / "catalog.raw.json"
    if hashlib.sha256(raw.read_bytes()).hexdigest() != document["catalog_sha256"]:
        raise ValueError("Raw catalog hash mismatch")
    official = json.loads(raw.read_text())
    if (official["operator_count"] != document["required_operators"]
            or official["total_case_count"] != document["required_cases"]
            or official["submission_version"] != document["benchmark_version"]
            or official["slug"] != document["benchmark_slug"]):
        raise ValueError("Manifest scope/version disagrees with official catalog")
    identities = lambda entries: {(o["operator"], o["function_name"], o["rel_path"], o["case_count"]) for o in entries}
    if identities(operators) != identities(official["multi_operators"]):
        raise ValueError("Manifest omits or changes official operators")
    for op in operators:
        if len(op["case_ids"]) != op["case_count"]:
            raise ValueError("Operator case count mismatch")
        directory = "tasks/" + op["function_name"] + "/"
        required = {directory + x for x in ("proto.yaml", "cases.yaml", "golden.py", "desc.md", "task.raw.json")}
        if set(op["task_files"]) != required:
            raise ValueError("Missing or unexpected task contract files")
        for name, digest in op["task_files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Task file escapes snapshot directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Task contract hash mismatch: " + name)
        cases = json.loads((root / directory / "cases.yaml").read_text())["cases"]
        task = json.loads((root / directory / "task.raw.json").read_text())
        if (task["operator"] != op["operator"] or task["function_name"] != op["function_name"]
                or task["benchmark_slug"] != document["benchmark_slug"]
                or task["cases_yaml"] != cases
                or task["proto"] != json.loads((root / directory / "proto.yaml").read_text())["operator"]
                or task["golden_source"] != (root / directory / "golden.py").read_text()
                or task["desc_md"] != (root / directory / "desc.md").read_text()):
            raise ValueError("Task contract disagrees with raw official task response")
        expected_ids = [op["rel_path"] + "_" + str(c["case_id"]) for c in cases]
        if expected_ids != op["case_ids"]:
            raise ValueError("Task case identities disagree with catalog manifest")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    data = validate(parser.parse_args().snapshot)
    print(f"Verified {data['required_operators']} operators / {data['required_cases']} cases")
