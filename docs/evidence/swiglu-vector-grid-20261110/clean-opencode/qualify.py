"""Extract the untouched model response and qualify it with the existing David profiler."""
import ast
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(events, destination):
    records = [json.loads(line) for line in events.read_text().splitlines() if line]
    assert not any(e["type"] in {"error", "tool_use"} for e in records)
    response = "".join(e["part"]["text"] for e in records if e["type"] == "text")
    fenced = re.fullmatch(r"\s*```(?:python|py)?[ \t]*\r?\n(.*?)^```\s*", response, re.DOTALL | re.MULTILINE)
    source = fenced.group(1) if fenced else response
    tree = ast.parse(source)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "swi_glu" for n in tree.body)
    with destination.open("xb") as stream:
        stream.write(source.encode())
    print("candidate", digest(destination), "events", digest(events), flush=True)


def run_case(case, output, expected):
    from measure import measure
    assert digest(Path("gen/swi_glu.py")) == expected
    data = measure("gen", case)
    assert data["correct"] and len(data["samples_us"]) == 3
    assert data["median_us"] == statistics.median(data["samples_us"])
    data["candidate_sha256"] = expected
    output.write_text(json.dumps(data) + "\n")


def run_suite(destination, expected):
    destination.mkdir()
    assert digest(Path("gen/swi_glu.py")) == expected
    for case in range(1, 21):
        result = destination / f"{case:02}.json"
        with (destination / f"{case:02}.log").open("w") as log:
            try:
                proc = subprocess.run([sys.executable, __file__, "case", str(case), str(result), expected],
                                      stdout=log, stderr=subprocess.STDOUT, timeout=450)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 124
        if code:
            result.write_text(json.dumps(dict(case=case, correct=False, candidate_sha256=expected, returncode=code)) + "\n")
        record = json.loads(result.read_text())
        print(case, record["correct"], record.get("median_us"), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "extract":
        extract(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "case":
        run_case(int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    elif mode == "suite":
        run_suite(Path(sys.argv[2]), sys.argv[3])
    else:
        raise ValueError(mode)
