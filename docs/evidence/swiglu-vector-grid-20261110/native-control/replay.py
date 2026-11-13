"""Compare native ACLNN on David with the published short FP32 SwiGLU baselines.

Build swiglu.cpp as ./bench and run in a fresh directory with msprof and the
public evaluator available. Arguments are the Season 2 task directory and the
case number. The first five of seventeen task timings are warm-up calls.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np
import torch
from kernel_eval.data.data_generator import DataGenerator


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(task, case, direct_simt=False):
    with (task / "cases.csv").open() as f:
        row, = (r for r in csv.DictReader(f) if int(r["case_id"]) == case)
    shape, = json.loads(row["input_shape"])
    dtype, = json.loads(row["dtype"])
    dim = json.loads(row["attrs"])["dim"]
    bounds = json.loads(row["value_range"])
    spec = importlib.util.spec_from_file_location("native_reference", task / "golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    # A second dense finite oracle exercises every row on the all-NaN fixture.
    fixtures = [("public", bounds)]
    if any(isinstance(v, str) for v in bounds):
        fixtures.append(("finite", [-1, 1]))
    results = []
    for name, limits in fixtures:
        x = DataGenerator(seed=case).generate_input_tensor(shape, dtype, limits)
        y = golden.swi_glu(x, dim)
        root = Path(f"case{case}-{name}")
        root.mkdir()
        inp, out = root / "input.f32", root / "output.f32"
        x.numpy().tofile(inp)
        command = f"./bench {case} {inp} {out}"
        with (root / "msprof.log").open("w") as log:
            subprocess.run(["msprof", "--application=" + command, "--output=" + str(root)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        actual = torch.from_numpy(np.fromfile(out, dtype=np.float32)).reshape_as(y)
        torch.testing.assert_close(actual, y, equal_nan=True)
        summary, = root.glob("**/op_summary*.csv")
        with summary.open(encoding="utf-8-sig") as f:
            records = list(csv.DictReader(f))
        if len(records) != 17:
            raise ValueError(f"expected 17 task records in {summary}, found {len(records)}")
        def find_column(prefix):
            key, = (k for k in records[0] if k.strip().startswith(prefix))
            return key
        duration = find_column("Task Duration")
        if "(us)" not in duration.replace(" ", "").lower():
            raise ValueError(duration)
        grid, = (k for k in records[0] if k.strip() in {"Block Num", "Block Dim"})
        task_type = find_column("Task Type")
        for key in (grid, task_type, "Op Name"):
            if len({r[key] for r in records}) != 1:
                raise ValueError(f"task selection changed across calls: {key}")
        op_name = records[0]["Op Name"].lower()
        expected_symbol = {11: "short_rowsilj273elj509elj8ee", 13: "short_rowsilj143elj536elj4ee"}[case]
        valid_name = expected_symbol in op_name if direct_simt else op_name == "swiglu"
        if direct_simt and int(records[0][grid]) not in (0, {11: 35, 13: 36}[case]):
            raise ValueError(f"unexpected SIMT grid: {records[0][grid]}")
        if not valid_name or records[0][task_type] != "AI_VECTOR_CORE":
            raise ValueError(f"unexpected native task: {records[0]}")
        times = [float(r[duration]) for r in records]
        if not all(math.isfinite(t) and t > 0 for t in times):
            raise ValueError(times)
        results.append(dict(fixture=name, case=case, shape=shape, dim=dim, correct=True,
                            task=records[0]["Op Name"], task_type=records[0][task_type],
                            block_dim=records[0][grid], samples_us=times,
                            **({"requested_blocks": {11: 35, 13: 36}[case]} if direct_simt else {}),
                            median_us=statistics.median(times[5:]), cases_sha256=digest(task/"cases.csv"),
                            golden_sha256=digest(task/"golden.py"), summary_sha256=digest(summary)))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("case", type=int, choices=(11, 13))
    parser.add_argument("--direct-simt", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.task, args.case, args.direct_simt)))
