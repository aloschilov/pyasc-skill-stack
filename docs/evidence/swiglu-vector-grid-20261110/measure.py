"""Profile a packaged SwiGLU candidate against the Season 2 task on an active NPU.

Run from a directory containing generation/task/{cases.csv,golden.py}, the
candidate package, and grid_template.py. Put the CANNBench src on PYTHONPATH.
"""

import csv
import importlib
import json
import math
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from asc.lib import runtime
from asc.lib.profiling import Profiler
from generation.task.golden import swi_glu as golden
from kernel_eval.data.data_generator import DataGenerator


def measure(candidate, case):
    with Path("generation/task/cases.csv").open() as stream:
        row, = (r for r in csv.DictReader(stream) if int(r["case_id"]) == case)
    shape, = json.loads(row["input_shape"])
    dtype, = json.loads(row["dtype"])
    dim = json.loads(row["attrs"])["dim"]
    op = importlib.import_module(candidate + ".swi_glu").swi_glu
    x = DataGenerator(seed=case).generate_input_tensor(shape, dtype, json.loads(row["value_range"]))
    launch = runtime.launch_kernel
    grids = []
    dispatch = []
    allocations = []

    def record_launch(fn, blocks, *args, **kwargs):
        value = launch(fn, blocks, *args, **kwargs)
        grids.append(blocks)
        return value

    def observe_warmup(frame, event, value):
        code = frame.f_code
        if event == "return" and code.co_filename == op.__code__.co_filename:
            integers = {k: v for k, v in frame.f_locals.items() if type(v) is int}
            if type(value) is int:
                integers["returned"] = value
            if integers:
                dispatch.append({"function": code.co_name, "integers": integers})
        if event == "call" and code.co_name == "check_memory_overflow":
            allocations.append(dict(frame.f_locals["memory_consumed"]))

    with patch.object(runtime, "launch_kernel", record_launch):
        previous = sys.getprofile()
        sys.setprofile(observe_warmup)
        try:
            output = op(x, dim)
        finally:
            sys.setprofile(previous)
        expected = golden(x, dim)
        assert output.shape == expected.shape and output.dtype == expected.dtype
        for actual, reference in zip(output.flatten().split(1048576), expected.flatten().split(1048576)):
            torch.testing.assert_close(actual, reference, equal_nan=True)
        del output, expected
        warmup_grid = grids.copy()
        profiler = Profiler()
        with profiler.profile():
            for _ in range(3):
                op(x, dim)

    tasks = [(t.name, t.type, t.duration) for t in profiler.last_result.tasks]
    kernels = [(n, t, d) for n, t, d in tasks if "swiglu" in n.lower().replace("_", "")
               or any(key in t.lower() for key in ("vector", "aiv", "aicore"))]
    count = len(warmup_grid)
    assert count and len(kernels) == 3 * count and grids == warmup_grid * 4, (grids, tasks)
    groups = [kernels[i:i + count] for i in range(0, len(kernels), count)]
    signatures = [[(n, t) for n, t, _ in group] for group in groups]
    assert signatures[0] == signatures[1] == signatures[2], signatures
    samples = [sum(duration for _, _, duration in group) for group in groups]
    assert all(math.isfinite(s) and s > 0 for s in samples)
    return dict(case=case, variant=candidate, shape=shape, dtype=dtype, dim=dim, correct=True,
                median_us=statistics.median(samples), samples_us=samples, launch_grid=grids,
                launches=count, dispatch=dispatch, allocations=allocations, tasks=tasks)


if __name__ == "__main__":
    name, number = sys.argv[1], int(sys.argv[2])
    result = measure(name, number)
    path = Path("results") / f"{number:02}-{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(number, result["correct"], result["median_us"], flush=True)
