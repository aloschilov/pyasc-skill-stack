"""Profile one generated unary operator against the supplied public task on a real NPU.

Call in a fresh process for each case. Expected cwd: generation/task/{cases.csv,
golden.py} and a Python package containing the candidate and runtime helper.
Raw tasks and launch grids are returned along with the median of three calls.
"""
import ast
import csv
import hashlib
import importlib
import json
import math
import statistics
import sys
from pathlib import Path
from unittest.mock import patch


def check_exp_host_casts(source):
    """Reject tensor conversion methods outside the public JIT in an Exp candidate.

    This scoped source check supplements output checks; it is not a sandbox.
    Exp has no index tensors or other host metadata requiring tensor conversion.
    """
    tree = ast.parse(source)
    for fn in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        decorators = [d.func if isinstance(d, ast.Call) else d for d in fn.decorator_list]
        if any(isinstance(d, ast.Attribute) and d.attr == "jit" and
               isinstance(d.value, ast.Name) and d.value.id == "asctile" for d in decorators):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"to", "float", "half", "bfloat16", "type", "type_as"}):
                raise ValueError(f"Exp host tensor conversion in {fn.name} at line {node.lineno}")


def measure(candidate, case, operator=None):
    import torch
    from asc.lib import runtime
    from asc.lib.profiling import Profiler
    from kernel_eval.data.data_generator import DataGenerator

    task = Path("generation/task")
    with (task / "cases.csv").open() as f:
        row, = (r for r in csv.DictReader(f) if int(r["case_id"]) == case)
    if operator is None:
        operator = {"Exp": "exp", "SwiGlu": "swi_glu"}[row["operator"]]
    if operator == "exp":
        check_exp_host_casts((Path(*candidate.split(".")) / "exp.py").read_text())
    op = getattr(importlib.import_module(f"{candidate}.{operator}"), operator)
    golden = getattr(importlib.import_module("generation.task.golden"), operator)
    shape, = json.loads(row["input_shape"])
    dtype, = json.loads(row["dtype"])
    attrs = json.loads(row["attrs"])
    value_range = json.loads(row["value_range"])
    make = DataGenerator(seed=case).generate_input_tensor

    def check(x):
        output = op(x, **attrs)
        expected = golden(x, **attrs)
        if output.shape != expected.shape or output.dtype != expected.dtype:
            raise ValueError("candidate changed shape or dtype")
        for actual, ref in zip(output.flatten().split(1048576), expected.flatten().split(1048576)):
            torch.testing.assert_close(actual, ref, equal_nan=True)

    def ordinary(bounds):
        return (len(bounds) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x)
                                         for x in bounds) and bounds[0] != bounds[1])

    if not ordinary(value_range):
        # At identical geometry, also reject implementations that only fit a
        # constant, infinity or NaN fixture. Keep this call outside the profiler.
        check(make(shape, dtype, [-1, 1]))
        make = DataGenerator(seed=case).generate_input_tensor

    x = make(shape, dtype, value_range)
    launch = runtime.launch_kernel
    grids = []

    def record(fn, blocks, *args, **kwargs):
        result = launch(fn, blocks, *args, **kwargs)
        grids.append(int(blocks))
        return result

    with patch.object(runtime, "launch_kernel", record):
        check(x)
        grid = grids.copy()
        if not grid:
            raise ValueError("no device launch")
        profiler = Profiler()
        with profiler.profile():
            for _ in range(3):
                op(x, **attrs)

    tasks = [(t.name, t.type, t.duration) for t in profiler.last_result.tasks]
    kernels = [(n, t, d) for n, t, d in tasks if t.upper() == "AI_VECTOR_CORE"]
    count = len(grid)
    if len(kernels) != 3 * count or grids != grid * 4:
        raise ValueError(f"device tasks and launches disagree: {tasks}, {grids}")
    groups = [kernels[i:i+count] for i in range(0, len(kernels), count)]
    if len({tuple((n,t) for n,t,_ in group) for group in groups}) != 1:
        raise ValueError(f"kernel selection changed across calls: {groups}")
    samples = [sum(d for _, _, d in group) for group in groups]
    if not all(math.isfinite(s) and s > 0 for s in samples):
        raise ValueError(f"invalid device timings: {samples}")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return dict(case=case, operator=operator, shape=shape, dtype=dtype, attrs=attrs,
                correct=True, finite_control=not ordinary(value_range), launches=count,
                launch_grid=grids, tasks=tasks, samples_us=samples,
                median_us=statistics.median(samples), cases_sha256=digest(task / "cases.csv"),
                golden_sha256=digest(task / "golden.py"), candidate_sha256=digest(op.__code__.co_filename))


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))
    package, op, case = sys.argv[1:]
    print(json.dumps(measure(package, int(case), op)))
