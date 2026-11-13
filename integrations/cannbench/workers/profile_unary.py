"""Profile one generated Level 1 operator against the supplied public task on a real NPU.

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
    class HostCalls(ast.NodeVisitor):
        def visit_FunctionDef(self, fn):
            decorators = [d.func if isinstance(d, ast.Call) else d for d in fn.decorator_list]
            if any(isinstance(d, ast.Attribute) and d.attr == "jit" and
                   isinstance(d.value, ast.Name) and d.value.id == "asctile" for d in decorators):
                return
            self.generic_visit(fn)

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and
                    node.func.attr in {"to", "float", "half", "bfloat16", "type", "type_as"}):
                raise ValueError(f"Exp host tensor conversion at line {node.lineno}")
            self.generic_visit(node)

    HostCalls().visit(ast.parse(source))


def measure(candidate, case, operator=None):
    import torch
    from asc.lib import runtime
    from asc.lib.profiling import Profiler
    from kernel_eval.data.data_generator import DataGenerator

    task = Path("generation/task")
    with (task / "cases.csv").open() as f:
        row, = (r for r in csv.DictReader(f) if int(r["case_id"]) == case)
    if operator is None:
        operator = {
            "Exp": "exp", "SwiGlu": "swi_glu", "Sigmoid": "sigmoid", "Mish": "mish",
            "Gelu": "gelu", "MaskedScale": "masked_scale", "ForeachNorm": "foreach_norm",
            "ForeachAddcdivScalar": "foreach_addcdiv_scalar",
        }[row["operator"]]
    if operator == "exp":
        check_exp_host_casts((Path(*candidate.split(".")) / "exp.py").read_text())
    op = getattr(importlib.import_module(f"{candidate}.{operator}"), operator)
    golden = getattr(importlib.import_module("generation.task.golden"), operator)
    shapes = json.loads(row["input_shape"])
    dtypes = json.loads(row["dtype"])
    attrs = json.loads(row["attrs"])
    value_range = json.loads(row["value_range"])
    ranges = value_range if isinstance(value_range[0], list) else [value_range] * len(shapes)
    if len(dtypes) == 1:
        dtypes *= len(shapes)
    if len(ranges) != len(shapes) or len(dtypes) != len(shapes):
        raise ValueError("case input shapes, dtypes and ranges disagree")

    def inputs(bounds):
        make = DataGenerator(seed=case).generate_input_tensor
        args = []
        for shape, dtype, limits in zip(shapes, dtypes, bounds):
            if shape and isinstance(shape[0], list):
                args.append([make(member, dtype, limits) for member in shape])
            else:
                args.append(make(shape, dtype, limits))
        return args

    def check(args):
        expected = golden(*args, **attrs)
        output = op(*args, **attrs)
        if isinstance(expected, (tuple, list)):
            if type(output) is not type(expected) or len(output) != len(expected):
                raise ValueError("candidate changed output sequence type or length")
        else:
            expected, output = [expected], [output]
        for tensor, reference in zip(output, expected):
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError("candidate changed output shape or dtype")
            for actual, ref in zip(tensor.flatten().split(1048576), reference.flatten().split(1048576)):
                torch.testing.assert_close(actual, ref, equal_nan=True)

    finite_control = any(
        len(bounds) != 2 or bounds[0] == bounds[1]
        or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bounds)
        for bounds in ranges
    )
    if finite_control:
        check(inputs([[0, 1] if dtype in ("uint8", "bool") else [-1, 1] for dtype in dtypes]))
    args = inputs(ranges)
    launch = runtime.launch_kernel
    grids = []

    def record(fn, blocks, *args, **kwargs):
        result = launch(fn, blocks, *args, **kwargs)
        grids.append(int(blocks))
        return result

    with patch.object(runtime, "launch_kernel", record):
        check(args)
        grid = grids.copy()
        if not grid:
            raise ValueError("no device launch")
        profiler = Profiler()
        with profiler.profile():
            for _ in range(3):
                op(*args, **attrs)

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
    return dict(case=case, operator=operator, shape=shapes[0] if len(shapes) == 1 else shapes,
                dtype=dtypes[0] if len(shapes) == 1 else dtypes, attrs=attrs,
                correct=True, finite_control=finite_control, launches=count,
                launch_grid=grids, tasks=tasks, samples_us=samples,
                median_us=statistics.median(samples), cases_sha256=digest(task / "cases.csv"),
                golden_sha256=digest(task / "golden.py"), candidate_sha256=digest(op.__code__.co_filename))


if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))
    package, op, case = sys.argv[1:]
    print(json.dumps(measure(package, int(case), op)))
