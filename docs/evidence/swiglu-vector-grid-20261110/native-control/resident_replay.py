"""Time the same generated launch with staged or reused device allocations.

Run from a candidate directory containing generation/task and the generated
package. Each process profiles one single-launch case. The resident control
replays the public launch while its original argument handles are still live.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
from unittest.mock import patch


def run(package, case, mode):
    import torch
    from asc.lib import runtime
    from asc.lib.profiling import Profiler
    from kernel_eval.data.data_generator import DataGenerator

    task = Path('generation/task')
    with (task / 'cases.csv').open() as f:
        row, = (r for r in csv.DictReader(f) if int(r['case_id']) == case)
    shape, = json.loads(row['input_shape'])
    dtype, = json.loads(row['dtype'])
    attrs = json.loads(row['attrs'])
    bounds = json.loads(row['value_range'])
    candidate = importlib.import_module(package + '.swi_glu').swi_glu
    golden = importlib.import_module('generation.task.golden').swi_glu
    make = DataGenerator(seed=case).generate_input_tensor
    if any(isinstance(v, str) or not math.isfinite(v) for v in bounds):
        finite = make(shape, dtype, [-1, 1])
        torch.testing.assert_close(candidate(finite, **attrs), golden(finite, **attrs))
    x = make(shape, dtype, bounds)
    ref = golden(x, **attrs)
    torch.testing.assert_close(candidate(x, **attrs), ref, equal_nan=True)
    profiler = Profiler()
    grids = []
    launch = runtime.launch_kernel

    def replay(fn, grid, args, **kw):
        for _ in range(9 if mode == 'resident' else 1):
            grids.append(int(grid))
            launch(fn, grid, args, **kw)
            if mode == 'resident':
                runtime.synchronize()

    with patch.object(runtime, 'launch_kernel', replay), profiler.profile():
        for _ in range(1 if mode == 'resident' else 9):
            result = candidate(x, **attrs)
    torch.testing.assert_close(result, ref, equal_nan=True)
    tasks = [(t.name, t.type, t.duration) for t in profiler.last_result.tasks
             if t.type.upper() == 'AI_VECTOR_CORE']
    if len(grids) != 9 or len(set(grids)) != 1 or len(tasks) != 9:
        raise ValueError((grids, tasks))
    if len({t[:2] for t in tasks}) != 1 or any(not math.isfinite(t[2]) or t[2] <= 0 for t in tasks):
        raise ValueError(tasks)
    paths = (Path(candidate.__code__.co_filename), task / 'cases.csv', task / 'golden.py')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(case=case, mode=mode, grid=grids[0], tasks=tasks, correct=True,
                median_us=statistics.median(t[2] for t in tasks[4:]), sha256=hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package')
    parser.add_argument('case', type=int, choices=(11, 13))
    parser.add_argument('mode', choices=('staged', 'resident'))
    opts = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    print(json.dumps(run(opts.package, opts.case, opts.mode)))
