"""Check nonfinite bases against the release golden on a resident finite tensor."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import torch
import torch_npu  # Registers the NPU device.


def check(package, task):
    sys.path[:0] = [str(package.parent), str(task)]
    candidate = importlib.import_module(f"{package.name}.exp").exp
    golden = importlib.import_module("golden").exp
    checks = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        values = torch.tensor([-1., 0., 1.], dtype=dtype)
        resident = values.to("npu:0")
        for label, base in (("finite", 2.), ("positive_inf", math.inf),
                            ("nan", math.nan), ("negative_inf", -math.inf),
                            ("positive_inf_again", math.inf)):
            expected = golden(values, base=base)
            actual = candidate(resident, base=base)
            assert actual.device == resident.device
            torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
            checks.append([str(dtype), label])
    source = package / "exp.py"
    return {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "golden_sha256": hashlib.sha256((task / "golden.py").read_bytes()).hexdigest(),
            "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("task", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.package.resolve(), args.task.resolve())))
