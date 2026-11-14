"""Factor UB tensor rank and compiler vector fusion at fixed row ownership."""
import argparse
import ast
import hashlib
from pathlib import Path

from row_math_cost import W62
from wave_rotation import prepare, run


def render(source, mode):
    if hashlib.sha256(source.encode()).hexdigest() != W62:
        raise ValueError("expected W62")
    if mode not in ("base", "flat", "fused", "flatfused"):
        raise ValueError(mode)
    kernel = next(n for n in ast.parse(source).body
                  if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    if mode in ("fused", "flatfused"):
        line = kernel.decorator_list[0].lineno - 1
        assert lines[line] == "@asctile.jit(reuse_alloc=2)\n"
        lines[line] = "@asctile.jit(reuse_alloc=2, vf_fusion=True)\n"
    if mode in ("flat", "flatfused"):
        line = next(n.lineno - 1 for n in kernel.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", "") == "gate_f")
        lines[line:kernel.end_lineno - 1] = [
            "    gate_f = asctile.reshape(gate, T * W)\n",
            "    value_f = asctile.reshape(value, T * W)\n",
            "    result = (gate_f / (1.0 + asctile.exp(-gate_f))) * value_f\n",
            "    out = asctile.reshape(result, T, W)\n",
        ]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "task", "worker"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--case", type=int, choices=(11, 13), required=True)
    args = parser.parse_args()
    finite = {11: [273, 1, 1, 1018], 13: [143, 1, 16, 67]}
    fixtures = {"public": None, "finite": finite[args.case]}
    modes = ("base", "flat", "fused", "flatfused")
    prepare(args.base.resolve(), args.task.resolve(), fixtures, modes, render, args.case)
    raise SystemExit(run(args.worker.resolve(), fixtures, modes, args.case))
