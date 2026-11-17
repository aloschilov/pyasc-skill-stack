"""Compare tile permutations of adjacent AIVs for a single wave of row DMA."""
import argparse
import ast
import hashlib
from pathlib import Path

from row_math_cost import W62
from wave_rotation import prepare, run


def render(source, order):
    if hashlib.sha256(source.encode()).hexdigest() != W62:
        raise ValueError("expected the verified W62 source")
    if order == "base":
        return source
    assignments = {
        "separate": "b = b // 2 + (b % 2) * long_half",
        "adjacent": "b = 2 * b if b < long_half else 2 * (b - long_half) + 1",
    }
    body = next(n.body for n in ast.parse(source).body
                if getattr(n, "name", "") == "row_single_wave_fp32")
    line = next(n.end_lineno for n in body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "b")
    lines = source.splitlines(keepends=True)
    # Launch grid is exactly ceil(R/T) in this one-wave specialization.
    lines[line:line] = ["    long_half = ((R + T - 1) // T + 1) // 2\n",
                       "    " + assignments[order] + "\n"]
    return "".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "task", "worker"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--case", choices=(11, 13), type=int, required=True)
    args = parser.parse_args()
    finite = {11: [273, 1, 1, 1018], 13: [143, 1, 16, 67]}
    fixtures = {"public": None, "finite": finite[args.case]}
    orders = ("base", "separate", "adjacent")
    prepare(args.base.resolve(), args.task.resolve(), fixtures, orders, render, args.case)
    raise SystemExit(run(args.worker.resolve(), fixtures, orders, args.case))
