"""Vary GM cache hints on the exact, four-buffer packed prefetch schedule."""
import argparse
import ast
import json
import os
from pathlib import Path
import random
from shutil import copyfile
import subprocess
import sys

from packed_stage_windows import render as stage_windows


def render(base, case, input_cache, output_cache):
    rows, half, height = {11: (273, 509, 8), 13: (143, 536, 4)}[case]
    source = stage_windows(base, rows, half, height, True, 0)
    fn = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "staged_rows")
    lines = source.splitlines(keepends=True)
    for i in range(fn.lineno - 1, fn.end_lineno):
        line = lines[i]
        for operand, cache in [("x", input_cache), ("y", output_cache)]:
            if not cache and f"asctile.global_tensor({operand} +" in line:
                lines[i] = line.rstrip().removesuffix(")") + ", l2_cache=False)\n"
    code = "".join(lines)
    compile(code, "swi_glu.py", "exec")
    return code


def measure(worker):
    base = Path("base/swi_glu.py").read_text()
    configurations = []
    for case in (11, 13):
        for src_cache in (True, False):
            for dst_cache in (True, False):
                package = Path(f"c{case}i{int(src_cache)}o{int(dst_cache)}")
                package.mkdir()
                (package / "swi_glu.py").write_text(render(base, case, src_cache, dst_cache))
                (package / "__init__.py").touch()
                copyfile("base/_pyasc_runtime.py", package / "_pyasc_runtime.py")
                configurations.append((case, str(package)))
    schedule = []
    for r in (1, 2):
        order = [(r, c, k) for c, k in configurations]
        random.Random(64170 + r).shuffle(order)
        schedule.extend(order)
    Path("plan.json").write_text(json.dumps(schedule))
    for r, c, k in schedule:
        root = Path("results") / k
        root.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYASC_CACHE_DIR": str((root / "cache").resolve()),
               "PYASC_DUMP_PATH": str((root / "dump").resolve())}
        with open(f"results/{r}.{c}.{k}.json", "w") as out, open(f"results/{r}.{c}.{k}.log", "w") as log:
            proc = subprocess.run([sys.executable, str(worker), k, "swi_glu", str(c)],
                                  stdout=out, stderr=log, env=env)
        with Path("status").open("a") as status:
            print(r, c, k, proc.returncode, file=status)
    Path("done").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker", type=Path)
    measure(parser.parse_args().worker.resolve())
