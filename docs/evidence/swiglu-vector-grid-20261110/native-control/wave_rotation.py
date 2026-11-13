"""Hold row ownership constant and vary integer lowering for a one-wave rotation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

from row_math_cost import W62


def render(base, mode):
    if hashlib.sha256(base.encode()).hexdigest() != W62:
        raise ValueError("expected the qualified W62 module")
    start = base.index("def row_single_wave_fp32(")
    stop = base.index("    r = tile_index * T", start)
    part = base[start:stop]
    if mode == "runtime":
        return base
    # The one-wave dispatcher checks that this exact tile count is launched.
    # Both R and T are static specialization arguments. For b < n and
    # offset = floor(n/2), at most one wrap is possible, including odd grids.
    part = part.replace("n = asctile.block_num()", "n = (R + T - 1) // T")
    if mode == "subtract":
        part = part.replace("tile_index = (b + n // 2) % n",
                            "shifted = b + n // 2\n        tile_index = shifted - n if shifted >= n else shifted")
    elif mode != "static":
        raise ValueError(mode)
    return base[:start] + part + base[stop:]


def prepare(base, task, shapes, modes, renderer=render):
    source = (base / "swi_glu.py").read_text()
    for key, shape in shapes.items():
        dest = Path(key)
        fixture = dest / "generation/task"
        fixture.mkdir(parents=True)
        shutil.copyfile(task/"golden.py", fixture/"golden.py")
        shapefile = fixture/"cases.csv"
        if shape is None:
            shutil.copyfile(task/"cases.csv", shapefile)
        else:
            with (task/"cases.csv").open(newline="") as f:
                reader = csv.DictReader(f)
                row, = (r for r in reader if r["case_id"] == "13")
            if json.loads(row["input_shape"]) != [[11, 13, 16, 67]]:
                raise ValueError(row["input_shape"])
            row["input_shape"] = json.dumps([shape])
            row["value_range"] = "[-2, 2]"
            with shapefile.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
        for mode in modes:
            pkg = dest / mode
            pkg.mkdir()
            (pkg/"__init__.py").touch()
            (pkg/"swi_glu.py").write_text(renderer(source, mode))
            shutil.copyfile(base/"_pyasc_runtime.py", pkg/"_pyasc_runtime.py")


def run(worker, shapes, modes):
    plan = []
    for r in range(1, 4):
        order = [(r,key,mode) for key in shapes for mode in modes]
        random.Random(73120 + r).shuffle(order)
        plan.extend(order)
    Path("plan.json").write_text(json.dumps(plan))
    for r, key, mode in plan:
        root = Path(key) / mode
        env = {**os.environ, "PYASC_CACHE_DIR": str((root/"cache").resolve()),
               "PYASC_DUMP_PATH": str((root/"dump").resolve())}
        with (root/f"{r}.json").open("w") as out, (root/f"{r}.log").open("w") as log:
            proc = subprocess.run([sys.executable, str(worker), mode, "swi_glu", "13"],
                                  cwd=key, env=env, stdout=out, stderr=log)
        with Path("status").open("a") as f:
            print(r, key, mode, proc.returncode, file=f)
    return int(any(not l.endswith(" 0") for l in Path("status").read_text().splitlines()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("base","task","worker"):
        p.add_argument(arg, type=Path)
    p.add_argument("--adjacent", action="store_true", help="compare odd/even grids with matched tails")
    args = p.parse_args()
    shapes = {"public": None, "neighbor": [11, 11, 16, 67]}
    modes = ("runtime", "static", "subtract")
    if args.adjacent:
        shapes = {f"rows{n}": [n, 1, 16, 67] for n in (139, 143, 140, 144)}
        modes = ("runtime", "subtract")
    prepare(args.base.resolve(), args.task.resolve(), shapes, modes)
    sys.exit(run(args.worker.resolve(), shapes, modes))
