"""Measure arithmetic substitutions at fixed DMA geometry on the two short FP32 rows."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

W62 = "26c6dad852aa263be1b44d7a8b258c498ba36915ec72d07bc5ee264f98a02453"
DIRECT = "(gate_f / (1.0 + asctile.exp(-gate_f))) * value.to(asctile.float32)"
EXPRESSIONS = {
    "direct": DIRECT,
    "reciprocal": "(gate_f * (1.0 / (1.0 + asctile.exp(-gate_f)))) * value.to(asctile.float32)",
    "multiply": "gate_f * value.to(asctile.float32)",
    "gate_copy": "gate_f",
}
CONTROL_ORACLES = {
    "multiply": "gate * value",
    "gate_copy": "gate.clone()",
}


def candidate(source, mode):
    if hashlib.sha256(source.encode()).hexdigest() != W62:
        raise ValueError("expected the qualified W62 source")
    fn = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[fn.lineno-1:fn.end_lineno])
    if body.count(DIRECT) != 1:
        raise ValueError("could not identify the arithmetic in the one-wave kernel")
    lines[fn.lineno-1:fn.end_lineno] = [body.replace(DIRECT, EXPRESSIONS[mode])]
    return "".join(lines)


def run(base, task, worker, rounds):
    source = (base / "swi_glu.py").read_text()
    for mode in EXPRESSIONS:
        root = Path(mode)
        pkg = root / "probe"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").touch()
        (pkg / "swi_glu.py").write_text(candidate(source, mode))
        shutil.copyfile(base / "_pyasc_runtime.py", pkg / "_pyasc_runtime.py")
        fixture = root / "generation/task"
        fixture.mkdir(parents=True)
        shutil.copyfile(task / "cases.csv", fixture / "cases.csv")
        if mode in CONTROL_ORACLES:
            # Private diagnostic fixture. Leave the public task intact.
            (fixture / "golden.py").write_text(
                "def swi_glu(input, dim=-1):\n"
                "    gate, value = input.chunk(2, dim=dim)\n"
                f"    return {CONTROL_ORACLES[mode]}\n"
            )
        else:
            shutil.copyfile(task / "golden.py", fixture / "golden.py")
    plan = []
    for r in range(1, rounds + 1):
        order = [(r, c, k) for c in (11, 13) for k in EXPRESSIONS]
        random.Random(65000 + r).shuffle(order)
        plan.extend(order)
    Path("plan.json").write_text(json.dumps(plan))
    for r, c, mode in plan:
        root = Path(mode)
        build = root / f"case{c}"
        build.mkdir(exist_ok=True)
        env = {**os.environ, "PYASC_CACHE_DIR": str((build/"cache").resolve()),
               "PYASC_DUMP_PATH": str((build/"dump").resolve())}
        with (root/f"{r}.{c}.json").open("w") as out, (root/f"{r}.{c}.log").open("w") as log:
            proc = subprocess.run([sys.executable, str(worker), "probe", "swi_glu", str(c)],
                                  cwd=root, env=env, stdout=out, stderr=log)
        with Path("status").open("a") as status:
            print(r, c, mode, proc.returncode, file=status)
    return int(any(not l.endswith(" 0") for l in Path("status").read_text().splitlines()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "task", "worker"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--rounds", type=int, default=3, choices=range(1, 10))
    opt = parser.parse_args()
    sys.exit(run(opt.base.resolve(), opt.task.resolve(), opt.worker.resolve(), opt.rounds))
