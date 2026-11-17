"""Factor local GM windows and UB reuse on the packed prefetch control."""
from pathlib import Path
from shutil import copyfile
import re

from row_stage_order import render as staged


def render(base, rows, half, height, local, reuse):
    if reuse not in (0, 2):
        raise ValueError("expected separate allocations or tensor allocation reuse")
    original = staged(base, rows, half, height, "prefetch")
    start = original.index("@asctile.jit(reuse_alloc=2)\ndef staged_rows(")
    end = original.index("\n@asctile.jit", start + 1)
    lines = []
    for line in original[start:end].splitlines():
        if local:
            if line.startswith(("    src = ", "    dst = ")):
                continue
            rd = re.fullmatch(r"        p(\d+) = asctile.copy_in\(src, \[2 \* \((.*)\), 0\], \[(\d+), W\], (.*)\)", line)
            if rd:
                n, row, count, extent = rd.groups()
                lines.append(f"        x{n} = asctile.global_tensor(x + 2 * ({row}) * H, [{count}, H])")
                line = f"        p{n} = asctile.copy_in(x{n}, [0, 0], [{count}, W], {extent})"
            wr = re.fullmatch(r"        asctile.copy_out\(v(\d+), dst, \[(.*), 0\], real_shape=\[(\d+), H\]\)", line)
            if wr:
                n, row, count = wr.groups()
                lines.append(f"        y{n} = asctile.global_tensor(y + ({row}) * H, [{count}, H])")
                line = f"        asctile.copy_out(v{n}, y{n}, [0, 0], real_shape=[{count}, H])"
        lines.append(line)
    body = "\n".join(lines).replace("jit(reuse_alloc=2)", f"jit(reuse_alloc={reuse})")
    if local and ("copy_in(src," in body or ", dst," in body):
        raise ValueError("stage origin was not rebased")
    return original[:start] + body + "\n\n" + original[end:]


if __name__ == "__main__":
    base = Path("base/swi_glu.py")
    for case, rows, half, height in [(11, 273, 509, 8), (13, 143, 536, 4)]:
        for local in (False, True):
            for reuse in (0, 2):
                package = Path(f"c{case}w{int(local)}r{reuse}")
                code = render(base.read_text(), rows, half, height, local, reuse)
                compile(code, str(package), "exec")
                package.mkdir()
                (package / "swi_glu.py").write_text(code)
                (package / "__init__.py").touch()
                copyfile(base.with_name("_pyasc_runtime.py"), package / "_pyasc_runtime.py")
    import json
    import os
    import random
    import subprocess
    import sys

    worker = Path(sys.argv[1]).resolve()
    schedule = []
    for round_id in (1, 2):
        entries = [(round_id, case, f"c{case}w{local}r{reuse}")
                   for case in (11, 13) for local in (0, 1) for reuse in (0, 2)]
        random.Random(6330 + round_id).shuffle(entries)
        schedule.extend(entries)
    Path("plan.json").write_text(json.dumps(schedule))
    for round_id, case, pkg in schedule:
        label = f"{round_id}.{case}.{pkg}"
        run = Path("results") / pkg
        run.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYASC_CACHE_DIR": str((run / "cache").resolve()),
               "PYASC_DUMP_PATH": str((run / "dump").resolve())}
        with open(f"results/{label}.json", "w") as out, open(f"results/{label}.log", "w") as err:
            proc = subprocess.run([sys.executable, str(worker), pkg, "swi_glu", str(case)],
                                  env=env, stdout=out, stderr=err)
        with open("status", "a") as log:
            print(label, proc.returncode, file=log)
    Path("done").touch()
