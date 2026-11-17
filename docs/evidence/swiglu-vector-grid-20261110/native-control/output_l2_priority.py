"""Isolate the output GlobalTensor cache attribute on two W79 FP32 specializations."""

import argparse
import hashlib
from pathlib import Path
import re

from row_dma_padding import run_control


W79 = "338ee63229612fc35cc287eff4a2d10bb1e9a1c47844b0d33bb0baa49da14202"


def render(source, case, mode):
    rows, width = {13: (143, 536), 11: (273, 509)}[case]
    if ("void row_single_wave_fp32(" not in source
            or not all(f"c{v}_i32 = {v};" in source for v in (rows, width))
            or source.count("SetL2CacheHint<") != 1
            or source.count("CacheMode::CACHE_MODE_DISABLE") != 1):
        raise ValueError("unexpected W79 one-wave kernel or input cache hint")
    output, = re.findall(r"^  (v\d+)\.SetGlobalBuffer\(v2_y\);$", source, re.MULTILINE)
    if mode == "base":
        return source
    cache = {"normal": "NORMAL", "persistent": "PERSISTENT"}[mode]
    setup = f"  {output}.SetGlobalBuffer(v2_y);"
    hint = f"  {output}.SetL2CacheHint<AscendC::CacheRwMode::WRITE>(AscendC::CacheMode::CACHE_MODE_{cache});"
    return source.replace(setup, setup + "\n" + hint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "normal", "persistent"))
    parser.add_argument("case", type=int, choices=(11, 13))
    for name in ("package", "dump", "original_dump", "worker"):
        parser.add_argument(name, type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if hashlib.sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != W79:
        raise ValueError("expected the independent W79 source")
    run_control(args, lambda src: render(src, args.case, args.mode), "priority.json")
