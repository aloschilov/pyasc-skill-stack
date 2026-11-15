"""Measure a packed FP32 input's guard fill at fixed DMA and vector code."""

import argparse
from hashlib import sha256
from pathlib import Path
import re

from packed_half_descriptors import PACKED
from row_dma_padding import run_control


def render(source, case, mode):
    capacity = {11: 6208, 13: 4352}[case]
    # Capture the rounded DMA byte count directly from the emitted fill guard.
    fill = re.compile(
        rf"bool (?P<test>v\d+) = (?P<rounded>v\d+) < c{capacity * 4}_i32;\n"
        r"  if \((?P=test)\) \{\n    get_buf\(PIPE_V, 0, 0\);\n"
        rf"    (?P<dup>AscendC::Duplicate\((?P<ub>v\d+), c0_f32, c{capacity}_i64\));"
    )
    match, = fill.finditer(source)
    if mode == "base":
        return source
    if mode != "suffix":
        raise ValueError(mode)
    # DMA initializes the data and right-pads the final 32-byte sector with zero.
    offset = f"{match['rounded']} / sizeof(float)"
    new = (f"const uint32_t firstUnused = {offset};\n"
           f"    AscendC::Duplicate({match['ub']}[firstUnused], c0_f32, "
           f"static_cast<int32_t>({capacity} - firstUnused))")
    start, end = match.span("dup")
    return source[:start] + new + source[end:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "suffix"))
    parser.add_argument("case", type=int, choices=(11, 13))
    for name in ("package", "dump", "original_dump", "worker"):
        parser.add_argument(name, type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != PACKED:
        raise ValueError("expected the packed W79 package")
    run_control(args, lambda source: render(source, args.case, args.mode), "guard-fill.json")
