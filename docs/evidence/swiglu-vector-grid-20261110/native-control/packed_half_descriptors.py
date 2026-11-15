"""Hold the packed FP32 vector function fixed while splitting its aligned GM read."""

import argparse
from hashlib import sha256
from pathlib import Path
import re

from row_dma_padding import run_control

PACKED = "f2db7ebbcaf48e1aad0dfdb43246852c4f0b860ab3ab7dd954c4748d5e59fa57"


def render(source, mode):
    if (source.count("AscendC::DataCopyPad(") != 2
            or "4352_i64 = 4352;" not in source or "(2*r+1)*536" not in source):
        raise ValueError("expected the aligned packed specialization")
    if mode == "base":
        return source
    if mode != "split":
        raise ValueError(mode)
    # The generated input element count is valid_rows * (2 * half_width).
    valid, = re.findall(r"int32_t v\d+ = (v\d+) \* c1072_i32;", source)
    call, = re.finditer(r"^  AscendC::DataCopyPad\((v\d+), (v\d+), (v\d+), (v\d+)\);$", source, re.M)
    ub, gm, _, pad = call.groups()
    transfer = f"""  AscendC::DataCopyExtParams halves{{static_cast<uint16_t>({valid}), 2144, 2144, 67, 0}};
  AscendC::DataCopyPad({ub}, {gm}, halves, {pad});
  AscendC::DataCopyPad({ub}[536], {gm}[536], halves, {pad});"""
    return source[:call.start()] + transfer + source[call.end():]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "split"))
    parser.add_argument("case", type=int, choices=(13,))
    for name in ("package", "dump", "original_dump", "worker"):
        parser.add_argument(name, type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != PACKED:
        raise ValueError("expected the packed W79 diagnostic")
    run_control(args, lambda cpp: render(cpp, args.mode), "descriptors.json")
