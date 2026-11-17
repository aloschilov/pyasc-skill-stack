"""Measure a constant full-vector predicate on the FP32 one-wave specialization."""

import argparse
from hashlib import sha256
from pathlib import Path
import re

from output_l2_priority import W79
from row_dma_padding import run_control


def render(source, case, mode):
    lanes = {11: 3072, 13: 2144}[case]
    if ("void row_single_wave_fp32(" not in source
            or source.count("AscendC::Reg::UpdateMask<float>") != 1):
        raise ValueError("expected one predicate update in the W79 FP32 vector loop")
    counter, = re.findall(rf"uint32_t (v\d+)\[1\]\{{c{lanes}_idx\}};", source)
    all_lanes, = re.findall(r"MaskReg (v\d+) = .*?CreateMask<float, .*?::ALL>\(\);", source)
    full_vectors, tail = divmod(lanes, 64)
    bound = f"c{full_vectors + bool(tail)}_idx"
    loop = re.search(
        rf"    for \(uint16_t (v\d+) = 0; \1 < static_cast<uint16_t>\({bound}\); \1 \+= 1\) \{{\n"
        r"(?P<body>.*?)^    \};", source, re.MULTILINE | re.DOTALL)
    if loop is None:
        raise ValueError("unexpected vector loop bounds")
    if mode == "base":
        return source
    if mode != "constant":
        raise ValueError(mode)
    update = f"AscendC::Reg::UpdateMask<float>(*{counter})"
    body = loop["body"]
    if body.count(update) != 1 or body.count("StoreAlign<") != 1:
        raise ValueError("expected one mask update and a single vector store")
    main = loop[0].replace(bound, str(full_vectors)).replace(update, all_lanes)
    if tail:
        # The original last load addresses and masked division/store are preserved.
        offset = f"{loop[1]} * c64_idx"
        if body.count(offset) != 1:
            raise ValueError("unexpected vector offset")
        epilogue = body.replace(offset, str(64 * full_vectors))
        main += f"\n    {{\n      *{counter} = {tail};\n" + epilogue + "    }"
    return source[:loop.start()] + main + source[loop.end():]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "constant"))
    parser.add_argument("case", type=int, choices=(11, 13))
    for name in ("package", "dump", "original_dump", "worker"):
        parser.add_argument(name, type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != W79:
        raise ValueError("expected the qualified W79 module")
    run_control(args, lambda src: render(src, args.case, args.mode), "mask.json")
