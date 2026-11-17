"""Measure grouping two co-consumed FP32 UB tiles under one hardware buffer ID."""
import argparse
from hashlib import sha256
from pathlib import Path
import re

from aligned_dma_units import CPP_DIGEST, PYTHON_DIGEST
from row_dma_padding import run_control


LISTINGS = {11: "4729822c5d8a5019b1f2c035393458555da4ab4f10eafae6fc9e11fbdc6a2c86",
            13: CPP_DIGEST}


def render(source, case, mode):
    if sha256(source.encode()).hexdigest() != LISTINGS[case]:
        raise ValueError("expected the qualified W74 listing")
    if mode == "base":
        return source
    if mode not in ("ordered", "shared"):
        raise ValueError(mode)
    # Both bounded reads consume whole rows and both zero fills target only
    # missing rows. Preserve both fills even when using a common buffer ID.
    first, second = re.finditer(r"  if \((v\d+)\) \{\n[^{}]+?\n  \}\n", source)
    if first[1] != second[1] or any(m[0].count("AscendC::Duplicate(") != 1 for m in (first, second)):
        raise ValueError("expected paired conditional terminal fills")
    if mode == "ordered":
        fill = first[0] + second[0]
    else:
        bodies = []
        for m in (first, second):
            body = m[0].splitlines(keepends=True)[1:-1]
            bodies.extend(line for line in body if "_buf(" not in line)
        fill = (f"  if ({first[1]}) {{\n    get_buf(PIPE_V, 0, 0);\n"
                + "".join(bodies) + "    rls_buf(PIPE_V, 0, 0);\n  }\n")
    out = source[:first.start()] + fill + source[first.end():second.start()] + source[second.end():]
    if mode == "shared":
        # The one token covers both physical UB addresses throughout the lifetime
        # of this single tile: optional zero fill -> both reads -> VF -> store.
        for marker in ("rls_buf(PIPE_MTE2, 0, 0);", "get_buf(PIPE_MTE2, 1, 0);",
                       "get_buf(PIPE_V, 1, 0);", "rls_buf(PIPE_V, 1, 0);"):
            line = f"  {marker}\n"
            if out.count(line) != 1:
                raise ValueError(f"expected one {marker}")
            out = out.replace(line, "")
        out = out.replace("rls_buf(PIPE_MTE2, 1, 0);", "rls_buf(PIPE_MTE2, 0, 0);")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("base", "ordered", "shared"))
    p.add_argument("case", type=int, choices=LISTINGS)
    for name in ("package", "dump", "original_dump", "worker"):
        p.add_argument(name, type=Path)
    p.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != PYTHON_DIGEST:
        raise ValueError("expected independent W74 input")
    run_control(args, lambda s: render(s, args.case, args.mode), "paired-sync.json")
