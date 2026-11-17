"""Report UB allocations for two forms of a unary negative exponent across ranks and dtypes."""
import argparse
import json
from pathlib import Path
import re

from lower_planar_nddma import lower_typed


KERNEL = r'''
from asc.experimental import asctile

@asctile.jit(reuse_alloc=2, vf_fusion=True)
def unary(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
          R: asctile.ConstExpr[int], H: asctile.ConstExpr[int],
          FLAT: asctile.ConstExpr[bool], LEFT: asctile.ConstExpr[bool]):
    if FLAT:
        gx = asctile.global_tensor(x, [R * H])
        gy = asctile.global_tensor(y, [R * H])
        raw = asctile.copy_in(gx, [0], [R * H])
    else:
        gx = asctile.global_tensor(x, [R, H])
        gy = asctile.global_tensor(y, [R, H])
        raw = asctile.copy_in(gx, [0, 0], [R, H])
    v = raw.to(asctile.float32)
    if LEFT:
        result = asctile.exp(0.0 - v)
    else:
        result = asctile.exp(-v)
    if FLAT:
        asctile.copy_out(result.to(raw.dtype), gy, [0])
    else:
        asctile.copy_out(result.to(raw.dtype), gy, [0, 0])
'''


def inspect(workers, destination):
    destination.mkdir()
    package = destination / "unary_exp"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "swi_glu.py").write_text(KERNEL)
    for dtype in ("float32", "float16", "bfloat16"):
        for flat in (False, True):
            for left in (False, True):
                cpp = destination / f"exp-{dtype}-{int(flat)}-{int(left)}.cpp"
                source = lower_typed(package, workers, "unary", (dtype, dtype),
                                     (4, 256, flat, left), cpp)
                yield {
                    "dtype": dtype, "flat": flat, "left": left,
                    "ub_tiles": source.count("{AscendC::TPosition::VECCALC,"),
                    "zero_tiles": len(re.findall(r"AscendC::Duplicate\(\w+, c0_f32,", source)),
                    "unary_math": re.findall(r"AscendC::Reg::(Exp|Mul|Sub|FusedExpSub)[(<]", source),
                    "dma": source.count("AscendC::DataCopyPad("),
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workers", type=Path, help="directory containing local_compile_gate.py")
    parser.add_argument("destination", type=Path, help="new directory for emitted C++")
    args = parser.parse_args()
    for result in inspect(args.workers.resolve(), args.destination.resolve()):
        print(json.dumps(result))
