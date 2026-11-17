"""Factor a scalar-left subtraction, register zero and fused exponent on the W75 FP32 route."""
import argparse
from hashlib import sha256
from pathlib import Path
import re

from row_dma_padding import run_control

ORIGINAL = "d2f91370bd596e49fc3bac470f7089246f05d7b31a27156f0b4985c8e288d39a"
SUBTRACT = "60b57fb02174728a977faa147108a2189db4ae4dfd4626d36195e3d90fcf289d"
BASE_CPP = {11: "969d0dafcb6ad86b14714e6ed48c768d6a3e8acbdf4bcbcafc2089023a72bcd5",
            13: "662d22665052c2d91fc8b800cfd5bd3d9451f638aef5a6d2329f4d3f0c9e006e"}
SUB_CPP = {11: "18ba788980cfa48a1bbf296a4c917ca05cf98dc3483ffc2b346e3284208efb03",
           13: "9d9ca064f07d232e37688966a9255b5299a617201b04e0dc15d9de2023c9686a"}


def render(source, case, mode):
    expected = SUB_CPP if mode == "sub" else BASE_CPP
    if sha256(source.encode()).hexdigest() != expected[case]:
        raise ValueError("unexpected W75 register listing")
    if mode in ("base", "sub"):
        return source
    seq = re.compile(r"AscendC::Reg::Duplicate\((?P<zero>v\d+), cm1_f32, (?P<mask>v\d+)\);\n"
                     r"      AscendC::Reg::Mul\((?P<neg>v\d+), (?P<gate>v\d+), (?P=zero), (?P=mask)\);\n"
                     r"      AscendC::Reg::Exp\((?P<out>v\d+), (?P=neg), (?P=mask)\);")
    m, = seq.finditer(source)
    zero, mask, neg, gate, out = m.groups()
    ops = [f"Duplicate({zero}, c0_f32, {mask})"]
    if mode == "regsub":
        ops.extend((f"Sub({neg}, {zero}, {gate}, {mask})", f"Exp({out}, {neg}, {mask})"))
    elif mode == "fused":
        ops.append(f"FusedExpSub({out}, {zero}, {gate}, {mask})")
    else:
        raise ValueError(mode)
    return source[:m.start()] + "\n      ".join(f"AscendC::Reg::{op};" for op in ops) + source[m.end():]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("base", "sub", "regsub", "fused"))
    p.add_argument("case", type=int, choices=BASE_CPP)
    for key in ("package", "dump", "original_dump", "worker"):
        p.add_argument(key, type=Path)
    p.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    expected = SUBTRACT if args.mode == "sub" else ORIGINAL
    if sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != expected:
        raise ValueError("unexpected source for the selected exponent control")
    run_control(args, lambda s: render(s, args.case, args.mode), "negative-exp.json")
