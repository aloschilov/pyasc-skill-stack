"""Compare pipeline setup on the two statically allocated W77 FP32 kernels."""
import argparse
import hashlib
import re
from pathlib import Path

from row_dma_padding import run_control

SOURCE = "dfc9cd891023b2aa6600018dee91984abbc8d07c8985e1c0bbf13c45cd2659ab"
LISTINGS = {
    11: "04005ff1799726d3164800afd5f03f16b93ed2c87f5feb89cc2a1def292d1750",
    13: "9f7e7370bf332ac35156280f0597ec5371fc4f0cd7089a186dff39b2551dbf3d",
}


def render(source, case, mode):
    if hashlib.sha256(source.encode()).hexdigest() != LISTINGS[case]:
        raise ValueError("expected the qualified static-allocation W77 listing")
    if mode == "base":
        return source
    if mode != "explicit" or len(re.findall(r"\bv3\b", source)) != 1:
        raise ValueError("expected a TPipe with no buffer or queue users")
    setup, end = "AscendC::TPipe v3;", "  return;"
    if source.count(setup) != 1 or source.count(end) != 1:
        raise ValueError("expected one pipeline entry and one exit")
    return source.replace(setup, "AscendC::InitSocState();").replace(
        end, "  AscendC::PipeBarrier<PIPE_ALL>();\n  return;")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "explicit"))
    parser.add_argument("case", type=int, choices=LISTINGS)
    for name in ("package", "dump", "original_dump", "worker"):
        parser.add_argument(name, type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if hashlib.sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != SOURCE:
        raise ValueError("unexpected candidate source")
    run_control(args, lambda src: render(src, args.case, args.mode), "init.json")
