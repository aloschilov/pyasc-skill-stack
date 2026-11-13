"""Measure two independent 32-byte gaps in the three UB buffers of W62."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys

from emitted_vf import driver
from vf_disassembly import device_elf, sections


SOURCES = {
    11: "166c9b36e31653e12935f88ff7e1b61824862c55ee467f705ac95102b429fb01",
    13: "e3a9c14a351bdbc5d549a5126d5282c2bf9479b44b40b5f331eaccbf9c88deac",
}
GAPS = tuple(product((0, 32), repeat=2))
HERE = Path(__file__).resolve().parent


def kernel(source, case, after_scratch, after_gate):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCES[case]:
        raise ValueError(f"unexpected W62 C++ for case {case}")
    if (after_scratch, after_gate) not in GAPS:
        raise ValueError("unknown layout")
    elements = {11: 4096, 13: 2144}[case]
    span = elements * 4
    offsets = (0, span + after_scratch, 2 * span + after_scratch + after_gate)
    assert offsets[2] + span <= 64 * 1024
    for i, offset in enumerate(offsets):
        before = f"v{4+i}{{AscendC::TPosition::VECCALC, {i*span}, {elements}}}"
        after = f"v{4+i}{{AscendC::TPosition::VECCALC, {offset}, {elements}}}"
        if source.count(before) != 1:
            raise ValueError(before)
        source = source.replace(before, after)
    return source.replace("row_single_wave_fp32", "staged_rows")


def name(case, first, second):
    return f"c{case}g{first}g{second}"


def build(dumps):
    cmake = ["cmake_minimum_required(VERSION 3.16)", "find_package(ASC REQUIRED)",
             "project(ub_spacing LANGUAGES ASC CXX)"]
    report = {}
    for case, (a, b) in product(SOURCES, GAPS):
        base = (dumps / f"case{case}/dump/ascendc.cpp").read_text()
        key = name(case, a, b)
        Path(key + ".asc").write_text(driver(case, kernel(base, case, a, b)))
        cmake += [f"add_executable({key} {key}.asc)",
                  f"target_compile_options({key} PRIVATE --npu-arch=dav-3510)"]
    Path("CMakeLists.txt").write_text("\n".join(cmake) + "\n")
    subprocess.run(["cmake", "-S", ".", "-B", "build"], check=True)
    subprocess.run(["cmake", "--build", "build", "-j2"], check=True)
    for case, (a, b) in product(SOURCES, GAPS):
        key = name(case, a, b)
        binary = device_elf((Path("build")/key).read_bytes())
        Path(key + ".o").write_bytes(binary)
        code = sections(binary)[".text"]
        original = sections((dumps / f"case{case}/dump/binary.o").read_bytes())[".text"]
        report[key] = dict(elf=hashlib.sha256(binary).hexdigest(),
                           text=hashlib.sha256(code).hexdigest(), size=len(code),
                           wheel_text=hashlib.sha256(original).hexdigest())
    Path("digests.json").write_text(json.dumps(report, indent=2))


def profile(raw, finite, task, host):
    for r in (1, 2):
        order = [(c, a, b, "public") for c, (a, b) in product(SOURCES, GAPS)]
        order.extend((13, a, b, "finite") for a, b in GAPS)
        random.Random(35283 + r).shuffle(order)
        for c, a, b, fixture in order:
            key = name(c, a, b)
            dest = f"r{r}-{key}-{fixture}"
            x = finite if fixture == "finite" else raw / f"case{c}-public/input.f32"
            with Path(dest + ".log").open("w") as log:
                proc = subprocess.run([sys.executable, str(HERE/"torch_resident_profile.py"),
                                       key + ".o", str(x), str(task), str(host), str(c),
                                       "cached", "conditioned", dest, "--symbol", "staged_rows"],
                                      stdout=log, stderr=log)
            with Path("status").open("a") as f:
                print(dest, proc.returncode, file=f)
            proc.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("build").add_argument("dumps", type=Path)
    test = sub.add_parser("profile")
    for arg in ("raw", "finite", "task", "host"):
        test.add_argument(arg, type=Path)
    args = parser.parse_args()
    if args.action == "build":
        build(args.dumps)
    else:
        profile(args.raw, args.finite, args.task, args.host)
