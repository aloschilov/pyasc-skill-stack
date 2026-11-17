"""Prepare full and in-place UB register-store controls for both short FP32 shapes."""
import argparse
from pathlib import Path

from emitted_vf import driver
from aligned_row_dma import extract


def prepare():
    hpp = Path(__file__).with_suffix(".hpp").read_text()
    cmake = ["cmake_minimum_required(VERSION 3.16)", "find_package(ASC REQUIRED)",
             "project(gate_store LANGUAGES ASC CXX)"]
    for case, rows, half, height in [(11,273,509,8),(13,143,536,4)]:
        pitch = (half + 7) // 8 * 8
        for inplace in (0,1):
            out = pitch * (1 + inplace)
            for count in (height, rows % height):
                # Each masked VF store owns exactly one gate row. A complete
                # vector load at the end stays within the allocated UB guard.
                for r in range(count):
                    for col in range(0, half, 64):
                        n = min(64, half - col)
                        assert r*out + col+n <= r*out + pitch
                        assert (2*r+1)*pitch + col+64 <= 2*height*pitch+64
                        assert not inplace or r*out + col+n <= (2*r+1)*pitch
                assert ((out-pitch)*4) % 32 == 0
            name = f"copy{int(case==13)}{inplace}"
            prefix = (f"#define TEST_ROWS {rows}\n#define TEST_HALF {half}\n"
                      f"#define TEST_HEIGHT {height}\n#define IN_GATE {inplace}\n")
            Path(name+".asc").write_text(driver(case, prefix+hpp))
            cmake.extend([f"add_executable({name} {name}.asc)",
                          f"target_compile_options({name} PRIVATE --npu-arch=dav-3510)"])
    Path("CMakeLists.txt").write_text("\n".join(cmake)+"\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("action", choices=("prepare","extract"))
    if args.parse_args().action == "prepare":
        prepare()
    else:
        extract(Path("build"))
