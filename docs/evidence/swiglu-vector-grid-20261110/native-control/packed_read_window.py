"""Audit and build the case-11 exact/expanded packed GM read comparison."""
import argparse
from pathlib import Path

from emitted_vf import driver
from aligned_row_dma import extract


def audit():
    rows, half, height, pitch, quantum = 273, 509, 8, 512, 64
    total = rows * half * 2
    extra = writes = 0
    for row in range(0, rows, height):
        count = min(height, rows - row)
        first, end = 2 * row * half, 2 * (row + count) * half
        begin = first // quantum * quantum
        stop = min(total, (end + quantum - 1) // quantum * quantum)
        lead = first - begin
        assert 0 <= begin <= first < end <= stop <= total and lead % 8 == 0
        for k in range(count):
            for value in (0, 1):
                start = lead + (2*k + value) * half
                assert begin + start == (2*(row+k) + value) * half
                assert start + half <= stop - begin
                assert start + pitch + quantum <= 2 * height * pitch + 3 * quantum
            assert writes == (row+k) * half
            writes += half
        extra += stop - begin - (end - first)
    assert writes == total // 2
    return extra * 4


def prepare():
    print(f"Extra bytes read across the grid: {audit()}")
    body = Path(__file__).with_suffix(".hpp").read_text()
    cmake = ["cmake_minimum_required(VERSION 3.16)", "find_package(ASC REQUIRED)",
             "project(packed_read_window LANGUAGES ASC CXX)"]
    for widen in (0, 1):
        name = f"copy{widen}0"
        src = driver(11, "#define WIDEN_WINDOW " + str(widen) + "\n" + body)
        Path(name + ".asc").write_text(src)
        cmake.extend([f"add_executable({name} {name}.asc)",
                      f"target_compile_options({name} PRIVATE --npu-arch=dav-3510)"])
    Path("CMakeLists.txt").write_text("\n".join(cmake) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "extract"))
    if parser.parse_args().action == "prepare":
        prepare()
    else:
        extract(Path("build"))
