"""Compare aligned and byte-granularity DMA at identical W62 geometry (H536)."""
import argparse
import hashlib
from pathlib import Path

from emitted_vf import driver
from vf_disassembly import device_elf, sections


SOURCE = "e3a9c14a351bdbc5d549a5126d5282c2bf9479b44b40b5f331eaccbf9c88deac"


def render(source, aligned_read, aligned_write):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE:
        raise ValueError("expected W62 emitted C++ for the aligned 143 by 536 case")
    # H536 in FP32 = 67 units of 32 bytes. GM row and UB pitch stay aligned
    # for all rows, including the last block's bounded three-row transfer.
    for enabled, param, count, gap, calls in [
        (aligned_read, "v29", "v23", 67, ["v5, v18", "v6, v32"]),
        (aligned_write, "v40", "v39", 0, ["v37, v6"]),
    ]:
        if enabled:
            lines = source.splitlines(keepends=True)
            i, = (i for i,line in enumerate(lines) if f"DataCopyExtParams {param}" in line)
            lines[i] = (f"  AscendC::DataCopyParams {param}{{static_cast<uint16_t>({count}), "
                        f"67, {gap}, 0}};\n")
            source = "".join(lines)
            for args in calls:
                old = f"AscendC::DataCopyPad({args}, {param}" + (", v30);" if gap else ");")
                if source.count(old) != 1:
                    raise ValueError(old)
                source = source.replace(old, f"AscendC::DataCopy({args}, {param});")
    return source


def prepare(dump):
    s = dump.read_text()
    cmake = ["cmake_minimum_required(VERSION 3.16)", "find_package(ASC REQUIRED)",
             "project(aligned_dma LANGUAGES ASC CXX)"]
    for reads in (0,1):
        for writes in (0,1):
            name = f"copy{reads}{writes}"
            kernel = render(s, reads, writes).replace("row_single_wave_fp32", "staged_rows")
            Path(name + ".asc").write_text(driver(13, kernel))
            cmake += [f"add_executable({name} {name}.asc)",
                      f"target_compile_options({name} PRIVATE --npu-arch=dav-3510)"]
    Path("CMakeLists.txt").write_text("\n".join(cmake) + "\n")


def extract(build):
    for image in sorted(build.glob("copy[01][01]")):
        binary = device_elf(image.read_bytes())
        text = sections(binary)[".text"]
        (image.with_suffix(".o")).write_bytes(binary)
        print(image.name, len(text), hashlib.sha256(text).hexdigest(),
              hashlib.sha256(binary).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare","extract"))
    p.add_argument("path", type=Path)
    args = p.parse_args()
    (prepare if args.action == "prepare" else extract)(args.path)
