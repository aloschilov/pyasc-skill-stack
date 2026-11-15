"""Test a leading partial vector on packed, DMA-aligned FP32 half-rows."""

import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from packed_half_descriptors import PACKED
from row_store_stream import vector_body


def render(source, height, half, exact):
    if sha256(source.encode()).hexdigest() != PACKED:
        raise ValueError("expected the packed W79 control")
    if half < 64 or half % 8 or half % 64 == 0:
        raise ValueError("expected a DMA-aligned row with a partial FP32 vector")
    prefix = half % 64
    body = vector_body(height, half, half, "padded")
    marker = f"uint16_t count = c == {(half - 1) // 64} ? {prefix} : 64;"
    if body.count(marker) != 1 or source.count(repr(body)) != 1:
        raise ValueError("expected the packed aligned vector function")
    body2 = body.replace("c*64", "offset").replace("c * 64", "offset")
    body2 = body2.replace(marker, f"uint16_t count = c == 0 ? {prefix} : 64;\n"
                          f"        uint16_t offset = c == 0 ? 0 : {prefix} + (c - 1) * 64;")
    updated = source.replace(repr(body), repr(body2))
    if exact:
        tile = 2 * half * height
        capacity = ((tile + 63) // 64 + 1) * 64
        old = f"[2 * r * H], [{capacity}],"
        if updated.count(old) != 1:
            raise ValueError("expected one allocation at this guard capacity")
        updated = updated.replace(old, f"[2 * r * H], [{tile}],")
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("guard", "exact"))
    args = parser.parse_args()
    source = render((args.package / "swi_glu.py").read_text(), 4, 536, args.mode == "exact")
    compile(source, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(source)
    for name in ("__init__.py", "_pyasc_runtime.py"):
        copyfile(args.package / name, args.output / name)
