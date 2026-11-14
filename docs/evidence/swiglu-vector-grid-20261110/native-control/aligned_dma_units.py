"""Compare byte-unit and 32-byte-unit DMA at fixed ownership on the W74 aligned FP32 path."""
import argparse
from hashlib import sha256
from pathlib import Path

from row_dma_padding import run_control


PYTHON_DIGEST = "ca0fbd3e6c5736786ec8006b58248fc4ed20f576a6331bc06fc3d10e8c31a08b"
CPP_DIGEST = "926c478a646c22f288ebefebdd9491ced908124f21aff99642f8fdf9701a60dc"
MODES = ("base", "read", "write", "both")


def render(source, mode):
    if sha256(source.encode()).hexdigest() != CPP_DIGEST or mode not in MODES:
        raise ValueError("expected the W74 H536/T4 specialization and a known DMA mode")
    if mode in ("read", "both"):
        # 536 floats occupy 67 sectors. Each input skips the opposite half-row in GM
        # and stores consecutive rows in UB. Keep the dynamic count on the final tile.
        begin = source.index("  AscendC::DataCopyExtParams v27")
        end = source.index("  get_buf(PIPE_MTE2", begin)
        source = (source[:begin]
                  + "  AscendC::DataCopyParams v27{static_cast<uint16_t>(v21), 67, 67, 0};\n"
                  + source[end:])
        for operands in ("v5, v18", "v4, v29"):
            source = source.replace(f"DataCopyPad({operands}, v27, v28)", f"DataCopy({operands}, v27)")
    if mode in ("write", "both"):
        # Both output pitches equal the logical width, with zero skipped sectors.
        begin = source.index("  AscendC::DataCopyExtParams v57")
        end = source.index("\n", begin)
        source = (source[:begin]
                  + "  AscendC::DataCopyParams v57{static_cast<uint16_t>(v56), 67, 0, 0};"
                  + source[end:])
        source = source.replace("DataCopyPad(v54, v5, v57)", "DataCopy(v54, v5, v57)")
    expected = (2 if mode in ("read", "both") else 0) + (1 if mode in ("write", "both") else 0)
    if source.count("AscendC::DataCopy(") != expected or source.count("AscendC::DataCopyPad(") != 3 - expected:
        raise ValueError("unexpected DMA calls")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("package", type=Path)
    parser.add_argument("dump", type=Path)
    parser.add_argument("original_dump", type=Path)
    parser.add_argument("worker", type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.case = 13
    if sha256((args.package / "swi_glu.py").read_bytes()).hexdigest() != PYTHON_DIGEST:
        raise ValueError("expected the independent W74 module")
    run_control(args, lambda source: render(source, args.mode), "dma-units.json")
