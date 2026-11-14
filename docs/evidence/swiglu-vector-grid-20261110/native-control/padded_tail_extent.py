"""Separate fixed DMA descriptors from shortening the last register tile."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import indent

from aligned_dma_units import PYTHON_DIGEST


def tile(valid, height):
    return f"""gx = asctile.global_tensor(x + r * (2 * H), [{valid}, 2 * H], l2_cache=False)
gy_tile = asctile.global_tensor(y + r * H, [{valid}, H])
value = asctile.copy_in(gx, [0, H], [{height}, W], real_shape=[{valid}, H], pad_value=0.0)
gate = asctile.copy_in(gx, [0, 0], [{height}, W], real_shape=[{valid}, H], pad_value=0.0)
gate_f = gate.to(asctile.float32)
out = (gate_f / (1.0 + asctile.exp(-gate_f))) * value.to(asctile.float32)
asctile.copy_out(out.to(gate.dtype), gy_tile, [0, 0], real_shape=[{valid}, H])
"""


def render(source, mode):
    if sha256(source.encode()).hexdigest() != PYTHON_DIGEST or mode not in ("fixed", "tight"):
        raise ValueError("expected the qualified W74 package and known extent control")
    start = source.index("        tile_index = b\n", source.index("def single_wave_fp32("))
    end = source.index("\n\n#", start)
    old = source[start:end]
    if old.count("asctile.copy_in(") != 2 or old.count("asctile.copy_out(") != 1:
        raise ValueError("unexpected padded short-row branch")
    # On an exact multiple the static remainder guard removes the tail body.
    # Both controls use identical full tiles and the same absolute GM intervals.
    parts = ["        r = b * T\n", "        if b < R // T:\n", indent(tile("T", "T"), " " * 12),
             "        else:\n", "            if R % T != 0:\n",
             indent(tile("R % T", "R % T" if mode == "tight" else "T"), " " * 16)]
    return source[:start] + "".join(parts) + source[end:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("fixed", "tight"))
    args = parser.parse_args()
    original = args.package / "swi_glu.py"
    src = render(original.read_text(), args.mode)
    compile(src, original.name, "exec")
    args.output.mkdir()
    (args.output / original.name).write_text(src)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
