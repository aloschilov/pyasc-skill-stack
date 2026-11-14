"""Use the wheel's public launch grid/stream tuple for one resident-input control."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil

from row_math_cost import W62


def render(source):
    if hashlib.sha256(source.encode()).hexdigest() != W62:
        raise ValueError("expected W62")
    sig = "row_single_wave_fp32[blocks](input, out, R, H, height, W)"
    old = "                " + sig
    if source.count(old) != 1:
        raise ValueError("expected exactly one single-wave dispatch")
    s = source.replace("import torch\n", "import ctypes\nimport torch\n", 1)
    # rt.Stream is ctypes.c_void_p. Passing a bare Python integer could
    # truncate the stream address at the runtime FFI boundary.
    s = s.replace(old, """                if input.device.type == "npu":
                    stream = ctypes.c_void_p(torch.npu.current_stream(input.device).npu_stream)
                    row_single_wave_fp32[blocks, stream](input, out, R, H, height, W)
                else:
    """ + old)
    ast.parse(s)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    s = render((args.base / "swi_glu.py").read_text())
    args.output.mkdir()
    (args.output / "__init__.py").touch()
    (args.output / "swi_glu.py").write_text(s)
    shutil.copyfile(args.base / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
