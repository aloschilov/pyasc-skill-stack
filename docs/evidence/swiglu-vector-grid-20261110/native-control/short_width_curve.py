"""Hold one-wave row ownership fixed across finite FP32 transfer widths."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from row_exact_divisor import W69
from row_order_neighbors import prepare


def render(source):
    if sha256(source.encode()).hexdigest() != W69:
        raise ValueError("expected the qualified W69 SwiGLU package")
    marker = "    # Two-wave BF16 multi-lane gather (before the ordinary slab)."
    if source.count(marker) != 1:
        raise ValueError("missing W69 dispatch entry")
    # Diagnostic-only admission. Every case uses the original compute JIT.
    begin = source.index(marker)
    after = source.find("\ndef ", begin)
    end = after if after >= 0 else len(source)
    return source[:begin] + """    if dtype != torch.float32:
        raise ValueError("FP32 width control requires FP32 input")
    height = 4 if H % 8 == 0 else 6
    width = (H + 7) // 8 * 8
    blocks = (R + height - 1) // height
    if blocks > aivs or 28 * width * height > usable_ub:
        raise ValueError("shape exceeds the single-wave control")
    grid = (blocks, _current_stream_handle(input)) if input.device.type == "npu" else blocks
    row_single_wave_fp32[grid](input, out, R, H, height, width)
    return out

""" + source[end:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("package", "public_task", "output"):
        parser.add_argument(arg, type=Path)
    opt = parser.parse_args()
    pkg = opt.output / "probe"
    pkg.mkdir()
    source = render((opt.package / "swi_glu.py").read_text())
    compile(source, "swi_glu.py", "exec")
    (pkg / "swi_glu.py").write_text(source)
    (pkg / "__init__.py").touch()
    copyfile(opt.package / "_pyasc_runtime.py", pkg / "_pyasc_runtime.py")
    for family in ("aligned-width", "padded-width"):
        prepare(opt.public_task, opt.output, family)
