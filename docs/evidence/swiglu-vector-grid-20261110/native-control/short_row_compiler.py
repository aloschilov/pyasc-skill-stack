"""Cross allocation reuse with VF fusion on the original short-row FP32 kernel."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from bf16_slab_fusion import W72
from lower_planar_nddma import lower_fp32
from row_exact_divisor import W69


OPTIONS = {
    "fresh": "reuse_alloc=0",
    "fused": "reuse_alloc=2, vf_fusion=True",
    "fresh_fused": "reuse_alloc=0, vf_fusion=True",
}


FUSED_STORAGE = {
    "legacy": "reuse_alloc=1, vf_fusion=True",
    "modern_pipe": "reuse_alloc=2, vf_fusion=True, static_alloc=False",
    "legacy_pipe": "reuse_alloc=1, vf_fusion=True, static_alloc=False",
}


def render(source, mode):
    current = mode in FUSED_STORAGE
    expected = W72 if current else W69
    if sha256(source.encode()).hexdigest() != expected:
        raise ValueError(f"expected the unchanged {'W72' if current else 'W69'} package")
    old = "reuse_alloc=2, vf_fusion=True" if current else "reuse_alloc=2"
    options = FUSED_STORAGE[mode] if current else OPTIONS[mode]
    anchor = f"@asctile.jit({old})\ndef row_single_wave_fp32("
    if source.count(anchor) != 1:
        raise ValueError("unexpected kernel decorator")
    return source.replace(anchor, anchor.replace(old, options))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=(*OPTIONS, *FUSED_STORAGE))
    parser.add_argument("--lower", type=Path, metavar="WORKERS")
    args = parser.parse_args()
    code = render((args.package / "swi_glu.py").read_text(), args.mode)
    compile(code, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(code)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
    if args.lower:
        for rows, half, height, pitch in ((273, 509, 6, 512), (143, 536, 4, 536)):
            lower_fp32(args.output.resolve(), args.lower.resolve(), "row_single_wave_fp32",
                       (rows, half, height, pitch), args.output / f"h{half}.cpp")
