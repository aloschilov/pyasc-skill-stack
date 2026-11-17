"""Compare VF fusion on two BF16 gather kernels with unchanged indices and DMA."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from lower_planar_nddma import lower_typed


W72 = "ab029b42633521f2b68bf118e4157893bb45e9c28e3dc4e01dd801d8409fcc86"
GEOMETRIES = {"two_wave_swiglu": (133221, 7, 928), "ordinary_slab": (1000003, 1, 4096, 2)}


def render(source, fused):
    if sha256(source.encode()).hexdigest() != W72:
        raise ValueError("expected the unchanged independent W72 candidate")
    for name in GEOMETRIES:
        anchor = f"@asctile.jit(reuse_alloc=2)\ndef {name}("
        if source.count(anchor) != 1:
            raise ValueError(f"unexpected decorator for {name}")
        if fused:
            source = source.replace(anchor, anchor.replace("reuse_alloc=2", "reuse_alloc=2, vf_fusion=True"))
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mode", choices=("base", "fused"))
    parser.add_argument("--lower", type=Path, metavar="WORKERS")
    args = parser.parse_args()
    code = render((args.package / "swi_glu.py").read_text(), args.mode == "fused")
    compile(code, "swi_glu.py", "exec")
    args.output.mkdir()
    (args.output / "swi_glu.py").write_text(code)
    (args.output / "__init__.py").touch()
    copyfile(args.package / "_pyasc_runtime.py", args.output / "_pyasc_runtime.py")
    if args.lower:
        for name, geometry in GEOMETRIES.items():
            lower_typed(args.output.resolve(), args.lower.resolve(), name,
                        ("bfloat16", "bfloat16", "int16"), geometry, args.output / f"{name}.cpp")
