"""Place the GM descriptor at the stage origin so its logical bounds are static."""
import re
from pathlib import Path
from shutil import copyfile

from tile_stage_order import render as make_tile_control


def render(base, rows, half, height, mode):
    fused = mode.endswith("fused")
    order = mode.removesuffix("fused")
    source = make_tile_control(base, rows, half, height, order)
    start = source.index("def tile_stages(")
    end = source.index("\n\n@asctile.jit", start)
    fn = source[start:end]
    fn = fn.replace("    src = asctile.global_tensor(x, [R, 2 * H], l2_cache=False)\n", "")
    fn = fn.replace("    dst = asctile.global_tensor(y, [R, H])\n", "")
    lines = []
    for line in fn.splitlines():
        gate = re.fullmatch(r"        gate(\d+) = asctile.copy_in\(src, \[(.+), 0\], \[(\d+), W\], (.*)\)", line)
        if gate:
            index, row, count, _ = gate.groups()
            lines.append(f"        x{index} = asctile.global_tensor(x + 2 * ({row}) * H, [{count}, 2 * H], l2_cache=False)")
            line = line.replace(f"src, [{row},", f"x{index}, [0,")
        value = re.fullmatch(r"        value(\d+) = asctile.copy_in\(src, \[(.+?), H\], (.*)", line)
        if value:
            index, row, _ = value.groups()
            line = line.replace(f"src, [{row},", f"x{index}, [0,")
        store = re.fullmatch(r"        asctile.copy_out\(out(\d+), dst, \[(.+), 0\], real_shape=\[(\d+), H\]\)", line)
        if store:
            index, row, count = store.groups()
            lines.append(f"        y{index} = asctile.global_tensor(y + ({row}) * H, [{count}, H])")
            line = line.replace(f"dst, [{row},", f"y{index}, [0,")
        lines.append(line)
    emitted="\n".join(lines)
    if "copy_in(src" in emitted or ", dst," in emitted:
        raise ValueError("stage access was not rebased")
    source = (source[:start]+emitted+source[end:]).replace("tile_stages","local_windows")
    if fused:
        source=source.replace("@asctile.jit(reuse_alloc=2)\ndef local_windows(",
                              "@asctile.jit(reuse_alloc=2, vf_fusion=True)\ndef local_windows(")
    return source


if __name__=="__main__":
    base=Path("base/swi_glu.py").read_text()
    for case,rows,half,height in [(11,273,509,8),(13,143,536,4)]:
        for mode in ("whole","prefetch","prefetchfused"):
            code=render(base,rows,half,height,mode)
            compile(code,"swi_glu.py","exec")
            dest=Path(f"v{case}{mode}")
            dest.mkdir()
            (dest/"swi_glu.py").write_text(code)
            (dest/"__init__.py").touch()
            copyfile("base/_pyasc_runtime.py",dest/"_pyasc_runtime.py")
