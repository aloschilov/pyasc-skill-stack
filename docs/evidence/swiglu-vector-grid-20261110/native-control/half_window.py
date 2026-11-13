"""Factor input pointer origin and compiler VF fusion on an unsplit row tile."""
from pathlib import Path
from shutil import copyfile

from local_dma_window import render as make_window


def render(base, rows, half, height, separate, fused):
    source = make_window(base, rows, half, height, "wholefused" if fused else "whole")
    if not separate:
        return source
    lines = []
    matches = 0
    for line in source.splitlines():
        if line.startswith("        x0 = asctile.global_tensor("):
            lines.append(line)
            # Only the first H elements of each 2H row are accessed through
            # this descriptor, including the final row of the input.
            line = line.replace("x0 = ", "xv = ").replace(" * H, [", " * H + H, [")
            matches += 1
        if line.startswith("        value0 = asctile.copy_in(x0, [0, H],"):
            line = line.replace("x0, [0, H]", "xv, [0, 0]")
            matches += 1
        lines.append(line)
    if matches != 4:
        raise ValueError("expected two fixed input windows and two value copies")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    base = Path("base/swi_glu.py").read_text()
    for case, rows, half, height in ((11, 273, 509, 8), (13, 143, 536, 4)):
        for separate in (False, True):
            for fused in (False, True):
                code = render(base, rows, half, height, separate, fused)
                dest = Path(f"v{case}_{int(separate)}_{int(fused)}")
                compile(code, str(dest), "exec")
                dest.mkdir()
                (dest / "swi_glu.py").write_text(code)
                (dest / "__init__.py").touch()
                copyfile("base/_pyasc_runtime.py", dest / "_pyasc_runtime.py")
