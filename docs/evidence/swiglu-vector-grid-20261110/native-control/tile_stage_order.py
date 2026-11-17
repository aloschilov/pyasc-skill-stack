"""Change stage issue order using public ranked copies and tile arithmetic."""
from pathlib import Path
from shutil import copyfile

from row_stage_order import render as register_control


def tile_stage(rows, height, mode):
    def read(i, count, start):
        return [f"{name}{i} = asctile.copy_in(src, [{start}, {col}], [{count}, W], "
                f"real_shape=[{count}, H], pad_value=0.0)"
                for name,col in (("gate","0"),("value","H"))]

    def write(i, count, start):
        return [f"out{i} = (gate{i} / (1.0 + asctile.exp(-gate{i}))) * value{i}",
                f"asctile.copy_out(out{i}, dst, [{start}, 0], real_shape=[{count}, H])"]

    lines=[]
    for count,first,header in [(rows % height, "((R // T) * T)", "    if b == R // T:"),
                               (height, "(b * T)", "    else:")]:
        part=height//2
        if mode=="whole" or count<=part:
            body=read(0,count,first)+write(0,count,first)
        else:
            second=f"{first} + {part}"
            a,b=read(0,part,first),read(1,count-part,second)
            c,d=write(0,part,first),write(1,count-part,second)
            body=a+b+c+d if mode=="prefetch" else a+c+b+d
        lines.extend([header,*["        "+stmt for stmt in body]])
    return "\n".join(lines)


def render(base, rows, half, height, mode):
    # Reuse the public dispatch and exact terminal extent from the matched control.
    source=register_control(base,rows,half,height,mode)
    start=source.index("def staged_rows(")
    begin=source.index("    src =",start)
    end=source.index("\n\n@asctile.jit",begin)
    body="""    src = asctile.global_tensor(x, [R, 2 * H], l2_cache=False)
    dst = asctile.global_tensor(y, [R, H])
    b = asctile.block_idx()
"""
    return (source[:begin]+body+tile_stage(rows,height,mode)+source[end:]).replace(
        "staged_rows", "tile_stages")


if __name__=="__main__":
    base=Path("base/swi_glu.py").read_text()
    for case,rows,half,height in [(11,273,509,8),(13,143,536,4)]:
        for mode in ("whole","serial","prefetch"):
            code=render(base,rows,half,height,mode)
            compile(code,"swi_glu.py","exec")
            dest=Path(f"v{case}{mode}")
            dest.mkdir()
            (dest/"swi_glu.py").write_text(code)
            (dest/"__init__.py").touch()
            copyfile("base/_pyasc_runtime.py",dest/"_pyasc_runtime.py")
