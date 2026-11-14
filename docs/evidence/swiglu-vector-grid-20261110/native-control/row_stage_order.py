"""Replay two disjoint packed DMA stages through the public AscTile emitter."""
from pathlib import Path
from shutil import copyfile

from fp32_stream_remainder import SOURCE_SHA
from hashlib import sha256
from packed_vector_loop import loop_body


def emit_body(height, half, mode):
    pitch = (half + 7) // 8 * 8
    part = height // 2
    def read(i, count, row):
        return (f"p{i} = asctile.copy_in(src, [2 * ({row}), 0], [{2*count}, W], "
                f"real_shape=[{2*count}, H], pad_value=0.0)")

    def compute(i, count, row):
        vf = loop_body(count, pitch)
        return [f"v{i} = asctile.inline_vf({vf!r}, [{count}, W], asctile.float32, [p{i}])",
                f"asctile.copy_out(v{i}, dst, [{row}, 0], real_shape=[{count}, H])"]

    def stage(count, first):
        if mode == "whole" or count <= part:
            return [read(0, count, first), *compute(0, count, first)]
        second = f"{first} + {part}"
        a, b = read(0, part, first), read(1, count - part, second)
        ca, cb = compute(0, part, first), compute(1, count - part, second)
        return [a, b, *ca, *cb] if mode == "prefetch" else [a, *ca, b, *cb]

    # The two public diagnostics share an even height and have different tails.
    return stage


def render(source, rows, half, height, mode):
    if sha256(source.encode()).hexdigest() != SOURCE_SHA or mode not in ("whole","serial","prefetch"):
        raise ValueError("expected W59 and one of the three qualified controls")
    assert rows % height and height % 2 == 0
    stage = emit_body(height, half, mode)
    kernel = [
        "@asctile.jit(reuse_alloc=2)",
        "def staged_rows(x: asctile.GlobalAddress, y: asctile.GlobalAddress,",
        "                R: asctile.ConstExpr[int], H: asctile.ConstExpr[int],",
        "                T: asctile.ConstExpr[int], W: asctile.ConstExpr[int]):",
        "    src = asctile.global_tensor(x, [2 * R, H])",
        "    dst = asctile.global_tensor(y, [R, H])",
        "    b = asctile.block_idx()",
        "    if b == R // T:",
        *["        "+s for s in stage(rows % height, "((R // T) * T)")],
        "    else:",
        *["        "+s for s in stage(height, "(b * T)")],
    ]
    a = source.index("@asctile.jit",source.index("def single_wave_rows("))
    marker = "                single_wave_rows[blocks](input, out, rows, half, h, width, rotate)"
    assert source.count(marker)==1
    src = source[:a]+"\n".join(kernel)+"\n\n"+source[a:]
    return src.replace(marker, f"                if (rows, half, h) == ({rows}, {half}, {height}):\n"
                       "                    staged_rows[blocks](input, out, rows, half, h, width)\n"
                       "                else:\n    "+marker)


if __name__ == "__main__":
    base=Path("base/swi_glu.py").read_text()
    for case,rows,half,height in [(11,273,509,8),(13,143,536,4)]:
        for mode in ("whole","serial","prefetch"):
            dest=Path(f"v{case}{mode}")
            code=render(base,rows,half,height,mode)
            compile(code,str(dest),"exec")
            dest.mkdir()
            (dest/"swi_glu.py").write_text(code)
            (dest/"__init__.py").touch()
            copyfile("base/_pyasc_runtime.py",dest/"_pyasc_runtime.py")
