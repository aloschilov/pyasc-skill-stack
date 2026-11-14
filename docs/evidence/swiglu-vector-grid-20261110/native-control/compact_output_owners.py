"""Partition an unaligned FP32 row group at a burst-aligned flat output offset."""
import argparse
import ast
import hashlib
from itertools import groupby
from pathlib import Path
from shutil import copyfile
from textwrap import indent

W65 = "1f0e5562b4a41228792ea34b61e8feaa7e295508d78986b670810d19958a0ff6"


def intervals(rows, half, owners):
    """Split one nonempty group at the nearest 32-byte FP32 output boundary."""
    length = rows * half
    if owners == 1:
        return [(0, length)]
    middle = ((length + 8) // 16) * 8
    return [(0, middle), (middle, length)] if 0 < middle < length else [(0, length)]


def emit_interval(half, begin, end):
    # Load only rows intersecting this owner's interval. Input reads can overlap.
    first, last = begin // half, (end - 1) // half
    rows = last - first + 1
    pitch = (half + 63) // 64 * 64
    parts = []
    for r in range(first, last + 1):
        lo = max(0, begin - r * half)
        hi = min(half, end - r * half)
        parts.append((r-first, lo, hi))
    vf = """using namespace AscendC::Reg;
auto* src = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
auto* out = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
UnalignRegForStore stream;
"""
    # Combine consecutive rows with the same live column interval.
    for (lo, hi), group in groupby(parts, key=lambda part: part[1:]):
        batch = list(group)
        rb = batch[0][0]
        row_stop = batch[-1][0] + 1
        vf += f"""for (uint16_t r = {rb}; r < {row_stop}; ++r) {{
    auto* gp = src + (2*r)*{pitch} + {lo};
    auto* vp = src + (2*r+1)*{pitch} + {lo};
"""
        unaligned = lo % 8 != 0
        if unaligned:
            # LoadUnAlign may fetch one further aligned register.
            # Inactive lanes may enter the next local row. Both the active
            # columns and the full physical access must remain in bounds.
            last_value = (2 * (row_stop - 1) + 1)*pitch + lo
            if last_value + ((hi-lo+63)//64+1)*64 > 2*rows*pitch:
                raise ValueError("terminal unaligned read needs a UB guard")
            vf += """    UnalignRegForLoad gs, vs;
    LoadUnAlignPre(gs, gp);
    LoadUnAlignPre(vs, vp);
"""
        vf += f"""    for (uint16_t c = 0; c < {hi-lo}; c += 64) {{
        uint16_t count = {hi-lo} - c < 64 ? {hi-lo} - c : 64;
        uint32_t active = count;
        auto mask = UpdateMask<float>(active);
        RegTensor<float> gate, value, acc;
"""
        vf += ("        LoadUnAlign(gate, gs, gp, 64);\n        LoadUnAlign(value, vs, vp, 64);\n" if unaligned
               else "        LoadAlign(gate, gp + c);\n        LoadAlign(value, vp + c);\n")
        vf += """        Muls(acc, gate, -1.0f, mask);
        Exp(acc, acc, mask);
        Adds(acc, acc, 1.0f, mask);
        Div(acc, gate, acc, mask);
        Mul(acc, acc, value, mask);
        StoreUnAlign(out, acc, stream, count);
    }
}
"""
    vf += "StoreUnAlignPost(out, stream, 0);"
    size = end-begin
    return f"""src = asctile.global_tensor(x + (r + {first}) * 2 * H, [{2*rows}, H], l2_cache=False)
dst = asctile.global_tensor(y + r * H + {begin}, [{size}])
packed = asctile.copy_in(src, [0, 0], [{2*rows}, {pitch}], real_shape=[{2*rows}, H], pad_value=0.0)
res = asctile.inline_vf({vf!r}, [{(size+7)//8*8}], asctile.float32, [packed])
asctile.copy_out(res, dst, [0], real_shape=[{size}])
"""


def render(source, rows, half, height, owners):
    if hashlib.sha256(source.encode()).hexdigest() != W65 or min(rows, half, height) < 1 or owners not in (1, 2):
        raise ValueError("expected W65 and one or two owners per row group")
    if height*half % 8:
        raise ValueError("full row group must end on a burst boundary")
    groups, tail = divmod(rows, height)
    full = intervals(height, half, owners)
    # One owner finishes the entire partial last group.
    grid = groups*len(full) + bool(tail)
    fn = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    header = lines[fn.decorator_list[0].lineno - 1:fn.body[0].lineno - 1]
    kernel = "".join(header).replace("row_single_wave_fp32", "compact_owners")
    kernel += f"    b = asctile.block_idx()\n    r = (b // {len(full)}) * T\n"
    branches = []
    if tail:
        branches.append((f"b == {groups * len(full)}", emit_interval(half, 0, tail * half)))
    for idx, (begin, end) in enumerate(full):
        branches.append((f"b % {len(full)} == {idx}", emit_interval(half, begin, end)))
    for i, (condition, body) in enumerate(branches):
        if len(branches) == 1:
            kernel += indent(body, "    ")
        else:
            keyword = "if" if i == 0 else "elif"
            branch = f"    {keyword} {condition}:\n" if i + 1 < len(branches) else "    else:\n"
            kernel += branch + indent(body, "        ")
    lines.insert(fn.decorator_list[0].lineno - 1, kernel + "\n\n")
    src = "".join(lines)
    marker = "            row_single_wave_fp32[launch_grid](input, out, R, H, height, width)"
    if src.count(marker) != 1:
        raise ValueError("unexpected W65 host routing")
    # The incoming grid already uses the framework's stream when on the NPU.
    extra = f"""            if (R, H, height) == ({rows}, {half}, {height}) and aivs >= {grid}:
                owner_grid = ({grid}, launch_grid[1]) if isinstance(launch_grid, tuple) else {grid}
                compact_owners[owner_grid](input, out, R, H, height, width)
            else:
"""
    return src.replace(marker, extra + "    " + marker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    for field in ("rows", "half", "height", "owners"):
        parser.add_argument(field, type=int)
    arg = parser.parse_args()
    source = (arg.package / "swi_glu.py").read_text()
    code = render(source, arg.rows, arg.half, arg.height, arg.owners)
    compile(code, "swi_glu.py", "exec")
    arg.output.mkdir()
    (arg.output / "swi_glu.py").write_text(code)
    (arg.output / "__init__.py").touch()
    copyfile(arg.package / "_pyasc_runtime.py", arg.output / "_pyasc_runtime.py")
