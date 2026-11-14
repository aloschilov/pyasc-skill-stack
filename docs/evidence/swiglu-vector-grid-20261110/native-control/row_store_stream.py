"""Compare padded and continuous register stores with the same packed input DMA."""
import argparse
import ast
import hashlib
from pathlib import Path
import shutil


def vector_body(height, half, pitch, mode):
    dense = mode != "padded"
    state = "auto* out = dst;\nUnalignRegForStore stream;"
    begin = state if mode == "tile" else ""
    row_begin = state.replace("= dst;", f"= dst + r * {half};") if mode == "row" else ""
    store = ("StoreUnAlign(out, acc, stream, count);" if dense
             else f"StoreAlign(dst + r * {pitch} + c * 64, acc, mask);")
    flush = "StoreUnAlignPost(out, stream, 0);"
    return f"""using namespace AscendC::Reg;
auto* src = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
auto* dst = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
{begin}
for (uint16_t r = 0; r < {height}; ++r) {{
    {row_begin}
    for (uint16_t c = 0; c < {(half+63)//64}; ++c) {{
        uint16_t count = c == {(half-1)//64} ? {(half-1)%64+1} : 64;
        uint32_t active = count;
        auto mask = UpdateMask<float>(active);
        RegTensor<float> gate, value, acc;
        LoadAlign(gate, src + 2*r*{pitch} + c*64);
        LoadAlign(value, src + (2*r+1)*{pitch} + c*64);
        Muls(acc, gate, -1.0f, mask);
        Exp(acc, acc, mask);
        Adds(acc, acc, 1.0f, mask);
        Div(acc, gate, acc, mask);
        Mul(acc, acc, value, mask);
        {store}
    }}
    {flush if mode == "row" else ""}
}}
{flush if mode == "tile" else ""}"""


def render(source, height, half, mode):
    if hashlib.sha256(source.encode()).hexdigest() != "1f0e5562b4a41228792ea34b61e8feaa7e295508d78986b670810d19958a0ff6":
        raise ValueError("expected the qualified W65 source")
    pitch = (half + 7) // 8 * 8
    if height < 1 or half < 1 or pitch % 64 or height * half % 8 or mode not in ("padded", "row", "tile"):
        raise ValueError("require aligned full-register loads and burst-aligned tile ownership")
    body = vector_body(height, half, pitch, mode)
    func, = (f for f in ast.parse(source).body if getattr(f, "name", "") == "row_single_wave_fp32")
    lines = source.splitlines(keepends=True)
    head = "".join(lines[func.decorator_list[0].lineno-1:func.body[0].lineno-1])
    head = head.replace("row_single_wave_fp32", "row_store_stream")
    dense = mode != "padded"
    out_shape = "[R * H]" if dense else "[R, H]"
    tile = "[T * W]" if dense else "[T, W]"
    offset = "[r * H]" if dense else "[r, 0]"
    real = "[valid * H]" if dense else "[valid, H]"
    lines.insert(func.decorator_list[0].lineno-1, head + f"""    gx = asctile.global_tensor(x, [2 * R, H], l2_cache=False)
    gy = asctile.global_tensor(y, {out_shape})
    r = asctile.block_idx() * T
    valid = min(T, R - r)
    packed = asctile.copy_in(gx, [2 * r, 0], [2 * T, W], real_shape=[2 * valid, H], pad_value=0.0)
    out = asctile.inline_vf({body!r}, {tile}, asctile.float32, [packed])
    asctile.copy_out(out, gy, {offset}, real_shape={real})


""")
    source = "".join(lines)
    marker = "            row_single_wave_fp32[launch_grid](input, out, R, H, height, width)"
    if source.count(marker) != 1:
        raise ValueError("unexpected single-wave dispatch")
    return source.replace(marker, f"            if (height, H) == ({height}, {half}):\n"
                          "                row_store_stream[launch_grid](input, out, R, H, height, width)\n"
                          "            else:\n    " + marker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("height", type=int)
    parser.add_argument("half", type=int)
    parser.add_argument("mode", choices=("padded", "row", "tile"))
    args = parser.parse_args()
    code = render((args.package / "swi_glu.py").read_text(), args.height, args.half, args.mode)
    compile(code, "swi_glu.py", "exec")
    args.dest.mkdir()
    (args.dest / "swi_glu.py").write_text(code)
    (args.dest / "__init__.py").touch()
    shutil.copyfile(args.package / "_pyasc_runtime.py", args.dest / "_pyasc_runtime.py")
