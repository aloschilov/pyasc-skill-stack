"""Compare a single flat register loop over planar and row-interleaved padded loads."""
import argparse
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from packed_math_cost import W81


def render(source, slab, row_loop=False):
    if sha256(source.encode()).hexdigest() != W81:
        raise ValueError("expected W81")
    start = source.index("def row_single_wave_fp32(")
    end = source.index("    if W == H:", source.index("    valid = min(T, R - r)", start))
    header = source[start:end].replace("def row_single_wave_fp32(", "def padded_register_rows(")
    if slab:
        reads = """    gx = asctile.global_tensor(x, [2 * R, H], l2_cache=False)
    slab = asctile.copy_in(gx, [2 * r, 0], [2 * T, W], real_shape=[2 * valid, H], pad_value=0.0)
"""
        inputs = "[slab]"
        pointers = "auto* src = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());"
        loads = """uint32_t input_offset = offset + (i / {W // 64}) * {W};
            LoadAlign(gate, src + input_offset);
            LoadAlign(value, src + input_offset + {W});"""
    else:
        reads = """    gx = asctile.global_tensor(x, [R, 2 * H], l2_cache=False)
    g = asctile.copy_in(gx, [r, 0], [T, W], real_shape=[valid, H], pad_value=0.0)
    v = asctile.copy_in(gx, [r, H], [T, W], real_shape=[valid, H], pad_value=0.0)
"""
        inputs = "[g, v]"
        pointers = """auto* g = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
        auto* v = reinterpret_cast<__ubuf__ float*>($2.GetPhyAddr());"""
        loads = """LoadAlign(gate, g + offset);
            LoadAlign(value, v + offset);"""
    vf = '''    result = asctile.inline_vf(f"""
        using namespace AscendC::Reg;
        POINTERS
        auto* dst = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
        auto mask = CreateMask<uint8_t>();
        for (uint16_t i = 0; i < {T * W // 64}; ++i) {{
            uint32_t offset = i * 64;
            RegTensor<float> gate, value, acc;
            LOADS
            Muls(acc, gate, -1.0f, mask);
            Exp(acc, acc, mask);
            Adds(acc, acc, 1.0f, mask);
            Div(acc, gate, acc, mask);
            Mul(acc, acc, value, mask);
            StoreAlign(dst + offset, acc, mask);
        }}
    """, [T, W], asctile.float32, INPUTS)
    gy = asctile.global_tensor(y, [R, H])
    asctile.copy_out(result, gy, [r, 0], real_shape=[valid, H])
'''.replace("POINTERS", pointers).replace("LOADS", loads).replace("INPUTS", inputs)
    if row_loop:
        # Expose the two constant strides to the vector address generator.
        flat_head = "        for (uint16_t i = 0; i < {T * W // 64}; ++i) {{\n"
        first, body = vf.split(flat_head)
        body, after = body.split("        }}\n", 1)
        body = body.replace("uint32_t offset = i * 64;", "uint32_t offset = row * {W} + col * 64;")
        body = body.replace("offset + (i / {W // 64}) * {W}", "2 * row * {W} + col * 64")
        nested = ("        for (uint16_t row = 0; row < {T}; ++row) {{\n"
                  "            for (uint16_t col = 0; col < {W // 64}; ++col) {{\n")
        vf = first + nested + "".join("    " + line for line in body.splitlines(True))
        vf += "            }}\n        }}\n" + after
    insert = source.index("@asctile.jit", end)
    kernel = "@asctile.jit(reuse_alloc=2, vf_fusion=True)\n" + header + reads + vf
    dispatch = "            row_single_wave_fp32[grid](x, out, rows, half, height, width)"
    select = (
        "            if width != half and width % 64 == 0 and 12 * height * width <= usable_ub:\n"
        "                padded_register_rows[grid](x, out, rows, half, height, width)\n"
        "                return out\n"
    )
    assert source.count(dispatch)==1
    return (source[:insert] + kernel + "\n\n" + source[insert:]).replace(dispatch, select + dispatch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--rows",action="store_true",help="also generate affine row and column loops")
    args = parser.parse_args()
    origin = (args.package / "swi_glu.py").read_text()
    modes = ("planes","slab","planes_rows","slab_rows") if args.rows else ("planes","slab")
    for mode in modes:
        root = args.output / mode
        root.mkdir(parents=True)
        text = render(origin, slab=mode.startswith("slab"), row_loop=mode.endswith("_rows"))
        compile(text, str(root/"swi_glu.py"), "exec")
        (root/"swi_glu.py").write_text(text)
        (root/"__init__.py").touch()
        copyfile(args.package/"_pyasc_runtime.py",root/"_pyasc_runtime.py")
