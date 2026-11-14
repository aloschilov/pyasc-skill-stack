"""Compare looped vs static VF and source L2 on the same padded two-case DMA."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from packed_row_vf import register_body


def loop_body(height, pitch):
    return f"""using namespace AscendC::Reg;
auto* x = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());
auto* y = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());
auto mask = CreateMask<uint8_t>();
for (uint16_t row = 0; row < {height}; ++row) {{
    for (uint16_t part = 0; part < {(pitch + 63) // 64}; ++part) {{
        uint16_t col = part * 64 + 64 <= {pitch} ? part * 64 : {pitch - 64};
        RegTensor<float> gate, value, result;
        LoadAlign(gate, x + 2 * row * {pitch} + col);
        LoadAlign(value, x + (2 * row + 1) * {pitch} + col);
        Muls(result, gate, -1.0f, mask);
        Exp(result, result, mask);
        Adds(result, result, 1.0f, mask);
        Div(result, gate, result, mask);
        Mul(result, result, value, mask);
        StoreAlign(y + row * {pitch} + col, result, mask);
    }}
}}"""


def render(source, height, half, loop, cache):
    if sha256(source.encode()).hexdigest() != '18ea91c301937c1f8c901f6d9fa529e952c8cea40c06a4458f322b51113fd607':
        raise ValueError('expected the unedited a7f4048 source')
    width = (half + 7) // 8 * 8
    vf = loop_body(height, width) if loop else register_body(height, width, 1)
    start = source.index('def single_wave_fp32(')
    a = source.index('    gx =', start)
    z = source.index('\n\n@asctile.jit', a)
    marker = 'single_wave_fp32[blocks](input, out, R, H, height, width)'
    if source.count(marker) != 1:
        raise ValueError('unexpected public launch')
    fn = source[start:a].replace('def single_wave_fp32(', 'def packed_vector(')
    fn += f"""    gx = asctile.global_tensor(x, [2 * R, H], l2_cache={cache})
    gy = asctile.global_tensor(y, [R, H])
    b = asctile.block_idx()
    first = b * T
    valid = min(T, R - first)
    packed = asctile.copy_in(gx, [2 * first, 0], [2 * T, W],
                             real_shape=[2 * valid, H], pad_value=0.0)
    out = asctile.inline_vf({vf!r}, [T, W], asctile.float32, [packed])
    asctile.copy_out(out, gy, [first, 0], real_shape=[valid, H])
"""
    source = source[:z] + '\n\n@asctile.jit(reuse_alloc=2)\n' + fn + source[z:]
    return source.replace(marker, f"""(packed_vector if (H, height, width) == ({half}, {height}, {width})
                     else single_wave_fp32)[blocks](input, out, R, H, height, width)""")


if __name__ == '__main__':
    s = Path('base/swi_glu.py').read_text()
    for case, h, t in [(11,509,8),(13,536,4)]:
        for loop, cache in [(False,False),(False,True),(True,False),(True,True)]:
            name = f"v{case}{'loop' if loop else 'static'}{'cache' if cache else 'direct'}"
            dest = Path(name)
            code = render(s,t,h,loop,cache)
            compile(code,name,'exec')
            dest.mkdir()
            (dest/'__init__.py').touch()
            (dest/'swi_glu.py').write_text(code)
            copyfile('base/_pyasc_runtime.py',dest/'_pyasc_runtime.py')
