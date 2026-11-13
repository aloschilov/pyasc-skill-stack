"""Compare static independent register chains on the same packed FP32 row DMA."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile


def register_body(height, width, group_rows):
    # Local half-row pitch is a multiple of 32 bytes. Keeping the last full
    # vector inside that pitch can recompute lanes, all with identical inputs.
    code = [
        'using namespace AscendC::Reg;',
        'auto* x = reinterpret_cast<__ubuf__ float*>($1.GetPhyAddr());',
        'auto* y = reinterpret_cast<__ubuf__ float*>($0.GetPhyAddr());',
        'auto mask = CreateMask<uint8_t>();',
    ]
    cols = [min(c, width - 64) for c in range(0, width, 64)]
    if min(cols) < 0 or any(c % 8 or c + 64 > width for c in cols):
        raise ValueError('invalid aligned register load')
    for first in range(0, height, group_rows):
        for col in cols:
            code.append('{')
            ids = list(range(first, min(first + group_rows, height)))
            for row in ids:
                code += [f'RegTensor<float> g{row}, v{row}, z{row};',
                         f'LoadAlign(g{row}, x + {2 * row * width + col});',
                         f'LoadAlign(v{row}, x + {(2 * row + 1) * width + col});']
            for op in ('Muls(z{r},g{r},-1.0f,mask);', 'Exp(z{r},z{r},mask);',
                       'Adds(z{r},z{r},1.0f,mask);', 'Div(z{r},g{r},z{r},mask);',
                       'Mul(z{r},z{r},v{r},mask);'):
                code.extend(op.format(r=row) for row in ids)
            for row in ids:
                code.append(f'StoreAlign(y + {row * width + col}, z{row}, mask);')
            code.append('}')
    return '\n'.join(code)


def build(base, rows, half, height, group):
    width = (half + 7) // 8 * 8
    vf = register_body(height, width, group)
    start = base.index('def single_wave_fp32_rows(')
    a = base.index('    gx =', start)
    geometry = base[base.index('    b = asctile.block_idx()', a):base.index('    if W == H:', base.index('    rows = min(T, R - r)', a))]
    body = f'''    gx = asctile.global_tensor(x, [2 * R, H], l2_cache=False)
    gy = asctile.global_tensor(y, [R, H])
{geometry}    slab = asctile.copy_in(gx, [2*r, 0], [2*T, W],
                           real_shape=[2*rows, H], pad_value=0.0)
    out = asctile.inline_vf({vf!r}, [T, W], asctile.float32, [slab])
    asctile.copy_out(out, gy, [r, 0], real_shape=[rows, H])
'''
    variant = base[start:a].replace('def single_wave_fp32_rows(', 'def packed_register_rows(') + body
    marker = '        if dtype == torch.float32 and tiles <= aivs:'
    if base.count(marker) != 1:
        raise ValueError('unexpected source dispatcher')
    override = f"""        if dtype == torch.float32 and tiles <= aivs and (R, H, height, width) == ({rows}, {half}, {height}, {width}):
            packed_register_rows[blocks](input, out, R, H, height, width)
            return out
"""
    return (base[:start] + variant + '\n\n@asctile.jit(reuse_alloc=2)\n'
            + base[start:].replace(marker, override + marker))


if __name__ == '__main__':
    base = Path('base/swi_glu.py').read_text()
    if sha256(base.encode()).hexdigest() != '30bc199acb49eecb1f262ca4afb5e16e3ce871f5f365954eb167107a61db412a':
        raise ValueError('expected the qualified 5ac7788 generation')
    for case, rows, half, height in [(13, 143, 536, 4), (11, 273, 509, 8)]:
        for group in (1, 4):
            dest = Path(f'p{case}g{group}')
            code = build(base, rows, half, height, group)
            compile(code, str(dest), 'exec')
            dest.mkdir()
            (dest/'swi_glu.py').write_text(code)
            copyfile('base/_pyasc_runtime.py', dest/'_pyasc_runtime.py')
