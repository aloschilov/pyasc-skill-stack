"""Compare tail ownership on the qualified medium single-row BF16 stream."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile


def make_kernel():
    return '''@asctile.jit(reuse_alloc=2)
def four_chunk_stream(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                      H: asctile.ConstExpr[int], T: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [2 * H])
    gy = asctile.global_tensor(y, [H])
    first = 4 * asctile.block_idx()
    for i in asctile.range(4, unroll_factor=2):
        offset = (first + i) * T
        valid = min(T, H - offset)
        gate = asctile.copy_in(gx, [offset], [T], real_shape=[valid], pad_value=0.0)
        value = asctile.copy_in(gx, [H + offset], [T], real_shape=[valid], pad_value=0.0)
        g = gate.to(asctile.float32)
        out = (g / (1.0 + asctile.exp(-g))) * value.to(asctile.float32)
        asctile.copy_out(out.to(gate.dtype), gy, [offset], real_shape=[valid])


'''


if __name__ == '__main__':
    base = Path('base/swi_glu.py').read_text()
    if sha256(base.encode()).hexdigest() != '30bc199acb49eecb1f262ca4afb5e16e3ce871f5f365954eb167107a61db412a':
        raise ValueError('expected the qualified 5ac7788 source')
    marker = '        tile = medium_bf16_two_stage_chunk(R, H, dtype, aivs, usable_ub)\n'
    if base.count(marker) != 1:
        raise ValueError('unexpected dispatcher')
    for name, launch in [('ceilflat','swiglu_flat[aivs](input, out, R, H, rounded, 2, True)'),
                         ('four','four_chunk_stream[aivs](input, out, H, rounded)')]:
        selection = '''        if tile > 0:
            rounded = ((H + 4 * aivs * 16 - 1) // (4 * aivs * 16)) * 16
            if 20 * rounded <= usable_ub and (4*aivs - 1)*rounded < H:
                ''' + launch + '''
                return out
'''
        code = base.replace(marker, marker + selection)
        if name == 'four':
            code = code.replace('def swi_glu(',make_kernel()+'def swi_glu(',1)
        compile(code, name, 'exec')
        dest = Path(name)
        dest.mkdir()
        (dest/'swi_glu.py').write_text(code)
        copyfile('base/_pyasc_runtime.py',dest/'_pyasc_runtime.py')
