"""Compare one-wave FP32 row tiles with independently owned residual columns."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile
from textwrap import indent

W61_SHA = 'c14e3e3a937a593d16a46e77625c416c42b21dbfac0af559f8c64a952143a1eb'


def render(source, group):
    if sha256(source.encode()).hexdigest() != W61_SHA:
        raise ValueError("expected the unedited W61 module")
    if group not in (1, 2, 4):
        raise ValueError("expected the same owner or a twofold/fourfold fringe height")
    start = source.index('@asctile.jit', source.index('def single_wave_fp32('))
    # Distinct statically shaped branches let the compiler size vector arithmetic
    # for the main columns independently of the smaller padded residual.
    kernel = '''@asctile.jit(reuse_alloc=2)
def column_fringe(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                  R: asctile.ConstExpr[int], H: asctile.ConstExpr[int],
                  T: asctile.ConstExpr[int], P: asctile.ConstExpr[int],
                  F: asctile.ConstExpr[int], N: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [R, 2 * H], l2_cache=False)
    gy = asctile.global_tensor(y, [R, H])
    b = asctile.block_idx()
'''
    main = '''first = b * T
valid = min(T, R - first)
gate = asctile.copy_in(gx, [first, 0], [T, P], real_shape=[valid, P], pad_value=0.0)
value = asctile.copy_in(gx, [first, H], [T, P], real_shape=[valid, P], pad_value=0.0)
out = (gate / (1.0 + asctile.exp(-gate))) * value
asctile.copy_out(out, gy, [first, 0], real_shape=[valid, P])
'''
    fringe_rank = 'b' if group == 1 else '(b - N)'
    fringe = f'''first = {fringe_rank} * F
valid = min(F, R - first)
width = (H - P + 7) // 8 * 8
gate = asctile.copy_in(gx, [first, P], [F, width], real_shape=[valid, H - P], pad_value=0.0)
value = asctile.copy_in(gx, [first, H + P], [F, width], real_shape=[valid, H - P], pad_value=0.0)
out = (gate / (1.0 + asctile.exp(-gate))) * value
asctile.copy_out(out, gy, [first, P], real_shape=[valid, H - P])
'''
    if group == 1:
        kernel += indent(main + fringe, '    ')
    else:
        kernel += '    if b < N:\n' + indent(main, '        ')
        kernel += '    else:\n' + indent(fringe, '        ')
    kernel += '\n\n'
    original = '                single_wave_fp32[tiles](input, out, R, H, height, width)'
    if source.count(original) != 1:
        raise ValueError("missing matched host branch")
    extra = '' if group == 1 else ' + (R + fringe_height - 1) // fringe_height'
    aligned_owners = 'height * H % 8 == 0' if group == 1 else 'H % 8 == 0'
    launch = f'''                bulk = H // 64 * 64
                fringe_height = {group} * height
                total = tiles{extra}
                fringe_width = (H - bulk + 7) // 8 * 8
                if (0 < bulk < H and total <= aivs
                        and {aligned_owners}
                        and 28 * max(height * bulk, fringe_height * fringe_width) <= usable_ub):
                    column_fringe[total](input, out, R, H, height, bulk, fringe_height, tiles)
                else:
    {original}'''
    return (source[:start] + kernel + source[start:]).replace(original, launch)


if __name__ == '__main__':
    base = Path('base/swi_glu.py')
    for group in (1, 2, 4):
        package = Path(f'fringe{group}')
        code = render(base.read_text(), group)
        compile(code, str(package), 'exec')
        package.mkdir()
        (package / 'swi_glu.py').write_text(code)
        (package / '__init__.py').touch()
        copyfile(base.with_name('_pyasc_runtime.py'), package / '_pyasc_runtime.py')
