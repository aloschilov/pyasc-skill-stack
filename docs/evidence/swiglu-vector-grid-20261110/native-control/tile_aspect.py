"""Compare taller, narrower tiles on the same single-wave SwiGLU grid."""
import hashlib
from pathlib import Path
from shutil import copyfile

src = Path('base/swi_glu.py').read_text()
expected = '75a622ff52e5dc69f72fd62d3ef5a9c90cab1dab7658010513ac08dd1b442635'
if hashlib.sha256(src.encode()).hexdigest() != expected:
    raise ValueError('This diagnostic requires the measured W52 module')
start = src.index('    r = tile_index * T', src.index('def row_single_wave('))
end = src.index('\n\n', start)
original = src[start:end]
call = 'row_single_wave[grid](input, out, R, H, height, width)'
assert src.count(call) == 1
for strips in (1, 2, 4):
    # Round the column pitch up to one 32-byte burst. At fixed grid, each
    # stripe is served by floor(N/S) or ceil(N/S) cores.
    setup = f"""    stripe = tile_index % {strips}
    rank = tile_index // {strips}
    owners = n // {strips} + min(1, max(0, n % {strips} - stripe))
    r = rank * (R // owners) + min(rank, R % owners)
    rows = R // owners + min(1, max(0, R % owners - rank))
    c = stripe * SW
    cols = min(SW, H - c)
"""
    fn = original[original.index('    if W > H:'):]
    fn = fn.replace('[r, 0]', '[r, c]').replace('[r, H]', '[r, H + c]')
    fn = fn.replace('[T, W]', '[A, SW]').replace('[rows, H]', '[rows, cols]')
    new = src[:start] + setup + fn + src[end:]
    a = new.index('    W: asctile.ConstExpr[int],', new.index('def row_single_wave('))
    new = new[:a] + '    A: asctile.ConstExpr[int],\n    SW: asctile.ConstExpr[int],\n' + new[a:]
    args = f'(R + grid // {strips} - 1) // (grid // {strips}), ((H + {strips} * 8 - 1) // ({strips} * 8)) * 8'
    new = new.replace(call, f'row_single_wave[grid](input, out, R, H, height, {args}, width)')
    out = Path(f's{strips}')
    out.mkdir()
    compile(new, str(out / 'swi_glu.py'), 'exec')
    (out / 'swi_glu.py').write_text(new)
    copyfile('base/_pyasc_runtime.py', out / '_pyasc_runtime.py')
