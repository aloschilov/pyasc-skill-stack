"""Compare two-wave BF16 slab height rounding and active grid independently."""
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

from column_fringe import W61_SHA


def render(source, mode):
    if sha256(source.encode()).hexdigest() != W61_SHA:
        raise ValueError("expected the unedited W61 module")
    if mode not in ('aligned', 'power', 'compact'):
        raise ValueError(mode)
    start = source.index('def two_wave_tile(')
    stop = source.index('    if 2 * lanes - 1', start)
    replacement = '''    if dtype != torch.bfloat16 or rows <= 0 or aivs <= 0 or not 1 < half <= 16:
        return 0
    waves = 2 * aivs
    needed = (rows + waves - 1) // waves
    base = (needed + 15) // 16 * 16
    if (rows + base - 1) // base != waves:
        return 0
'''
    if mode == 'aligned':
        replacement += '''    quantum = 64 // math.gcd(64, half)
    tile = (base + quantum - 1) // quantum * quantum
'''
    else:
        replacement += '''    tile = 1 << (base - 1).bit_length()
'''
    replacement += '''    count = (rows + tile - 1) // tile
    if not aivs < count <= waves:
        return 0
    lanes = tile * half
'''
    head = source.index('    if dtype', start)
    source = source[:head] + replacement + source[stop:]
    if mode == 'compact':
        call = '            two_wave_swiglu[aivs](input, out, idx, R, H, tw)'
        if source.count(call) != 1:
            raise ValueError('missing two-wave launch')
        source = source.replace(call, '''            count = (R + tw - 1) // tw
            grid = min(aivs, 2 * ((count + 3) // 4))
            two_wave_swiglu[grid](input, out, idx, R, H, tw)''')
    return source


if __name__ == '__main__':
    base = Path('base/swi_glu.py')
    for mode in ('aligned', 'power', 'compact'):
        code = render(base.read_text(), mode)
        package = Path(mode)
        compile(code, str(package), 'exec')
        package.mkdir()
        (package / 'swi_glu.py').write_text(code)
        (package / '__init__.py').touch()
        copyfile(base.with_name('_pyasc_runtime.py'), package / '_pyasc_runtime.py')
