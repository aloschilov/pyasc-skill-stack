"""Change the physical position of the short tile in a single-wave row cover."""
import hashlib
from pathlib import Path
from shutil import copyfile


def main():
    source = Path('base/swi_glu.py').read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != '30bc199acb49eecb1f262ca4afb5e16e3ce871f5f365954eb167107a61db412a':
        raise ValueError('expected the unedited SwiGLU from the 5ac7788 skill')
    start = source.index('def single_wave_fp32_rows(')
    end = source.index('@asctile.jit', start)
    common = '    r = tile_index * T\n    rows = min(T, R - r)'
    kernel = source[start:end]
    if kernel.count(common) != 1:
        raise ValueError('unexpected row assignment')
    for name, edge in [('head', '0'), ('middle', 'n // 2'), ('end', 'n - 1')]:
        # Moving only the short interval shifts subsequent row starts by the slack.
        geometry = f'''    edge = {edge}
    slack = n * T - R
    r = tile_index * T - (slack if tile_index > edge else 0)
    rows = T - (slack if tile_index == edge else 0)'''
        candidate = source[:start] + kernel.replace(common, geometry) + source[end:]
        dest = Path(name)
        dest.mkdir()
        (dest / 'swi_glu.py').write_text(candidate)
        copyfile('base/_pyasc_runtime.py', dest / '_pyasc_runtime.py')


if __name__ == '__main__':
    main()
