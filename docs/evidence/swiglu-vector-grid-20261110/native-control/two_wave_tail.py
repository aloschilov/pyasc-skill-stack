"""Generate matched aligned FP16 two-wave diagnostics from the measured W53 source."""
import ast
import hashlib
from pathlib import Path
from shutil import copyfile
from textwrap import indent


def transfer(start, height, suffix=""):
    return f"""r{suffix} = {start}
gate{suffix} = asctile.copy_in(gx, [r{suffix}, 0], [{height}, H])
value{suffix} = asctile.copy_in(gx, [r{suffix}, H], [{height}, H])
gate_f{suffix} = gate{suffix}.to(asctile.float32)
out{suffix} = (gate_f{suffix} / (1.0 + asctile.exp(-gate_f{suffix}))) * value{suffix}.to(asctile.float32)
asctile.copy_out(out{suffix}.to(gate{suffix}.dtype), gy, [r{suffix}, 0])"""


def build(mode):
    common = """@asctile.jit(reuse_alloc=2)
def paired_rows(x: asctile.GlobalAddress, y: asctile.GlobalAddress,
                R: asctile.ConstExpr[int], H: asctile.ConstExpr[int],
                T: asctile.ConstExpr[int], N: asctile.ConstExpr[int]):
    gx = asctile.global_tensor(x, [R, 2 * H])
    gy = asctile.global_tensor(y, [R, H])
    b = asctile.block_idx()
    tail = R - 2 * N * T
"""
    if mode == 'separate':
        return common + """    for i in asctile.range(2, unroll_factor=2):
""" + indent(transfer('(b + i*N)*T', 'T'), '        ') + """
    if tail > 0:
        if b == 0:
""" + indent(transfer('2*N*T', 'tail'), '            ')
    # Adjacent and strided variants both fold the edge into the final DMA.
    first, second = ('2*b*T', '(2*b + 1)*T') if mode == 'adjacent' else ('b*T', '(N+b)*T')
    return (common + indent(transfer(first, 'T'), '    ') + '\n    if b == N - 1:\n'
            + indent(transfer(second, 'T + tail', '_last'), '        ') + '\n    else:\n'
            + indent(transfer(second, 'T', '_other'), '        '))


def main():
    src = Path('base/swi_glu.py').read_text()
    if hashlib.sha256(src.encode()).hexdigest() != '91c41acfa5c67a776e12d75aa7dc7063dcaa1d21e2510d5cb49cef62c785444c':
        raise ValueError('expected the unedited W53 module')
    call = 'bounded_rows[aivs](input, out, R, H, height, H, 2)'
    assert src.count(call) == 1
    for mode in ('separate', 'strided', 'adjacent'):
        kernel = build(mode)
        code = src.replace('# Public entry point', kernel + '\n\n# Public entry point')
        code = code.replace(call, 'paired_rows[aivs](input, out, R, H, height, aivs)')
        tree = ast.parse(code)
        assert any(getattr(node, 'name', None) == 'paired_rows' for node in tree.body)
        out = Path(mode)
        out.mkdir()
        (out / 'swi_glu.py').write_text(code)
        copyfile('base/_pyasc_runtime.py', out / '_pyasc_runtime.py')


if __name__ == '__main__':
    main()
