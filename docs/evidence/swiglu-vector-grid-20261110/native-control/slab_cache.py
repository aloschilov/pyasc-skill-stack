"""Isolate input and shared BF16 reindex-table L2 policy at fixed two-wave geometry."""
import ast
import hashlib
from pathlib import Path
from shutil import copyfile


def build(source, bypass):
    fn, = (n for n in ast.parse(source).body if getattr(n, 'name', '') == 'two_wave_swiglu')
    lines = source.splitlines(keepends=True)
    changed = set()
    for node in fn.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in bypass:
            continue
        call = node.value
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != 'asctile.global_tensor':
            raise ValueError(name)
        if call.keywords or call.lineno != call.end_lineno:
            raise ValueError('unexpected descriptor')
        idx = call.lineno - 1
        pos = call.end_col_offset - 1
        assert lines[idx][pos] == ')'
        lines[idx] = lines[idx][:pos] + ', l2_cache=False' + lines[idx][pos:]
        changed.add(name)
    assert changed == set(bypass)
    return ''.join(lines)


if __name__ == '__main__':
    source = Path('base/swi_glu.py').read_text()
    if hashlib.sha256(source.encode()).hexdigest() != '18ea91c301937c1f8c901f6d9fa529e952c8cea40c06a4458f322b51113fd607':
        raise ValueError('expected the unedited a7f4048 generation')
    for pkg, bypass in [('inputonly', ['source']), ('indexonly', ['table']), ('both', ['source', 'table'])]:
        dest = Path(pkg)
        code = build(source, bypass)
        compile(code, pkg, 'exec')
        dest.mkdir()
        (dest/'swi_glu.py').write_text(code)
        copyfile('base/_pyasc_runtime.py', dest/'_pyasc_runtime.py')
