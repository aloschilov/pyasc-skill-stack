"""Regenerate the strided SwiGLU kernel and its host geometry using OpenCode."""
import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

evidence = Path(__file__).resolve().parents[1]
reference = evidence.parents[2] / 'skills/asctile-api-patterns/references/strided-row-tiles.md'
source = (evidence / 'rows-2d/swi_glu.py').read_text()
functions = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
names = ('swiglu_rows', 'row_geometry')


def extent(fn):
    return min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1, fn.end_lineno


lines = source.splitlines(keepends=True)
original = [''.join(lines[slice(*extent(functions[name]))]) for name in names]
request = """Regenerate these two functions for a public AscTile SwiGLU operator
using the attached skill and the existing functions as their interface. Preserve
the kernel signature, decorator, valid extents, copy geometry, and existing
validation. The host caller supplies row_geometry(rows, half, itemsize, ub_bytes,
aivs, is_bf16) for FP16, BF16 and FP32. Return exactly the two complete
functions in fenced Python, without imports or other top-level code."""
prompt = '\n\n'.join([request, reference.read_text(), *original])
Path('prompt.txt').write_text(prompt)
with Path('events.jsonl').open('w') as output:
    subprocess.run(['opencode', 'run', '--agent', 'dispatch', '-m', 'cannbot/GLM-5.3',
                    '--format', 'json', prompt], stdout=output, check=True, timeout=480)
texts = [event['part']['text'] for line in Path('events.jsonl').read_text().splitlines()
         if (event := json.loads(line)).get('type') == 'text']
blocks = re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```', '\n'.join(texts), re.S)
generated = '\n\n'.join(block.strip() for block in blocks) + '\n'
nodes = ast.parse(generated).body
assert all(isinstance(n, ast.FunctionDef) for n in nodes)
assert sorted(n.name for n in nodes) == sorted(names)
emitted = generated.splitlines(keepends=True)
for n in sorted(nodes, key=lambda n: functions[n.name].lineno, reverse=True):
    lines[slice(*extent(functions[n.name]))] = emitted[slice(*extent(n))]
operator = ''.join(lines)
old_call = '        tile, unroll = row_geometry(rows, half, itemsize, ub_bytes, aivs)'
assert operator.count(old_call) == 1
operator = operator.replace(old_call, old_call[:-1] + ', is_bf16)')
ast.parse(operator)
ns = {}
exec(operator[operator.index('def row_geometry('):operator.index('ROUTES =')], ns)
for rows, half, size, ub, cores, bf16, expected in [
    (3003, 506, 2, 253952, 72, False, (16, 2)),
    (3003, 506, 2, 163840, 72, False, (16, 2)),
    (3003, 506, 2, 163839, 72, False, (16, 1)),
    (3003, 506, 2, 253952, 72, True, (16, 1)),
    (143, 536, 4, 253952, 72, False, (4, 1)),
    (52224, 50, 4, 253952, 72, False, (64, 2)),
]:
    assert ns['select'](rows, half, size, False, bf16, ub, cores) == ('batched_rows', *expected)
out = Path('fresh')
out.mkdir(exist_ok=True)
(out / 'swi_glu.py').write_text(operator)
shutil.copyfile(evidence / 'dma/_pyasc_runtime.py', out / '_pyasc_runtime.py')
Path('emitted.py').write_text(generated)
print(hashlib.sha256(operator.encode()).hexdigest(), flush=True)
