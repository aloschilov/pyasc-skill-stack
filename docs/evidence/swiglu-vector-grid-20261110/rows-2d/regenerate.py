"""Regenerate the public SwiGLU selector with OpenCode in an experiment directory.

Run with the campaign's configured opencode.json in the current directory.
The second phase checks the emitted operator on the previous profile inputs.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

root = Path(__file__).resolve().parents[4]
evidence = Path(__file__).resolve().parents[1]
prior = evidence / 'pair-reindex'
source = (prior / 'swi_glu.py').read_text()
before, sep, after = source.partition('def select(')
old, routes, rest = (sep + after).partition('ROUTES =')
assert before and routes
request = (
    'Regenerate the host selector for the public SwiGLU operator on the archive-bundled '
    '950PR build. Apply the supplied skill to the ordinary strided 2D copy branch. '
    'Retain the signature and the existing short-half fallback, gather and flat routes. '
    'Return the complete select function in one fenced Python block.'
)
reference = root / 'skills/asctile-api-patterns/references/strided-row-tiles.md'
prompt = '\n\n'.join([request, old, reference.read_text()])
Path('prompt.txt').write_text(prompt)
with Path('events.jsonl').open('w') as output:
    subprocess.run(['opencode', 'run', '--agent', 'dispatch', '-m', 'cannbot/GLM-5.3',
                    '--format', 'json', prompt], stdout=output, check=True, timeout=480)
texts = [e['part']['text'] for line in Path('events.jsonl').read_text().splitlines()
         if (e := json.loads(line)).get('type') == 'text']
blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', '\n'.join(texts), re.S)
selected, = [s.strip() + '\n' for s in blocks if s.lstrip().startswith('def select(')]
tree = ast.parse(selected)
assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
namespace = {}
exec(compile(tree, 'generated selector', 'exec'), namespace)
select = namespace['select']
previous = json.loads((prior / 'measurements.json').read_text())
routes_by_name = {'swiglu_rows': 'batched_rows', 'swiglu_slab': 'index_gather',
                  'swiglu_flat': 'flat_chunks', '_swiglu_pair_groups': 'packed_pairs'}
for record in previous['cases']:
    h, = [d['integers'] for d in record['dispatch'] if d['function'] == 'swi_glu']
    pair = h['dim'] == len(record['shape']) - 1 and h['half'] == 1
    choice = select(h['rows'], h['half'], h['itemsize'], pair, record['dtype'] == 'bfloat16',
                    h['ub_bytes'], h['aivs'])
    old_route = routes_by_name[next(n for n, k, _ in record['tasks'] if k == 'AI_VECTOR_CORE')]
    if old_route == 'batched_rows':
        assert choice[0] == old_route, (record['case'], choice)
    else:
        assert choice == (old_route, h['tile'], h['unroll']), (record['case'], choice)
for rows, half, ub, aivs, tile, unroll in [
    (52224, 50, 253952, 72, 64, 2),
    (52224, 50, 90000, 72, 64, 1),
    (52224, 50, 52000, 72, 32, 2),
    (143, 536, 253952, 72, 4, 1),
    (512, 80, 253952, 4, 32, 2),
    (512, 80, 253952, 72, 32, 1),
]:
    assert select(rows, half, 4, False, False, ub, aivs) == ('batched_rows', tile, unroll)
out = Path('fresh')
out.mkdir(exist_ok=True)
(out / 'swi_glu.py').write_text(before + selected + '\n' + routes + rest)
shutil.copyfile(evidence / 'dma/_pyasc_runtime.py', out / '_pyasc_runtime.py')
Path('selector.py').write_text(selected)
print(hashlib.sha256((out / 'swi_glu.py').read_bytes()).hexdigest(), flush=True)
