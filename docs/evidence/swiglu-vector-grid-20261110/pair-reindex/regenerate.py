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
prior = evidence / 'dma'
source = (prior / 'swi_glu.py').read_text()
before, sep, after = source.partition('def select(')
old, routes, rest = (sep + after).partition('ROUTES =')
assert before and routes
request = (
    'Regenerate the select function of this public AscTile SwiGLU operator for the '
    'archive-bundled 950PR target. Public reindex and cached int32 indices are available. '
    'Apply the supplied skill to the dense BF16 pair dispatch. Keep the existing '
    'function signature, route names and all unrelated geometry. Return just the '
    'complete selector in one fenced Python block.'
)
reference = root / 'skills/asctile-api-patterns/references'
prompt = '\n\n'.join([request, old, (reference / 'packed-bf16-pairs.md').read_text(),
                     (reference / 'reindex-slabs.md').read_text()])
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
    expected = ('index_gather', 4096, 2) if record['case'] == 12 else (old_route, h['tile'], h['unroll'])
    assert choice == expected, (record['case'], choice, expected)
for ub, tile, unroll in [(105000, 4096, 1), (60000, 2048, 2)]:
    assert select(1000003, 1, 2, True, True, ub, 72) == ('index_gather', tile, unroll)
assert select(72, 1, 2, True, True, 253952, 72)[0] == 'batched_rows'
out = Path('fresh')
out.mkdir(exist_ok=True)
(out / 'swi_glu.py').write_text(before + selected + '\n' + routes + rest)
shutil.copyfile(prior / '_pyasc_runtime.py', out / '_pyasc_runtime.py')
Path('selector.py').write_text(selected)
print(hashlib.sha256((out / 'swi_glu.py').read_bytes()).hexdigest(), flush=True)
