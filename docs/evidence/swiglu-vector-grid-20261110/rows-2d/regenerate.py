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
start = source.index('    # Short half-rows (wider):')
end = source.index('    # Long contiguous half-row:', start)
request = (
    'Generate a standalone Python function row_geometry(rows, half, itemsize, ub_bytes, aivs) '
    'for the ordinary strided SwiGLU 2D copy on the archive-bundled 950PR build. '
    'The caller has positive dimensions, a physical width aligned to 32 bytes, and '
    'a separate fallback for small gathers. Return (height, unroll). Raise ValueError '
    'if one row cannot fit. Use the supplied skill. Return only the function in one '
    'fenced Python block.'
)
reference = root / 'skills/asctile-api-patterns/references/strided-row-tiles.md'
prompt = '\n\n'.join([request, reference.read_text()])
Path('prompt.txt').write_text(prompt)
with Path('events.jsonl').open('w') as output:
    subprocess.run(['opencode', 'run', '--agent', 'dispatch', '-m', 'cannbot/GLM-5.3',
                    '--format', 'json', prompt], stdout=output, check=True, timeout=480)
texts = [e['part']['text'] for line in Path('events.jsonl').read_text().splitlines()
         if (e := json.loads(line)).get('type') == 'text']
blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', '\n'.join(texts), re.S)
selected, = [s.strip() + '\n' for s in blocks if s.lstrip().startswith('def row_geometry(')]
tree = ast.parse(selected)
assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
call = ('    if half_bytes <= 3072:\n'
        '        tile, unroll = row_geometry(rows, half, itemsize, ub_bytes, aivs)\n'
        '        return ("batched_rows", tile, unroll)\n\n')
operator = (source[:start] + call + source[end:]).replace('def select(', selected + '\ndef select(', 1)
dispatch = operator[operator.index('def row_geometry('):operator.index('ROUTES =')]
namespace = {}
exec(dispatch, namespace)
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
try:
    select(512, 80, 4, False, False, 512, 72)
except ValueError:
    pass
else:
    raise AssertionError('a single padded row exceeds the available UB')
out = Path('fresh')
out.mkdir(exist_ok=True)
(out / 'swi_glu.py').write_text(operator)
shutil.copyfile(evidence / 'dma/_pyasc_runtime.py', out / '_pyasc_runtime.py')
Path('selector.py').write_text(dispatch.rstrip() + '\n')
print(hashlib.sha256((out / 'swi_glu.py').read_bytes()).hexdigest(), flush=True)
