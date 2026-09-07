"""Read recorded outputs; never infer numerical success from compilation."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
source=ROOT.parent/'gelu-asctile-jit-20260907/analyze_lowering.py'
spec=importlib.util.spec_from_file_location('lowering_analysis',source)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
rows=[]
for p in sorted((ROOT/'evidence').glob('*/result.json')):
    r=json.loads(p.read_text());a=r.get('args',{})
    row=dict(name=p.parent.name,variant=a.get('variant'),phase=a.get('phase'),dtype=a.get('dtype'),
             tile=a.get('tile'),unroll=a.get('unroll'),cores=a.get('cores'),reuse=a.get('reuse'),
             static=a.get('static'),vf=a.get('vf'),elements=r.get('elements'),completed=r.get('completed'),
             compiled=r.get('compile_passed'),numerical=r.get('numerical_passed'),
             ub_bytes=(r.get('memory_consumed') or {}).get('UB'),median_ticks=r.get('median_ticks'),
             warm_ticks=r.get('warm_ticks'),failed_segments=[c['segment'] for c in r.get('checks',[]) if not c['passed']],
             changed_guard_elements=r.get('writes_outside_logical_size'),
             source_sha256=r.get('source_sha256'),evidence=str(p.relative_to(ROOT)),
             error=(r.get('error') or '').splitlines()[-1:])
    cpp=p.parent/'diagnostic/ascendc.cpp'
    if cpp.exists():
        row['cpp_sha256']=hashlib.sha256(cpp.read_bytes()).hexdigest()
        if a.get('static')=='false':row['tpipe_accounting']=helper.tpipe_ub(cpp.read_text())
    rows.append(row)
matrix=[r for r in rows if r['name'].startswith('matrix-')]
pairs=[]
for r in matrix:
    if r['static']=='default':
        mate=next(x for x in matrix if x['reuse']==r['reuse'] and x['vf']==r['vf'] and x['static']=='true')
        pairs.append(dict(reuse=r['reuse'],vf=r['vf'],cpp_equal=r.get('cpp_sha256')==mate.get('cpp_sha256')))
summary=dict(runtime_pin='adadd7d66ed0ee16d33d79487bf584899a26ef1e',scope='Local CaModel only; tanh and exact diagnostic variants, no remote submission',
             results=rows,default_true_equivalence=pairs,capacity_bytes=253952)
(ROOT/'evidence/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fields=['name','variant','phase','dtype','tile','unroll','cores','reuse','static','vf','elements','completed','compiled','numerical','ub_bytes','median_ticks','failed_segments','evidence']
with (ROOT/'evidence/results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps({'results':len(rows),'matrix':len(matrix),'default_true_equivalence':pairs}))
