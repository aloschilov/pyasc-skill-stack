"""Select measured valid configurations; avoid shadowing Python's select module."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'evidence/perf-summary.json').read_text())
assert len(rows)==14 and all(r.get('completed') for r in rows.values())
selection={};reasons={}
for dtype in ['float32','float16']:
    for mode in ['none','tanh']:
        valid=[r for r in rows.values() if r['valid'] and r['args']['dtype']==dtype and r['args']['mode']==mode]
        assert valid
        best=min(valid,key=lambda r:r['median_ticks'])
        normal=min((r for r in valid if r['args']['reuse']==1),key=lambda r:r['median_ticks'])
        selected=normal if normal['median_ticks']<=best['median_ticks']*1.03 else best
        baseline_tile=1024 if dtype=='float32' and mode=='none' else 15872 if dtype=='float32' else 8192
        baseline=next(r for r in valid if r['args']['tile']==baseline_tile and r['args']['reuse']==1)
        if baseline['median_ticks']<=selected['median_ticks']*1.03:
            selected=baseline
        c=selected['args'];route=dtype+'-'+mode
        selection[route]={k:c[k] for k in ['dtype','mode','tile','unroll','reuse','vf']}
        reasons[route]={'selected_ticks':selected['median_ticks'],'fastest_ticks':best['median_ticks'],
            'policy':'Prefer reuse=1 within 3% of minimum; retain baseline geometry within 3%; no artificial UB-fill target',
            'elements':selected['elements'],'cores':c['cores'],'ub_bytes':selected['memory_consumed']['UB']}
(ROOT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
(ROOT/'evidence/selection-reasons.json').write_text(json.dumps(reasons,indent=2)+'\n')
print(json.dumps(selection,indent=2))
