"""Matched, bounded GeLU geometry/allocation experiment; never submits remotely."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
from run_suite import run

ROOT=Path(__file__).resolve().parent

def execute(item):
    name,args=item
    row=run(name,args)
    errors=[s for s in (ROOT/'evidence'/name/'stdout.log').read_text(errors='replace').splitlines() if '[error]' in s.lower()]
    special=args[0]=='numerical' and any(c['segment']=='special_values' and c['passed'] for c in row.get('checks',[]))
    row['unexpected_errors']=[s for s in errors if not (special and '[vec_err_idata_inf_nan_t0]' in s)]
    row['valid']=bool(row.get('completed') and row.get('ub_fit') and not row['unexpected_errors']
                      and (args[0]=='compile' or (row.get('numerical_passed') and all(row.get('repeat_accuracy',[True])))))
    (ROOT/'evidence'/name/'result.json').write_text(json.dumps(row,indent=2)+'\n')
    return name,row

def config(dtype,mode,tile,unroll,reuse,vf):
    name=f'{dtype}-{mode}-t{tile}-u{unroll}-r{reuse}-vf{vf}'
    args=['--variant','submission','--tune-geometry','--dtype',dtype,'--mode',mode,
          '--tile',str(tile),'--unroll',str(unroll),'--reuse',str(reuse),'--vf',str(vf)]
    return name,args

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['matrix','perf','qualify']);p.add_argument('--skip-running',action='store_true');a=p.parse_args()
    jobs=[]
    if a.phase=='matrix':
        # Matched JIT grid, then UB-boundary geometry grid. BF16 is independently qualified later.
        for dtype,mode in itertools.product(['float32','float16'],['none','tanh']):
            baseline=1024 if dtype=='float32' and mode=='none' else 15872 if dtype=='float32' else 8192
            for reuse,vf in itertools.product([0,1,2],[0,1]):
                name,args=config(dtype,mode,baseline,1 if baseline==1024 else 2,reuse,vf)
                jobs.append(('compile-'+name,['compile',*args]))
            tiles=[4096,5120,8192] if dtype=='float32' and mode=='none' else [12288,15872,20480]
            for tile,reuse,vf in itertools.product(tiles,[1,2],[0,1]):
                name,args=config(dtype,mode,tile,1 if dtype=='float32' and mode=='none' else 2,reuse,vf)
                jobs.append(('compile-'+name,['compile',*args]))
    elif a.phase=='perf':
        for name,row in json.loads((ROOT/'evidence/matrix-summary.json').read_text()).items():
            if not row['valid']:continue
            c=row['args'];key,args=config(c['dtype'],c['mode'],c['tile'],c['unroll'],c['reuse'],c['vf'])
            # Initial matched search: preserve historical VF choice. VF alternatives
            # are compiled above and timed separately at the selected geometry.
            if c['vf'] != int(c['dtype']=='float32' and c['mode']=='none'):continue
            allowed={('float32','none'):{(1024,0),(1024,1),(1024,2),(5120,1),(5120,2)},
                     ('float32','tanh'):{(15872,1),(15872,2)},
                     ('float16','none'):{(8192,1),(20480,1),(15872,2)},
                     ('float16','tanh'):{(8192,1),(15872,1),(20480,2)}}
            if (c['tile'],c['reuse']) not in allowed[(c['dtype'],c['mode'])]:continue
            # Identical logical workload, including the authored guarded tail path.
            # Padding overhead is a real property of each selected geometry.
            n=8192 if c['dtype']=='float32' and c['mode']=='none' else 32768
            jobs.append((f'perf-n{n}-'+key,['perf',*args,'--size',str(n),'--perf-range','-6','2']))
        # Aligned high-UB control separates larger useful tiles from padding cost.
        key,args=config('float16','none',16384,2,1,0)
        jobs.append(('perf-n32768-'+key,['perf',*args,'--size','32768','--perf-range','-6','2']))
    else:
        selected=json.loads((ROOT/'selection.json').read_text())
        for route,c in selected.items():
            for dtype in ([c['dtype'],'bfloat16'] if c['dtype']=='float16' else [c['dtype']]):
                key,args=config(dtype,c['mode'],c['tile'],c['unroll'],c['reuse'],c['vf'])
                jobs.append(('numerical-'+key,['numerical',*args,'--compact-stress','--repeat','2']))
                jobs.append(('tail-'+key,['tail',*args,'--size',str(c['tile']*9+1),'--cores','8','--repeat','2']))
    if a.skip_running:
        jobs=[(name,args) for name,args in jobs if not ((ROOT/'evidence'/name/'result.json').exists()
            and not json.loads((ROOT/'evidence'/name/'result.json').read_text()).get('completed'))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=dict(pool.map(execute,jobs))
    (ROOT/'evidence'/f'{a.phase}-summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'{a.phase}: {sum(r["valid"] for r in results.values())}/{len(results)} valid',flush=True)

if __name__=='__main__':main()
