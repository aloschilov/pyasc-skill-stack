"""Bounded two-process final qualification; simulator errors fail the gate."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
from run_suite import run

ROOT = Path(__file__).resolve().parent
jobs=[]
for mode in ['none','tanh']:
    for dtype in ['float32','float16','bfloat16']:
        tile=1024 if mode=='none' and dtype=='float32' else 15872 if dtype=='float32' else 8192
        base=['--variant','submission','--mode',mode,'--dtype',dtype]
        jobs.extend([
            (f'final-v2-numerical-{mode}-{dtype}',['numerical',*base,'--compact-stress','--repeat','2']),
            (f'final-v2-tail-{mode}-{dtype}',['tail',*base,'--size',str(tile*9+1),'--cores','8','--repeat','2']),
            (f'final-v2-perf-{mode}-{dtype}',['perf',*base,'--size',str(tile*2),'--perf-range','-6','2'])])


def execute(job):
    name,args=job
    row=run(name,args)
    log=ROOT/'evidence'/name/'stdout.log'
    errors=[line for line in log.read_text(errors='replace').splitlines() if '[error]' in line.lower()]
    # VF reports deliberate NaN/Inf inputs as diagnostics. Keep their count,
    # but distinguish them from DMA, memory, synchronization or other faults.
    special_probe='numerical' in name and any(c['segment']=='special_values' and c['passed'] for c in row.get('checks',[]))
    expected=[line for line in errors if special_probe and '[vec_err_idata_inf_nan_t0]' in line]
    unexpected=[line for line in errors if line not in expected]
    row['simulator_error_count']=len(errors)
    row['simulator_error_sample']=errors[:10]
    row['expected_special_value_diagnostics']=len(expected)
    row['unexpected_simulator_error_count']=len(unexpected)
    row['unexpected_simulator_error_sample']=unexpected[:10]
    row['clean_model_run']=row.get('completed') and row.get('numerical_passed') and not unexpected
    path=ROOT/'evidence'/name/'result.json'
    path.write_text(json.dumps(row,indent=2)+'\n')
    return name,row


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.audit_only:
        assert all((ROOT/'evidence'/name/'result.json').exists() for name,_ in jobs), 'Audit-only never launches missing checks'
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=dict(pool.map(execute,jobs))
    (ROOT/'evidence/final-v2-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert all(r.get('clean_model_run') for r in rows.values()), 'Failed or diagnostic-error Model run: inspect evidence'
    assert all(r['source_sha256']==hashlib.sha256((ROOT/'candidate/gelu.py').read_bytes()).hexdigest() for r in rows.values())
    print('All 18 final validation jobs passed without unexpected simulator errors.',flush=True)
