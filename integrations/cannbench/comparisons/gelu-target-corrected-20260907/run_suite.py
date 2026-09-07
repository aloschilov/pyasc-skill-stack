"""Run bounded Model checks serially in isolated working directories."""
import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent


def run(name,args):
    out=ROOT/'evidence'/name;out.mkdir(parents=True,exist_ok=True)
    path=out/'result.json'
    if path.exists():return json.loads(path.read_text())
    with tempfile.TemporaryDirectory(prefix='gelu-target-') as cwd, (out/'stdout.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(ROOT/'check.py'),*args,'--output',str(path)],
                              cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=360)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    result=json.loads(path.read_text()) if path.exists() else {}
    if code:
        result.update(process_exit=code,completed=False)
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'name':name,'completed':result.get('completed'),'numerical':result.get('numerical_passed'),
                      'ticks':result.get('median_ticks'),'UB':result.get('memory_consumed'),'error':result.get('error')}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['numerical','perf','matrix','extensions','exact']);a=p.parse_args()
    if a.phase=='numerical':
        jobs=[('original-f32',['numerical','--variant','original']),
              ('corrected-f16',['numerical','--dtype','float16']),
              ('corrected-bf16',['numerical','--dtype','bfloat16']),
              ('corrected-f32-vf1',['numerical','--vf','1']),
              ('corrected-f32-r2-vf1',['numerical','--reuse','2','--vf','1']),
              ('tail-cores8',['tail','--tile','1024','--unroll','1','--cores','8','--size','2049'])]
    elif a.phase=='perf':
        jobs=[('perf-original-15872',['perf','--variant','original']),
              ('perf-corrected-15872',['perf']),
              ('perf-corrected-1024',['perf','--tile','1024']),
              ('perf-corrected-15872-vf1',['perf','--vf','1']),
              ('perf-stable-15872',['perf','--variant','stable'])]
    elif a.phase=='extensions':
        jobs=[('promoted-f16',['numerical','--variant','promoted','--dtype','float16','--tile','8192']),
              ('promoted-bf16',['numerical','--variant','promoted','--dtype','bfloat16','--tile','8192']),
              ('perf-stable-1024',['perf','--variant','stable','--tile','1024','--unroll','1'])]
    elif a.phase=='exact':
        jobs=[(f'exact-{dtype}-r1-vf0',['numerical','--variant','exact','--dtype',dtype,'--tile','8192'])
              for dtype in ['float32','float16','bfloat16']]
        jobs.append(('exact-float32-r1-vf1',['numerical','--variant','exact','--tile','8192','--vf','1']))
    else:
        jobs=[(f'matrix-r{r}-s{s}-vf{vf}',['compile','--reuse',str(r),'--static',s,'--vf',str(vf),'--size','150994944','--cores','72'])
              for r,s,vf in itertools.product([0,1,2],['default','false','true'],[0,1])]
    results={name:run(name,args) for name,args in jobs}
    (ROOT/'evidence'/f'{a.phase}-summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
