"""One credit-aware, hash-bound GeLU tiling diagnostic. Never retry ambiguous uploads."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]/'integrations/cannbench/workers'))
from evalqueue import EvalQueue,TERMINAL_STATUSES,_unwrap_job
OUT=ROOT/'evidence/remote'
TAG='pyasc-adadd7d-gelu-tiling-reuse-20260908'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k not in {'token','api_key','aggregation_token'}}
    if isinstance(v,list):return [clean(x) for x in v]
    return v
def save(name,v):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(clean(v),indent=2)+'\n')

def qualify():
    m=json.loads((ROOT/'evidence/package.json').read_text())
    e=json.loads((ROOT/'evidence/package-x86.json').read_text())
    assert e['status']=='passed' and e['compile_passed']==20 and e['installed_files_match_wheel']
    assert e['wheel_sha256']==m['wheel_sha256']==sha(ROOT/'bundle/wheel/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl')
    assert m['candidate_sha256']==sha(ROOT/'candidate/gelu.py')
    assert m['archive_sha256']==sha(ROOT/'diagnostic.zip')
    def kernel_ast(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='gelu_kernel'))
    assert kernel_ast(ROOT/'candidate/gelu.py')==kernel_ast(ROOT.parent/'gelu-target-corrected-20260907/candidate/gelu.py')
    perf=json.loads((ROOT/'evidence/perf-summary.json').read_text())
    assert len(perf)==14 and all(r['valid'] for r in perf.values())
    assert json.loads((ROOT/'evidence/gate-options.json').read_text())['status']=='passed'
    selected=json.loads((ROOT/'selection.json').read_text())
    assert selected==m['routes']
    q=json.loads((ROOT/'evidence/qualify-summary.json').read_text())
    assert len(q)==12 and all(r['valid'] for r in q.values())
    assert len({(r['args']['dtype'],r['args']['mode']) for r in q.values()})==6
    for name,r in q.items():
        assert r['source_sha256']==m['candidate_sha256'] and not r['unexpected_errors']
        assert r['cached_repeats'] and all(v['same_output_including_nan'] and v['accuracy_passed'] for v in r['cached_repeats'])
        route=('float16' if r['args']['dtype']=='bfloat16' else r['args']['dtype'])+'-'+r['args']['mode']
        assert all(r['args'][k]==selected[route][k] for k in ['tile','unroll','reuse','vf'])
        if name.startswith('tail-'):assert r['writes_outside_logical_size']==0 and r['outer_guard_untouched']
    audit=json.loads((ROOT/'evidence/dispatch.json').read_text())
    assert audit['source_sha256']==m['candidate_sha256'] and audit['source_gate']=='passed'
    save('local-gates.json',{'passed':True,'candidate_sha256':m['candidate_sha256'],'checks':list(q)})
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--retry-incomplete',action='store_true');a=p.parse_args()
    q=EvalQueue()
    if not (OUT/'submission.json').exists():
        credits=q._client.get_credits();jobs=q._client.list_jobs(limit=100)
        save('credits-before.json',credits);save('jobs-before.json',jobs)
        matches=[j for j in jobs.get('jobs',[]) if j.get('job_tag')==TAG]
        if matches:
            assert len(matches)==1
            save('submission.json',{'job_id':matches[0]['id'],'tag':TAG,'reconciled':True})
        elif a.submit:
            attempt='attempt.json'
            if a.retry_incomplete:
                incomplete=json.loads((OUT/'incomplete-transport.json').read_text())
                previous=json.loads((OUT/'attempt.json').read_text())
                assert incomplete['curl_exit_code']==-15 and not Path(f"/proc/{incomplete['curl_pid']}").exists()
                assert incomplete['archive_fd_position_before_abort']<incomplete['archive_size_bytes']==(ROOT/'diagnostic.zip').stat().st_size
                assert incomplete['archive_sha256']==previous['archive_sha256']==sha(ROOT/'diagnostic.zip')
                assert incomplete['reconciled_job_created_after_abort'] is False
                assert incomplete['reconciled_remaining_after_abort']==credits['credits']['remaining']
                attempt='attempt-direct-route.json'
            assert not (OUT/attempt).exists(),'Ambiguous previous attempt; reconcile without retry'
            m=qualify()
            assert credits['credits'].get('unlimited') or credits['credits']['remaining']>0
            assert not [j for j in jobs.get('jobs',[]) if not j.get('job_completed') and j.get('status') not in TERMINAL_STATUSES]
            save(attempt,{'tag':TAG,'started_unix':time.time(),'archive_sha256':m['archive_sha256']})
            response=q._submit_streaming(str(ROOT/'diagnostic.zip'),['gelu'],TAG)
            save('submission-response.json',response)
            job_id=response.get('job_id') or (response.get('job') or {}).get('id')
            assert job_id,'No job ID; reconcile rather than retry'
            save('submission.json',{'job_id':job_id,'tag':TAG,'archive_sha256':m['archive_sha256']})
        else:
            print(json.dumps({'remaining':credits['credits']['remaining'],'submitted':False}));return
    job_id=json.loads((OUT/'submission.json').read_text())['job_id']
    payload=q._client.get_job(job_id);save('job-latest.json',payload)
    job=_unwrap_job(payload)
    if job.get('job_completed') or job.get('status') in TERMINAL_STATUSES:
        save('job.json',payload);save('credits-after.json',q._client.get_credits())
    print(json.dumps({'job_id':job_id,'status':job['status'],'passed_cases':job.get('passed_cases')}),flush=True)

if __name__=='__main__':main()
