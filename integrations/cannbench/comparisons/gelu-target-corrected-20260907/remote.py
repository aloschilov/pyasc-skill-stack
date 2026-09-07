"""One authorized GeLU upload, protected by hashes, local gates and job reconciliation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]/'integrations/cannbench/workers'))
from evalqueue import EvalQueue, TERMINAL_STATUSES, _unwrap_job
TAG = 'pyasc-adadd7d-gelu-target-corrected-cf6-20260907'
OUT = ROOT/'evidence/remote'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def clean(v):
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items() if k not in {'aggregation_token','token','api_key'}}
    if isinstance(v,list): return [clean(x) for x in v]
    return v


def save(name,v):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(clean(v),indent=2)+'\n')


def qualify():
    manifest=json.loads((ROOT/'evidence/package.json').read_text())
    evidence=json.loads((ROOT/'evidence/package-x86.json').read_text())
    assert evidence['status']=='passed' and evidence['compile_passed']==20
    assert evidence['installed_files_match_wheel'] and evidence['wheel_sha256']==manifest['wheel_sha256']
    assert sha(ROOT/'candidate/gelu.py')==manifest['candidate_sha256']
    assert sha(ROOT/'diagnostic.zip')==manifest['archive_sha256']
    assert sha(ROOT/'bundle/wheel/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl')==manifest['wheel_sha256']
    checked=[]
    for mode in ['none','tanh']:
        for dtype in ['float32','float16','bfloat16']:
            for name in [f'final-v2-{phase}-{mode}-{dtype}' for phase in ['numerical','tail','perf']]:
                row=json.loads((ROOT/'evidence'/name/'result.json').read_text())
                assert row['completed'] and row['numerical_passed'],name
                assert row['clean_model_run'] and row['unexpected_simulator_error_count']==0,name
                assert row['source_sha256']==manifest['candidate_sha256'],name
                assert row['memory_consumed']['UB']<=253952,name
                if 'tail' in name:
                    assert row['writes_outside_logical_size']==0 and row['outer_guard_untouched'],name
                    assert row['cached_repeats'] and all(x['same_output_including_nan'] and x['accuracy_passed'] for x in row['cached_repeats']),name
                if 'perf' in name:
                    assert len(row['warm_ticks'])==3 and all(row['repeat_accuracy']) and row['repeated_outputs_equal'],name
                checked.append(name)
    assert json.loads((ROOT/'evidence/gate-options.json').read_text())['status']=='passed'
    review=json.loads((ROOT/'workers-submission-v2/provenance.json').read_text())
    assert review['exit_code']==0 and review['session_ids']
    assert review['inputs'][str(ROOT/'candidate/gelu.py')]==manifest['candidate_sha256']
    assert (ROOT/'REVIEW_ADJUDICATION.md').exists()
    save('local-gates.json',{'status':'passed','candidate_sha256':manifest['candidate_sha256'],
        'checks':checked,'review_session_ids':review['session_ids'],'scope':'sampled Model; all20 compile; hardware pending'})
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true')
    p.add_argument('--retry-rejected-layout',action='store_true');a=p.parse_args()
    q=EvalQueue()
    if not (OUT/'submission.json').exists():
        credits=q._client.get_credits();save('credits-before.json',credits)
        jobs=q._client.list_jobs(limit=100);save('jobs-before.json',jobs)
        matches=[j for j in jobs.get('jobs',[]) if j.get('job_tag')==TAG]
        if matches:
            assert len(matches)==1,'Multiple matching jobs; manual reconciliation needed'
            save('submission.json',{'job_id':matches[0]['id'],'tag':TAG,'recovered_from_list':True})
        elif a.submit:
            attempt_name='attempt.json'
            if a.retry_rejected_layout:
                rejection=json.loads((OUT/'layout-rejection.json').read_text())
                previous=json.loads((OUT/'attempt.json').read_text())
                assert rejection['error_code']=='WEB-ZIP-007' and rejection['job_created'] is False
                assert rejection['rejected_archive_sha256']==previous['archive_sha256']
                assert rejection['remaining_after']==credits['credits']['remaining']
                attempt_name='attempt-corrected-layout.json'
            assert not (OUT/attempt_name).exists(),'Ambiguous prior attempt; never retry automatically'
            manifest=qualify()
            info=credits['credits'];assert info.get('unlimited') or info.get('remaining',0)>0,'No credits'
            active=[j for j in jobs.get('jobs',[]) if not j.get('job_completed') and j.get('status') not in TERMINAL_STATUSES]
            assert not active,'Pending jobs: defer upload'
            save(attempt_name,{'tag':TAG,'started_unix':time.time(),'archive_sha256':manifest['archive_sha256'],
                'candidate_sha256':manifest['candidate_sha256'],'diagnostic_only':True,'known_local_numerical_failure':False})
            response=q._submit_streaming(str(ROOT/'diagnostic.zip'),['gelu'],TAG)
            save('submission-response.json',response)
            job_id=response.get('job_id') or (response.get('job') or {}).get('id')
            assert job_id,'No job ID: reconcile response, do not retry'
            save('submission.json',{'job_id':job_id,'tag':TAG,'archive_sha256':manifest['archive_sha256']})
            print(json.dumps({'submitted':job_id}),flush=True)
        else:
            print(json.dumps({'submitted':False,'credits':credits['credits']['remaining']}));return
    record=json.loads((OUT/'submission.json').read_text())
    payload=q._client.get_job(record['job_id']);save('job-latest.json',payload)
    job=_unwrap_job(payload)
    terminal=bool(job.get('job_completed') or job.get('status') in TERMINAL_STATUSES)
    if terminal:
        save('job.json',payload)
        save('logs.json',q._client.get_job_logs(record['job_id']))
        save('credits-after.json',q._client.get_credits())
    print(json.dumps({'job_id':record['job_id'],'status':job.get('status'),'terminal':terminal}),flush=True)


if __name__=='__main__': main()
