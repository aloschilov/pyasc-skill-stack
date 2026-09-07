"""Submit one frozen candidate or collect it; never retry an ambiguous POST."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]/'integrations/cannbench/workers'))
from evalqueue import EvalQueue, TERMINAL_STATUSES, _unwrap_job
from package_submission import digest

def clean(value):
    if isinstance(value,dict):
        return {k:clean(v) for k,v in value.items()
                if k.lower() not in ('aggregation_token','api_key','token','authorization','access_token',
                                     'download_url','submission_download_url')}
    if isinstance(value,list): return [clean(v) for v in value]
    return value

def save(name,value):
    out=ROOT/'remote_runs'
    out.mkdir(exist_ok=True)
    (out/name).write_text(json.dumps(clean(value),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['submit','status'])
    a=p.parse_args()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    archive=ROOT/manifest['archive']['path']
    assert digest(archive)==manifest['archive']['sha256']
    assert digest(ROOT/'candidate/gelu.py')==manifest['candidate_sha256']
    for evidence in manifest['local_evidence']:
        assert digest(ROOT/evidence['path'])==evidence['sha256']
    stress=json.loads((ROOT/'final-pipeline-stress-f16.json').read_text())
    assert stress['checker_passed'] and stress['source_sha256']==manifest['candidate_sha256']
    assert stress['repeat']>=2 and stress['size']>=32771 and stress['cores']==1
    lock=(ROOT/'.submission.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=EvalQueue()
    jobs=q._client.list_jobs(limit=100).get('jobs',[])
    matches=[j for j in jobs if j.get('job_tag')==manifest['tag']]
    if len(matches)>1: raise RuntimeError('Multiple matching jobs: refuse to spend credit')
    if matches:
        job_id=matches[0]['id']
    else:
        record=ROOT/'remote_runs/submission.json'
        if record.exists():
            job_id=json.loads(record.read_text())['job_id']
        elif a.action=='status':
            print(json.dumps(dict(status='no_matching_submission')))
            return
        else:
            intent=ROOT/'remote_runs/upload-intent.json'
            if intent.exists():
                raise RuntimeError('Previous POST may have been accepted. Reconcile before any retry.')
            credits=q._client.get_credits()
            save('credits-before.json',credits)
            if not credits['credits'].get('unlimited') and credits['credits']['remaining']<1:
                print(json.dumps(dict(status='waiting_for_credits',credits=credits['credits'])))
                return
            save('upload-intent.json',dict(tag=manifest['tag'],sha256=manifest['archive']['sha256'],
                 at=datetime.now(timezone.utc).isoformat()))
            response=q._submit_streaming(str(archive),['gelu'],manifest['tag'])
            job_id=response.get('job_id') or response.get('job',{}).get('id')
            if not job_id: raise RuntimeError('Submission response has no job id; reconcile manually')
            save('submission.json',dict(job_id=job_id,response=response,tag=manifest['tag'],
                 archive_sha256=manifest['archive']['sha256']))
            print(json.dumps(dict(submitted=job_id,url='https://cannbench.com/workspace/jobs/'+job_id)),flush=True)
    payload=q._client.get_job(job_id)
    job=_unwrap_job(payload)
    save('job-latest.json',payload)
    save('credits-latest.json',q._client.get_credits())
    if job.get('status') in TERMINAL_STATUSES or job.get('job_completed'):
        save('job.json',payload)
        save('logs.json',q._client.get_job_logs(job_id))
        if job.get('results'): save('results.json',job['results'])
    print(json.dumps(dict(job_id=job_id,status=job.get('status'),url='https://cannbench.com/workspace/jobs/'+job_id)))

if __name__=='__main__': main()
