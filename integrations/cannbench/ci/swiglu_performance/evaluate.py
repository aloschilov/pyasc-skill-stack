"""Bind Actions-generated PyASC, its CaModel checks and the package returned by CANNBench."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from statistics import geometric_mean
import subprocess
import sys
import tempfile
import time
import zipfile

from geometry import DTYPES, all_specs, full_row_probes, key
from qualification_control import accept, source_hashes, unchanged

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import single_operator as ci

sha, save, load = ci.smoke.sha, ci.save, ci.load
LEDGER = Path.home()/'.local/state/pyasc-cannbench-smoke'
IMAGE = 'ghcr.io/compiler-team-ru/pyasc-sim-dev@sha256:1efdcc8585b769715f640e9c9078c2e03cdf2157082d05b0c4b87974beea7407'


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def restore(root, origin):
    # Read a named Actions artifact from this repository and check its producing job.
    require(re.fullmatch(r'[1-9][0-9]*',origin), 'Invalid generation run ID')
    repo = os.environ['GITHUB_REPOSITORY']
    def api(endpoint):
        return json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/actions/'+endpoint]))
    run = api('runs/'+origin)
    attempt = run['run_attempt']
    require(run['head_repository']['full_name']==repo and (run['event']=='workflow_dispatch' or (run['event']=='push' and run['head_branch']=='main'))
            and run['path']=='.github/workflows/ci.yml', 'Wrong generation workflow')
    jobs = api(f'runs/{origin}/attempts/{attempt}/jobs')['jobs']
    require(any(j['name'] in ('swiglu-performance / generate','swiglu-on-skill-change / generate') and j['conclusion']=='success' for j in jobs),
            'Generation job did not succeed')
    dest = root/'generation'
    dest.mkdir()
    subprocess.run(['gh','run','download',origin,'--repo',repo,'--name',f'swiglu-generation-{origin}-{attempt}',
                    '--dir',str(dest)],check=True)
    g = load(dest/'generation.json')
    require(str(g['run_id'])==origin and int(g['run_attempt'])==attempt and g['revision']==run['head_sha']
            and g['model']==ci.MODEL and set(g['loaded_skills'])==set(ci.SKILLS)
            and sha(dest/'project/candidate.py')==g['candidate_sha256']
            and sha(dest/'project/policy.json')==g['policy_sha256']
            and sha(dest/'policy-prompt.txt')==g['policy_prompt_sha256']
            and sha(dest/'prompt.txt')==g['prompt_sha256']
            and sha(dest/'project/opencode.json')==g['profile_sha256'], 'Generation receipt mismatch')
    ci.check_hashes(dest/'context',g['context_hashes'])


def package(root):
    gen = root/'generation'
    g = load(gen/'generation.json')
    package = root/'prepared'
    package.mkdir()
    src = root/'qualification/package-source'
    checks = load(root/'qualification/camodel.json')
    # A retained numerical PASS is insufficient: the native log boundary can
    # reject an exit-zero run after detecting illegal DMA.
    require(checks.get('status') == 'passed' and checks.get('child_exit_code') == 0
            and checks.get('diagnostics', {}).get('illegal_dma') == 0
            and checks.get('numerics_only') is False, 'Missing clean full-bank CaModel qualification')
    current = source_hashes(gen)
    unchanged(checks.get('source_sha256_before'), current, 'Qualified generation sources')
    unchanged(checks.get('source_sha256_after'), current, 'Qualified generation sources')
    accept(checks, checks['child_exit_code'], checks['diagnostics'], current, current)
    probes = full_row_probes(load(gen/'project/policy.json'))
    for shape in probes:
        require({c['seed'] for c in checks['checks'] if c['dtype']=='float32' and c['shape']==list(shape)}=={19,23},
                'Missing full-row or boundary control')
    require(checks['candidate_sha256']==sha(gen/'project/candidate.py')==g['candidate_sha256']
            and checks['policy_sha256']==sha(gen/'project/policy.json')==g['policy_sha256']
            and len(checks['checks'])==24+2*len(probes)
            and all(c['positive'] and c['suppressed_rejected']
                    and c['real_launches']==c['intercepted_launches']==1 for c in checks['checks']),
            'Missing CaModel numerical evidence')
    cpp = src/'csrc/ops/swi_glu/op_kernel'
    require(checks['sources']==ci.hashes(cpp)
            and set(load(src/'flags.json'))=={key(*s) for s in all_specs(load(gen/'project/policy.json'))}, 'Incomplete lowering')
    catalog = load(gen/'catalog.json')
    require(catalog['benchmark_version'] in ('2.0.0','2.1.0') and catalog['required_cases']==20, 'Unexpected catalog')
    entries = {str(p.relative_to(src)):p.read_bytes() for p in [src/'flags.json', *sorted(cpp.glob('*.cpp'))]}
    entries.update({
        'build.sh':b'#!/usr/bin/env bash\nset -eu\ncd "$(dirname "$0")"\npython3 build_device.py\npython3 setup.py bdist_wheel\n',
        'setup.py':b'from setuptools import setup\nsetup(name="cann_bench", version="2.0.0", packages=["cann_bench"], package_data={"cann_bench":["device/*.o","policy.json"]})\n',
        'cann_bench/__init__.py':b'from .swi_glu import swi_glu\n',
        'cann_bench/swi_glu.py':(HERE/'device_host.py').read_bytes(),
        'cann_bench/policy.json':(gen/'project/policy.json').read_bytes(),
        'cann_bench/geometry.py':(HERE/'geometry.py').read_bytes(),
        'build_device.py':(HERE/'build_device.py').read_bytes(),
        'pyasc_source/candidate.py':(gen/'project/candidate.py').read_bytes(),
        'pyasc_source/packed-bf16-swiglu.py':(gen/'context/skills/asctile-api-patterns/templates/packed-bf16-swiglu.py').read_bytes()})
    archive = package/'submission.zip'
    with zipfile.ZipFile(archive,'x') as z:
        for name,data in sorted(entries.items()):
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o100755 if name=='build.sh' else 0o100644)<<16
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED)
    require(archive.stat().st_size<=10*1024**2,'Source package exceeds service size limit')
    for name in ('catalog.json','task.json'):
        shutil.copyfile(gen/name,package/name)
    save(package/'manifest.json',{'scope':'CI-generated SwiGLU, separate packed BF16 template',
         'catalog':catalog,'target_hardware':'950pr','aggregation_token':os.environ['CANNBENCH_AGGREGATION_TOKEN'],
         'generation':g,'simulator_image':IMAGE,'evaluation_revision':ci.ci_identity()['GITHUB_SHA'],
         'files':{p:sha(package/p) for p in ('submission.zip','catalog.json','task.json')}})


def verify_swiglu_transition(old_catalog, old_task, catalog, task):
    """2.1 makes explicit the output dtype already produced by this SwiGLU launcher."""
    require(old_catalog['benchmark_slug']=='official-tasks' and old_catalog['benchmark_version']=='2.0.0'
            and old_catalog['required_cases']==20 and old_task['function_name']=='swi_glu'
            and catalog==dict(old_catalog,benchmark_version='2.1.0'), 'Different official SwiGLU catalog')
    expected = deepcopy(old_task)
    output, = expected['proto']['outputs']
    require('dtype_relation' not in output, 'Unexpected source output contract')
    output['dtype_relation'] = {'kind':'same_as','input':'input'}
    require(expected==task, 'Public SwiGLU task, cases or numerical contract changed')


def service_package(root, queue, stage, version):
    prepared = root/'prepared'
    if version:
        require(stage=='reopen' and version=='2.1.0', 'A catalog transition requires a prepared 2.1.0 submission')
    migrated = root/'service-package'
    if not version and not (stage=='resume' and migrated.exists()):
        return prepared
    original = ci.smoke.verify(prepared)
    if migrated.exists():
        saved = ci.smoke.verify(migrated)
        require(saved['qualified_manifest_sha256']==sha(prepared/'manifest.json')
                and saved['files']['submission.zip']==original['files']['submission.zip']
                and saved['catalog']['benchmark_version']=='2.1.0', 'Qualified checkpoint differs')
        return migrated
    catalog, task = ci.smoke.task_snapshot(queue)
    require(original['target_hardware']==queue.target_hardware=='950pr', 'Hardware target changed')
    verify_swiglu_transition(original['catalog'],load(prepared/'task.json'),catalog,task)
    with zipfile.ZipFile(prepared/'submission.zip') as z:
        require(z.read('cann_bench/swi_glu.py')==(HERE/'device_host.py').read_bytes(),
                'Cannot verify output dtype on this archived host launcher')
    # Publish the new snapshot atomically. The original manifest stays bound to the original catalog.
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        package = Path(tmp)/'package'
        package.mkdir()
        os.link(prepared/'submission.zip',package/'submission.zip')
        save(package/'catalog.json',catalog)
        save(package/'task.json',task)
        files = {name:sha(package/name) for name in ('submission.zip','catalog.json','task.json')}
        save(package/'manifest.json',dict(original, catalog=catalog, files=files,
             qualified_manifest_sha256=sha(prepared/'manifest.json'),
             qualification_benchmark_version=original['catalog']['benchmark_version']))
        package.rename(migrated)
    return migrated


def service(root, stage):
    package = root/'prepared'
    manifest = ci.smoke.verify(package)
    queue = ci.ci_client()
    LEDGER.mkdir(parents=True,exist_ok=True)
    requested_version = os.environ.get('PREPARED_CATALOG_VERSION','')
    require(not requested_version or stage=='reopen', 'Select performance-submit-prepared for a catalog transition')
    if stage in ('submit','reopen'):
        if stage=='reopen':
            origin = os.environ['EVALUATION_RUN']
            require(re.fullmatch(r'[1-9][0-9]*',origin), 'Invalid prepared evaluation ID')
            with tempfile.TemporaryDirectory() as folder:
                subprocess.run(['gh','run','download',origin,'--repo',os.environ['GITHUB_REPOSITORY'],
                                '--name','swiglu-prepared-'+origin,'--dir',folder],check=True)
                archived, = Path(folder).rglob('submission.zip')
                require(sha(archived)==manifest['files']['submission.zip'], 'Central prepared package differs')
        else:
            require(os.environ.get('PREPARED_ARTIFACT_ID'), 'Missing central prepared checkpoint')
        status = queue._client._get_json('/api/public/service-status')['maintenance']
        require(status['enabled'] is False, 'CANNBench maintenance active; retain the prepared package')
        ci.require_quota(queue._client.get_credits())
        with (LEDGER/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            package = service_package(root,queue,stage,requested_version)
            manifest = ci.smoke.verify(package)
            state = ci.smoke.submit_once(queue,package,LEDGER,retry_maintenance=stage=='reopen')
        print('CANNBench job',state.get('job_id'), 'submission',state.get('submission_id'),flush=True)
    else:
        package = service_package(root,queue,stage,requested_version)
        manifest = ci.smoke.verify(package)
    end = time.monotonic()+4*3600
    visible_by = time.monotonic()+300
    while True:
        with (LEDGER/'lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                code, result = ci.smoke.collect(queue,package,LEDGER)
            except Exception as exc:
                # A confirmed POST can precede job visibility on the read replica.
                if getattr(exc,'status',None)!=404 or time.monotonic()>=visible_by:
                    raise
                recorded = load(package/'state.json')
                if not recorded.get('job_id') or not recorded.get('submission_id'):
                    raise
                code, result = 2, dict(recorded,status='awaiting job visibility')
        print(ci.smoke.now(),result.get('job_id'),result.get('status','terminal'), flush=True)
        if code!=2:
            break
        if time.monotonic()>=end:
            raise TimeoutError('Service is still pending; use performance-resume with the evaluation run ID')
        time.sleep(60)
    job = load(package/'job.json')['job']
    evaluated = ci.download(queue, job['submission_download_url'])
    require(hashlib.sha256(evaluated).hexdigest()==manifest['files']['submission.zip'],
            'CANNBench returned a different submission')
    if code:
        save(package/'job-logs.json',queue._client.get_job_logs(job['id']))
        raise ValueError('CANNBench failed; inspect the retained job and logs')
    require(job.get('case_set','standard')=='standard','Unexpected CANNBench case set')
    op, = job['results']['operators']
    require(op['anti_cheat_failed_cases']==0 and math.isfinite(job['result_score']), 'Invalid service score')
    cases = []
    for case in op['cases']:
        device = case['op_times']['device_kernels']
        compute = set(device) & {'rows','slab','flat','_swiglu_pair_groups'}
        require(len(compute)==1 and case['accuracy']['passed'] is True,'Cannot attribute hardware case')
        name, = compute
        service_fields = {k:case[k] for k in ('perf_score','t_hw_us') if case.get(k) is not None}
        require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in service_fields.values()),
                'Invalid CANNBench case metric')
        cases.append({**service_fields,'id':case['case_id'],'kernel':name,'elapsed_us':case['elapsed_us'],
                      'baseline_us':case['baseline_perf_us']})
    require({c['id'] for c in cases}=={f'level1/swi_glu_{i}' for i in range(1,21)}
            and len(cases)==20 and {c['id'] for c in cases if c['kernel']=='_swiglu_pair_groups'}=={'level1/swi_glu_12'},
            'Wrong hardware cohort')
    generated = [c for c in cases if c['kernel']!='_swiglu_pair_groups']
    require(all(math.isfinite(c[k]) and c[k]>0 for c in cases for k in ('elapsed_us','baseline_us')),
            'Invalid per-case timing')
    public = root/'public'
    public.mkdir(exist_ok=True)
    shutil.copyfile(package/'submission.zip',public/'submission.zip')
    shutil.copyfile(root/'generation/project/candidate.py',public/'candidate.py')
    g = manifest['generation']
    attribution = {k:g[k] for k in ('policy_sha256','skills_revision','kernel_origin','policy_profile','kernel_profile_sha256') if k in g}
    if 'policy_sha256' in g:
        require(sha(root/'generation/project/policy.json')==g['policy_sha256'], 'Published policy identity differs')
    save(public/'performance.json',{'schema':1,**attribution,'job_id':job['id'],'submission_id':job['submission_id'],
        'origin_run':g['run_id'],'origin_attempt':g['run_attempt'],'revision':g['revision'],
        'evaluation_run':os.environ['EVALUATION_RUN'],'evaluation_revision':manifest['evaluation_revision'],
        'candidate_sha256':g['candidate_sha256'],'package_sha256':manifest['files']['submission.zip'],
        'model':g['model'],'opencode':g['opencode'],'loaded_skills':g['loaded_skills'],
        'service_score':job['result_score'],'cases':cases,'benchmark_version':job['benchmark_version'],
        'qualification_benchmark_version':manifest.get('qualification_benchmark_version',manifest['catalog']['benchmark_version']),
        'hardware':job['target_hardware'],'simulator_image':IMAGE,'ca_model_pairs':len(load(root/'qualification/camodel.json')['checks']),
        'queued_at':job['queued_at'],'claimed_at':job['claimed_at'],'finished_at':job['finished_at'],
        'generated_ratio':geometric_mean(c['baseline_us']/c['elapsed_us'] for c in generated)})
    (public/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(public.iterdir()) if p.name!='SHA256SUMS'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['restore','package','submit','resume','reopen'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--origin',default='')
    args = parser.parse_args()
    ci.ci_identity()
    if args.stage=='restore':
        restore(args.root,args.origin)
    elif args.stage=='package':
        package(args.root)
    else:
        service(args.root,args.stage)
