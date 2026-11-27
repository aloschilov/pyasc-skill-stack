"""Task-local synchronous owned commands; no provider access, server or replay.

Deployment must place this recipe and session-config.json outside model-writable
project paths. Admission and CPU calibration are mandatory before any execution.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from parameter_guard import validate, BASELINE_SHA256
from consumer_interface import ACTIONS, commands
from selection import select
from non_tool_budget import CAPS, process_startticks

if not __debug__:raise RuntimeError('Assertions must remain enabled; optimized Python is unsupported')
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def publish(p,x):
    # Watchdog reads must never observe a partially written clock or receipt.
    # Unique temporary file and link preserve the original no-overwrite rule.
    p=Path(p);temporary=p.with_name(p.name+'.publishing')
    write(temporary,x)
    os.link(temporary,p)
    temporary.unlink()

def verify_index(root,index):
    for name,h in index.items():
        p=(root/name).resolve()
        assert p.is_relative_to(root.resolve()) and p.is_file() and sha(p)==h,name

def cpu_gate(path,fixtures):
    rows=read(path);keys={f['id']+'/'+str(s) for f in fixtures for s in f['seeds']}
    assert set(rows)==keys
    for f in fixtures:
        for seed in f['seeds']:
            row=rows[f['id']+'/'+str(seed)]
            assert row['outcomes']=={'native':True,'zero':False,'poison':False}
            assert row['finite'] is True and row['dtype']=='float16' and row['shape']==f['shape']
            assert row['attrs']==f['attrs'] and row['torch_version']=='2.7.1+cpu'
            for k in ('input_sha256','native_sha256','fp64_sha256'):
                assert re.fullmatch('[a-f0-9]{64}',row[k])

def host_projection(source):
    proof=validate((ROOT/'baseline.py').read_bytes(),source)
    return dict(proof,status='APPROVED_UNCHANGED_FIRST_SESSION_SOURCE',
                review_kind='Supplied reviewed baseline plus exact parameter-only AST proof; not a claim of independent human review',
                unchanged_snapshot=True, domain='finite contiguous 2D FP16, dim=-1, frozen three fixtures only')

def copy_probe(dst,cpu,telemetry=False):
    shutil.copytree(ROOT/'engine/probe',dst)
    shutil.copyfile(cpu,dst/'cpu-preflight.json')
    plan=read(dst/'qualification-plan.json')
    plan.update(per_candidate_seconds=180,child_phase_seconds=60,inner_seconds=210,whole_seconds=240)
    if telemetry:
        for name in ('telemetry.py','qualification_observe.py'):shutil.copyfile(ROOT/'engine'/name,dst/name)
        bindings=read(ROOT/'timing-template.json')
        write(dst/'qualification-telemetry.json',{'telemetry_sha256':sha(dst/'telemetry.py'),'runtime_source_hashes':bindings['runtime_source_hashes'],'execute_kernel_sha256':plan['checker_hashes']['tests/tools/execute_kernel.py']})
        wrapper=dst/'trusted-fixture-draft.py';original=wrapper.read_text()
        old="if __name__ == '__main__':\n    main()"
        new="if __name__ == '__main__':\n    import sys\n    sys.path.insert(0, '/probe')\n    from qualification_observe import observed\n    observed(main)"
        assert original.count(old)==1
        adapted=original.replace(old,new);assert adapted.replace(new,old)==original
        wrapper.write_text(adapted);plan['wrapper_sha256']=sha(wrapper)
    (dst/'qualification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')

def prepare_qualification(scope,source,cpu):
    scope.mkdir();copy_probe(scope/'probe',cpu,telemetry=True);(scope/'candidate').mkdir()
    (scope/'candidate/candidate.py').write_bytes(source)
    write(scope/'candidate/host-review.json',host_projection(source))
    return ['/probe:/probe:ro','/candidate:/candidate:ro'], ['timeout','-k','5','210','bash','-c','unset PYASC_DRY_RUN; exec python3.12 /probe/qualify.py small-tail']

def qualification(scope,source_hash):
    terminal=read(scope/'terminal.json');result=read(scope/'out/result.json')
    assert terminal['exit_code']==0 and terminal['original_client_reaped'] is True
    assert terminal['original_container_absent'] is True and terminal['name_absent'] is True and not terminal['errors']
    assert result['candidate_sha256']==source_hash and result['status']=='passed' and len(result['checks'])==6
    fixtures=read(scope/'probe/fixtures.json')['small-tail'];expected={(f['id'],s) for f in fixtures for s in f['seeds']}
    assert {(r['fixture'],r['seed']) for r in result['checks']}==expected
    compiled={}
    for row in result['checks']:
        assert row['exit_code']==0 and row['verification_level']=='numerically_verified' and row['launcher_calls']>0
        real=row['evidence']['execution'];negative=row['evidence']['control']
        assert real['exit_code']==0 and real['report']['mock_launch'] is False and len(real['report']['numerical_checks'])==4
        assert negative['exit_code']==1 and negative['report']['mock_launch'] is True and negative['report'].get('numerical_failure')
        for phase in (real,negative):
            assert not phase['report'].get('fatal_model_diagnostics')
            assert not re.search(r'^\s*\[(?:error|fatal)\]',phase['log'],re.M|re.I),'Unexpected native diagnostic'
        case=scope/'out'/(row['fixture']+'-'+str(row['seed']))
        assert sha(case/'candidate.py')==source_hash
        dump=case/'dumps';ub=re.findall(r'asc.memory_consumed = \{UB = ([0-9]+) : i64\}',(dump/'ascir.mlir').read_text());assert len(ub)==1
        pins={'CPP_sha256':sha(dump/'ascendc.cpp'),'ELF_sha256':sha(dump/'binary.o'),'compiler_UB_bytes':int(ub[0])}
        observed=read(case/'qualification-real-telemetry.json');suppressed=read(case/'qualification-suppressed-telemetry.json')
        assert observed['phase']=='real' and observed['status']=='passed' and observed['actual_runtime_launches_observed']==1
        assert suppressed['phase']=='suppressed' and suppressed['status']=='raised' and suppressed['actual_runtime_launches_observed']==0
        assert observed['binding']['binary_sha256']==pins['ELF_sha256'] and observed['binding']['compiler_memory_bytes']['UB']==pins['compiler_UB_bytes']
        pins['observed_binding']=observed['binding']
        pins['resource_queries']=[e for e in observed['events']if e['kind']=='resource_query']
        assert pins['resource_queries'],'Actual baseline resource query is required'
        if row['fixture'] in compiled: assert compiled[row['fixture']]==pins,'Codegen differs between seeds'
        compiled[row['fixture']]=pins
    return {'source_sha256':source_hash,'rows':compiled,'result_sha256':sha(scope/'out/result.json'),'terminal_sha256':sha(scope/'terminal.json')}

def prepare_comparison(scope,work,cpu):
    scope.mkdir();bundle=scope/'bundle';bundle.mkdir();copy_probe(bundle/'probe',cpu)
    for n in ('runtime_gate.py','telemetry.py','protocol.py','series_control.py','run_timing.py'):
        shutil.copyfile(ROOT/'engine'/n,bundle/n)
    manifest=read(ROOT/'timing-template.json');manifest['secondary']={};manifest['schedule']=[]
    for name in ('baseline','variant'):
        source=work/name/'candidate';dst=bundle/'candidates'/name;shutil.copytree(source,dst)
        manifest['secondary'][name]=qualification(work/name,sha(source/'candidate.py'))
    for fixture in read(bundle/'probe/fixtures.json')['small-tail']:
        controlled=set()
        for order in ('ABBA','BAAB'):
            for pos,arm in enumerate(order):
                manifest['schedule'].append(dict(dtype='float16',fixture_id=fixture['id'],shape=fixture['shape'],seed=19,order=order,position=pos,arm=arm,suppressed_before_warmup=arm not in controlled,warmups=1,samples=3));controlled.add(arm)
    write(bundle/'manifest.json',manifest)
    write(bundle/'input-hashes.json',{str(p.relative_to(bundle)):sha(p)for p in sorted(bundle.rglob('*'))if p.is_file()})
    return ['/bundle:/bundle:ro'],['timeout','-k','5','510','bash','-c','unset PYASC_DRY_RUN; exec python3.12 /bundle/run_timing.py float16']

def prepare_command(scope,mounts,inner,whole,launches):
    name='row-tune-'+hashlib.sha256(str(scope).encode()).hexdigest()[:20]
    image=read(ROOT/'engine/probe/qualification-plan.json')['runtime']['image']
    argv=['docker','run','--user',str(os.getuid())+':'+str(os.getgid()),'--rm','--pull=never','--network','none','--cpus','2','--memory','8g','--ulimit','nofile=65536:65536','-e','HOME=/tmp','-e','PYTHONPYCACHEPREFIX=/tmp/pycache','--workdir','/out','--name',name,'--cidfile',str(scope/'container.cid'),'--label','pyasc.owner=consumer-row-tuning','--label','pyasc.scope='+scope.name]
    for mount in mounts:
        local,target,mode=mount.split(':');argv+=['-v',str(scope/local.lstrip('/'))+':'+target+':'+mode]
    argv+=['-v',str(scope/'out')+':/out',image]+inner
    write(scope/'command.json',{'argv':argv,'whole_seconds':whole,'launches_maximum':launches})
    write(scope/'input-hashes.json',{str(p.relative_to(scope)):sha(p)for p in sorted(scope.rglob('*'))if p.is_file()})
    template=(ROOT/'supervisor-template.py.txt').read_text()
    for k,v in {'__WHOLE__':str(whole),'__CLIENT__':str(whole-25),'__LAUNCHES__':str(launches),'__COMMAND_SHA__':sha(scope/'command.json'),'__INPUT_SHA__':sha(scope/'input-hashes.json')}.items():template=template.replace(k,v)
    compile(template,'prepared-supervisor','exec');(scope/'owned-invocation.py').write_text(template)

def gates(config):
    verify_index(ROOT,read(ROOT/'recipe-input-hashes.json'))
    admission=read(ROOT/'admission.json')
    assert admission['status']=='ADMITTED' and admission['session_config_sha256']==sha(ROOT/'session-config.json')
    assert admission['recipe_input_sha256']==sha(ROOT/'recipe-input-hashes.json')
    assert time.time()<admission['deadline_epoch']
    clock=read(Path(config['attempt'])/'session-clock.json')
    assert clock['run_dir']==config['attempt'] and time.monotonic()<clock['deadline_monotonic']
    ledger=Path(config['ledger']);active=read(ledger/'active.json')
    assert Path(active['run_dir']).resolve()==Path(config['attempt']).resolve() and active['status']=='active'
    assert sha(config['cpu_receipt'])==admission['cpu_receipt_sha256']
    fixtures=read(ROOT/'engine/probe/fixtures.json')['small-tail'];cpu_gate(config['cpu_receipt'],fixtures)
    return admission

def main(action):
    tool_started=time.monotonic()
    assert action in ACTIONS
    config=read(ROOT/'session-config.json');admission=gates(config)
    fixture_ids=[f['id']for f in read(ROOT/'engine/probe/fixtures.json')['small-tail']]
    work=Path(config['work']);project=Path(config['project']);work.mkdir(exist_ok=True)
    with (work/'recipe.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scope=work/action
        for prior in work.glob('*/public-receipt.json'):
            assert read(prior).get('status')!='unknown','Unresolved prior action; original receipt only, no further action'
        if scope.exists():
            # Read the known original receipt only. No replacement or replay.
            receipt=scope/'public-receipt.json'
            assert receipt.is_file(),'Prior action unresolved; stop and reconcile original'
            print(receipt.read_text(),end='');print('RECEIPT_SHA256 '+sha(receipt));return
        clock_path=work/(action+'-clock.json')
        ticks=process_startticks(os.getpid());assert ticks is not None
        publish(clock_path,{'status':'active','action':action,'attempt':config['attempt'],
                'recipe_index_sha256':admission['recipe_input_sha256'],'native_command':commands(ROOT)[action],
                'pid':os.getpid(),'startticks':ticks,'seconds_limit':CAPS[action],'tool_monotonic_start':tool_started})
        report={'action':action,'native_command':commands(ROOT)[action],'status':'failed','evidence_level':'CaModel only',
                'tool_clock_sha256':sha(clock_path)}
        try:
            if action=='decision':
                scope.mkdir();q=qualification(work/'baseline',BASELINE_SHA256)
                comparison_receipt=read(work/'compare/public-receipt.json') if (work/'compare/public-receipt.json').exists() else {'status':'not-run'}
                assert comparison_receipt.get('status')!='unknown','Unresolved original comparison'
                result=read(work/'compare/out/result.json') if comparison_receipt.get('status')=='passed' else {'status':'not-run'}
                chosen=select(result,fixture_ids);submitted=read(project/'decision.json')
                target=sha(work/chosen['selected']/'candidate/candidate.py')
                assert submitted['selected_candidate_sha256']==target and isinstance(submitted['rationale'],str) and submitted['rationale'].strip()
                measured=sha(work/'compare/public-receipt.json') if (work/'compare/public-receipt.json').exists()else None
                assert submitted.get('comparison_receipt_sha256')==measured
                assert submitted.get('baseline_sha256')==BASELINE_SHA256
                if (work/'variant/candidate/candidate.py').exists():
                    assert submitted.get('variant_sha256')==sha(work/'variant/candidate/candidate.py')
                    proposal=read(work/'variant/hypothesis-before-evaluation.json')
                    assert all(submitted.get(k)==proposal[k]for k in ('hypothesis','parameter_change','baseline_receipt_sha256'))
                report.update(status='passed',selection=chosen,selected_candidate_sha256=target,decision_sha256=sha(project/'decision.json'),comparison_receipt_sha256=measured,baseline_sha256=BASELINE_SHA256,variant_sha256=sha(work/'variant/candidate/candidate.py') if (work/'variant/candidate/candidate.py').exists()else None)
            else:
                # Exact synchronous CPU/Model work only; do not allow an existing
                # unrelated container or silently share a simulator allocation.
                running=subprocess.check_output(['docker','ps','-q'],text=True,timeout=10)
                assert not running.strip(),'Existing Docker workload; stop rather than overlap'
                image=read(ROOT/'engine/probe/qualification-plan.json')['runtime']['image']
                subprocess.run(['docker','image','inspect',image],check=True,stdout=subprocess.DEVNULL,timeout=10)
                if action in ('baseline','variant'):
                    if action=='variant':
                        qualification(work/'baseline',BASELINE_SHA256)
                        proposal=read(project/'decision.json')
                        assert all(isinstance(proposal.get(k),str)and proposal[k].strip()for k in ('hypothesis','parameter_change'))
                        assert proposal['baseline_receipt_sha256']==sha(work/'baseline/public-receipt.json')
                    source=(ROOT/'baseline.py' if action=='baseline' else project/'candidate.py').read_bytes()
                    proof=validate((ROOT/'baseline.py').read_bytes(),source)
                    if action=='variant':assert proof['ast_changed'],'Unchanged source is a tie, not a new Model attempt'
                    mounts,inner=prepare_qualification(scope,source,config['cpu_receipt']);whole=240;launches=6
                    if action=='variant':
                        shutil.copyfile(project/'decision.json',scope/'hypothesis-before-evaluation.json')
                        (project/'candidate.py').chmod(0o444)
                else:
                    mounts,inner=prepare_comparison(scope,work,config['cpu_receipt']);whole=540;launches=96
                assert time.time()+whole+10<admission['deadline_epoch'],'No time for bounded command plus cleanup'
                clock=read(Path(config['attempt'])/'session-clock.json')
                assert time.monotonic()+whole+10<clock['deadline_monotonic'],'Evaluator would cross actual native session deadline'
                prepare_command(scope,mounts,inner,whole,launches)
                # The independently pinned supervisor owns CID/PID cleanup. Its
                # timeout exceeds the inner workload and precedes bash timeout.
                done=subprocess.run([sys.executable,'-B',str(scope/'owned-invocation.py')],capture_output=True,text=True,timeout=whole+10)
                (scope/'supervisor.stdout').write_text(done.stdout);(scope/'supervisor.stderr').write_text(done.stderr)
                terminal=read(scope/'terminal.json');assert terminal['original_container_absent'] is True and terminal['original_client_reaped'] is True and not terminal['errors'],'Unknown cleanup; no subsequent action'
                if action in ('baseline','variant'):
                    q=qualification(scope,sha(scope/'candidate/candidate.py'));report.update(status='passed',qualification=q,real_passes=6,rejecting_controls=6)
                else:
                    assert done.returncode==0;result=read(scope/'out/result.json');assert result['status']=='passed' and len(result['series'])==24 and len(result['controls'])==6
                    assert result['source_manifest_sha256']==sha(scope/'bundle/manifest.json')
                    actual_real=sum(r['report']['launcher_calls']for r in result['series'])
                    actual_samples=sum(len(r['samples'])for r in result['series'])
                    assert actual_real==96 and actual_samples==72
                    for r in result['series']:
                        assert r['report']['status']=='pass' and r['report']['launcher_calls']==4 and len(r['samples'])==3 and len(r['report']['numerical_checks'])==16
                        assert all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in r['samples'])
                    for r in result['controls']:assert r['code']==1 and r['report']['mock_launch'] is True and r['report'].get('numerical_failure')
                    report.update(status='passed',selection=select(result,fixture_ids),analysis=result['analysis'],bindings=[{'fixture':r['series']['fixture_id'],'arm':r['series']['arm'],'binding':r['binding']}for r in result['series'] if r['series']['order']=='ABBA' and r['series']['position']<2],real_passes=actual_real,rejecting_controls=len(result['controls']),samples=actual_samples,result_sha256=sha(scope/'out/result.json'),manifest_sha256=sha(scope/'bundle/manifest.json'),sources={name:sha(scope/'bundle/candidates'/name/'candidate.py')for name in ('baseline','variant')})
        except BaseException as exc:
            if not scope.exists():scope.mkdir()
            terminal=read(scope/'terminal.json') if (scope/'terminal.json').exists() else {}
            dispatched=(scope/'run-intent.json').exists()
            cleaned=terminal.get('original_container_absent') is True and terminal.get('original_client_reaped') is True and not terminal.get('errors')
            report.update(status='unknown' if dispatched and not cleaned else 'failed', error=type(exc).__name__+': '+str(exc),no_replay=True)
        report.update(tool_monotonic_start=tool_started,tool_monotonic_end=time.monotonic())
        publish(scope/'public-receipt.json',report);print((scope/'public-receipt.json').read_text(),end='');print('RECEIPT_SHA256 '+sha(scope/'public-receipt.json'))
        if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':
    assert len(sys.argv)==2;main(sys.argv[1])
