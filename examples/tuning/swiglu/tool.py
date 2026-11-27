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
import statistics
import subprocess
import sys
import time
from parameter_guard import validate, BASELINE_SHA256
from consumer_interface import ACTIONS, COMPARE_ACTIONS, commands
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

def qualification_plan():
    plan=read(ROOT/'engine/probe/qualification-plan.json')
    plan.update(per_candidate_seconds=180,child_phase_seconds=60,inner_seconds=210,whole_seconds=240)
    return plan

def copy_probe(dst,cpu,telemetry=False):
    shutil.copytree(ROOT/'engine/probe',dst)
    shutil.copyfile(cpu,dst/'cpu-preflight.json')
    plan=qualification_plan()
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

def probe_hashes(cpu_hash,telemetry=False):
    expected={str(p.relative_to(ROOT/'engine/probe')):sha(p)for p in (ROOT/'engine/probe').rglob('*')if p.is_file()}
    plan=qualification_plan();expected['cpu-preflight.json']=cpu_hash
    if telemetry:
        expected.update({name:sha(ROOT/'engine'/name)for name in ('telemetry.py','qualification_observe.py')})
        data={'telemetry_sha256':expected['telemetry.py'],'runtime_source_hashes':read(ROOT/'timing-template.json')['runtime_source_hashes'],'execute_kernel_sha256':plan['checker_hashes']['tests/tools/execute_kernel.py']}
        expected['qualification-telemetry.json']=hashlib.sha256((json.dumps(data,indent=2)+'\n').encode()).hexdigest()
        original=(ROOT/'engine/probe/trusted-fixture-draft.py').read_text()
        old="if __name__ == '__main__':\n    main()"
        new="if __name__ == '__main__':\n    import sys\n    sys.path.insert(0, '/probe')\n    from qualification_observe import observed\n    observed(main)"
        assert original.count(old)==1
        expected['trusted-fixture-draft.py']=hashlib.sha256(original.replace(old,new).encode()).hexdigest()
        plan['wrapper_sha256']=expected['trusted-fixture-draft.py']
    expected['qualification-plan.json']=hashlib.sha256((json.dumps(plan,indent=2)+'\n').encode()).hexdigest()
    return expected

def tree_hashes(root,exclude=()):
    paths=list(root.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return {str(p.relative_to(root)):sha(p)for p in paths if p.is_file()and str(p.relative_to(root))not in exclude}

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

TIMING_HELPERS=('runtime_gate.py','telemetry.py','protocol.py','series_control.py','run_timing.py')
TIMING_INNER=['timeout','-k','5','330','bash','-c','unset PYASC_DRY_RUN; exec python3.12 /bundle/run_timing.py float16']

def comparison_parent(work):
    """Derive identity from this actual session and verified source/qualification."""
    recipe_index=read(ROOT/'recipe-input-hashes.json');verify_index(ROOT,recipe_index)
    assert tree_hashes(ROOT/'engine')=={name[len('engine/'):]:h for name,h in recipe_index.items()if name.startswith('engine/')}
    config=read(ROOT/'session-config.json');admission=read(ROOT/'admission.json')
    assert Path(config['work']).resolve()==work.resolve()
    assert admission['status']=='ADMITTED' and admission['session_config_sha256']==sha(ROOT/'session-config.json')
    assert admission['recipe_input_sha256']==sha(ROOT/'recipe-input-hashes.json')
    cpu=Path(config['cpu_receipt']);assert sha(cpu)==admission['cpu_receipt_sha256']
    cpu_gate(cpu,read(ROOT/'engine/probe/fixtures.json')['small-tail'])
    parent={'session_config_sha256':sha(ROOT/'session-config.json'),'attempt':config['attempt'],
            'recipe_index_sha256':sha(ROOT/'recipe-input-hashes.json'),'fixtures_sha256':sha(ROOT/'engine/probe/fixtures.json'),
            'timing_template_sha256':sha(ROOT/'timing-template.json'),'cpu_receipt_sha256':sha(cpu),'qualifications':{}}
    for name in ('baseline','variant'):
        scope=work/name;source=scope/'candidate/candidate.py';host=scope/'candidate/host-review.json'
        assert tree_hashes(scope/'probe')==probe_hashes(parent['cpu_receipt_sha256'],telemetry=True)
        assert read(host)==host_projection(source.read_bytes())
        proof=qualification(scope,sha(source));receipt=read(scope/'public-receipt.json')
        assert receipt['action']==name and receipt['status']=='passed' and receipt['qualification']==proof
        parent['qualifications'][name]={'proof':proof,'receipt_sha256':sha(scope/'public-receipt.json'),'host_review_sha256':sha(host)}
    return parent

def comparison_manifest(parent,fixture_id):
    manifest=read(ROOT/'timing-template.json');manifest['parent_binding']=parent
    manifest['secondary']={name:row['proof']for name,row in parent['qualifications'].items()};manifest['schedule']=[]
    fixtures=read(ROOT/'engine/probe/fixtures.json')['small-tail']
    assert fixture_id in [f['id']for f in fixtures]
    manifest['comparison_fixture_id']=fixture_id
    fixture=next(f for f in fixtures if f['id']==fixture_id);controlled=set()
    for order in ('ABBA','BAAB'):
        for pos,arm in enumerate(order):
            manifest['schedule'].append(dict(dtype='float16',fixture_id=fixture['id'],shape=fixture['shape'],seed=19,order=order,position=pos,arm=arm,suppressed_before_warmup=arm not in controlled,warmups=1,samples=3));controlled.add(arm)
    return manifest

def prepare_comparison(scope,work,cpu,fixture_id):
    parent=comparison_parent(work);assert sha(cpu)==parent['cpu_receipt_sha256']
    scope.mkdir();bundle=scope/'bundle';bundle.mkdir();copy_probe(bundle/'probe',cpu)
    for name in TIMING_HELPERS:shutil.copyfile(ROOT/'engine'/name,bundle/name)
    for name in ('baseline','variant'):shutil.copytree(work/name/'candidate',bundle/'candidates'/name)
    write(bundle/'manifest.json',comparison_manifest(parent,fixture_id))
    write(bundle/'input-hashes.json',{str(p.relative_to(bundle)):sha(p)for p in sorted(bundle.rglob('*'))if p.is_file()})
    return ['/bundle:/bundle:ro'],list(TIMING_INNER)

def comparison_closure(scope,parent,fixture_id):
    """Expected bytes come from the trusted recipe/session, not the bundle index."""
    bundle=scope/'bundle';manifest=read(bundle/'manifest.json')
    assert manifest==comparison_manifest(parent,fixture_id)
    expected={'probe/'+name:h for name,h in probe_hashes(parent['cpu_receipt_sha256']).items()}
    expected.update({name:sha(ROOT/'engine'/name)for name in TIMING_HELPERS})
    for name,row in parent['qualifications'].items():
        expected['candidates/'+name+'/candidate.py']=row['proof']['source_sha256']
        expected['candidates/'+name+'/host-review.json']=row['host_review_sha256']
    expected['manifest.json']=sha(bundle/'manifest.json')
    actual=tree_hashes(bundle,exclude=('input-hashes.json',))
    assert actual==expected and read(bundle/'input-hashes.json')==expected
    assert read(scope/'command.json')==command_spec(scope,['/bundle:/bundle:ro'],TIMING_INNER,360,32)
    intent=read(scope/'run-intent.json');assert intent['argv']==read(scope/'command.json')['argv'] and intent['kernel_launches_permitted']==32
    terminal=read(scope/'terminal.json')
    assert terminal['exit_code']==0 and all(terminal[k]is True for k in ('original_client_reaped','original_container_absent','name_absent'))and not terminal['errors']
    return manifest

def comparison_result(scope,fixture_id,parent=None):
    """Validate one exact shape result; no partial or cross-candidate pooling."""
    parent=comparison_parent(scope.parent) if parent is None else parent
    manifest=comparison_closure(scope,parent,fixture_id);result=read(scope/'out/result.json')
    assert result['status']=='passed' and result['source_manifest_sha256']==sha(scope/'bundle/manifest.json')
    assert manifest['comparison_fixture_id']==fixture_id
    schedule=manifest['schedule'];expected=[];controlled=set()
    fixture=next(f for f in read(scope/'bundle/probe/fixtures.json')['small-tail']if f['id']==fixture_id)
    for order in ('ABBA','BAAB'):
        for position,arm in enumerate(order):
            expected.append(dict(dtype='float16',fixture_id=fixture_id,shape=fixture['shape'],seed=19,order=order,position=position,arm=arm,suppressed_before_warmup=arm not in controlled,warmups=1,samples=3));controlled.add(arm)
    assert schedule==expected and len(result['series'])==8 and len(result['controls'])==2
    assert len(result['analysis'])==1 and result['analysis'][0]['fixture']==fixture_id
    for index,(row,spec) in enumerate(zip(result['series'],expected)):
        sid='baseline'if spec['arm']=='A'else'variant'
        assert row['series']==dict(spec,session=sid,index=index)
        assert row['report']['status']=='pass' and row['report']['launcher_calls']==4 and len(row['samples'])==3 and len(row['report']['numerical_checks'])==16
        assert all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in row['samples'])
    ratios={}
    for order in ('ABBA','BAAB'):
        rows=[r for r in result['series']if r['series']['order']==order]
        arms={a:[statistics.median(r['samples'])for r in rows if r['series']['arm']==a]for a in 'AB'}
        assert all(len(values)==2 for values in arms.values())
        ratios[order]=math.exp(sum(math.log(v)for v in arms['A'])/2-sum(math.log(v)for v in arms['B'])/2)
    expected_analysis=[{'fixture':fixture_id,'order_A_over_B':ratios,'primary_A_over_B':math.sqrt(ratios['ABBA']*ratios['BAAB']),'order_direction_agrees':(ratios['ABBA']>=1)==(ratios['BAAB']>=1),'no_shape_pooling':True}]
    actual=result['analysis'][0];canonical=expected_analysis[0]
    assert set(actual)==set(canonical) and actual['fixture']==fixture_id
    assert set(actual['order_A_over_B'])=={'ABBA','BAAB'}
    for name in ('order_direction_agrees','no_shape_pooling'):
        assert type(actual[name])is bool and actual[name]is canonical[name]
    ratio_pairs=[(actual['order_A_over_B'][order],ratios[order])for order in ('ABBA','BAAB')]
    ratio_pairs.append((actual['primary_A_over_B'],canonical['primary_A_over_B']))
    # Host and pinned runtime may use different libm versions. Tolerance is only
    # for receipt consistency; selection always receives recomputed ratios.
    assert all(type(value)in(int,float)and math.isfinite(value)and value>0 and math.isclose(value,expected,rel_tol=1e-12,abs_tol=0)for value,expected in ratio_pairs)
    result=dict(result,analysis=expected_analysis)
    assert all(row['median']==statistics.median(row['samples'])for row in result['series'])
    assert {r['key']for r in result['controls']}=={'baseline/'+fixture_id,'variant/'+fixture_id}
    for row in result['controls']:
        assert row['code']==1 and row['report']['mock_launch'] is True and row['report'].get('numerical_failure')
    counts={'real_passes':sum(r['report']['launcher_calls']for r in result['series']),
            'samples':sum(len(r['samples'])for r in result['series']),'rejecting_controls':len(result['controls'])}
    assert counts=={'real_passes':32,'samples':24,'rejecting_controls':2}
    sources={name:sha(scope/'bundle/candidates'/name/'candidate.py')for name in ('baseline','variant')}
    assert all(manifest['secondary'][name]['source_sha256']==value for name,value in sources.items())
    return result,counts,sources

def collected_comparison(work,fixture_ids):
    """All three immutable receipts, in order, or retain the baseline."""
    assert len(fixture_ids)==3 and len(set(fixture_ids))==3
    parent=None
    receipts={action:None for action in COMPARE_ACTIONS};analysis=[];bound_sources=None;totals={'real_passes':0,'samples':0,'rejecting_controls':0}
    for action,fixture_id in zip(COMPARE_ACTIONS,fixture_ids):
        scope=work/action;path=scope/'public-receipt.json'
        if not path.exists():continue
        receipt=read(path);receipts[action]=sha(path)
        assert receipt['status']!='unknown','Unresolved original comparison; stop'
        if receipt['status']!='passed':continue
        assert receipt['action']==action and receipt['fixture_id']==fixture_id
        assert receipt['result_sha256']==sha(scope/'out/result.json') and receipt['manifest_sha256']==sha(scope/'bundle/manifest.json')
        if parent is None:parent=comparison_parent(work)
        assert receipt['parent_binding']==parent and receipt['terminal_sha256']==sha(scope/'terminal.json')
        assert receipt['command_sha256']==sha(scope/'command.json') and receipt['bundle_index_sha256']==sha(scope/'bundle/input-hashes.json')
        result,counts,sources=comparison_result(scope,fixture_id,parent)
        assert receipt['sources']==sources
        assert sources=={name:sha(work/name/'candidate/candidate.py')for name in ('baseline','variant')}
        if bound_sources is not None:assert sources==bound_sources
        bound_sources=sources;analysis.extend(result['analysis'])
        for k,v in counts.items():assert receipt[k]==v;totals[k]+=v
    if len(analysis)!=3:return {'status':'incomplete','analysis':analysis},receipts
    assert totals=={'real_passes':96,'samples':72,'rejecting_controls':6}
    assert [row['fixture']for row in analysis]==fixture_ids
    return {'status':'passed','analysis':analysis,'sources':bound_sources,'counts':totals},receipts

def command_spec(scope,mounts,inner,whole,launches):
    name='row-tune-'+hashlib.sha256(str(scope).encode()).hexdigest()[:20]
    image=read(ROOT/'engine/probe/qualification-plan.json')['runtime']['image']
    argv=['docker','run','--user',str(os.getuid())+':'+str(os.getgid()),'--rm','--pull=never','--network','none','--cpus','2','--memory','8g','--ulimit','nofile=65536:65536','-e','HOME=/tmp','-e','PYTHONPYCACHEPREFIX=/tmp/pycache','--workdir','/out','--name',name,'--cidfile',str(scope/'container.cid'),'--label','pyasc.owner=consumer-row-tuning','--label','pyasc.scope='+scope.name]
    for mount in mounts:
        local,target,mode=mount.split(':');argv+=['-v',str(scope/local.lstrip('/'))+':'+target+':'+mode]
    argv+=['-v',str(scope/'out')+':/out',image]+inner
    return {'argv':argv,'whole_seconds':whole,'launches_maximum':launches}

def prepare_command(scope,mounts,inner,whole,launches):
    write(scope/'command.json',command_spec(scope,mounts,inner,whole,launches))
    write(scope/'input-hashes.json',{str(p.relative_to(scope)):sha(p)for p in sorted(scope.rglob('*'))if p.is_file()})
    template=(ROOT/'supervisor-template.py.txt').read_text()
    for k,v in {'__WHOLE__':str(whole),'__CLIENT__':str(whole-25),'__CLEANUP_RESERVE__':str(25),'__LAUNCHES__':str(launches),'__COMMAND_SHA__':sha(scope/'command.json'),'__INPUT_SHA__':sha(scope/'input-hashes.json')}.items():template=template.replace(k,v)
    compile(template,'prepared-supervisor','exec');(scope/'owned-invocation.py').write_text(template)

class TimingBudgetExceeded(ValueError):
    """Operational time bound, never a candidate numerical failure."""


def comparison_wait_seconds(started,now):
    # Native600 > action400 >= actual preparation + wait370 + postprocessing10.
    # The360-second owner includes the335-second client and25-second cleanup.
    if now<started or now+370+10>started+CAPS['compare-1']:
        raise TimingBudgetExceeded('Comparison preparation left insufficient native timeout; no Model dispatch')
    return 370


def invoke_owned(scope,whole,action,started):
    wait=comparison_wait_seconds(started,time.monotonic()) if action in COMPARE_ACTIONS else whole+10
    return subprocess.run([sys.executable,'-B',str(scope/'owned-invocation.py')],capture_output=True,text=True,timeout=wait)


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
                result,measured=collected_comparison(work,fixture_ids)
                chosen=select(result,fixture_ids);submitted=read(project/'decision.json')
                target=sha(work/chosen['selected']/'candidate/candidate.py')
                assert submitted['selected_candidate_sha256']==target and isinstance(submitted['rationale'],str) and submitted['rationale'].strip()
                assert submitted.get('comparison_receipt_sha256s')==measured
                assert submitted.get('baseline_sha256')==BASELINE_SHA256
                if (work/'variant/candidate/candidate.py').exists():
                    assert submitted.get('variant_sha256')==sha(work/'variant/candidate/candidate.py')
                    proposal=read(work/'variant/hypothesis-before-evaluation.json')
                    assert all(submitted.get(k)==proposal[k]for k in ('hypothesis','parameter_change','baseline_receipt_sha256'))
                report.update(status='passed',selection=chosen,selected_candidate_sha256=target,decision_sha256=sha(project/'decision.json'),comparison_receipt_sha256s=measured,baseline_sha256=BASELINE_SHA256,variant_sha256=sha(work/'variant/candidate/candidate.py') if (work/'variant/candidate/candidate.py').exists()else None)
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
                    position=COMPARE_ACTIONS.index(action)
                    for prior_action in COMPARE_ACTIONS[:position]:
                        assert read(work/prior_action/'public-receipt.json')['status']=='passed','Fixed fixture order requires completed preceding shape'
                    mounts,inner=prepare_comparison(scope,work,config['cpu_receipt'],fixture_ids[position]);whole=360;launches=32
                assert time.time()+whole+10<admission['deadline_epoch'],'No time for bounded command plus cleanup'
                clock=read(Path(config['attempt'])/'session-clock.json')
                assert time.monotonic()+whole+10<clock['deadline_monotonic'],'Evaluator would cross actual native session deadline'
                prepare_command(scope,mounts,inner,whole,launches)
                # The independently pinned supervisor owns CID/PID cleanup. Its
                # timeout exceeds the inner workload and precedes bash timeout.
                done=invoke_owned(scope,whole,action,tool_started)
                (scope/'supervisor.stdout').write_text(done.stdout);(scope/'supervisor.stderr').write_text(done.stderr)
                terminal=read(scope/'terminal.json');assert terminal['original_container_absent'] is True and terminal['original_client_reaped'] is True and not terminal['errors'],'Unknown cleanup; no subsequent action'
                if action in ('baseline','variant'):
                    q=qualification(scope,sha(scope/'candidate/candidate.py'));report.update(status='passed',qualification=q,real_passes=6,rejecting_controls=6)
                else:
                    assert done.returncode==0
                    result,counts,sources=comparison_result(scope,fixture_ids[COMPARE_ACTIONS.index(action)])
                    report.update(status='passed',parent_binding=read(scope/'bundle/manifest.json')['parent_binding'],terminal_sha256=sha(scope/'terminal.json'),command_sha256=sha(scope/'command.json'),bundle_index_sha256=sha(scope/'bundle/input-hashes.json'),fixture_id=fixture_ids[COMPARE_ACTIONS.index(action)],analysis=result['analysis'],bindings=[{'fixture':r['series']['fixture_id'],'arm':r['series']['arm'],'binding':r['binding']}for r in result['series'] if r['series']['order']=='ABBA' and r['series']['position']<2],**counts,result_sha256=sha(scope/'out/result.json'),manifest_sha256=sha(scope/'bundle/manifest.json'),sources=sources)
            if action in COMPARE_ACTIONS and time.monotonic()>tool_started+CAPS[action]:
                raise TimingBudgetExceeded('Comparison postprocessing exceeded native timeout reserve; retained measurements are not accepted')
        except BaseException as exc:
            if not scope.exists():scope.mkdir()
            terminal=read(scope/'terminal.json') if (scope/'terminal.json').exists() else {}
            dispatched=(scope/'run-intent.json').exists()
            cleaned=terminal.get('original_container_absent') is True and terminal.get('original_client_reaped') is True and not terminal.get('errors')
            report.update(status='unknown' if dispatched and not cleaned else 'failed', error=type(exc).__name__+': '+str(exc),no_replay=True)
            if isinstance(exc,TimingBudgetExceeded):report['failure_kind']='timing_resource_budget'
            if action in COMPARE_ACTIONS and (scope/'out/result.json').is_file():
                budget_failure=read(scope/'out/result.json').get('budget_failure')
                if budget_failure:
                    report.update(failure_kind='timing_resource_budget',timing_budget_failure=budget_failure)

        report.update(tool_monotonic_start=tool_started,tool_monotonic_end=time.monotonic())
        publish(scope/'public-receipt.json',report);print((scope/'public-receipt.json').read_text(),end='');print('RECEIPT_SHA256 '+sha(scope/'public-receipt.json'))
        if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':
    assert len(sys.argv)==2;main(sys.argv[1])
