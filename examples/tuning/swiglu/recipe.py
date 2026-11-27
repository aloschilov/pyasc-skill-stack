#!/usr/bin/env python3
"""Prepare and run one bounded SwiGLU consumer tuning session.

No downloads, submissions, background jobs or automatic retries. Engine source
is a separately published hash-pinned artifact; runtime is an existing image.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

if not __debug__:
    raise RuntimeError('Optimized Python disables required checks; use python3 -B')

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[2]
SKILLS = ('asctile-cannbench-kernel', 'asctile-syntax-constraints',
          'asctile-api-patterns', 'asctile-build-run-verify')
CODE = ('tool.py', 'parameter_guard.py', 'consumer_interface.py', 'selection.py',
        'session_driver.py', 'non_tool_budget.py', 'tool_delivery.py', 'native_preflight.py', 'terminal_usage.py', 'baseline.py',
        'prompt.txt', 'TUNING_TASK.md', 'timing-template.json',
        'supervisor-template.py.txt', 'engine-lock.json', 'fixtures.json')
PREFLIGHT_VARIANT = '02d85c679c30b1ac5167d0799a92a1f4c6f2b9eb40717d1a4a6e383f92939dee'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def git(*args):
    result = subprocess.run(['git', '-C', str(CHECKOUT), *args], capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError('Local Git operation failed: ' + ' '.join(args[:2]))
    return result.stdout


def extract_sources(archive, destination, lock):
    """Extract only the exact regular members; never links or traversal paths."""
    if sha(archive) != lock['sha256'] or Path(archive).stat().st_size != lock['bytes']:
        raise ValueError('Engine artifact differs from engine-lock.json')
    expected = set(lock['source_files']) | {'source-index.json', 'README.md'}
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != expected:
            raise ValueError('Unexpected engine member inventory')
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or member.size > 8*1024*1024:
                raise ValueError('Unsafe engine member')
            data = source.extractfile(member).read()
            expected_hash = lock['source_files'].get(member.name)
            if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError('Engine source hash mismatch: ' + member.name)
            target = destination/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if sha(destination/'source-index.json') != lock['source_index_sha256']:
        raise ValueError('Engine source index changed')
    if load(destination/'source-index.json') != lock['source_files']:
        raise ValueError('Engine source index is not the locked inventory')


def install_context(directory, revision):
    source = directory/'skill-source'
    source.mkdir()
    raw = git('archive', revision, 'install.sh', *(f'skills/{name}' for name in SKILLS))
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Skill archive must contain regular source files only')
            target = source/path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    context = directory/'context'
    result = subprocess.run(['bash', str(source/'install.sh'), '--target', str(context/'skills')],
                            capture_output=True, text=True, timeout=30)
    (directory/'install.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise ValueError('Existing installer failed; retained install.log')
    pins = {}
    for name in SKILLS:
        installed = context/'skills'/name
        if not installed.is_symlink() or installed.resolve() != (source/'skills'/name).resolve():
            raise ValueError('Installer link differs from the pinned skill source')
        for path in sorted(installed.resolve().rglob('*')):
            if path.is_file():
                pins['skills/'+name+'/'+str(path.relative_to(installed.resolve()))] = sha(path)
    save(directory/'context-hashes.json', pins)
    return context


def prepare(directory, engine, skills_ref, state_dir):
    directory = directory.resolve()
    state_dir = state_dir.expanduser().resolve()
    if directory.exists() or directory.is_relative_to(CHECKOUT.resolve()) or state_dir.is_relative_to(directory):
        raise ValueError('Use a new external consumer directory and the existing project state directory')
    revision = git('rev-parse', '--verify', skills_ref+'^{commit}').decode().strip()
    if not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Full skill revision required')
    lock = load(HERE/'engine-lock.json')
    # Check the large immutable input before creating a partially prepared tree.
    if sha(engine) != lock['sha256']:
        raise ValueError('Wrong engine artifact; preparation did not start')
    directory.mkdir(parents=True)
    recipe = directory/'recipe'
    recipe.mkdir()
    for name in CODE:
        shutil.copyfile(HERE/name, recipe/name)
    staging = directory/'source-artifact'
    staging.mkdir()
    extract_sources(engine, staging, lock)
    shutil.move(staging/'engine', recipe/'engine')
    # Reuse the checkout's existing verification source, never a Git-vendored copy.
    for name, expected in lock['checker_hashes'].items():
        if sha(CHECKOUT/name) != expected:
            raise ValueError('Checkout checker differs from the recipe: ' + name)
        target = recipe/'engine/probe/checker'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CHECKOUT/name, target)
    if sha(recipe/'engine/probe/fixtures.json') != lock['fixture_sha256']:
        raise ValueError('Fixture source differs')
    if sha(recipe/'baseline.py') != lock['baseline_sha256']:
        raise ValueError('Supplied baseline differs')
    context = install_context(directory, revision)
    sources = {str(p.relative_to(recipe)):sha(p) for p in sorted(recipe.rglob('*')) if p.is_file()}
    save(recipe/'recipe-input-hashes.json', sources)
    # Bind the exact executor source closure without copying it into the recipe.
    common = {str(p.relative_to(CHECKOUT)):sha(p) for p in sorted((CHECKOUT/'tests/tools').glob('*.py'))}
    common['integrations/opencode/profiles/cloud-glm-5.3-low.json'] = sha(CHECKOUT/'integrations/opencode/profiles/cloud-glm-5.3-low.json')
    prepared = {'schema_version':1, 'status':'prepared', 'checkout':str(CHECKOUT),
                'checkout_revision':git('rev-parse','HEAD').decode().strip(), 'entry_sha256':sha(Path(__file__)),
                'checkout_dirty':bool(git('status','--porcelain')),
                'directory':str(directory), 'context':str(context), 'skills_revision':revision,
                'state_dir':str(state_dir), 'recipe_index_sha256':sha(recipe/'recipe-input-hashes.json'),
                'context_index_sha256':sha(directory/'context-hashes.json'),
                'engine_sha256':lock['sha256'], 'common_checkout_sources':common,
                'mode':'unexecuted', 'cpu_calibration':'not-run', 'native_preflight':'not-run'}
    save(directory/'prepared.json', prepared)
    return prepared


def verify(directory):
    directory = directory.resolve()
    prepared = load(directory/'prepared.json')
    if Path(prepared['checkout']).resolve() != CHECKOUT.resolve() or prepared['directory'] != str(directory):
        raise ValueError('Prepared directory belongs to another checkout/location')
    if sha(Path(__file__)) != prepared['entry_sha256'] or git('rev-parse','HEAD').decode().strip() != prepared['checkout_revision']:
        raise ValueError('Preparing entry or checkout revision changed')
    recipe = directory/'recipe'
    if sha(recipe/'recipe-input-hashes.json') != prepared['recipe_index_sha256']:
        raise ValueError('Recipe source index changed')
    for name, expected in load(recipe/'recipe-input-hashes.json').items():
        path = (recipe/name).resolve()
        if not path.is_relative_to(recipe) or sha(path) != expected:
            raise ValueError('Prepared recipe source changed: ' + name)
    for name, expected in prepared['common_checkout_sources'].items():
        if sha(CHECKOUT/name) != expected:
            raise ValueError('Executor checkout source changed: ' + name)
    if sha(directory/'context-hashes.json') != prepared['context_index_sha256']:
        raise ValueError('Installed skill index changed')
    for name, expected in load(directory/'context-hashes.json').items():
        if sha(directory/'context'/name) != expected:
            raise ValueError('Installed skill changed: ' + name)
    return prepared


def modules(recipe):
    # All helper sources were just verified. Resolve their sibling imports from
    # the prepared closure, not the consumer's writable project.
    sys.path.insert(0, str(recipe))
    spec = importlib.util.spec_from_file_location('owned_swiglu_tool', recipe/'tool.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready(prepared):
    if sys.platform != 'linux' or os.getuid() == 0:
        raise ValueError('Execution requires an ordinary Linux user; preparation is portable')
    if prepared['checkout_dirty'] or git('status','--porcelain').strip():
        raise ValueError('Commit/review the recipe first; execution requires the prepared clean checkout')
    running = subprocess.check_output(['docker','ps','-q'],text=True,timeout=10)
    if running.strip():
        raise ValueError('Docker workload active; no shared machine availability established')
    image = load(HERE/'engine-lock.json')['runtime']['image']
    rows = json.loads(subprocess.check_output(['docker','image','inspect',image],text=True,timeout=10))
    if len(rows)!=1 or image not in rows[0].get('RepoDigests',[]):
        raise ValueError('Exact pinned image is unavailable; never pull or substitute automatically')
    return {'image':image,'image_id':rows[0]['Id'],'uid':os.getuid(),'gid':os.getgid(),'docker_empty':True}


def owned_cpu(directory, helper):
    scope=directory/'cpu-calibration'
    scope.mkdir()
    shutil.copytree(directory/'recipe/engine/probe',scope/'probe')
    # Existing CPU script imports torch/comparator only, never asc or candidate.
    inner=['timeout','-k','5','90','python3.12','/probe/cpu-preflight.py','/out/cpu-preflight.json']
    helper.prepare_command(scope,['/probe:/probe:ro'],inner,120,0)
    result=subprocess.run([sys.executable,'-B',str(scope/'owned-invocation.py')],capture_output=True,text=True,timeout=130)
    (scope/'supervisor.stdout').write_text(result.stdout)
    (scope/'supervisor.stderr').write_text(result.stderr)
    terminal=load(scope/'terminal.json')
    if result.returncode or terminal.get('errors') or terminal.get('exit_code')!=0 or not all(terminal.get(k)is True for k in ('original_client_reaped','original_container_absent','name_absent')):
        raise ValueError('Original CPU preflight failed; preserve it, no retry')
    receipt=scope/'out/cpu-preflight.json'
    helper.cpu_gate(receipt,load(scope/'probe/fixtures.json')['small-tail'])
    lines=(scope/'run.log').read_text().splitlines()
    summaries=[json.loads(line)for line in lines if line.startswith('{') and 'cpu_fixture_seed_checks'in line]
    if len(summaries)!=1 or summaries[0]!={'cpu_fixture_seed_checks':6,'status':'passed','kernel_launches':0,'model_imported':False}:
        raise ValueError('CPU phase did not prove six checks and no Model import/launch')
    return receipt


def configure_owned(directory, prepared, attempt, cpu, seconds):
    recipe=directory/'recipe';project=attempt/'project';project.mkdir()
    shutil.copyfile(recipe/'baseline.py',project/'baseline.py')
    shutil.copyfile(recipe/'TUNING_TASK.md',project/'TUNING_TASK.md')
    # Model session_driver creates these paths itself. For native preflight they
    # are explicitly provisioned before the zero-model client starts.
    config={'attempt':str(attempt),'project':str(project),'work':str(directory/'evaluation'),
            'ledger':prepared['state_dir'],'cpu_receipt':str(cpu)}
    save(recipe/'session-config.json',config)
    save(attempt/'session-clock.json',{'run_dir':str(attempt),'deadline_monotonic':time.monotonic()+seconds,'whole_session_seconds':seconds})
    save(recipe/'admission.json',{'status':'ADMITTED','session_config_sha256':sha(recipe/'session-config.json'),
         'recipe_input_sha256':prepared['recipe_index_sha256'],'cpu_receipt_sha256':sha(cpu),'deadline_epoch':time.time()+seconds,
         'origin':'Derived by the explicit public execute entry after source/availability/ledger checks; not hand-authored review metadata'})
    return config


def missing_container(cid):
    result=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=10)
    return result.returncode==1 and result.stdout.strip()=='[]' and result.stderr.strip().lower()=='error: no such object: '+cid


def retire_owned(directory, prepared, attempt, lease, provider=False):
    """Retire only this terminal owner; missing/unknown evidence retains latch."""
    for scope in [directory/'cpu-calibration',*(directory/'evaluation').glob('*')]:
        if not scope.is_dir() or not (scope/'run-intent.json').exists():
            continue
        terminal=load(scope/'terminal.json')
        if terminal.get('errors') or not all(terminal.get(k)is True for k in ('original_client_reaped','original_container_absent','name_absent')):
            raise ValueError('Owned evaluator cleanup is unresolved')
        if not missing_container(terminal['container_id']):
            raise ValueError('Original owned container absence is not confirmed')
    command_name='generation'if provider else'native-preflight'
    command_path=attempt/(command_name+'.command.json')
    if command_path.exists():
        command=load(command_path)
        if command.get('status')not in ('completed','blocked','interrupted') or type(command.get('exit_code'))is not int:
            raise ValueError('Original client did not have a complete terminal outcome')
        pid=command.get('pid')
        if type(pid)is not int:
            raise ValueError('Missing original client identity')
        try:os.killpg(pid,0)
        except ProcessLookupError:pass
        else:raise ValueError('Original client process group still exists')
    if provider:
        from generation_provider import response_usage
        from terminal_usage import classify
        if not classify(attempt,response_usage)['outcome_known']:
            raise ValueError('Original provider outcome unresolved; retain shared latch')
    if load(lease)!={'run_dir':str(attempt),'status':'active'}:
        raise ValueError('Executor latch ownership changed')
    save(directory/'owner-cleanup.json',{'status':'terminal-owned-cleanup-confirmed','attempt':str(attempt),'provider_accounting_checked':provider})
    lease.unlink()


def preflight(directory):
    directory=directory.resolve();prepared=verify(directory)
    if (directory/'execution-intent.json').exists():
        raise ValueError('Existing preflight/run must be inspected; never replayed')
    readiness=ready(prepared)
    sys.path.insert(0,str(CHECKOUT/'tests/tools'))
    from generation_executor import exclusive
    helper=modules(directory/'recipe');attempt=directory/'attempt'
    with exclusive(prepared['state_dir'],attempt)as lease:
        attempt.mkdir(mode=0o700)
        save(directory/'execution-intent.json',{'mode':'zero-model-native-preflight','readiness':readiness,'no_retry':True})
        try:
            cpu=owned_cpu(directory,helper)
            config=configure_owned(directory,prepared,attempt,cpu,1650)
            baseline=(directory/'recipe/baseline.py').read_text()
            old='    T = max(T, 1)\n'
            new='    T = min(T, _pow2_ceil((R - 1) // aiv + 1))\n'+old
            if baseline.count(old)!=1:raise ValueError('Retained preflight source anchor changed')
            variant=baseline.replace(old,new)
            if hashlib.sha256(variant.encode()).hexdigest()!=PREFLIGHT_VARIANT:
                raise ValueError('Retained preflight derivative hash mismatch')
            (attempt/'retained-variant.py').write_text(variant)
            spec=importlib.util.spec_from_file_location('native_swiglu_preflight',directory/'recipe/native_preflight.py')
            native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
            result=native.run(CHECKOUT,directory/'recipe',Path(config['project']),Path(prepared['context']),Path(config['work']),attempt,time.monotonic()+1500)
            retire_owned(directory,prepared,attempt,lease)
            report={'status':'passed','synthetic_native_fixture':True,'model_calls':0,'service_submissions':0,
                    'recipe_index_sha256':prepared['recipe_index_sha256'],'engine_sha256':prepared['engine_sha256'],
                    'cpu_receipt_sha256':sha(cpu),'cpu_receipt':str(cpu),'native_proof_sha256':sha(attempt/'native-preflight-proof.json'),
                    'owner_cleanup_sha256':sha(directory/'owner-cleanup.json'),'actual_outcomes':{a:load(directory/'evaluation'/a/'public-receipt.json')for a in ('baseline','variant','compare','decision')}}
            save(directory/'preflight.json',report)
            return report
        except BaseException as exc:
            cleanup='unresolved-preserved'
            if lease.exists() and load(lease)=={'run_dir':str(attempt),'status':'active'}:
                try:
                    retire_owned(directory,prepared,attempt,lease)
                    cleanup='confirmed-and-retired'
                except Exception:
                    pass  # Original terminal/identity evidence remains authoritative.
            save(directory/'preflight-failure.json',{'status':'failed'if cleanup=='confirmed-and-retired'else'unresolved',
                 'exception_type':type(exc).__name__,'cleanup':cleanup,'model_calls':0,'no_replay':True})
            raise



def run_consumer(directory, preflight_directory):
    directory=directory.resolve();prepared=verify(directory)
    if (directory/'execution-intent.json').exists():raise ValueError('Existing session is never replayed')
    prior=verify(preflight_directory.resolve());proof=load(preflight_directory/'preflight.json')
    if proof['status']!='passed'or proof['model_calls']!=0 or proof['recipe_index_sha256']!=prepared['recipe_index_sha256']or prior['recipe_index_sha256']!=prepared['recipe_index_sha256']:
        raise ValueError('A matching complete clean-consumer preflight is required')
    if sha(preflight_directory/'attempt/native-preflight-proof.json')!=proof['native_proof_sha256']or sha(preflight_directory/'owner-cleanup.json')!=proof['owner_cleanup_sha256']:
        raise ValueError('Preflight provenance changed')
    readiness=ready(prepared)
    if subprocess.check_output(['opencode','--version'],text=True,timeout=10).strip()!='1.19.0':
        raise ValueError('OpenCode 1.19.0 required')
    sys.path.insert(0,str(CHECKOUT/'tests/tools'))
    from generation_plan import build_plan
    plan=build_plan(root=CHECKOUT,model_override={'profile':'cloud-glm-5.3-low','response_model':'glm-5-3-260814'})
    plan['skills']=list(SKILLS);plan['runtime']=load(HERE/'engine-lock.json')['runtime']
    profile=load(CHECKOUT/'integrations/opencode/profiles/cloud-glm-5.3-low.json')
    attempt=directory/'attempt';recipe=directory/'recipe'
    # run_once acquires the existing shared executor lock and creates attempt.
    config={'attempt':str(attempt),'project':str(attempt/'project'),'work':str(directory/'evaluation'),
            'ledger':prepared['state_dir'],'cpu_receipt':str(directory/'cpu-preflight.json')}
    cpu=Path(proof['cpu_receipt'])
    if sha(cpu)!=proof['cpu_receipt_sha256']:raise ValueError('Qualified CPU receipt changed')
    shutil.copyfile(cpu,directory/'cpu-preflight.json')
    save(recipe/'session-config.json',config)
    save(recipe/'admission.json',{'status':'ADMITTED','session_config_sha256':sha(recipe/'session-config.json'),
         'recipe_input_sha256':prepared['recipe_index_sha256'],'cpu_receipt_sha256':sha(cpu),'deadline_epoch':time.time()+2820,
         'origin':'Derived by explicit run command after immutable preflight, source and ownership checks'})
    save(directory/'execution-intent.json',{'mode':'one-glm-5.3-tuning-session','preflight_sha256':sha(preflight_directory/'preflight.json'),'readiness':readiness,'no_retry':True})
    sys.path.insert(0,str(recipe))
    spec=importlib.util.spec_from_file_location('consumer_tuning_session',recipe/'session_driver.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    failure=None
    try:
        driver.run_once(CHECKOUT,recipe,attempt,Path(prepared['context']),Path(prepared['state_dir']),plan,profile)
    except Exception as exc:
        failure=type(exc).__name__  # Never serialize provider/config exception text.
    finally:
        from generation_provider import response_usage
        from terminal_usage import classify
        try:
            classification=classify(attempt,response_usage)
        except Exception as exc:
            classification={'outcome_known':False,'kind':'invalid-retained-accounting','exception_type':type(exc).__name__}
        cleanup=None
        ledger=Path(prepared['state_dir'])
        import fcntl
        try:
            with (ledger/'executor.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                lease=ledger/'active.json'
                owned=lease.exists()and load(lease)=={'run_dir':str(attempt),'status':'active'}
                if owned and classification['outcome_known']:
                    try:
                        retire_owned(directory,prepared,attempt,lease,provider=True)
                        cleanup='confirmed-and-retired'
                    except Exception as exc:
                        cleanup='unresolved-'+type(exc).__name__
                elif not owned:
                    cleanup='no-owned-latch; unrelated state untouched'
                else:
                    cleanup='unknown-preserved'
        except (OSError, ValueError) as exc:
            cleanup='cleanup-lock-or-evidence-unavailable-'+type(exc).__name__
        outcome=load(attempt/'tuning-accounting.json')if (attempt/'tuning-accounting.json').exists()else{}
        resolved=classification['outcome_known'] and cleanup=='confirmed-and-retired'
        success=resolved and failure is None and outcome.get('procedure',{}).get('completed') is True
        blocked_before_dispatch=(classification['outcome_known'] and classification.get('requests')==0 and cleanup=='no-owned-latch; unrelated state untouched')
        report={'status':'completed'if success else'failed'if resolved else'blocked'if blocked_before_dispatch else'unresolved',
                'exception_type':failure,'classification':classification,'cleanup':cleanup,'procedure':outcome.get('procedure'),
                'skills_revision':prepared['skills_revision'],'model':'glm-5-3-260814','hardware':'not-run'}
        save(directory/'result.json',report)
    return report



def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('prepare', help='offline source verification and installed consumer setup')
    prep.add_argument('--directory', type=Path, required=True)
    prep.add_argument('--engine', type=Path, required=True)
    prep.add_argument('--skills-ref', default='HEAD')
    prep.add_argument('--state-dir', type=Path, required=True, help='existing generation executor/CI state directory; never a replacement for an unresolved ledger')
    check = commands.add_parser('verify', help='offline recheck of all prepared source and installed skill hashes')
    check.add_argument('--directory', type=Path, required=True)
    pre=commands.add_parser('preflight',help='one CPU and zero-model native/Model evaluator preflight; consumes no model quota')
    pre.add_argument('--directory',type=Path,required=True)
    run=commands.add_parser('run',help='one fresh GLM-5.3 session using a matching completed preflight')
    run.add_argument('--directory',type=Path,required=True)
    run.add_argument('--preflight',type=Path,required=True)
    return result


def main():
    args = parser().parse_args()
    if args.action == 'prepare':
        value = prepare(args.directory, args.engine, args.skills_ref, args.state_dir)
    elif args.action=='verify':
        value=verify(args.directory)
    elif args.action=='preflight':
        value=preflight(args.directory)
    else:
        value=run_consumer(args.directory,args.preflight)
    print(json.dumps(value, indent=2))
    if args.action=='run' and value['status']!='completed':
        raise SystemExit(2 if value['status']=='blocked' else 1)


if __name__ == '__main__':
    main()
