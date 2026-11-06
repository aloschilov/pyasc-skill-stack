"""Explicit, serial generation executor. No schedule, hardware or automatic resume."""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from generation_adapters import static_check, task_contract, write_probes
from generation_plan import ROOT, DEFAULT_MATRIX, build_plan, canonical, digest, resource
from generation_provider import BudgetGate, BudgetError, budget_server, known_budget_exhaustion


class Blocked(ValueError):
    pass


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def external(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents: raise ValueError('Evidence/state must be outside checkout')
    return path


def validate_plan(path, matrix=DEFAULT_MATRIX):
    original = json.loads(Path(path).read_text())
    plan = dict(original)
    signature = plan.pop('plan_sha256')
    if digest(canonical(plan)) != signature: raise ValueError('Plan digest mismatch')
    plan.pop('checkout_revision', None)
    if canonical(plan) != canonical(build_plan(matrix=matrix)): raise ValueError('Plan differs from current inputs/limits')
    return original


@contextmanager
def exclusive(state_dir, run):
    """OS lock plus durable unresolved-run latch, shared across all plans."""
    state_dir = external(state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir/'executor.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise Blocked('Another executor holds the budget lock') from None
        active = state_dir/'active.json'
        if active.exists(): raise Blocked('Unresolved previous run: inspect '+str(active))
        save(active, {'run_dir': str(run), 'status': 'active'})
        # Caller clears only after every command/request has a known outcome.
        yield active


def command(argv, directory, name, seconds, env=None, abort=None, cwd=None):
    """Kill the whole local process group on timeout/cancellation, including children."""
    if seconds <= 0: raise Blocked('Execution deadline exhausted')
    directory = Path(directory); start = time.monotonic()
    record = {'argv': argv, 'status': 'starting', 'timeout_seconds': seconds}
    meta = directory/(name+'.command.json'); save(meta, record)
    proc = None
    try:
        with (directory/(name+'.stdout')).open('wb') as out, (directory/(name+'.stderr')).open('wb') as err:
            proc = subprocess.Popen(argv, cwd=cwd or directory, env=env, stdout=out, stderr=err, start_new_session=True)
            record.update(pid=proc.pid, status='running'); save(meta, record)
            while proc.poll() is None:
                if abort and abort(): raise Blocked('Provider gate rejected the request; inspect usage.json')
                if time.monotonic()-start >= seconds: raise TimeoutError(name+' exceeded deadline')
                if out.tell()+err.tell() > 32*1024*1024: raise Blocked('Command output limit exceeded')
                time.sleep(.05)
            record.update(status='completed', exit_code=proc.returncode)
    except BaseException as exc:
        record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'blocked', error=type(exc).__name__)
        raise
    finally:
        if proc:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait(timeout=10)
            record['exit_code'] = proc.returncode
        record['duration_seconds'] = time.monotonic()-start
        for stream in ('stdout', 'stderr'):
            p = directory/(name+'.'+stream)
            if p.exists(): record[stream+'_sha256'] = digest(p.read_bytes())
        save(meta, record)
    return record


def native_trace(events, skills, *, allow_client_error=False):
    sessions, loaded = set(), set()
    for line in events.splitlines():
        event = json.loads(line)  # Non-JSON output is not accepted as attribution.
        if event.get('type') == 'error' and not allow_client_error: raise Blocked('OpenCode reported an error')
        if event.get('sessionID'): sessions.add(event['sessionID'])
        part = event.get('part') or {}; state = part.get('state') or {}
        if part.get('tool') == 'skill' and state.get('status') == 'completed':
            name = (state.get('input') or {}).get('name')
            directory = (state.get('metadata') or {}).get('dir')
            if name not in skills or not directory or Path(directory).resolve() != skills[name].resolve():
                raise Blocked('Unpinned skill load')
            loaded.add(name)
    if len(sessions) != 1 or loaded != set(skills): raise Blocked('Missing native session/skill attribution')
    return {'session_id': sessions.pop(), 'loaded_skills': sorted(loaded)}


def configure(config, plan, context, project):
    limits = plan['limits']; model = plan['model']['requested']; provider, model_id = model.split('/')
    if config['provider'][provider].get('npm') != '@ai-sdk/openai-compatible':
        raise Blocked('Budget gate supports only OpenAI-compatible Chat Completions')
    config['enabled_providers'] = [provider]
    config['small_model'] = model  # Auxiliary/title calls share the same gate and budget.
    config['skills'] = {'paths': [str(context/'skills')]}
    config['agent'] = {'build': {'steps': limits['rounds_per_attempt']}}
    config['compaction'] = {'auto': False}
    config['permission'] = {'*': 'deny', 'read': 'allow', 'glob': 'allow', 'grep': 'allow', 'skill': 'allow',
        'edit': {'*': 'deny', 'candidate.py': 'allow', str(project/'candidate.py'): 'allow'},
        'external_directory': {'*': 'deny', str(context/'skills'/'*'): 'allow'}}
    config['provider'][provider]['models'][model_id]['limit']['output'] = limits['output_tokens_per_response']
    if plan.get('intervention'):
        from generation_protocols import context_files
        # OpenCode glob/grep permissions match the search expression, not its
        # directory. Internal tool-output paths also receive native exceptions.
        # Paired trials therefore expose only explicit file reads and skill loads.
        config['permission'].update(glob='deny', grep='deny')
        config['permission']['read'] = {'*': 'deny', 'candidate.py': 'allow', str(project/'candidate.py'): 'allow',
            # OpenCode 1.19 read asks relative to the isolated Git worktree;
            # external_directory asks separately with an absolute directory glob.
            **{os.path.relpath(context/n, project): 'allow' for n in context_files(plan) if not n.startswith('integrations/')}}
        config['permission']['external_directory'] = {'*': 'deny',
            **{str(context/('skills/'+s)/'*'): 'allow' for s in plan['skills']}}
        if plan['intervention']['id'] == 'P4':
            config['permission']['external_directory'][str(context/'tests/generation/context'/'*')] = 'allow'
    return config


def run_session(attempt, plan, project, config, prompt, observe_request=None):
    """One fresh OpenCode session using the shared measured provider gate."""
    limits = plan['limits']; model = plan['model']['requested']; provider, model_id = model.split('/')
    state = attempt/'private'
    state.mkdir(mode=0o700, exist_ok=True)
    if any(state.iterdir()): raise Blocked('Existing private session is never replayed')
    (attempt/'prompt.txt').write_text(prompt)
    deadline = time.monotonic() + limits['generation_seconds_per_attempt']
    # OpenCode 1.19 resolves edit rules against its VCS worktree. Without a
    # repository it can use '/', so candidate.py never matches the allow rule.
    initialized = command(['git', '-c', 'init.templateDir=', 'init', '--quiet', str(project)],
                          attempt, 'project-init', min(20, limits['generation_seconds_per_attempt']))
    if initialized['exit_code']: raise Blocked('Could not initialize isolated OpenCode project')
    def capture_response(number, raw):
        # Private wire evidence permits reconciliation without another POST.
        path = state/f'provider-response-{number}.bin'
        with path.open('xb') as handle: handle.write(raw)
        path.chmod(0o600)
    gate = BudgetGate(os.environ['OPENCODE_BASE_URL'], os.environ['OPENCODE_API_KEY'], model_id,
                      limits, lambda value: save(attempt/'usage.json', value), deadline, capture_response,
                      response_model=plan['model']['expected_response'], observe_request=observe_request)
    save(attempt/'usage.json', gate.snapshot())
    # No inherited credentials/config/plugins: only the gate knows the provider key.
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'TMPDIR') if k in os.environ}
    env.update(HOME=str(state), XDG_DATA_HOME=str(state/'data'), XDG_CONFIG_HOME=str(state/'config'),
        XDG_CACHE_HOME=str(state/'cache'), OPENCODE_DISABLE_EXTERNAL_SKILLS='1',
        OPENCODE_DISABLE_CLAUDE_CODE_SKILLS='1', OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=str(limits['output_tokens_per_response']))
    with budget_server(gate) as endpoint:
        config['provider'][provider]['options'] = {'baseURL': endpoint, 'apiKey': '{env:PYASC_GATE_TOKEN}'}
        save(attempt/'model-config.json', config)
        env.update(PYASC_GATE_TOKEN=gate.token, OPENCODE_CONFIG_CONTENT=json.dumps(config))
        result = None
        try:
            result = command(['opencode', 'run', '--pure', '--format', 'json', '--dir', str(project), '-m', model, prompt],
                             attempt, 'generation', max(.01, deadline-time.monotonic()), env, abort=lambda: gate.error, cwd=project)
        except Blocked:
            usage = gate.snapshot()
            if not known_budget_exhaustion(usage, limits): raise
        finally:
            save(attempt/'usage.json', gate.snapshot())
    usage = gate.snapshot()
    return result, usage


def generate(attempt, plan, contract, context):
    limits = plan['limits']; model = plan['model']['requested']; provider, model_id = model.split('/')
    project = attempt/'project'; project.mkdir()
    state = attempt/'private'; state.mkdir(mode=0o700)
    skills = {s: context/'skills'/s for s in plan['skills']}
    config = configure(json.loads((context/f'integrations/opencode/profiles/{plan["model"]["profile"]}.json').read_text()), plan, context, project)
    prompt = ('Load these pinned skills with the skill tool: '+', '.join(skills)+'.\n'
        'Write candidate.py for this task: '+json.dumps(contract)+'.\n'
        'Use import asc and from asc.experimental import asctile. Define @asctile.jit device functions '
        'and def launch('+', '.join(contract['arguments'])+'). The host launch body must be exactly '
        'one kernel[block_count](input tensors, out, optional literal scalar arguments) call; '
        'block_count is '+str(contract['blocks'])+'. No Torch, CPU arithmetic, imports inside functions, '
        'runtime configuration, main, tests, file I/O or function defaults in candidate.py. '
        'Supported device calls are asctile APIs, tensor reshape/broadcast_to and scalar min/max/range/len/int/float/bool/abs. '
        'All inputs are CPU tensors for Model simulation; the harness owns the oracle and poisoned output. '
        'Do not copy a reference kernel. Do not run commands. Write only candidate.py.')
    intervention = plan.get('intervention')
    observe_request = None
    if intervention:
        from generation_protocols import context_files, prompt as paired_prompt, request_audit
        prompt = paired_prompt(contract, intervention['id'], skills, context,
            project=project if intervention.get('context_path_format') == 'relative-read-call-v1' else None)
        visible = context_files(plan)
        save(attempt/'isolation.json', {'protocol': intervention, 'context_sha256': visible,
            'project': str(project), 'context': str(context), 'private': str(state),
            'fresh_project': not any(project.iterdir()), 'fresh_private': not any(state.iterdir()),
            'permission': config['permission']})
        def observe_request(number, body, payload):
            path = state/f'provider-request-{number}.bin'
            with path.open('xb') as handle: handle.write(payload)
            path.chmod(0o600)
            try: return request_audit(body, prompt, intervention['id'], intervention['selected_skills'], context)
            except (ValueError, KeyError, TypeError) as exc:
                raise BudgetError('Paired request context/schema audit failed') from exc
    result, usage = run_session(attempt, plan, project, config, prompt, observe_request)
    budget_failure = known_budget_exhaustion(usage, limits)
    if budget_failure and not intervention:
        raise ValueError('Generation budget exhausted; complete provider usage retained')
    if not budget_failure and (result is None or result['exit_code'] or usage['error'] or not usage['usage_complete']):
        raise Blocked('Generation or provider usage incomplete; inspect retained evidence')
    from generation_protocols import context_files
    for name, sha in context_files(plan).items():
        if digest(resource(context, name)) != sha:
            raise Blocked('Generation context changed')
    if intervention:
        from generation_protocols import audit_trace
        events = (attempt/'generation.stdout').read_text(); empty_session = None
        if budget_failure and not events.strip() and intervention['id'] in ('P2', 'P3'):
            from generation_native import early_output_cap, capture_empty_session
            from generation_protocols import provider_prompt
            if early_output_cap(usage, limits):
                import sqlite3
                try: empty_session = capture_empty_session(state, project, model, provider_prompt(prompt))
                except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                    raise Blocked('Could not attribute the empty native session') from exc
                save(attempt/'native-session.json', empty_session)
        trace = audit_trace(events, project, context, plan, allow_client_error=bool(budget_failure),
                            empty_session=empty_session, usage=usage)
        audits = [r.get('audit', {}) for r in usage['requests']]
        schemas = {a.get('tool_schema_sha256') for a in audits if a.get('tool_schema_sha256')}
        if not any(a.get('prompt_present') for a in audits) or len(schemas) != 1:
            raise Blocked('Paired provider prompt/tool attribution incomplete')
        if {s for a in audits for s in a['visible_skills']} != set(skills):
            raise Blocked('Pinned skill contents did not reach the provider')
        if intervention['id'] == 'P4' and not any(a['baseline_visible'] for a in audits):
            raise Blocked('Baseline notes did not reach the provider')
        trace.update(tool_schema_sha256=schemas.pop(), prompt_sha256=digest(prompt.encode()))
    else:
        trace = native_trace((attempt/'generation.stdout').read_text(), skills)
    observed = {r['usage']['observed_model'] for r in usage['requests']}
    if observed != {plan['model']['expected_response']}: raise Blocked('Provider model attribution mismatch')
    trace.update(requested_model=model, observed_model=plan['model']['expected_response'],
                 provider_model_revision=plan['model']['expected_response'])
    save(attempt/'attribution.json', trace)
    if budget_failure:
        save(attempt/'generation-outcome.json', {'status': 'budget-exhausted', 'reason': budget_failure})
        raise ValueError('Generation budget exhausted; complete provider usage retained')
    candidate = project/'candidate.py'
    if candidate.is_symlink() or not candidate.is_file(): raise ValueError('Missing regular candidate.py')
    if candidate.stat().st_size > 1024*1024: raise ValueError('Candidate too large')
    shutil.copyfile(candidate, attempt/'candidate.py')
    return usage


def qualify(attempt, plan, contract, source):
    before = digest((attempt/'candidate.py').read_bytes())
    save(attempt/'static.json', static_check((attempt/'candidate.py').read_text(), contract))
    write_probes(attempt, contract)
    name = 'pyasc-generation-'+uuid.uuid4().hex
    seconds = plan['limits']['qualification_seconds_per_attempt']
    argv = ['docker', 'run', '--rm', '--pull', 'never', '--name', name, '--network', 'none',
        '--ulimit', 'nofile=65536:65536', '-v', str(source)+':/repo:ro', '-v', str(attempt)+':/evidence',
        '-w', '/tmp', '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYASC_CACHE_DIR=/tmp/cache',
        '-e', 'PYASC_DUMP_PATH=/tmp/dumps', plan['runtime']['image'], 'python3.11',
        '/repo/tests/tools/generation_qualify.py', '/evidence', '--seconds', str(seconds)]
    try:
        record = command(argv, attempt, 'qualification', seconds)
    finally:
        # Killing docker's client alone does not stop its container.
        removed = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=20)
        gone = removed.returncode == 0 or b'No such container' in removed.stderr
        save(attempt/'container-cleanup.json', {'name': name, 'confirmed': gone})
        if not gone: raise Blocked('Container cleanup unconfirmed')
    if digest((attempt/'candidate.py').read_bytes()) != before: raise Blocked('Candidate changed during qualification')
    report = json.loads((attempt/'qualification.json').read_text())
    if record['exit_code'] == 2 or report['status'] == 'blocked': raise Blocked('Simulator unavailable')
    if (record['exit_code'] or report['status'] != 'passed' or report.get('inputs_unchanged') is not True
            or report.get('verification_level') != 'numerically_verified'):
        raise ValueError('Candidate did not qualify')
    return report


def execute(plan, output, state_dir, *, _lease=None):
    """Run cells; a paired coordinator may retain its already-held shared lease."""
    output = external(output)
    if output.exists(): raise ValueError('Use a new results directory; existing attempts are never replayed')
    for key in ('OPENCODE_API_KEY', 'OPENCODE_BASE_URL'):
        if not os.environ.get(key): raise Blocked('Missing '+key)
    contracts = {c['id']: task_contract(ROOT, c) for c in plan['cells']}
    with (exclusive(state_dir, output) if _lease is None else nullcontext(_lease)) as latch:
        output.mkdir(parents=True)
        save(output/'plan.json', plan)
        revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
            'checkout_revision': revision.stdout.strip() if revision.returncode == 0 else None,
            'plan_sha256': plan['plan_sha256'], 'cells': [
            {'id': c['id'], 'status': 'not-run', 'attempts': []} for c in plan['cells']],
            'reserved_output_tokens': 0, 'observed_output_tokens': 0, 'reserved_execution_seconds': 0}
        def checkpoint(): save(output/'state.json', state)
        checkpoint()
        source = output/'source'; source.mkdir()
        try:
            for name, sha in plan['source_sha256'].items():
                data = resource(ROOT, name)
                if digest(data) != sha: raise Blocked('Source changed after planning')
                target = source/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
            # Public model context contains selected skills/profile only, never oracle kernels.
            context = output/'context'; context.mkdir()
            from generation_protocols import context_files
            for name in context_files(plan):
                dest = context/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source/name, dest)
            if plan.get('intervention'): (context/'skills').mkdir(exist_ok=True)
            version = command(['opencode', '--version'], output, 'opencode-version', 20)
            image = command(['docker', 'image', 'inspect', plan['runtime']['image']], output, 'image-inspect', 20)
            if version['exit_code'] or image['exit_code']: raise Blocked('Provision OpenCode and pinned simulator first')
            start = time.monotonic(); limits = plan['limits']
            for cell in state['cells']:
                seen = set()
                for number in range(1, limits['attempts_per_cell']+1):
                    tokens = limits['rounds_per_attempt']*limits['output_tokens_per_response']
                    seconds = limits['generation_seconds_per_attempt']+limits['qualification_seconds_per_attempt']
                    if (state['reserved_output_tokens']+tokens > limits['max_output_tokens']
                            or state['reserved_execution_seconds']+seconds > limits['max_execution_seconds']
                            or time.monotonic()-start+seconds > limits['max_execution_seconds']+1):
                        raise Blocked('Aggregate budget exhausted before next attempt')
                    state['reserved_output_tokens'] += tokens; state['reserved_execution_seconds'] += seconds
                    attempt = output/cell['id']/f'attempt-{number}'; attempt.mkdir(parents=True)
                    entry = {'number': number, 'status': 'generating'}; cell['attempts'].append(entry)
                    cell['status'] = 'running'; checkpoint()
                    before = time.monotonic()
                    try:
                        generate(attempt, plan, contracts[cell['id']], context)
                        sha = digest((attempt/'candidate.py').read_bytes()); entry['candidate_sha256'] = sha
                        if sha in seen: raise Blocked('Unchanged failed candidate; refusing repeated qualification')
                        seen.add(sha)
                        entry['status'] = 'qualifying'; checkpoint()
                        qualify(attempt, plan, contracts[cell['id']], source)
                        entry['status'] = cell['status'] = 'passed'
                    except (Blocked, TimeoutError, OSError, KeyboardInterrupt):
                        entry['status'] = cell['status'] = 'blocked'
                        raise
                    except (ValueError, SyntaxError) as exc:
                        entry['failure_stage'] = entry['status']
                        entry['status'] = cell['status'] = 'failed'
                        entry['error'] = str(exc)
                    finally:
                        entry['duration_seconds'] = time.monotonic()-before
                        if (attempt/'usage.json').is_file():
                            observed = json.loads((attempt/'usage.json').read_text())
                            entry['usage'] = observed
                            state['observed_output_tokens'] += observed['observed_output_tokens']
                        checkpoint()
                    if cell['status'] == 'passed': break
            state['status'] = 'passed' if all(c['status']=='passed' for c in state['cells']) else 'failed'
            state['inputs_unchanged'] = all(digest(resource(ROOT, name))==sha for name, sha in plan['source_sha256'].items())
            if not state['inputs_unchanged']: raise Blocked('Checkout changed during execution')
            if _lease is None:
                latch.unlink()  # A paired coordinator releases only after all trials finish.
        except BaseException as exc:
            state.update(status='blocked', error=type(exc).__name__ + ': ' + str(exc))
            # Keep the latch: do not assume interrupted/unknown provider work is safe to replay.
        finally:
            state['finished_at'] = datetime.now(timezone.utc).isoformat()
            checkpoint()
            save(output/'files.json', {str(p.relative_to(output)): digest(p.read_bytes()) for p in output.rglob('*')
                if p.is_file() and not p.is_symlink() and 'private' not in p.parts and p.name != 'files.json'})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--matrix', default=DEFAULT_MATRIX)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--state-dir', type=Path, default=Path.home()/'.local/state/pyasc-generation')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approve-plan', help='Exact digest from the reviewed plan; required for model calls')
    args = parser.parse_args(); plan = validate_plan(args.plan, args.matrix)
    for cell in plan['cells']: task_contract(ROOT, cell)
    if not args.execute:
        print(json.dumps({'status': 'not-run', 'plan_sha256': plan['plan_sha256'], 'adapters': len(plan['cells'])})); return 0
    if args.approve_plan != plan['plan_sha256'] or not args.results_dir:
        raise ValueError('Explicit matching --approve-plan and new --results-dir required')
    result = execute(plan, args.results_dir, args.state_dir)
    print(json.dumps(result)); return 0 if result['status']=='passed' else 2 if result['status']=='blocked' else 1


if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try: sys.exit(main())
    except (ValueError, OSError) as exc: raise SystemExit('Generation stopped: '+str(exc))
