#!/usr/bin/env python3
"""One CI-owned OpenCode -> exact package -> CANNBench -> durable release run.

Only collect may resume an existing submission; it never POSTs. All evidence is
outside checkout and checkpointed by the workflow before/after the upload.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(HERE.parent / 'workers')]
import smoke
from operator_recipe import load_recipe, run_recipe
from submission import save

PIN = json.loads((HERE / 'single-runtime.json').read_text())
SKILLS = ('pyasc-cannbench-kernel', 'pyasc-syntax-constraints', 'pyasc-api-patterns', 'pyasc-build-run-verify')
MODEL = PIN['model']


def load(path):
    return json.loads(Path(path).read_text())


def retry_get(request):
    """Bounded retry for transient GET failures only; never wrap submission."""
    delays = (15, 30, 60, 60)
    for attempt in range(len(delays)+1):
        try: return request()
        except Exception as exc:
            status = getattr(exc, 'status', getattr(exc, 'code', None))
            if status not in (502,503,504) or attempt == len(delays): raise
            print(f'Transient GET HTTP {status}; retry in {delays[attempt]}s', flush=True)
            time.sleep(delays[attempt])


def ci_client():
    queue = smoke.client(None)
    get = queue._client._get_json
    queue._client._get_json = lambda *args, **kwargs: retry_get(lambda: get(*args, **kwargs))
    return queue


def scrub(text):
    for key in ('OPENCODE_API_KEY', 'BENCHSITE_API_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN'):
        value = os.environ.get(key)
        if value:
            text = text.replace(value, '<redacted>')
    return text


def hashes(root):
    return {str(p.relative_to(root)): smoke.sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}


def check_hashes(root, expected):
    for name, digest in expected.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or smoke.sha(p) != digest:
            raise ValueError('Evidence changed: ' + name)


def ci_identity():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise ValueError('This workflow must execute in GitHub Actions')
    return {key: os.environ[key] for key in ('GITHUB_REPOSITORY', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA')}


def call(command, log, *, cwd=ROOT, env=None, timeout=900):
    start = time.monotonic()
    print(f'CI stage: {log.name} ({command[0]})', flush=True)
    try:
        p = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        code = 124
        out = e.stdout or b''; err = e.stderr or b''
        if isinstance(out, bytes): out = out.decode(errors='replace')
        if isinstance(err, bytes): err = err.decode(errors='replace')
    log.write_text(scrub(out + '\n=== STDERR ===\n' + err))
    save(log.with_suffix(log.suffix + '.meta.json'), {'command': command, 'exit': code,
         'elapsed_seconds': time.monotonic() - start, 'log_sha256': smoke.sha(log)})
    print(f'CI stage finished: {log.name}, exit={code}, elapsed={time.monotonic()-start:.2f}s', flush=True)
    if code:
        raise ValueError(f'{command[0]} exited {code}; retained {log.name}')
    return out


def trace(stdout, skill_root):
    sessions, loaded, reads = set(), set(), []
    for line in stdout.splitlines():
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        if event.get('sessionID'): sessions.add(event['sessionID'])
        part = event.get('part') or {}; state = part.get('state') or {}
        if event.get('type') == 'tool_use' and part.get('tool') == 'skill' and state.get('status') == 'completed':
            name = (state.get('input') or {}).get('name')
            directory = (state.get('metadata') or {}).get('dir')
            if (not isinstance(name,str) or Path(name).name != name or not directory
                    or Path(directory).resolve() != (skill_root/name).resolve()):
                raise ValueError('Native skill came from an unpinned source')
            loaded.add(name); reads.append({'name': name, 'directory': directory})
    if len(sessions) != 1 or not set(SKILLS) <= loaded:
        raise ValueError('Missing unique native session or selected skill-load evidence')
    return {'session_id': next(iter(sessions)), 'loaded_skills': sorted(loaded), 'skill_calls': reads}


def capture_session(database, session_id, output):
    """Capture this native session read-only; CLI stdout export can truncate large JSON."""
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('BEGIN')
        row = connection.execute('SELECT * FROM session WHERE id=?', (session_id,)).fetchone()
        if row is None: raise ValueError('Native session missing')
        session = dict(row)
        model = json.loads(session['model'])
        if model['providerID']+'/'+model['id'] != MODEL:
            raise ValueError('Native session model differs from configured model')
        data = {'format':'opencode-sqlite-session-v1', 'session':session}
        for table in ('message','part'):
            data[table] = [dict(row) for row in connection.execute(
                f'SELECT * FROM {table} WHERE session_id=? ORDER BY time_created,id', (session_id,))]
            if not data[table]: raise ValueError('Incomplete native session '+table)
        save(output, json.loads(scrub(json.dumps(data))))
    finally:
        connection.close()


def run_root(origin):
    if not re.fullmatch(r'[0-9]+', origin):
        raise ValueError('Origin run must be a numeric GitHub run ID')
    base = Path(os.environ.get('PYASC_CI_STATE_ROOT', str(Path.home() / '.local/state/pyasc-generated-ci')))
    return smoke.external(base / ('run-' + origin))


def inputs(root):
    meta = load(root / 'run.json')
    if meta['repository'] != os.environ['GITHUB_REPOSITORY']:
        raise ValueError('Run repository mismatch')
    check_hashes(root / 'source', meta['source_hashes'])
    check_hashes(root / 'context', meta['context_hashes'])
    if 'operator_recipe' in meta:
        recipe = run_recipe(root)
        selected = load(root/'catalog.json')['operators']
        if (len(selected) != 1 or selected[0]['operator'] != recipe['operator']
                or selected[0]['function_name'] != recipe['function']):
            raise ValueError('Run task differs from its operator recipe')
    if (root/'repair-lineage.json').exists():
        lineage = load(root/'repair-lineage.json')
        check_hashes(root/'parent-ci', lineage['parent_files'])
        check_hashes(root, {lineage['candidate']: lineage['candidate_sha256']})
    if (root/'continuation-lineage.json').exists():
        lineage = load(root/'continuation-lineage.json')
        check_hashes(root/'qualified-parent', lineage['parent_files'])
        parent = load(root/'qualified-parent/run.json')
        for key in ('skill_stack_commit', 'source_hashes', 'context_hashes', 'opencode_version', 'operator_recipe'):
            if meta.get(key) != parent.get(key): raise ValueError('Continuation changed generation attribution: '+key)
        check_hashes(root/'orchestration-source', meta['orchestration_source_hashes'])
    return meta


def qualified_binding(root, gen, catalog):
    iteration = root/'generation'/f"attempt-{gen['attempt']}"
    lowering = load(iteration/'lowering.json'); numerical = load(iteration/'simulator.json')
    if (lowering.get('status') != 'passed' or lowering.get('compile_passed') != catalog['required_cases']
            or lowering.get('cases') != catalog['required_cases']
            or lowering.get('pyasc_commit') != PIN['pyasc_commit']
            or lowering.get('runtime_bundle_sha256') != PIN['runtime_bundle_sha256']
            or not lowering.get('runtime_files_verified')
            or lowering.get('candidate_sha256') != gen['candidate_sha256']):
        raise ValueError('Exact generated candidate lacks full pinned-runtime lowering')
    if (numerical.get('passed') is not True or len(numerical.get('cases',[])) != run_recipe(root)['numerical_cases']
            or any(c.get('passed') is not True for c in numerical['cases'])
            or numerical.get('candidate_sha256') != gen['candidate_sha256']):
        raise ValueError('Exact generated candidate lacks simulator preflight')


def generated_binding(root):
    meta = inputs(root); gen = load(root / 'generation.json')
    if smoke.sha(root/'model-config.json') != gen['model_configuration_sha256']:
        raise ValueError('Model configuration changed')
    check_hashes(root, gen['files'])
    if gen['model'] != MODEL or gen['provider_model_revision'] != 'unknown':
        raise ValueError('Unexpected model attribution')
    if not set(SKILLS) <= set(gen['native_trace']['loaded_skills']):
        raise ValueError('Incomplete native skill evidence')
    qualified_binding(root, gen, load(root/'catalog.json'))
    staged = smoke.verify(root / 'package')
    if staged['files']['candidate/'+run_recipe(root)['function']+'.py'] != gen['candidate_sha256']:
        raise ValueError('Generated/packaged candidate mismatch')
    if staged['files']['candidate/_pyasc_runtime.py'] != smoke.sha(root / 'source/integrations/cannbench/smoke-kernel/_pyasc_runtime.py'):
        raise ValueError('Runtime helper mismatch')
    if staged['runtime_bundle_sha256'] != PIN['runtime_bundle_sha256']:
        raise ValueError('Runtime bundle mismatch')
    if staged['catalog'] != load(root / 'catalog.json') or load(root / 'package/task.json') != load(root / 'task.json'):
        raise ValueError('Generated/package task mismatch')
    with zipfile.ZipFile(root / 'package/submission.zip') as z:
        wheel = z.read(next(n for n in z.namelist() if n.startswith('vendor/') and n.endswith('.whl')))
        if hashlib.sha256(wheel).hexdigest() != staged['wheel_sha256']:
            raise ValueError('Packaged wheel mismatch')
        with zipfile.ZipFile(io.BytesIO(wheel)) as w:
            for name in (run_recipe(root)['function']+'.py', '_pyasc_runtime.py'):
                b = (root / 'package/candidate' / name).read_bytes()
                if z.read('cann_bench/' + name) != b or w.read('cann_bench/' + name) != b:
                    raise ValueError('Source/wheel candidate mismatch')
    return meta, gen, staged


def summarize_api(context):
    """Extract public signatures/docstrings from the exact wheel, never a kernel."""
    requested = {
        'global_tensor.py': ('global_tensor',),
        'memory_ops.py': ('copy_in', 'copy_out'),
        'prog_model_ops.py': ('block_idx', 'block_num'),
        'unary_ops.py': ('exp', 'abs', 'negative'),
        'binary_ops.py': ('add', 'sub', 'mul', 'div', 'maximum', 'minimum', 'equal', 'not_equal'),
        'creation_ops.py': ('cast', 'full'),
        'indexing_ops.py': ('where',),
        'shape_ops.py': ('reshape',),
        'range.py': ('range',),
    }
    lines = ['# Public API extracted from the hash-verified runtime wheel',
             'Source/license: runtime-api/asctile; full licensed sources remain alongside this summary.',
             'This is API documentation, not a candidate implementation.', '']
    for filename, names in requested.items():
        path=context/'runtime-api/asctile/language'/filename
        tree=ast.parse(path.read_text())
        functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name in names:
            node=functions.get(name)
            if node is None: continue
            signatures = [node] if isinstance(node,ast.FunctionDef) else [
                n for n in node.body if isinstance(n,ast.FunctionDef) and n.name == '__init__']
            lines += [f'## asctile.{name} ({filename}:{node.lineno})']
            lines += [f'`{name}({ast.unparse(n.args)})`' for n in signatures]
            lines += [ast.get_docstring(node) or '', '']
    (context/'API.md').write_text('\n'.join(lines)+'\n')


def worker_permissions(context):
    # OpenCode uses simple wildcards, not globstar: include root-relative names.
    return {'read': 'allow', 'glob': 'allow', 'grep': 'allow', 'skill': 'allow',
        'edit': {'*': 'deny', 'candidate.py': 'allow', 'design.md': 'allow',
                 '**/candidate.py': 'allow', '**/design.md': 'allow'}, 'bash': 'deny',
        'task': 'deny', 'webfetch': 'deny', 'websearch': 'deny', 'question': 'deny',
        'doom_loop': 'deny', 'external_directory': {'*': 'deny', str(context / '*'): 'allow'}}


def qualification_feedback(iteration, error):
    parts = [str(error).split('\n',1)[0]]
    report = iteration/'lowering.json'
    if report.exists() and load(report).get('status') != 'passed':
        parts.append(report.read_text()[-8000:])
    for name in ('lowering.log','simulator.log'):
        log = iteration/name; meta = iteration/(name+'.meta.json')
        if log.exists() and meta.exists() and load(meta).get('exit'):
            parts.append(log.read_text(errors='replace').split('=== STDERR ===')[-1][-6000:])
    return scrub('\n'.join(parts))


def repair_seed(root, origin):
    """Import the failed CI's central evidence; never repair an uncertain submission."""
    parent = root/'parent-ci'
    previous = restore_checkpoint(parent, origin)
    if previous['status'] != 'completed': raise ValueError('Parent CI is still active')
    if load(parent/'task.json') != load(root/'task.json') or load(parent/'catalog.json') != load(root/'catalog.json'):
        raise ValueError('Task changed since failed generation; repair requires the same task snapshot')
    for name in ('submission-intent.json', 'submission.json', 'package/state.json', 'generation.json'):
        if (parent/name).exists():
            raise ValueError('Parent may have reached submission; use GET-only resume, not repair')
    candidates = []
    for attempt in (parent/'generation').glob('attempt-*'):
        if all((attempt/name).is_file() for name in ('original-candidate.py','static.json','failure.json')):
            if load(attempt/'static.json').get('passed') and ((attempt/'lowering.json').exists() or (attempt/'lowering.log.meta.json').exists()):
                candidates.append((int(attempt.name.split('-')[-1]),attempt))
    if not candidates: raise ValueError('No recorded qualified-or-failed candidate to repair')
    attempt = max(candidates)[1]
    candidate = attempt/'original-candidate.py'
    if (attempt/'lowering.json').exists():
        if load(attempt/'lowering.json').get('candidate_sha256') != smoke.sha(candidate):
            raise ValueError('Parent candidate does not match its qualification')
    else:
        log = attempt/'lowering.log'; meta = load(attempt/'lowering.log.meta.json')
        if not meta.get('exit') or smoke.sha(log) != meta['log_sha256']:
            raise ValueError('Missing exact interrupted-compiler evidence')
        for path in ('project/candidate.py','candidate/'+run_recipe(parent)['function']+'.py'):
            if smoke.sha(attempt/path) != smoke.sha(candidate):
                raise ValueError('Interrupted compiler candidate mismatch')
    save(root/'repair-lineage.json', {'origin_run_id': origin, 'parent_commit': load(parent/'run.json')['skill_stack_commit'],
        'candidate': str(candidate.relative_to(root)), 'candidate_sha256': smoke.sha(candidate),
        'parent_files': hashes(parent)})
    feedback = qualification_feedback(attempt, load(attempt/'failure.json')['error'])
    return candidate, feedback


def qualified_generation(root):
    """Reconstruct attribution from retained native records, without invoking a model."""
    meta = inputs(root)
    config = load(root/'model-config.json')
    attempts = sorted((root/'generation').glob('attempt-*'), key=lambda p: int(p.name.split('-')[-1]))
    if not attempts: raise ValueError('No generated candidate retained')
    iteration = attempts[-1]
    original = iteration/'original-candidate.py'
    digest = smoke.sha(original)
    for name in ('project/candidate.py', 'candidate/'+run_recipe(root)['function']+'.py'):
        if smoke.sha(iteration/name) != digest: raise ValueError('Qualified candidate changed')
    if load(iteration/'static.json').get('passed') is not True: raise ValueError('Static checks did not pass')
    native = trace((iteration/'events.log').read_text(), Path(config['skills']['paths'][0]))
    if native != load(iteration/'skill-trace.json'): raise ValueError('Native skill trace changed')
    session = load(iteration/'session.json')
    model = json.loads(session['session']['model'])
    if (session['session']['id'] != native['session_id'] or model['providerID']+'/'+model['id'] != MODEL
            or not session['message'] or not session['part']):
        raise ValueError('Native generation session mismatch')
    gen = {'model': MODEL, 'provider_model_revision': 'unknown', 'opencode_version': meta['opencode_version'],
        'attempt': int(iteration.name.split('-')[-1]), 'native_trace': native, 'candidate_sha256': digest,
        'skill_hashes': {name: hashes(root/'context/skills'/name) for name in native['loaded_skills']},
        'simulator_image': load(root/'source/reference/images/ENV.json')['dev_image']['ref'],
        'original_candidate': str(original.relative_to(root)),
        'model_configuration_sha256': smoke.sha(root/'model-config.json'),
        'files': {str(p.relative_to(root)): smoke.sha(p) for p in (root/'generation').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}}
    qualified_binding(root, gen, load(root/'catalog.json'))
    return gen


def prepare_qualified(root, generation, queue):
    runtime = Path(os.environ['CANNBENCH_RUNTIME_BUNDLE']).resolve()
    package_source = root/'generation'/f"attempt-{generation['attempt']}"/'candidate'
    recipe = run_recipe(root)
    args = SimpleNamespace(operator=recipe['operator'], function=recipe['function'], run_dir=root/'package', kernel_dir=package_source, runtime_bundle=runtime,
        runtime_sha256=PIN['runtime_bundle_sha256'], aggregation_token=os.environ['CANNBENCH_AGGREGATION_TOKEN'])
    prepared = smoke.prepare(args, queue)
    if prepared['catalog'] != load(root/'catalog.json') or load(root/'package/task.json') != load(root/'task.json'):
        raise ValueError('Official task changed during generation')
    save(root/'generation.json', generation)
    generated_binding(root)
    save(root/'submission-intent.json', {'phase': 'prepared-not-posted', 'ci': ci_identity(),
        'archive_sha256': prepared['files']['submission.zip'],
        'job_tag': 'pyasc-smoke-'+prepared['files']['submission.zip'][:24],
        'manifest_sha256': smoke.sha(root/'package/manifest.json'),
        'resume_policy': 'collect/reconcile only; never repeat an uncertain POST'})


def continue_qualified(root, origin):
    """New CI packaging attempt; preserve qualified output and original model provenance."""
    if root.exists(): raise ValueError('Continuation state already exists; never overwrite it')
    parent = root/'qualified-parent'
    previous = restore_checkpoint(parent, origin)
    if previous['status'] != 'completed': raise ValueError('Parent CI is still active')
    for name in ('submission-intent.json', 'submission.json', 'prepared-artifact.json', 'package/state.json', 'generation.json'):
        if (parent/name).exists(): raise ValueError('Parent may have reached submission; use GET-only resume')
    generation = qualified_generation(parent)
    # Keep the complete interrupted package and lineage in qualified-parent.
    # Copy only immutable generation inputs/outputs into the new run.
    for name in ('source', 'context', 'generation', 'parent-ci'):
        if (parent/name).exists(): shutil.copytree(parent/name, root/name)
    for name in ('task.json', 'catalog.json', 'model-config.json', 'opencode-version.log', 'repair-lineage.json'):
        if (parent/name).exists(): shutil.copy2(parent/name, root/name)
    identity = ci_identity(); meta = load(parent/'run.json')
    orchestration = root/'orchestration-source'; orchestration.mkdir()
    archive = subprocess.check_output(['git', 'archive', identity['GITHUB_SHA']], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as z: z.extractall(orchestration, filter='data')
    meta['generation_ci'] = meta.get('generation_ci') or {k: meta[k] for k in
        ('ci_url', 'origin_run_id', 'origin_run_attempt', 'skill_stack_commit')}
    meta.update(origin_run_id=identity['GITHUB_RUN_ID'], origin_run_attempt=identity['GITHUB_RUN_ATTEMPT'],
        ci_url=f"https://github.com/{identity['GITHUB_REPOSITORY']}/actions/runs/{identity['GITHUB_RUN_ID']}",
        orchestration_commit=identity['GITHUB_SHA'], orchestration_source_hashes=hashes(orchestration))
    save(root/'run.json', meta)
    save(root/'continuation-lineage.json', {'parent_run_id': origin, 'parent_files': hashes(parent),
        'candidate_sha256': generation['candidate_sha256'], 'model_invoked': False})
    inputs(root)
    prepare_qualified(root, generation, ci_client())


def generate(root, repair_origin=''):
    if (root / 'generation.json').exists():
        generated_binding(root)
        return
    if root.exists():
        raise ValueError('Interrupted generation retained; start a distinct CI run, do not overwrite it')
    recipe = load_recipe(os.environ.get('CANNBENCH_OPERATOR', 'swi_glu'))
    root.mkdir(parents=True)
    identity = ci_identity()
    source = root / 'source'; source.mkdir()
    archive = subprocess.check_output(['git', 'archive', identity['GITHUB_SHA']], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as z: z.extractall(source, filter='data')
    recipe = load_recipe(os.environ.get('CANNBENCH_OPERATOR', 'swi_glu'), source/'integrations/cannbench/ci/operators')
    context = root / 'context'; context.mkdir()
    shutil.copytree(source / 'skills', context / 'skills')
    runtime = Path(os.environ['CANNBENCH_RUNTIME_BUNDLE']).resolve()
    if smoke.sha(runtime) != PIN['runtime_bundle_sha256']:
        raise ValueError('Provisioned runtime does not match pinned runtime')
    with zipfile.ZipFile(runtime) as z:
        for name in z.namelist():
            if name.startswith(('asctile/', 'asc/')) and name.endswith('.py') and '..' not in Path(name).parts:
                p = context / 'runtime-api' / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(z.read(name))
    summarize_api(context)
    queue = ci_client()
    catalog, task = smoke.task_snapshot(queue, recipe['operator'], recipe['function'])
    save(root / 'catalog.json', catalog); save(root / 'task.json', task)
    task_dir = context / 'task'; task_dir.mkdir()
    save(task_dir / 'proto.yaml', {'operator': task['proto']})
    save(task_dir / 'cases.yaml', {'cases': task['cases_yaml']})
    (task_dir / 'golden.py').write_text(task['golden_source']); (task_dir / 'desc.md').write_text(task['desc_md'])
    version = call(['opencode', '--version'], root / 'opencode-version.log', timeout=30).strip()
    meta = {'repository': identity['GITHUB_REPOSITORY'], 'origin_run_id': identity['GITHUB_RUN_ID'],
            'origin_run_attempt': identity['GITHUB_RUN_ATTEMPT'], 'skill_stack_commit': identity['GITHUB_SHA'],
            'ci_url': f"https://github.com/{identity['GITHUB_REPOSITORY']}/actions/runs/{identity['GITHUB_RUN_ID']}",
            'operator_recipe': os.environ.get('CANNBENCH_OPERATOR', 'swi_glu'), 'pyasc': PIN, 'opencode_version': version, 'source_hashes': hashes(source),
            'context_hashes': hashes(context), 'created_at': smoke.now()}
    save(root / 'run.json', meta)
    config = load(source / 'integrations/opencode/profiles/cloud-glm-5.3.json')
    config['skills'] = {'paths': [str(context / 'skills')]}
    config['agent'] = {'build': {'steps': 12}}
    config['provider']['gateway']['models']['glm-5.3']['limit']['output'] = 24000
    config['permission'] = worker_permissions(context)
    config['provider']['gateway']['options']['baseURL'] = os.environ['OPENCODE_BASE_URL']
    save(root / 'model-config.json', config)
    env = dict(os.environ)
    for k in ('GH_TOKEN', 'GITHUB_TOKEN', 'BENCHSITE_API_TOKEN'): env.pop(k, None)
    env.update(OPENCODE_CONFIG_CONTENT=json.dumps(config), OPENCODE_DISABLE_EXTERNAL_SKILLS='1',
               OPENCODE_DISABLE_CLAUDE_CODE_SKILLS='1')
    state = root.parent / ('opencode-private-' + identity['GITHUB_RUN_ID']); state.mkdir(mode=0o700)
    env['XDG_DATA_HOME'] = str(state)
    env['XDG_CONFIG_HOME'] = str(state / 'config')
    prompt = (source/'integrations/cannbench/ci/operators'/recipe['prompt']).read_text().format(
        skills=', '.join(SKILLS), task_dir=task_dir, pyasc_commit=PIN['pyasc_commit'], api=context/'API.md')
    seed, feedback = repair_seed(root, repair_origin) if repair_origin else (None, '')
    from contracts import static_check
    for attempt in range(1, 4):
        iteration = root / 'generation' / f'attempt-{attempt}'; project = iteration / 'project'; project.mkdir(parents=True)
        prior = root/'generation'/f'attempt-{attempt-1}'/'project/candidate.py' if attempt > 1 else seed
        if prior and prior.is_file(): shutil.copy2(prior, project/'candidate.py')
        before = smoke.sha(project/'candidate.py') if (project/'candidate.py').exists() else None
        message = prompt + ('\nRepair YOUR previous candidate. Previous qualification feedback:\n' + feedback if feedback else '')
        (iteration / 'prompt.txt').write_text(message)
        save(project / 'opencode.json', config)
        try:
            out = call(['opencode', 'run', '--pure', '--format', 'json', '--dir', str(project), '-m', MODEL, message],
                       iteration / 'events.log', cwd=project, env=env, timeout=1200)
            native = trace(out, context / 'skills'); save(iteration / 'skill-trace.json', native)
            capture_session(state/'opencode/opencode.db', native['session_id'], iteration/'session.json')
            candidate = project / 'candidate.py'
            if candidate.is_file(): shutil.copy2(candidate, iteration / 'original-candidate.py')
            if before and candidate.is_file() and smoke.sha(candidate) == before:
                raise ValueError('Model did not change the failed candidate; no blind repair retry')
            if not (project / 'design.md').is_file(): raise ValueError('Missing generated design')
            problems = static_check(candidate, recipe['function'], task_dir)
            save(iteration / 'static.json', {'passed': not problems, 'errors': problems})
            if problems: raise ValueError('; '.join(problems))
            original = iteration / 'original-candidate.py'; shutil.copy2(candidate, original)
            package_source = iteration / 'candidate'; package_source.mkdir()
            shutil.copy2(candidate, package_source / (recipe['function']+'.py'))
            shutil.copy2(source / 'integrations/cannbench/smoke-kernel/_pyasc_runtime.py', package_source / '_pyasc_runtime.py')
            if recipe.get('reference') and smoke.sha(candidate) == smoke.sha(source / 'integrations/cannbench'/recipe['reference']):
                raise ValueError('Reference candidate substitution rejected')
            actual = subprocess.check_output(['docker', 'image', 'inspect', PIN['compile_image'], '--format', '{{.Id}}'], text=True).strip()
            if actual != PIN['compile_image_id']: raise ValueError('Compiler image differs from qualified image')
            call(['docker', 'run', '--rm', '--network', 'none', '--platform', 'linux/amd64',
                  '--label', 'pyasc.ci.run='+identity['GITHUB_RUN_ID'], '--entrypoint', 'timeout',
                  '--label', 'pyasc.ci.repository='+identity['GITHUB_REPOSITORY'],
                  '-v', str(source)+':/workspace:ro', '-v', str(root)+':/evidence', '-v', str(runtime)+':/runtime.whl:ro',
                  PIN['compile_image'], '-k', '10', '540', 'python', '-u', 'integrations/cannbench/ci/single_qualify.py',
                  '--candidate', '/evidence/'+str((package_source/(recipe['function']+'.py')).relative_to(root)),
                  '--function', recipe['function'], '--task-dir', '/evidence/context/task', '--runtime', '/runtime.whl',
                  '--output', '/evidence/'+str((iteration/'lowering.json').relative_to(root))], iteration/'lowering.log', timeout=600)
            image = load(source/'reference/images/ENV.json')['dev_image']['ref']
            call(['docker', 'run', '--rm', '--network', 'none', '--ulimit', 'nofile=65536:65536',
                  '--label', 'pyasc.ci.run='+identity['GITHUB_RUN_ID'],
                  '--label', 'pyasc.ci.repository='+identity['GITHUB_REPOSITORY'],
                  '-v', str(source)+':/workspace:ro', '-v', str(root)+':/evidence', '-w', '/tmp',
                  '-e', 'HOME=/tmp', '-e', 'PYASC_CACHE_DIR=/tmp/cache', '-e', 'PYASC_DUMP_PATH=/tmp/dumps',
                  image, 'timeout', '300', 'python3.11', '/workspace/integrations/cannbench/ci/operators/'+recipe['numerical_probe'],
                  '/evidence/'+str(package_source.relative_to(root)), '/evidence/'+str((iteration/'simulator.json').relative_to(root))],
                  iteration/'simulator.log', timeout=330)
            if smoke.sha(original) != smoke.sha(candidate) or smoke.sha(candidate) != smoke.sha(package_source/(recipe['function']+'.py')):
                raise ValueError('Candidate changed during qualification')
            inputs(root)
            generation = {'model': MODEL, 'provider_model_revision': 'unknown', 'opencode_version': version,
                'attempt': attempt, 'native_trace': native, 'candidate_sha256': smoke.sha(original),
                'skill_hashes': {name: hashes(context/'skills'/name) for name in native['loaded_skills']},
                'simulator_image': image, 'original_candidate': str(original.relative_to(root)),
                'files': {str(p.relative_to(root)): smoke.sha(p) for p in (root/'generation').rglob('*') if p.is_file() and '__pycache__' not in p.parts}}
            generation['model_configuration_sha256'] = smoke.sha(root/'model-config.json')
            prepare_qualified(root, generation, queue)
            return
        except Exception as exc:
            feedback = qualification_feedback(iteration, exc)
            print(f'Model attempt {attempt} did not qualify: {feedback[:2000]}', flush=True)
            if not (iteration/'session.json').exists() and (iteration/'events.log').exists():
                try:
                    native = trace((iteration/'events.log').read_text(), context/'skills')
                    save(iteration/'skill-trace.json', native)
                    capture_session(state/'opencode/opencode.db', native['session_id'], iteration/'session.json')
                except Exception as export_error:
                    save(iteration/'session-export-error.json', {'error':scrub(str(export_error))})
            save(iteration/'failure.json', {'error': feedback})
            if not (project/'candidate.py').is_file(): raise  # No candidate means no repairable compiler/numerical feedback.
            if before and smoke.sha(project/'candidate.py') == before: raise
            if (root/'package').exists(): raise  # A partially staged package must not be silently replaced.
    raise ValueError('Three recorded model attempts failed qualification; see CI artifacts')


def require_quota(credits):
    c = credits.get('credits') or {}
    remaining = c.get('remaining')
    used = c.get('used', c.get('used_today'))
    limit = c.get('daily_limit', c.get('limit'))
    if used is None and isinstance(limit, int) and isinstance(remaining, int):
        used = limit - remaining
    if (type(remaining) is not int or type(used) is not int
            or remaining < 1 or not 0 <= used < 10):
        raise ValueError('Daily quota unknown or exhausted (working budget 10 submissions/day)')


def download(queue, url):
    # Only same-service authenticated download paths, never redirect credentials to a supplied host.
    if not url.startswith('/api/') or '..' in url.split('/'):
        raise ValueError('Unexpected service download URL')
    req = urllib.request.Request(queue._client._url(url), headers=queue._client._headers())
    with queue._client._open(req, 120) as response:
        return response.read()


def submit(root):
    _, _, prepared = generated_binding(root)
    if not os.environ.get('PREPARED_ARTIFACT_ID'): raise ValueError('Central prepared checkpoint required before POST')
    save(root/'prepared-artifact.json', {'id': os.environ['PREPARED_ARTIFACT_ID'], 'ci': ci_identity()})
    queue = ci_client()
    credits = queue._client.get_credits(); save(root/'credits-before.json', credits)
    require_quota(credits)
    ledger = Path.home()/'.local/state/pyasc-cannbench-smoke'; ledger.mkdir(parents=True, exist_ok=True)
    with (ledger/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = smoke.submit_once(queue, root/'package', ledger)
        save(root/'submission.json', state)
    print('Submission recorded:', state.get('job_id'), flush=True)


def restore_checkpoint(root, origin):
    """Restore only CI artifacts of this repository's manual origin run."""
    repo = os.environ['GITHUB_REPOSITORY']
    run = json.loads(subprocess.check_output(['gh', 'api', f'repos/{repo}/actions/runs/{origin}']))
    if run['event'] != 'workflow_dispatch' or run['head_repository']['full_name'] != repo:
        raise ValueError('Resume origin must be a trusted manual run of this repository')
    payload = json.loads(subprocess.check_output(['gh', 'api', f'repos/{repo}/actions/runs/{origin}/artifacts?per_page=100']))
    prefixes = {f'{kind}-{phase}-{origin}-': rank
                for kind in ('generated-swiglu', 'generated-operator')
                for rank, phase in enumerate(('prepared','submitted','evidence'))}
    choices = []
    for artifact in payload['artifacts']:
        for prefix, rank in prefixes.items():
            tail = artifact['name'].removeprefix(prefix)
            if artifact['name'].startswith(prefix) and tail.isdigit() and not artifact['expired']:
                choices.append((int(tail),rank,artifact['name']))
    if not choices: raise ValueError('No retained central checkpoint; provision the original durable runner state')
    name = max(choices)[2]
    root.mkdir(parents=True)
    subprocess.run(['gh','run','download',origin,'--repo',repo,'--name',name,'--dir',str(root)],check=True)
    meta = inputs(root)
    if meta['origin_run_id'] != origin or meta.get('orchestration_commit',meta['skill_stack_commit']) != run['head_sha']:
        raise ValueError('Restored checkpoint does not match origin CI revision')
    return run


def reconcile_ledger(root, ledger):
    """Lost local ledger may only be reconstructed as a GET-only intent."""
    manifest = smoke.verify(root/'package')
    digest = manifest['files']['submission.zip']; record = ledger/(digest+'.json')
    if record.exists(): return
    if (root/'package/state.json').exists():
        state = load(root/'package/state.json')
    else:
        intent = load(root/'submission-intent.json')
        state = dict(intent, phase='ambiguous', job_id=None, benchmark_slug=manifest['catalog']['benchmark_slug'])
    if state['archive_sha256'] != digest or state['manifest_sha256'] != smoke.sha(root/'package/manifest.json'):
        raise ValueError('Restored submission intent differs from package')
    state['run_dir'] = str(root/'package')
    save(record,state)


def collect(root, seconds):
    generated_binding(root)
    queue = ci_client(); ledger = Path.home()/'.local/state/pyasc-cannbench-smoke'
    ledger.mkdir(parents=True, exist_ok=True)
    archive = root.parent/(root.name+'-qualification.tar.gz')
    if archive.exists():
        verify_archive(archive, root)
        return 0  # Publication retry must not mutate the already sealed qualification.
    end = time.monotonic() + seconds
    while True:
        with (ledger/'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            reconcile_ledger(root, ledger)
            code, result = smoke.collect(queue, root/'package', ledger)
        print(smoke.now(), result.get('status', 'terminal'), result.get('job_id'), flush=True)
        if code != 2:
            state = load(root/'package/state.json'); jobid = state['job_id']
            save(root/'package/job-logs.json', queue._client.get_job_logs(jobid))
            artifacts = queue._client._get_json('/api/jobs/'+jobid+'/artifacts')
            save(root/'package/artifact-index.json', artifacts)
            target = root/'package/service-artifacts'; target.mkdir(exist_ok=True)
            for a in artifacts['artifacts']:
                if Path(a['name']).name != a['name']: raise ValueError('Unsafe artifact name')
                data = download(queue, a['download_url'])
                if hashlib.sha256(data).hexdigest() != a['checksum']: raise ValueError('Service artifact checksum mismatch')
                (target/a['name']).write_bytes(data)
            job = load(root/'package/job.json')['job']
            evaluated = download(queue, job['submission_download_url'])
            (root/'package/evaluated-submission.zip').write_bytes(evaluated)
            if hashlib.sha256(evaluated).hexdigest() != smoke.sha(root/'package/submission.zip'):
                raise ValueError('Service evaluated package differs from CI-submitted package')
            generated_binding(root)
            return code
        if time.monotonic() >= end:
            print('Still pending; resume this origin run through CI, do not submit again.', flush=True)
            return 2
        time.sleep(60)


def publish(root):
    meta, gen, prepared = generated_binding(root)
    archive = root.parent / (root.name+'-qualification.tar.gz')
    if archive.exists():
        verify_archive(archive, root)
        publication_outputs(root, archive, meta)
        return
    job = load(root/'package/job.json')['job']
    report = smoke.evaluate(prepared['catalog'], [{'job': job}], prepared['target_hardware'])
    state = load(root/'package/state.json')
    if (not report['gate_passed'] or job['id'] != state['job_id'] or job['submission_id'] != state['submission_id']
            or job['job_tag'] != 'pyasc-smoke-'+prepared['files']['submission.zip'][:24]):
        raise ValueError('Hardware result is not bound to a successful exact generated candidate')
    if smoke.sha(root/'package/evaluated-submission.zip') != prepared['files']['submission.zip']:
        raise ValueError('Missing exact service-returned submission continuity')
    score = job['results'].get('overall_score')
    if not isinstance(score, (int,float)) or not math.isfinite(score): raise ValueError('Missing finite service score')
    cases = job['results']['operators'][0]['cases']
    for c in cases:
        if c.get('accuracy', {}).get('passed') is not True or not c.get('op_times',{}).get('device_kernels'):
            raise ValueError('Missing case numerical or device-kernel evidence')
    generation_ci = meta.get('generation_ci', meta)
    rows = [f"# Generated {run_recipe(root)['operator']}: verified CANNBench result", '',
            f"Model: `{MODEL}`; provider revision: unknown. OpenCode: `{meta['opencode_version']}`.",
            f"Skill stack: `{meta['skill_stack_commit']}`. PyASC: `{PIN['pyasc_commit']}`.",
            f"[Generation CI]({generation_ci['ci_url']}), attempt {generation_ci['origin_run_attempt']}; model attempt {gen['attempt']}.",
            f"[Evaluation CI]({meta['ci_url']}); orchestration commit `{meta.get('orchestration_commit',meta['skill_stack_commit'])}`.",
            f"CANNBench job: `{job['id']}`, submission `{job['submission_id']}`.",
            f"Correct/measured cases: {report['correct_cases']}/{report['required_cases']}; anti-cheat failures: 0.",
            f"Service performance score (`overall_score`): **{score}**.",
            f"Unweighted geometric mean baseline/candidate timing ratio: **{report['true_geometric_mean_speedup']:.8f}x**.",
            'A ratio above 1 means faster than baseline; below 1 means slower. It is not the service score.',
            '', '| Case | Candidate us | Baseline us | Baseline/candidate |', '|---|---:|---:|---:|']
    for c in cases: rows.append(f"| {c['case_id']} | {c['elapsed_us']} | {c['baseline_perf_us']} | {c['baseline_perf_us']/c['elapsed_us']:.8f} |")
    for label, a, b in (('Queue','queued_at','claimed_at'),('Execution','claimed_at','finished_at')):
        duration=(datetime.fromisoformat(job[b])-datetime.fromisoformat(job[a])).total_seconds()
        rows.append(f'\n{label}: {duration:.3f} seconds.')
    rows += ['', f"Candidate SHA256: `{gen['candidate_sha256']}`.", f"ZIP SHA256: `{prepared['files']['submission.zip']}`.",
             'Scope: one model-generated operator on the recorded task snapshot/hardware; no broad performance claim.']
    (root/'RESULT.md').write_text('\n'.join(rows)+'\n')
    save(root/'acceptance.json', {'passed': True, 'origin': meta, 'generation': gen,
        'package_manifest_sha256': smoke.sha(root/'package/manifest.json'), 'hardware': report,
        'service_score': score, 'publishing_ci': ci_identity()})
    # Never include authentication databases or provider credentials in published evidence.
    for p in root.rglob('*'):
        if p.is_symlink(): raise ValueError('Symlink in evidence')
        if p.is_file() and p.suffix in ('.json','.log','.md','.txt','.yaml','.py'):
            data=p.read_text(errors='replace')
            if scrub(data)!=data: raise ValueError('Credential detected in evidence: '+str(p.relative_to(root)))
    manifest = {n: h for n, h in hashes(root).items() if n != 'EVIDENCE.json'}
    save(root/'EVIDENCE.json', {'files': manifest, 'created_by_ci': ci_identity()})
    archive = root.parent / (root.name+'-qualification.tar.gz')
    if archive.exists(): raise ValueError('Existing qualification archive: reconcile release rather than overwrite')
    with tarfile.open(archive, 'x:gz') as z: z.add(root, arcname='evidence')
    with tarfile.open(archive) as z:
        for n,h in manifest.items():
            if hashlib.sha256(z.extractfile('evidence/'+n).read()).hexdigest()!=h: raise ValueError('Archive mismatch')
    publication_outputs(root, archive, meta)


def verify_archive(archive, root):
    with tarfile.open(archive) as z:
        evidence = json.load(z.extractfile('evidence/EVIDENCE.json'))
        check_hashes(root, evidence['files'])
        for name, digest in evidence['files'].items():
            if hashlib.sha256(z.extractfile('evidence/'+name).read()).hexdigest() != digest:
                raise ValueError('Existing archive hash mismatch: '+name)
        accepted = json.load(z.extractfile('evidence/acceptance.json'))
        if accepted.get('passed') is not True:
            raise ValueError('Archive has no acceptance record')


def publication_outputs(root, archive, meta):
    operator = meta.get('operator_recipe', 'swiglu')  # Preserve historical release tags.
    checksum=archive.with_suffix(archive.suffix+'.sha256')
    checksum.write_text(smoke.sha(archive)+'  '+archive.name+'\n')
    with open(os.environ['GITHUB_OUTPUT'],'a') as out:
        out.write(f'archive={archive}\nchecksum={checksum}\ntag=qualification-{operator}-{meta["origin_run_id"]}\ncommit={meta["skill_stack_commit"]}\n')
    with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as out: out.write((root/'RESULT.md').read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('initialize','generate','continue','submit','collect','publish'))
    ap.add_argument('--origin-run',default='');ap.add_argument('--repair-run',default='');ap.add_argument('--qualified-run',default='');ap.add_argument('--wait-seconds',type=int,default=9000)
    args=ap.parse_args();identity=ci_identity();origin=args.origin_run or identity['GITHUB_RUN_ID'];root=run_root(origin)
    if args.action=='initialize':
        if args.origin_run:
            if not root.exists(): restore_checkpoint(root, origin)
            inputs(root)
        with open(os.environ['GITHUB_ENV'],'a') as out: out.write(f'CANNBENCH_RESULTS={root}\nCANNBENCH_ORIGIN_RUN={origin}\n')
        return 0
    if args.action=='generate': generate(root,args.repair_run)
    elif args.action=='continue': continue_qualified(root,args.qualified_run)
    elif args.action=='submit': submit(root)
    elif args.action=='collect': return collect(root,args.wait_seconds)
    else: publish(root)
    return 0


if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc:
        print('CI CANNBench stopped: '+scrub(str(exc)),file=sys.stderr)
        sys.exit(1)
