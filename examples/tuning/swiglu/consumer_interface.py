"""Narrow native-bash interface layered on the existing run_session gate."""
import hashlib
import json
import os
from pathlib import Path

COMPARE_ACTIONS = ('compare-1', 'compare-2', 'compare-3')
ACTIONS = ('baseline', 'variant', *COMPARE_ACTIONS, 'decision')
LIMITS = dict(rounds_per_attempt=12, generation_seconds_per_attempt=2700,
              max_output_tokens=200000, output_tokens_per_response=32768)

def commands(recipe):
    recipe = Path(recipe).resolve()
    if any(c in str(recipe) for c in " \n\t'\"$`*?[];|&()<>\\"):
        raise ValueError('Recipe install path must be shell-literal-safe')
    return {action: f'python3 -B {recipe}/tool.py {action}' for action in ACTIONS}

def tool_config(config, recipe, project):
    config = json.loads(json.dumps(config))
    project = Path(project).resolve(); allowed = commands(recipe)
    config['permission']['bash'] = {'*': 'deny', **{c: 'allow' for c in allowed.values()}}
    config['permission']['edit'] = {'*': 'deny', **{n: 'allow' for n in ('candidate.py', 'decision.json')},
                                  **{str(project/n): 'allow' for n in ('candidate.py', 'decision.json')}}
    for n in ('baseline.py','TUNING_TASK.md','candidate.py','decision.json'):
        config['permission']['read'][n]='allow';config['permission']['read'][str(project/n)]='allow'
    config['permission']['external_directory'][str(Path(recipe).resolve()/'*')] = 'allow'
    return config

def validate_bash(data, recipe, project):
    # This is the native-trace/receipt audit in addition to native permissions.
    # Never use shell tokenization to accept prefixes, trailing commands or aliases.
    rows = commands(recipe); command = data.get('command')
    if command not in rows.values() or set(data) - {'command', 'description', 'timeout', 'workdir'}:
        raise ValueError('Unapproved command or argument')
    if Path(data.get('workdir', project)).resolve() != Path(project).resolve():
        raise ValueError('Different tool workdir')
    timeout = data.get('timeout')
    action = next(a for a,c in rows.items() if c == command)
    expected = 600000 if action in COMPARE_ACTIONS else 300000
    if type(timeout) is not int or timeout != expected:
        raise ValueError('Explicit native timeout insufficient or outside OpenCode cap')
    return action

def request_audit(body):
    schemas = {t['function']['name']:t['function']['parameters'] for t in body.get('tools', [])}
    if 'bash' not in schemas or set(schemas) - {'bash','read','write','edit','skill'}:
        raise ValueError('Native tool schema missing bash or contains an unplanned tool')
    schema = schemas['bash']; props = schema.get('properties', {})
    if 'command' not in props or 'timeout' not in props:
        raise ValueError('Native bash declaration is incomplete')
    return {'tool_names':sorted(schemas), 'tool_schema_sha256':hashlib.sha256(json.dumps(schemas,sort_keys=True).encode()).hexdigest()}

def audit_events(events, recipe, project, receipt_root):
    seen=[]
    for event in events:
        part=event.get('part') or {}; state=part.get('state') or {}
        if part.get('tool') != 'bash' or state.get('status') not in ('completed','error'):
            continue
        data=state.get('input') or {}
        if state.get('status')=='error' and str(state.get('error','')).startswith('The user has specified a rule which prevents you from using this specific tool call.'):
            seen.append({'denied':True,'command_sha256':hashlib.sha256(str(data.get('command')).encode()).hexdigest()});continue
        action=validate_bash(data,recipe,project)
        receipt=Path(receipt_root)/action/'public-receipt.json'
        if not receipt.is_file():
            raise ValueError('Native completed tool has no owned receipt; no synthetic feedback')
        raw=receipt.read_text(); report=json.loads(raw)
        if report.get('action') != action or report.get('native_command') != data['command'] or raw.strip() not in state.get('output',''):
            raise ValueError('Native tool output is not bound to actual owned receipt')
        seen.append({'action':action,'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'status':report['status']})
    return seen

def native_source_binding(events,project,receipt_root):
    """Reconstruct final native writes and bind evaluated snapshot, no source import."""
    project=Path(project).resolve();constructed={};sealed=None;writes=[]
    for number,event in enumerate(events,1):
        part=event.get('part') or {};state=part.get('state') or {};data=state.get('input') or {};tool=part.get('tool')
        if state.get('status')not in('completed','error'):continue
        if tool in ('write','edit') and state.get('status')=='completed':
            path=Path(data.get('filePath',''));path=(path if path.is_absolute()else project/path).resolve()
            if path.parent!=project or path.name not in ('candidate.py','decision.json'):
                raise ValueError('Write outside the two consumer artifacts')
            if path.name=='candidate.py' and sealed is not None:
                raise ValueError('Post-evaluation candidate edit; retain outcome, no repair')
            if tool=='write' and isinstance(data.get('content'),str):constructed[path.name]=data['content']
            elif tool=='edit':
                old,new=data.get('oldString'),data.get('newString');prior=constructed.get(path.name)
                if not isinstance(prior,str)or not isinstance(old,str)or not old or not isinstance(new,str)or old not in prior:
                    raise ValueError('Cannot reconstruct native edit')
                if not data.get('replaceAll') and prior.count(old)!=1:
                    raise ValueError('Ambiguous edit reconstruction')
                constructed[path.name]=prior.replace(old,new) if data.get('replaceAll')else prior.replace(old,new,1)
            else:raise ValueError('Missing native write content')
            writes.append({'event':number,'file':path.name,'sha256':hashlib.sha256(constructed[path.name].encode()).hexdigest()})
        if tool=='bash' and str(data.get('command','')).endswith('/tool.py variant'):
            snapshot=Path(receipt_root)/'variant/candidate/candidate.py'
            if snapshot.exists():
                current=constructed.get('candidate.py')
                if current is None or hashlib.sha256(current.encode()).hexdigest()!=hashlib.sha256(snapshot.read_bytes()).hexdigest():
                    raise ValueError('Evaluated snapshot lacks native write binding')
                proposal=Path(receipt_root)/'variant/hypothesis-before-evaluation.json'
                if not proposal.is_file()or constructed.get('decision.json') is None or proposal.read_bytes()!=constructed['decision.json'].encode():
                    raise ValueError('Hypothesis not bound to a native write before evaluation')
                sealed=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    final={}
    for name,content in constructed.items():
        p=project/name
        if not p.is_file()or p.read_bytes()!=content.encode():raise ValueError('Final bytes changed outside native writes')
        final[name]=hashlib.sha256(content.encode()).hexdigest()
    return {'evaluated_variant_sha256':sealed,'final_files':final,'native_write_events':writes}

def installed_context_permissions(config, context, project, skills):
    """Exact alias/canonical spellings for the four verified installer targets."""
    context,project=Path(context).resolve(),Path(project).resolve()
    config['permission']['external_directory']={'*':'deny'}
    for name in skills:
        installed=context/'skills'/name
        resolved=installed.resolve()
        for root in (installed,resolved):
            config['permission']['external_directory'][str(root)]='allow'
            config['permission']['external_directory'][str(root/'*')]='allow'
        for path in resolved.rglob('*'):
            if path.is_file():
                linked=installed/path.relative_to(resolved)
                for spelling in (str(path),str(linked),os.path.relpath(path,project),os.path.relpath(linked,project)):
                    config['permission']['read'][spelling]='allow'
    return config
