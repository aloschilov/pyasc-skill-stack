"""Explicit paired interventions. Historical P-labels do not imply old evidence."""
import json
from pathlib import Path

from generation_plan import canonical, digest

BASELINE = 'tests/generation/context/baseline.md'
PROTOCOLS = {
    'P2': {'name': 'minimal', 'guidance': False, 'baseline_context': False, 'skills': False},
    'P3': {'name': 'guided', 'guidance': True, 'baseline_context': False, 'skills': False},
    'P4': {'name': 'baseline-notes-v2', 'guidance': True, 'baseline_context': True, 'skills': False},
    'P6': {'name': 'skills', 'guidance': True, 'baseline_context': False, 'skills': True},
}
# P6-P3 isolates the skills package/retrieval intervention. P6-P4 replaces two
# different context packages and cannot be reported as the pure skills effect.
CONTRASTS = [('P6', 'P3'), ('P3', 'P2'), ('P4', 'P3'), ('P6', 'P4')]
GUIDANCE = ('Check the output shape and each input role before writing the kernel. '
    'Partition work consistently with the declared block count. Account for tail '
    'elements and preserve the task dtype. Check device-tensor shapes through '
    'broadcasts and reductions. Do not add a CPU fallback or change the task.')


def common_prompt(contract):
    # This necessary ABI/acceptance information is identical in every leg.
    return ('Write candidate.py for this task: '+json.dumps(contract, sort_keys=True)+'.\n'
        'Use import asc and from asc.experimental import asctile. Define @asctile.jit device functions '
        'and def launch('+', '.join(contract['arguments'])+'). The host launch body must be exactly '
        'one kernel[block_count](input tensors, out, optional literal scalar arguments) call; '
        'block_count is '+str(contract['blocks'])+'. No Torch, CPU arithmetic, imports inside functions, '
        'runtime configuration, main, tests, file I/O or function defaults in candidate.py. '
        'Supported device calls are asctile APIs, tensor reshape/broadcast_to and scalar min/max/range/len/int/float/bool/abs. '
        'All inputs are CPU tensors for Model simulation; the harness owns the oracle and poisoned output. '
        'Do not copy a reference kernel. Do not run commands or search directories. '
        'Read only explicitly provided context files or candidate.py. Write only candidate.py.')


def prompt(contract, protocol, skills, context):
    definition = PROTOCOLS[protocol]
    parts = [common_prompt(contract)]
    if definition['guidance']: parts.append(GUIDANCE)
    if definition['baseline_context']:
        parts.append('Read the complete pinned baseline notes with the read tool before writing: '+str(context/BASELINE))
    if definition['skills']:
        parts.append('Load these pinned skills with the skill tool before writing: '+', '.join(skills)+'.')
    return '\n\n'.join(parts)


def context_files(plan):
    """Exact generation-visible inputs; source snapshots/oracles are elsewhere."""
    selected = plan['skills']
    names = [n for n in plan['source_sha256'] if n.startswith('integrations/opencode/profiles/') or
        any(n.startswith('skills/'+s+'/') for s in selected)]
    if plan.get('intervention', {}).get('id') == 'P4': names.append(BASELINE)
    return {n: plan['source_sha256'][n] for n in sorted(names)}


def skill_body(text):
    return text.split('---', 2)[-1].strip() if text.startswith('---\n') else text.strip()


def provider_prompt(text):
    # OpenCode v1.19.0 cli/cmd/run.ts:288-290 renders each message argument.
    # Match this exact lossless representation, not fuzzy/unescaped substrings.
    return '"'+text.replace('"', '\\"')+'"' if ' ' in text else text


def request_audit(body, text, protocol, selected_skills, context):
    """Record actual provider inputs without publishing messages or credentials."""
    messages = json.dumps(body.get('messages', []), ensure_ascii=False)
    if protocol != 'P6' and any(s in messages for s in selected_skills):
        raise ValueError('Unselected skill appeared in provider context')
    tools = body.get('tools', [])
    # Skill availability/descriptions belong to the context intervention. Tool
    # names and argument schemas must stay constant, including for empty skills.
    schemas = sorted([{'name': t['function']['name'], 'parameters': t['function']['parameters']}
                      for t in tools], key=lambda t: t['name'])
    def strings(value):
        if isinstance(value, str): return [value]
        if isinstance(value, list): return [s for v in value for s in strings(v)]
        if isinstance(value, dict): return [s for v in value.values() for s in strings(v)]
        return []
    user = '\n'.join(s for m in body.get('messages', []) if m.get('role') == 'user'
                     for s in strings(m.get('content')))
    contents = '\n'.join(s for m in body.get('messages', []) for s in strings(m.get('content')))
    visible = [s for s in selected_skills if protocol == 'P6'
               and skill_body((context/'skills'/s/'SKILL.md').read_text()) in contents]
    baseline = False
    if protocol == 'P4':
        numbered = '\n'.join(f'{i}: {line}' for i, line in enumerate((context/BASELINE).read_text().splitlines(), 1))
        baseline = numbered in contents
    rendered = provider_prompt(text)
    return {'protocol': protocol, 'prompt_present': rendered in user,
        'prompt_encoding': 'opencode-1.19-single-argument',
        'provided_prompt_sha256': digest(text.encode()), 'provider_prompt_sha256': digest(rendered.encode()),
        'visible_skills': visible, 'baseline_visible': baseline,
        'messages_sha256': digest(canonical(body.get('messages', []))),
        'tool_schema_sha256': digest(canonical(schemas)) if schemas else None,
        'tool_names': [t['name'] for t in schemas]}


def audit_trace(events, project, context, plan, *, source_context=None, allow_client_error=False, empty_session=None, usage=None):
    """Verify observed file/skill access, not just a requested skills-on flag."""
    from generation_executor import Blocked, native_trace
    skills = {s: context/'skills'/s for s in plan['skills']}
    source_context = source_context or context
    if empty_session is not None:
        from generation_native import early_output_cap, validate_empty_session
        from generation_adapters import task_contract
        from generation_plan import ROOT
        if events.strip() or not allow_client_error or plan['intervention']['id'] not in ('P2', 'P3') or skills or not early_output_cap(usage, plan['limits']):
            raise Blocked('Empty native identity cannot replace protocol access evidence')
        text = prompt(task_contract(ROOT, plan['cells'][0]), plan['intervention']['id'], [], context)
        validate_empty_session(empty_session, project, plan['model']['requested'], provider_prompt(text))
        trace = {'session_id': empty_session['session_id'], 'loaded_skills': [],
            'session_source': 'private-db-empty-session', 'session_receipt_sha256': digest(canonical(empty_session))}
    else:
        trace = native_trace(events, skills, allow_client_error=allow_client_error)
    allowed = context_files(plan)
    # Profile configuration is harness input, not model-readable context.
    readable = {str((context/n).resolve()): sha for n, sha in allowed.items()
                if not n.startswith('integrations/')}
    candidate = str((project/'candidate.py').resolve())
    baseline_read = False; accesses = []; denied = []
    def path(value):
        if not isinstance(value, str): raise Blocked('Missing native access path')
        p = Path(value); return str((p if p.is_absolute() else project/p).resolve())
    for line in events.splitlines():
        part = json.loads(line).get('part') or {}; tool = part.get('tool')
        if not tool: continue
        state = part.get('state') or {}; data = state.get('input') or {}; meta = state.get('metadata') or {}
        if state.get('status') not in ('completed', 'error'): continue
        if tool not in ('read', 'glob', 'grep', 'skill', 'write', 'edit'):
            raise Blocked('Undeclared tool in paired generation')
        if meta.get('loaded'): raise Blocked('Unplanned automatic instruction load')
        # This pinned native denial occurs before the requested tool access.
        # Retain the attempted access without pretending its contents were read.
        if state['status'] == 'error' and state.get('error', '').startswith(
                'The user has specified a rule which prevents you from using this specific tool call.'):
            denied.append({'tool': tool, 'path': data.get('filePath', data.get('path')), 'skill': data.get('name')})
            continue
        if tool == 'skill':
            if data.get('name') not in skills: raise Blocked('Unselected skill call')
            if state['status'] == 'completed' and (meta.get('truncated') or skill_body(
                    (source_context/'skills'/data['name']/'SKILL.md').read_text()) not in state.get('output', '')):
                raise Blocked('Missing/truncated native skill content')
        elif tool in ('read', 'write', 'edit'):
            target = path(data.get('filePath'))
            if target != candidate and (tool != 'read' or target not in readable):
                raise Blocked('Unplanned file access')
            if tool == 'read' and state['status'] == 'completed':
                if target in readable:
                    accesses.append({'path': str(Path(target).relative_to(context)), 'sha256': readable[target]})
                if target == str((context/BASELINE).resolve()):
                    display = meta.get('display') or {}
                    baseline_read = (not meta.get('truncated') and display.get('lineStart') == 1
                        and display.get('text') == (source_context/BASELINE).read_text().rstrip('\n'))
        else:
            raise Blocked('Directory search is disabled in paired trials')
    if plan['intervention']['id'] == 'P4' and not baseline_read:
        raise Blocked('Complete baseline context read was not observed')
    for name, expected in allowed.items():
        if digest((source_context/name).read_bytes()) != expected: raise Blocked('Paired context changed')
    trace.update(protocol=plan['intervention']['id'], context_sha256=allowed,
        baseline_read=baseline_read, observed_reads=accesses, denied_accesses=denied, protocol_valid=True)
    return trace
