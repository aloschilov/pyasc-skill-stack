"""Narrow identity receipt for an OpenCode 1.19 response capped before CLI events.

The stopped client's private SQLite backup stays private. This cannot establish
skill loads, file accesses, a candidate or successful generation.
"""
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

from generation_plan import digest, exact_keys
from generation_provider import known_budget_exhaustion


def early_output_cap(usage, limits):
    if not isinstance(usage, dict): return False
    rows = usage.get('requests', [])
    return bool(known_budget_exhaustion(usage, limits) == 'output-limit'
        and rows[-1].get('audit', {}).get('tool_schema_sha256')
        and rows[-1]['audit'].get('tool_names') == ['edit', 'read', 'skill', 'write']
        and all(r.get('audit', {}).get('tool_schema_sha256') is None
            and r.get('audit', {}).get('tool_names') == [] and r['usage'].get('finish_reason') == 'stop'
            for r in rows[:-1]))


def check(condition, message):
    if not condition: raise ValueError(message)


def validate_empty_session(value, project, model, rendered_prompt):
    exact_keys(value, ('schema_version', 'kind', 'session_id', 'session_version', 'project', 'model',
        'user_message_id', 'assistant_message_id', 'prompt_sha256', 'table_counts', 'snapshot_sha256'), 'empty session receipt')
    check(type(value['schema_version']) is int and value['schema_version'] == 1
        and value['kind'] == 'opencode-1.19-empty-session', 'Unsupported native receipt')
    check(value['session_version'] == '1.19.0' and value['project'] == str(project.resolve())
        and value['model'] == model and value['prompt_sha256'] == digest(rendered_prompt.encode()), 'Native session input mismatch')
    check(value['table_counts'] == {'project': 1, 'session': 1, 'message': 2, 'part': 1}
        and all(type(v) is int for v in value['table_counts'].values()), 'Native session has additional activity')
    for key, prefix in [('session_id', 'ses_'), ('user_message_id', 'msg_'), ('assistant_message_id', 'msg_')]:
        check(isinstance(value[key], str) and re.fullmatch(prefix+r'[A-Za-z0-9_-]+', value[key]), 'Invalid native identity')
    check(value['user_message_id'] != value['assistant_message_id'] and isinstance(value['snapshot_sha256'], str)
        and re.fullmatch(r'[0-9a-f]{64}', value['snapshot_sha256']), 'Invalid native snapshot binding')
    return value


def capture_empty_session(private, project, model, rendered_prompt):
    """After command cleanup only; no CLI call, provider request or DB mutation."""
    private, project = Path(private).resolve(), Path(project).resolve()
    database = private/'data/opencode/opencode.db'; snapshot = private/'empty-session.db'
    check(database.is_file() and database.resolve().is_relative_to(private) and not database.is_symlink()
        and not snapshot.exists() and not snapshot.is_symlink(), 'Native DB absent or snapshot already exists')
    check(not (project/'candidate.py').exists() and not (project/'candidate.py').is_symlink(),
        'Unexpected candidate before an empty native trace')
    snapshot.touch(mode=0o600, exist_ok=False)
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True)) as source:
        with closing(sqlite3.connect(snapshot)) as copy: source.backup(copy)
    snapshot.chmod(0o600)
    check(snapshot.stat().st_size <= 64*1024*1024, 'Native snapshot too large')
    with closing(sqlite3.connect(snapshot.as_uri()+'?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        counts = {t: db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ('project', 'session', 'message', 'part')}
        check(counts == {'project': 1, 'session': 1, 'message': 2, 'part': 1}, 'Native session has additional activity')
        session = db.execute('SELECT id,project_id,parent_id,directory,version FROM session').fetchone()
        owner = db.execute('SELECT id,worktree FROM project').fetchone()
        check(session['parent_id'] is None and session['project_id'] == owner['id']
            and owner['worktree'] == session['directory'] == str(project), 'Native session belongs to another project')
        messages = [(r, json.loads(r['data'])) for r in db.execute('SELECT id,session_id,data FROM message')]
        check(all(r['session_id'] == session['id'] for r, _ in messages), 'Cross-session messages')
        check(sorted(m.get('role', '') for _, m in messages) == ['assistant', 'user'], 'Unexpected native message roles')
        user, assistant = (next((r, m) for r, m in messages if m['role'] == role) for role in ('user', 'assistant'))
        provider, selected = model.split('/')
        check(user[1].get('model') == {'providerID': provider, 'modelID': selected} and user[1].get('agent') == 'build'
            and user[1].get('summary') in (None, {'diffs': []}), 'Unexpected native user request')
        check(assistant[1].get('parentID') == user[0]['id'] and assistant[1].get('modelID') == selected
            and assistant[1].get('providerID') == provider and assistant[1].get('agent') == assistant[1].get('mode') == 'build'
            and assistant[1].get('path') == {'cwd': str(project), 'root': str(project)}
            and not assistant[1].get('error') and not assistant[1].get('finish'), 'Unexpected native assistant state')
        part = db.execute('SELECT session_id,message_id,data FROM part').fetchone(); data = json.loads(part['data'])
        check(part['session_id'] == session['id'] and part['message_id'] == user[0]['id']
            and data == {'type': 'text', 'text': rendered_prompt}, 'Native prompt/parts differ')
        receipt = {'schema_version': 1, 'kind': 'opencode-1.19-empty-session', 'session_id': session['id'],
            'session_version': session['version'], 'project': str(project), 'model': model,
            'user_message_id': user[0]['id'], 'assistant_message_id': assistant[0]['id'],
            'prompt_sha256': digest(rendered_prompt.encode()), 'table_counts': counts,
            'snapshot_sha256': digest(snapshot.read_bytes())}
    return validate_empty_session(receipt, project, model, rendered_prompt)
