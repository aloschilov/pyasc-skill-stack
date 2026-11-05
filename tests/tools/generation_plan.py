"""Build an offline generation experiment plan; never invoke a model or runtime."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = 'tests/generation/matrix.json'
IDENTIFIER = r'[a-z][a-z0-9_-]*'
LIMITS = {
    'attempts_per_cell': 3, 'rounds_per_attempt': 12,
    'output_tokens_per_response': 24000, 'generation_seconds_per_attempt': 900,
    'qualification_seconds_per_attempt': 900, 'max_output_tokens': 1000000,
    'max_execution_seconds': 21600,
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'Unexpected fields in {label}')


def resource(root, name):
    """Only explicit files within this checkout; no external symlink targets."""
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe input path')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing or external input: ' + name)
    return path.read_bytes()


def declared_cases(data):
    """Read declared groups without importing torch/PyASC or running pytest."""
    match = re.search(r'# PYASC_TESTS_BEGIN\n(.+?)# PYASC_TESTS_END', data.decode(), re.S)
    if not match:
        raise ValueError('No declarative case inventory')
    items = ast.parse('[' + match[1] + ']', mode='eval').body.elts
    result = {}
    for row in items:
        if not isinstance(row, ast.Tuple) or not row.elts or not isinstance(row.elts[0], ast.Constant):
            raise ValueError('Unsupported case declaration')
        name = row.elts[0].value
        if not isinstance(name, str) or name in result:
            raise ValueError('Invalid or duplicate declared case')
        # Bind every parameter, including shape/dtype/attrs/tiling, without eval.
        result[name] = digest(ast.dump(row, include_attributes=False).encode())
    if not result:
        raise ValueError('Empty declared case inventory')
    return result


def build_plan(root=ROOT, matrix=DEFAULT_MATRIX):
    root = Path(root)
    inputs = {}
    def read(name):
        data = resource(root, name)
        inputs[name] = digest(data)
        return data
    spec = json.loads(read(matrix))
    exact_keys(spec, ('schema_version', 'profile', 'skills', 'limits', 'cells'), 'matrix')
    if type(spec['schema_version']) is not int or spec['schema_version'] != 1:
        raise ValueError('Unsupported matrix schema')
    limits = spec['limits']; exact_keys(limits, LIMITS, 'limits')
    for name, ceiling in LIMITS.items():
        if type(limits[name]) is not int or not 1 <= limits[name] <= ceiling:
            raise ValueError('Invalid limit: ' + name)
    cells = spec['cells']
    if not isinstance(cells, list) or not 1 <= len(cells) <= 6:
        raise ValueError('Select between one and six cells')
    attempts = len(cells) * limits['attempts_per_cell']
    tokens = attempts * limits['rounds_per_attempt'] * limits['output_tokens_per_response']
    seconds = attempts * (limits['generation_seconds_per_attempt'] + limits['qualification_seconds_per_attempt'])
    if tokens > limits['max_output_tokens'] or seconds > limits['max_execution_seconds']:
        raise ValueError('Matrix exceeds aggregate execution budget')
    profile = spec['profile']
    if not isinstance(profile, str) or not re.fullmatch(r'[a-z][a-z0-9_.-]*', profile):
        raise ValueError('Invalid profile')
    config = json.loads(read(f'integrations/opencode/profiles/{profile}.json'))
    model = config.get('model')
    if not isinstance(model, str) or model.count('/') != 1:
        raise ValueError('Profile must select an explicit provider/model')
    provider, model_name = model.split('/')
    entry = config.get('provider', {}).get(provider, {}).get('models', {}).get(model_name, {})
    model_limit = entry.get('limit', {}).get('output')
    if type(model_limit) is not int or limits['output_tokens_per_response'] > model_limit:
        raise ValueError('Response limit exceeds configured model output limit')
    image = json.loads(read('reference/images/ENV.json'))
    pin = json.loads(read('reference/asctile/PIN.json'))
    ref = image.get('dev_image', {}).get('ref', '')
    if not re.fullmatch(r'[^\s@]+@sha256:[0-9a-f]{64}', ref):
        raise ValueError('Digest-pinned simulator image required')
    if not re.fullmatch(r'[0-9a-f]{40}', image.get('source_sha', '')) or image['source_sha'] != pin.get('commit'):
        raise ValueError('Runtime and task source revisions differ')
    skills = spec['skills']
    if not isinstance(skills, list) or not skills or any(not isinstance(s, str) or not re.fullmatch(IDENTIFIER, s) for s in skills):
        raise ValueError('Explicit skill list required')
    if len(set(skills)) != len(skills):
        raise ValueError('Duplicate skills')
    for skill in skills:
        read('skills/' + skill + '/SKILL.md')
        for p in sorted((root/'skills'/skill).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.md', '.py', '.sh', '.json'):
                read(p.relative_to(root).as_posix())
    planned, seen, selected = [], set(), set()
    for cell in cells:
        exact_keys(cell, ('id', 'module', 'cases'), 'cell')
        name = cell['id']
        if not isinstance(name, str) or not re.fullmatch(IDENTIFIER, name) or name in seen:
            raise ValueError('Invalid or duplicate cell ID')
        seen.add(name)
        module = cell['module']
        if not isinstance(module, str) or not re.fullmatch(r'(?:[a-z0-9_]+/)*test_[a-z0-9_]+\.py', module):
            raise ValueError('Invalid task module')
        data = read('reference/asctile/target/' + module)
        if digest(data) != pin['files'].get(module):
            raise ValueError('Pinned task source drift: ' + module)
        declared = declared_cases(data)
        cases = cell['cases']
        if not isinstance(cases, list) or not cases or any(not isinstance(c, str) for c in cases) or len(set(cases)) != len(cases):
            raise ValueError('Nonempty unique case selection required')
        if any(c not in declared or (module, c) in selected for c in cases):
            raise ValueError('Unknown or repeated selected case')
        selected.update((module, c) for c in cases)
        planned.append(dict(cell, case_ast_sha256={c: declared[c] for c in cases}, status='not-run'))
    # Bind helper/oracle source dependencies too, not only selected test modules.
    for module, expected in sorted(pin['files'].items()):
        if digest(read('reference/asctile/target/' + module)) != expected:
            raise ValueError('Pinned task source drift: ' + module)
    read('reference/asctile/conftest.py')
    read('tests/tools/generation_plan.py')
    return {
        'schema_version': 1, 'kind': 'generation-plan', 'status': 'not-run',
        'execution_enabled': False, 'schedule_enabled': False,
        'model': {'requested': model, 'observed': None, 'profile': profile},
        'runtime': {'backend': 'Model', 'image': ref, 'source_revision': image['source_sha']},
        'skills': skills, 'cells': planned, 'limits': limits,
        'maximum_work': {'model_attempts': attempts, 'output_tokens': tokens, 'execution_seconds': seconds},
        'required_gates': ['static', 'jit', 'numerically_verified', 'suppressed-launch-control'],
        'source_sha256': inputs,
        'limitations': ['Planning only; no provider, runtime, NPU or service calls.',
            'Declared case groups are task anchors, not collected pytest instances or numerical coverage.',
            'Candidate-specific oracle/probe adapters and runtime enforcement are not implemented.',
            'Token bound covers output including reasoning, not input tokens, price or provider charges.',
            'Execution bound excludes setup, queue and archival overhead; schedule approval is separate.'],
    }


def write_plan(plan, directory):
    directory = Path(directory).expanduser().resolve()
    if directory == ROOT or ROOT in directory.parents:
        raise ValueError('Keep plans outside checkout')
    payload = dict(plan, plan_sha256=digest(canonical(plan)))
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'plan.json').write_text(json.dumps(payload, indent=2)+'\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', default=DEFAULT_MATRIX, help='Checkout-relative inventory')
    parser.add_argument('--results-dir', required=True, type=Path)
    args = parser.parse_args()
    plan = build_plan(matrix=args.matrix)
    # Git metadata is optional; input hashes remain authoritative for dirty trees.
    revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    plan['checkout_revision'] = revision.stdout.strip() if revision.returncode == 0 else None
    result = write_plan(plan, args.results_dir)
    print(json.dumps({'status': result['status'], 'cells': len(result['cells']),
                      'plan_sha256': result['plan_sha256'], 'execution_enabled': False}))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit('Generation planning stopped: ' + str(exc))
