"""Exact pinned target-case inventory; collection alone proves no numerics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'reference/asctile/target/'
CONTRACT = 'tests/target-suite/inventory.json'


def require(condition, message):
    if not condition: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sources(root=ROOT):
    """Verify upstream files, and also bind local fixture/collection helpers."""
    pin = read(root/'reference/asctile/PIN.json')
    target = root/PREFIX
    for name, expected in pin['files'].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Invalid pinned path')
        require(digest((target/name).read_bytes()) == expected, 'Upstream source drift: '+name)
    modules = {p.relative_to(target).as_posix() for p in target.rglob('test_*.py')}
    require(modules == {n for n in pin['files'] if PurePosixPath(n).name.startswith('test_')},
            'Pinned module inventory differs from checkout')
    paths = list((root/'reference/asctile').rglob('*.py'))
    paths += [root/'reference/__init__.py', root/'reference/asctile/PIN.json', root/'reference/images/ENV.json']
    require(all(not p.is_symlink() for p in paths), 'Symlink in target inputs')
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in sorted(paths)}


def normalize(nodeids):
    require(isinstance(nodeids, list) and bool(nodeids), 'Empty or invalid case inventory')
    result = []
    for node in nodeids:
        require(isinstance(node, str) and '\n' not in node and '\r' not in node and '::' in node,
                'Invalid case ID')
        node = node.removeprefix(PREFIX)
        module, case = node.split('::', 1)
        path = PurePosixPath(module)
        require(not path.is_absolute() and '..' not in path.parts and str(path) == module
                and path.name.startswith('test_') and path.suffix == '.py' and bool(case), 'Invalid case ID')
        result.append(node)
    require(len(result) == len(set(result)), 'Duplicate case ID')
    return sorted(result)


def derive(nodeids, root=ROOT):
    ids = normalize(nodeids)
    inputs = sources(root)
    pin, environment = read(root/'reference/asctile/PIN.json'), read(root/'reference/images/ENV.json')
    require(pin['commit'] == environment['source_sha'], 'Target/runtime upstream revisions differ')
    image = environment['dev_image']['ref']
    require(re.fullmatch(r'ghcr\.io/[^@\s]+@sha256:[0-9a-f]{64}', image), 'Unpinned runtime image')
    grouped = defaultdict(list)
    for node in ids: grouped[node.split('::', 1)[0]].append(node)
    expected = {n for n in pin['files'] if PurePosixPath(n).name.startswith('test_')}
    require(set(grouped) == expected, 'Missing or unexpected collected module')
    return {'schema_version': 1, 'upstream_revision': pin['commit'], 'runtime_image': image,
        'platform': environment['image']['simulator'], 'inventory_format': 'target-relative-pytest-nodeids-v1',
        'case_count': len(ids), 'case_ids_sha256': digest(canonical(ids)), 'source_sha256': inputs,
        'modules': {name: {'case_count': len(cases), 'case_ids_sha256': digest(canonical(cases))}
                    for name, cases in sorted(grouped.items())}}


def validate(nodeids, root=ROOT):
    value = derive(nodeids, root)
    require(value == read(root/CONTRACT), 'Case inventory, runtime or source drift; review a new inventory')
    return value


def collect(directory, root=ROOT):
    """Bounded collection in the pinned development image, with no device/network."""
    directory = Path(directory).expanduser().resolve()
    require(directory != root and root not in directory.parents, 'Keep evidence outside checkout')
    directory.mkdir(parents=True, exist_ok=False)
    inputs = sources(root)
    env = read(root/'reference/images/ENV.json'); image = env['dev_image']['ref']
    require(image == read(root/CONTRACT)['runtime_image'], 'Inventory/runtime image mismatch')
    name = 'pyasc-target-collect-'+uuid.uuid4().hex
    argv = ['docker', 'run', '--rm', '--pull', 'never', '--name', name, '--network', 'none',
        '--cpus', '1', '--memory', '2g', '--ulimit', 'nofile=65536:65536',
        '-v', str(root)+':/repo:ro', '-w', '/repo', '-e', 'HOME=/tmp',
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1',
        image, env['image']['python_bin'], '-m', 'pytest', '-c', '/dev/null', '--rootdir', '/repo', PREFIX.rstrip('/'), '--collect-only',
        '-q', '-p', 'no:cacheprovider', '--backend', 'Model', '--platform', env['image']['simulator']]
    result = {'schema_version': 1, 'kind': 'target-collection', 'status': 'running', 'command': argv,
        'source_sha256': inputs, 'runtime_image': image, 'started_unix': time.time(),
        'collector_sha256': digest(Path(__file__).read_bytes()),
        'numerically_qualified_cases': 0, 'hardware_invoked': False, 'model_invoked': False}
    save(directory/'result.json', result)
    try:
        with (directory/'stdout').open('wb') as out, (directory/'stderr').open('wb') as err:
            process = subprocess.run(argv, stdout=out, stderr=err, timeout=180)
        result['exit_code'] = process.returncode
        require(process.returncode == 0, 'Collection command failed')
        ids = [line for line in (directory/'stdout').read_text().splitlines() if line.startswith(PREFIX) and '::' in line]
        value = validate(ids, root)
        save(directory/'nodeids.json', normalize(ids)); save(directory/'inventory.json', value)
        result.update(status='collected', case_count=len(ids), module_count=len(value['modules']))
    except Exception as exc:
        result.update(status='blocked', error=type(exc).__name__+': '+str(exc))
    finally:
        result.update(cleanup_confirmed=False, inputs_unchanged=False)
        try:
            removed = subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=20)
            result['cleanup_confirmed'] = removed.returncode == 0 or b'No such container' in removed.stderr
            result['inputs_unchanged'] = sources(root) == inputs
        except Exception as exc:
            result['finalization_error'] = type(exc).__name__+': '+str(exc)
        if not result['cleanup_confirmed'] or not result['inputs_unchanged']: result['status'] = 'blocked'
        result['seconds'] = time.time()-result['started_unix']
        result['artifact_sha256'] = {p.name: digest(p.read_bytes()) for p in directory.iterdir()
            if p.is_file() and p.name != 'result.json'}
        save(directory/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('collect').add_argument('--output', type=Path, required=True)
    for name in ('derive', 'check'):
        p = sub.add_parser(name); p.add_argument('--nodeids', type=Path, required=True)
        if name == 'derive': p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'collect':
        result = collect(args.output); print(json.dumps(result)); return 0 if result['status'] == 'collected' else 2
    value = (derive if args.command == 'derive' else validate)(read(args.nodeids))
    if args.command == 'derive':
        require(not args.output.exists(), 'Do not overwrite an inventory implicitly')
        save(args.output, value)
    print(json.dumps({'case_count': value['case_count'], 'module_count': len(value['modules']),
                      'case_ids_sha256': value['case_ids_sha256'], 'numerically_qualified_cases': 0}))
    return 0


if __name__ == '__main__': sys.exit(main())
