#!/usr/bin/env python3
"""Explicit build-only handoff to the maintained upstream wheel recipe (#26)."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'integrations/cannbench'))
import runtime_bundle as bundle

RECIPE = 'f5f95882a7a6c912d7a74cf6204fb9934d31e7a9'
SOURCE = 'd0f64a63c432761aa57648e67be7582780a17561'
TARGET = 'cp312-x86_64'
ENVIRONMENT = 'ghcr.io/compiler-team-ru/pyasc-env@sha256:8cf07d34df92d2433449099f7d78d20faa44b68769a89eec88bc5352943ab940'


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external(path):
    path = path.resolve()
    if path == ROOT or ROOT in path.parents or any((p/'.git').exists() for p in (path, *path.parents)):
        raise ValueError('Use an output directory outside every checkout')
    if ',' in str(path) or '\n' in str(path):
        raise ValueError('Unsupported Docker mount path')
    return path


def run(command, log=None, timeout=600, **kwargs):
    if log:
        with log.open('w') as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT,
                           timeout=timeout, **kwargs)
    else:
        return subprocess.check_output(command, text=True, timeout=timeout, **kwargs).strip()


def clone(repository, revision, destination, log):
    run(['git', 'init', '-q', str(destination)])
    run(['git', '-C', str(destination), 'remote', 'add', 'origin', 'https://github.com/'+repository+'.git'])
    run(['git', '-C', str(destination), 'fetch', '--depth=1', 'origin', revision], log)
    run(['git', '-C', str(destination), 'checkout', '-q', '--detach', 'FETCH_HEAD'])
    if run(['git', '-C', str(destination), 'rev-parse', 'HEAD']) != revision:
        raise ValueError('Unexpected source revision')


def native():
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise ValueError('Requires native Linux x86_64; emulated builds are not qualified')
    if run(['docker', 'info', '--format', '{{.OSType}}/{{.Architecture}}']) != 'linux/x86_64':
        raise ValueError('Requires a native Linux x86_64 Docker daemon')


def container(command, name, log, timeout):
    try:
        run(command, log, timeout)
    finally:
        # This exact unique container belongs to this invocation; no global prune.
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60, check=False)


def build(root, revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Supply a full lowercase source commit')
    native()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    recipe, source = root/'recipe', root/'source'
    for repo, rev, path in [('compiler-team-ru/asctile', RECIPE, recipe),
                            ('compiler-team-ru/pyasc', revision, source)]:
        clone(repo, rev, path, root/(path.name+'-fetch.log'))
    # The pinned upstream recipe remains the owner of dependency/toolchain locks.
    sys.path.insert(0, str(recipe/'ci'))
    spec = importlib.util.spec_from_file_location('upstream_resolve', recipe/'ci/resolve.py')
    resolver = importlib.util.module_from_spec(spec); spec.loader.exec_module(resolver)
    lock = json.loads((recipe/'ci/toolchain.json').read_text())[TARGET]
    version = '1.1.1.dev'+datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    manifest = {'source_repository': 'compiler-team-ru/pyasc', 'source_sha': revision,
                'commit_time': run(['git', '-C', str(source), 'show', '-s', '--format=%cI', revision]),
                'recipe_sha256': resolver.recipe_digest(recipe), 'recipe_commit': RECIPE,
                'recipe_repository': 'compiler-team-ru/asctile', 'target': TARGET,
                'artifact': 'local-source-runtime', 'build': True}
    save(recipe/'source.json', manifest)
    source_files = ['scripts/source_runtime.py', 'integrations/cannbench/runtime_bundle.py',
                    'integrations/cannbench/smoke.py', 'tests/integration/check-source-runtime.py',
                    'tests/fixtures/source-runtime/runtime_probe.py']
    save(root/'intent.json', dict(manifest, version=version, qualification_environment=ENVIRONMENT,
         stack_commit=run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
         stack_dirty=bool(run(['git', '-C', str(ROOT), 'status', '--porcelain'])),
         stack_files={n: sha(ROOT/n) for n in source_files}, hardware='not-run', publication='none'))
    for name in ('work', 'output'):
        (root/name).mkdir()
    image_file = root/'builder-image.txt'
    run(['docker', 'build', '--pull', '--iidfile', str(image_file),
         '--build-arg', 'BASE_IMAGE='+lock['base_image'], '--build-arg', 'BUILD_TARGET='+TARGET,
         '-f', str(recipe/'ci/Dockerfile'), str(recipe)], root/'builder.log', 1800)
    image = image_file.read_text().strip()
    name = 'pyasc-source-'+uuid.uuid4().hex
    command = ['docker', 'run', '--rm', '--name', name,
               '-v', str(source)+':/source:ro', '-v', str(recipe)+':/recipe:ro',
               '-v', str(root/'work')+':/work', '-v', str(root/'output')+':/out',
               '-e', 'SOURCE_SHA='+revision, '-e', 'BUILD_TARGET='+TARGET,
               '-e', 'WHEEL_VERSION='+version, '-e', 'BUILDER_IMAGE_ID='+image,
               image, 'bash', '/recipe/ci/build.sh']
    save(root/'build-command.json', command)
    container(command, name, root/'build.log', 3600)
    output = root/'output'
    passport = json.loads((output/'build-passport.json').read_text())
    deps = output/'dependencies'; deps.mkdir()
    for filename in passport['runtime_dependency_wheels']:
        shutil.copy2(root/'work/dependencies'/filename, deps/filename)
    runtime = output/'pyasc_runtime-0.0.0-cp312-cp312-linux_x86_64.whl'
    assembled = bundle.assemble(output/passport['wheel'], list(deps.glob('*.whl')), passport, runtime)
    save(output/'bundle-manifest.json', dict(assembled, runtime_sha256=sha(runtime)))
    # Source export and receipt survive clean-up and CI artifact expiration.
    shutil.copy2(root/'work/source.tar', output/'source.tar')
    save(output/'build-result.json', {'status': 'built', 'seconds': time.monotonic()-started,
                                     'source_sha': revision, 'runtime_sha256': sha(runtime),
                                     'qualification': 'not-run', 'hardware': 'not-run'})
    index = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    save(output/'files.json', index)
    print(json.dumps({'status': 'built', 'runtime_sha256': sha(runtime), 'output': str(output)}))


def check_build(root):
    output = root/'output'
    intent = json.loads((root/'intent.json').read_text())
    for filename, checksum in intent['stack_files'].items():
        if not (ROOT/filename).resolve().is_relative_to(ROOT) or sha(ROOT/filename) != checksum:
            raise ValueError('Qualification code differs from build intent: '+filename)
    for filename, checksum in json.loads((output/'files.json').read_text()).items():
        if not (output/filename).resolve().is_relative_to(output.resolve()) or sha(output/filename) != checksum:
            raise ValueError('Build artifact changed: '+filename)
    manifest = json.loads((output/'bundle-manifest.json').read_text())
    if manifest['source_sha'] != intent['source_sha'] or manifest['recipe_commit'] != intent['recipe_commit']:
        raise ValueError('Bundle source/recipe differs from build intent')
    return output


def qualify(root):
    native()
    output = check_build(root)
    destination = root/'qualification'; destination.mkdir(exist_ok=False)
    # Exercise the ordinary submission packager without live task GET or POST.
    import smoke
    candidate = destination/'candidate'; candidate.mkdir()
    shutil.copy2(ROOT/'tests/fixtures/source-runtime/runtime_probe.py', candidate/'runtime_probe.py')
    (candidate/'_pyasc_runtime.py').write_text('# No hardware dispatch in this fixed simulator fixture.\n')
    runtime = output/'pyasc_runtime-0.0.0-cp312-cp312-linux_x86_64.whl'
    wheel_hash = smoke.make_bundle(runtime, sha(runtime), candidate, '2.0.0',
                                  destination/'submission.zip', function='runtime_probe')
    wheel_name = 'cann_bench-2.0.0-cp312-cp312-linux_x86_64.whl'
    with zipfile.ZipFile(destination/'submission.zip') as archive:
        (destination/wheel_name).write_bytes(archive.read('vendor/'+wheel_name))
    save(destination/'package.json', {'wheel': wheel_name, 'wheel_sha256': wheel_hash,
         'candidate_sha256': sha(candidate/'runtime_probe.py'), 'submission_sha256': sha(destination/'submission.zip')})
    run(['docker', 'pull', ENVIRONMENT], destination/'pull.log', 1200)
    name = 'pyasc-runtime-'+uuid.uuid4().hex
    command = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
               '--ulimit', 'nofile=65536:65536', '-v', str(ROOT)+':/stack:ro',
               '-v', str(output)+':/input:ro', '-v', str(destination)+':/result',
               '-w', '/result', '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               ENVIRONMENT, 'bash', '-c',
               'set -euo pipefail\n'
               'python3.12 -m venv /tmp/runtime-venv\n'
               '/tmp/runtime-venv/bin/python -m pip install --no-index --no-deps /result/cann_bench-*.whl\n'
               'for mode in compile model control; do\n'
               ' /tmp/runtime-venv/bin/python -I /stack/tests/integration/check-source-runtime.py "$mode"\n'
               'done']
    save(destination/'command.json', command)
    started = time.monotonic()
    container(command, name, destination/'qualification.log', 600)
    results = {mode: json.loads((destination/(mode+'.json')).read_text())
               for mode in ('compile', 'model', 'control')}
    if [results[x]['status'] for x in results] != ['passed', 'passed', 'rejected-control']:
        raise ValueError('Incomplete runtime qualification')
    manifest = json.loads((output/'bundle-manifest.json').read_text())
    save(destination/'result.json', {'status': 'passed', 'seconds': time.monotonic()-started,
         'environment': ENVIRONMENT, 'source_sha': manifest['source_sha'],
         'runtime_sha256': manifest['runtime_sha256'], 'checks': results,
         'scope': 'fixed add-f32 128, compile/Model/negative-control; no NPU or performance claim'})
    print(json.dumps({'status': 'passed', 'qualification': str(destination)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'qualify'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--source', default=SOURCE)
    args = parser.parse_args()
    root = external(args.root)
    if args.action == 'build':
        build(root, args.source)
    else:
        qualify(root)


if __name__ == '__main__':
    main()
