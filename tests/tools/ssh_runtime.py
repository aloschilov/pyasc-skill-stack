"""Bounded SSH delegation with source-only staging and no shared-tree sync/delete."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import uuid

TOOLS = Path(__file__).resolve().parent
PAYLOAD_TOOLS = ('run_and_verify.py', 'execute_kernel.py', 'pytest_verify_kernel.py',
                 'numerical_checks.py', 'device_identity.py', 'ssh_worker.py')
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
TRANSFER_TIMEOUT = 120


def command_timeout(timeout):
    # Create/probe + transfer + execution (including identity) + fetch + overhead.
    return 2 * timeout + 120 + 60 + TRANSFER_TIMEOUT + 90


def stage(directory, kernel, project_dir=None):
    kernel = kernel.resolve()
    project = Path(project_dir).resolve() if project_dir else kernel.parent
    relative = kernel.relative_to(project)
    if any(part.startswith('.') for part in relative.parts):
        raise ValueError('Hidden kernel paths are not transferred')
    names = list(project.rglob('*.py')) if project_dir else list(project.glob('*.py'))
    for source in names:
        rel = source.relative_to(project)
        if any(part.startswith('.') or part == '__pycache__' for part in rel.parts):
            continue
        if source.is_symlink() or any((project/p).is_symlink() for p in rel.parents):
            raise ValueError(f'Symlinks are not allowed in SSH source payload: {rel}')
        dest = directory/'project'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    for name in PAYLOAD_TOOLS:
        dest = directory/'tests/tools'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TOOLS/name, dest)
    if not (directory/'project'/relative).is_file():
        raise ValueError('SSH kernel must be a Python source file inside project-dir')
    files = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(directory.rglob('*.py'))}
    return 'project/'+str(relative), files


def run(args):
    target = args.ssh_target
    base = args.ssh_remote_dir
    report = {'file':args.kernel_py, 'status':'SKIP', 'verification_level':None,
              'mode':args.mode+' (ssh)', 'detail':'', 'evidence':{'ssh':{}}}
    evidence = report['evidence']['ssh']
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]*', target or ''):
        report['detail'] = 'Set PYASC_SSH_TARGET or --ssh-target to a valid SSH alias'
        return 2, report
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', base) or '..' in Path(base).parts:
        report.update(status='FAIL', detail='SSH remote directory must be an absolute path without spaces or ..')
        return 1, report
    if args.backend != 'NPU':
        report.update(status='FAIL', detail='SSH hardware lane requires --backend NPU')
        return 1, report
    remote = base.rstrip('/')+'/run-'+uuid.uuid4().hex
    ssh = ['ssh', *SSH_OPTIONS, target]
    evidence.update(target=target, remote_dir=remote, stages=[])
    phase = 'prepare'
    def call(command, timeout, data=None):
        result = subprocess.run(command, input=data, capture_output=True, timeout=timeout)
        evidence['stages'].append({'phase':phase, 'exit_code':result.returncode,
                                   'log':result.stderr.decode(errors='replace')})
        return result
    try:
        with tempfile.TemporaryDirectory(prefix='pyasc-ssh-') as tmp:
            directory = Path(tmp)
            kernel, files = stage(directory, Path(args.kernel_py).resolve(), args.project_dir)
            request = {'kernel':kernel, 'sha256':files, 'mode':args.mode, 'platform':args.platform,
                       'timeout':args.timeout, 'scalar_args':args.jit_scalar_args,
                       'no_prove_numerics':args.no_prove_numerics}
            (directory/'request.json').write_text(json.dumps(request))
            result = call([*ssh, f'mkdir -p {shlex.quote(base)} && mkdir {shlex.quote(remote)} && '
                           '(command -v rsync || true)'], 30)
            if result.returncode:
                report['detail'] = 'SSH unavailable while preparing run: '+result.stderr.decode(errors='replace')[-1000:]
                return 2, report
            phase = 'transfer'
            if shutil.which('rsync') and result.stdout.strip().endswith(b'rsync'):
                result = call(['rsync', '-az', '-e', shlex.join(['ssh', *SSH_OPTIONS]),
                               str(directory)+'/', target+':'+remote+'/'], TRANSFER_TIMEOUT)
                evidence['transport'] = 'rsync'
            else:
                # Fresh CANNLab images may lack rsync. The tar contains only
                # the allowlisted staged sources, never credentials or VCS data.
                archive = io.BytesIO()
                with tarfile.open(fileobj=archive, mode='w:gz') as tar:
                    for path in sorted(directory.rglob('*')):
                        if path.is_file(): tar.add(path, arcname=str(path.relative_to(directory)))
                result = call([*ssh, f'tar xzf - -C {shlex.quote(remote)}'], TRANSFER_TIMEOUT, archive.getvalue())
                evidence['transport'] = 'tar'
            if result.returncode:
                report['detail'] = 'SSH source transfer unavailable: '+result.stderr.decode(errors='replace')[-1000:]
                return 2, report
            phase = 'execute'
            command = shlex.join([args.ssh_python, remote+'/tests/tools/ssh_worker.py', remote])
            if args.ssh_env:
                command = f'. {shlex.quote(args.ssh_env)} && '+command
            result = call([*ssh, 'bash -c '+shlex.quote(command)], 2 * args.timeout + 180)
            if result.returncode in (255, 126, 127):
                report['detail'] = 'SSH runtime unavailable: '+result.stderr.decode(errors='replace')[-1000:]
                return 2, report
            remote_code = result.returncode
            phase = 'fetch-report'
            result = call([*ssh, 'cat '+shlex.quote(remote+'/report.json')], 30)
            if result.returncode == 255:
                report['detail'] = 'SSH disconnected while fetching result; execution is unverified'
                return 2, report
            inner = json.loads(result.stdout)
            if (result.returncode or remote_code not in (0,1,2) or not isinstance(inner,dict)
                    or inner.get('status') != {0:'PASS',1:'FAIL',2:'SKIP'}[remote_code]):
                raise ValueError('Remote report/exit mismatch')
            if remote_code == 0 and (inner.get('device_identity',{}).get('status') != 'ready'
                    or inner.get('verification_level') not in ('compiled','executed','numerically_verified')
                    or inner.get('evidence',{}).get('ssh',{}).get('payload_sha256') != files
                    or inner.get('evidence',{}).get('ssh',{}).get('inputs_unchanged') is not True):
                raise ValueError('Remote PASS lacks device identity or unchanged payload evidence')
            inner.setdefault('evidence',{}).setdefault('ssh',{}).update(evidence)
            return remote_code, inner
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        code = 1 if phase == 'execute' else 2
        report.update(status='FAIL' if code == 1 else 'SKIP', detail=f'SSH {phase}: {exc}')
        return code, report
    except (OSError, ValueError) as exc:
        report.update(status='FAIL', detail=f'SSH {phase}: {exc}')
        return 1, report
