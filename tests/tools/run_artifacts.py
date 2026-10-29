"""External, append-only run directories. Never archives environment/auth files."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_hashes():
    files = []
    for directory in ('tests', 'skills', 'reference/images', 'integrations/opencode/profiles'):
        files.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.sh', '.md', '.json')
                     and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


class RunArtifacts:
    def __init__(self, directory, kind):
        self.path = Path(directory).expanduser().resolve()
        if self.path == ROOT or ROOT in self.path.parents:
            raise ValueError('Results must be outside the checkout')
        self.path.mkdir(parents=True, exist_ok=False)
        rev = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True)
        self.data = {'schema_version': 1, 'kind': kind,
                     'revision': os.environ.get('PYASC_SOURCE_REVISION') or rev.stdout.strip() or None,
                     'input_sha256': input_hashes(), 'image': None,
                     'started_unix': time.time(), 'commands': [], 'status': 'running'}
        self.save()

    def save(self):
        (self.path / 'manifest.json').write_text(json.dumps(self.data, indent=2)+'\n')

    def command(self, label, argv, timeout=900, env=None, cwd=ROOT):
        start = time.time()
        try:
            result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                                    text=True, timeout=timeout)
            code, out, err = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            code = 124
            out = exc.stdout or ''; err = exc.stderr or ''
            if isinstance(out, bytes): out = out.decode(errors='replace')
            if isinstance(err, bytes): err = err.decode(errors='replace')
            err += '\nouter command timed out'
        # Credentials are never part of a manifest. Redact known provider values
        # in tool output too; generated config files are never archived.
        for key in ('OPENCODE_API_KEY', 'OPENCODE_BASE_URL'):
            secret = (env or os.environ).get(key)
            if secret:
                out = out.replace(secret, '<redacted>'); err = err.replace(secret, '<redacted>')
        (self.path / (label+'.stdout')).write_text(out)
        (self.path / (label+'.stderr')).write_text(err)
        record = {'label': label, 'argv': list(map(str, argv)), 'cwd': str(cwd),
                  'exit_code': code, 'elapsed_s': time.time()-start}
        try:
            report = json.loads(out)
            (self.path / (label+'.json')).write_text(json.dumps(report, indent=2)+'\n')
        except ValueError:
            report = {}
        record['argv_input_sha256'] = {str(v): sha(v) for v in argv if Path(str(v)).is_file()}
        record['stdout_sha256'] = sha(self.path / (label+'.stdout'))
        record['stderr_sha256'] = sha(self.path / (label+'.stderr'))
        self.data['commands'].append(record); self.save()
        return code, report

    def finish(self, status):
        self.data.update(status=status, elapsed_s=time.time()-self.data['started_unix'])
        self.data['inputs_unchanged'] = self.data['input_sha256'] == input_hashes()
        if not self.data['inputs_unchanged']:
            self.data['status'] = 'failed-inputs-changed'
        self.save()
        if not self.data['inputs_unchanged']:
            raise RuntimeError('Qualification inputs changed during the run')
