"""Repair only the in-build integrity boundary; preserve all compute bytes."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    original = json.loads((ROOT/'package.json').read_text())
    assert sha((ROOT/'diagnostic.zip').read_bytes()) == original['archive_sha256']
    out = ROOT/'repair'
    out.mkdir()
    bundle = out/'bundle'
    with zipfile.ZipFile(ROOT/'diagnostic.zip') as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    old_manifest = members['source.sha256.txt']
    lines = old_manifest.decode().splitlines(keepends=True)
    removed = [line for line in lines if line.split(maxsplit=1)[1].strip() == 'build.sh']
    assert len(removed) == 1
    members['source.sha256.txt'] = ''.join(line for line in lines if line not in removed).encode()
    for name, content in members.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path = bundle/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    checks = []
    with tempfile.TemporaryDirectory(prefix='gelu-entry-check-') as temporary:
        work = Path(temporary)
        for name, content in members.items():
            target = work/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        def check(label, expected):
            result = subprocess.run(['sha256sum', '--check', 'source.sha256.txt'],
                                    cwd=work, capture_output=True, text=True)
            assert (result.returncode == 0) is expected, (label, result.stdout, result.stderr)
            checks.append(dict(label=label, expected_success=expected, returncode=result.returncode,
                               output=result.stdout, stderr=result.stderr))

        check('unchanged-repair', True)
        (work/'build.sh').write_bytes(b'# synthetic staging comment, not a recovered runner transform\n'+members['build.sh'])
        (work/'source.sha256.txt').write_bytes(old_manifest)
        check('old-manifest-rejects-entrypoint-edit', False)
        (work/'source.sha256.txt').write_bytes(members['source.sha256.txt'])
        check('repair-allows-entrypoint-edit', True)
        for name in ('cann_bench/kernel.py', 'cann_bench/_device_resources.py',
                     'metadata/build_payload.sh', 'wheel/cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl'):
            (work/name).write_bytes(members[name]+b'\n')
            check('rejects-payload-edit:'+name, False)
            (work/name).write_bytes(members[name])
    with zipfile.ZipFile(out/'diagnostic.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)
    record = dict(original, archive_sha256=sha((out/'diagnostic.zip').read_bytes()),
                  original_archive_sha256=original['archive_sha256'],
                  changed_members=['source.sha256.txt'],
                  members={name: sha(content) for name, content in members.items()})
    (out/'package.json').write_text(json.dumps(record, indent=2))
    (out/'entrypoint-tests.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps({k:v for k,v in record.items() if k != 'members'}, indent=2))


if __name__ == '__main__':
    main()
