"""Assemble an offline self-contained evaluator wheel, with fresh RECORD hashes."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = 'cann_bench-1.1.0-cp312-cp312-linux_x86_64.whl'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    wheels = list((ROOT/'x86-wheels').glob('*.whl'))
    assert len(wheels) == 1
    assert (ROOT/'bundle/cann_bench/kernel.py').read_bytes() == (ROOT/'kernel.py').read_bytes()
    members = {}
    with zipfile.ZipFile(wheels[0]) as source:
        for name in source.namelist():
            if name.startswith('asc/') and not name.endswith('/'):
                members[name] = source.read(name)
    assert any('asc/experimental/asctile/' in name for name in members)
    assert any(name.endswith('.so') for name in members)
    for base, prefix in ((ROOT/'bundle/cann_bench', 'cann_bench'),
                         (ROOT/'x86-build/venv/lib/python3.12/site-packages/pybind11', 'pybind11')):
        for path in base.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                members[prefix+'/'+str(path.relative_to(base))] = path.read_bytes()
    dist = 'cann_bench-1.1.0.dist-info/'
    members[dist+'METADATA'] = b'Metadata-Version: 2.1\nName: cann_bench\nVersion: 1.1.0\nSummary: Reference-DAG GeLU, pyasc v2 9069108e\nRequires-Python: >=3.12\n'
    members[dist+'WHEEL'] = b'Wheel-Version: 1.0\nGenerator: isolated-pyasc-reference-experiment\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n'
    members[dist+'LICENSE-pyasc'] = (ROOT/'x86-source/LICENSE').read_bytes()
    members[dist+'LICENSE-pybind11'] = (ROOT/'x86-build/venv/lib/python3.12/site-packages/pybind11-2.13.6.dist-info/LICENSE').read_bytes()
    members[dist+'LICENSE-LLVM'] = Path('/home/aloschilov/workspace/llvm-source/llvm/LICENSE.TXT').read_bytes()
    members[dist+'LICENSE-MLIR'] = Path('/home/aloschilov/workspace/llvm-source/mlir/LICENSE.TXT').read_bytes()
    members[dist+'REFERENCE-NOTICE'] = b'Reference mathematical DAG: Huawei ops-nn, activation/gelu_v2/op_kernel/arch35/gelu_v2_dag.h, revision f7a6b3e2c4f4c1bb4a2890dfb1e589170f21b307. CANN Open Software License Agreement Version 2.0. High-level AscTile reimplementation; not the original compiled reference.\n'
    metadata = dict(pyasc_pin='9069108e323746187c48d78fec4929c4afa2efc4',
                    upstream_wheel_sha256=sha(wheels[0].read_bytes()), kernel_sha256=sha((ROOT/'kernel.py').read_bytes()),
                    experiment='reference-DAG diagnostic; FP32 negative-tail stress fails locally')
    members['cann_bench/provenance.json'] = json.dumps(metadata, indent=2).encode()
    record = io.StringIO(newline='')
    writer = csv.writer(record, lineterminator='\n')
    for name, data in sorted(members.items()):
        writer.writerow((name, 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='), len(data)))
    writer.writerow((dist+'RECORD', '', ''))
    members[dist+'RECORD'] = record.getvalue().encode()
    wheel = ROOT/'bundle/wheel'/NAME
    with wheel.open('xb') as handle, zipfile.ZipFile(handle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    files = sorted(p for p in (ROOT/'bundle').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest = {str(p.relative_to(ROOT/'bundle')): sha(p.read_bytes()) for p in files}
    (ROOT/'bundle/source.sha256.txt').write_text(''.join(f'{h}  {name}\n' for name, h in manifest.items()))
    with (ROOT/'diagnostic.zip').open('xb') as handle, zipfile.ZipFile(handle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in (ROOT/'bundle').rglob('*') if p.is_file() and '__pycache__' not in p.parts):
            archive.write(path, str(path.relative_to(ROOT/'bundle')))
    metadata.update(archive_sha256=sha((ROOT/'diagnostic.zip').read_bytes()), wheel_sha256=sha(wheel.read_bytes()), members=manifest)
    (ROOT/'package.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != 'members'}))


if __name__ == '__main__':
    main()
