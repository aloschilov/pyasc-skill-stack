"""File-only assembly of a source-built CPython 3.12/x86_64 runtime bundle.

The upstream wheel builder owns compilation. This module verifies wheel RECORDs
and copies their payload without importing it or modifying runtime code.
"""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

NATIVE = 'asc/_C/libpyasc.cpython-312-x86_64-linux-gnu.so'
PREFIXES = ('asc/', 'pybind11/', 'numpy/', 'numpy.libs/')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload_name(name):
    return name.startswith(PREFIXES) or name == 'typing_extensions.py'


def read_wheel(path, expected):
    if digest(path.read_bytes()) != expected:
        raise ValueError('Wheel hash mismatch: ' + path.name)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel member')
        for item in archive.infolist():
            name = item.filename
            if (PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                    or '\\' in name or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Unsafe wheel member')
        files = {n: archive.read(n) for n in names if not n.endswith('/')}
    records = [n for n in files if n.endswith('.dist-info/RECORD')]
    if len(records) != 1:
        raise ValueError('Expected one wheel RECORD')
    rows = list(csv.reader(io.StringIO(files[records[0]].decode())))
    if len(rows) != len(files) or {r[0] for r in rows} != set(files):
        raise ValueError('Wheel RECORD inventory mismatch')
    for name, checksum, size in rows:
        if name == records[0]:
            if checksum or size:
                raise ValueError('Invalid RECORD self entry')
        else:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).decode().rstrip('=')
            if checksum != 'sha256=' + encoded or size != str(len(files[name])):
                raise ValueError('Wheel RECORD mismatch: ' + name)
    return files


def write_wheel(path, files, distribution='pyasc_runtime', version='0.0.0'):
    files = dict(files)
    meta = f'{distribution}-{version}.dist-info/'
    files[meta+'METADATA'] = (f'Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n'
                              'Requires-Python: >=3.12,<3.13\n\n').encode()
    files[meta+'WHEEL'] = (b'Wheel-Version: 1.0\nGenerator: pyasc-source-runtime\n'
                         b'Root-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n')
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='\n')
    for name, data in sorted(files.items()):
        checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        writer.writerow((name, 'sha256='+checksum, len(data)))
    writer.writerow((meta+'RECORD', '', ''))
    files[meta+'RECORD'] = stream.getvalue().encode()
    with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), data,
                             compress_type=zipfile.ZIP_DEFLATED)


def assemble(wheel, dependencies, passport, output):
    if passport['target'] != 'cp312-x86_64' or passport['llvm']['zstd_soname'] is not None:
        raise ValueError('Only the pinned cp312/x86_64 recipe is supported')
    expected = {passport['wheel']: passport['wheel_sha256'], **passport['runtime_dependency_wheels']}
    paths = {p.name: p for p in [wheel, *dependencies]}
    if set(paths) != set(expected) or len(paths) != 4:
        raise ValueError('Expected exact PyASC, numpy, pybind11 and typing_extensions wheels')
    payload = {}
    meta = 'pyasc_runtime-0.0.0.dist-info/'
    for filename, checksum in expected.items():
        files = read_wheel(paths[filename], checksum)
        for name, data in files.items():
            if payload_name(name):
                if name in payload:
                    raise ValueError('Conflicting runtime files: '+name)
                payload[name] = data
            elif '.dist-info/' in name:
                suffix = name.split('.dist-info/', 1)[1]
                if 'license' in suffix.lower() or 'copying' in suffix.lower():
                    package = filename.split('-')[0]
                    payload[meta+'licenses/'+package+'/'+suffix] = data
            else:
                raise ValueError('Unexpected runtime payload: '+name)
    required = (NATIVE, 'asc/__init__.py', 'asc/experimental/asctile/__init__.py',
                'pybind11/__init__.py', 'numpy/__init__.py', 'typing_extensions.py')
    if not all(n in payload for n in required) or any(n.startswith('asctile/') for n in payload):
        raise ValueError('Incomplete modern runtime payload')
    for package in ('pyasc_dev', 'numpy', 'pybind11', 'typing_extensions'):
        if not any(n.startswith(meta+'licenses/'+package+'/') for n in payload):
            raise ValueError('Missing license: '+package)
    manifest = {'schema_version': 1, 'source_sha': passport['source_sha'],
                'recipe_commit': passport['recipe_commit'], 'input_wheels': expected,
                'payload_sha256': {n: digest(b) for n, b in sorted(payload.items()) if payload_name(n)},
                'scope': 'assembled runtime; no compiler, numerical or NPU qualification implied'}
    payload[meta+'runtime-manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    write_wheel(output, payload)
    return manifest
