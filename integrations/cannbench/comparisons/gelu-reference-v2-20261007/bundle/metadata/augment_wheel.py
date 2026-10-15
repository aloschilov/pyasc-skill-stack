"""Add evaluator-built own metadata bridge to a source-qualified wheel.

No SDK files are transported by this step. It does not qualify source, kernels,
live-device binding or submissions. Caller must independently gate the output.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
from loaded_sdk import SDK_OBJECTS

PACKAGE = 'cann_bench/_gelu_metadata/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def augment(source, built, destination):
    source, built, destination = map(Path, (source,built,destination))
    result = json.loads((built/'RESULTS.json').read_text())
    start = json.loads((built/'START.json').read_text())
    binary = (built/'platform_capacity.so').read_bytes()
    if (result.get('passed') is not True or result.get('inputs_unchanged') is not True
            or result.get('binary_sha256') != sha(binary)):
        raise ValueError('build_identity_failed')
    sdk = Path(start['sdk']).resolve(strict=True)
    if not sdk.is_dir():
        raise ValueError('installed_sdk_missing')
    # Recheck all builder inputs, including SDK files and own C++ source. This
    # runs during build, not per inference. Do not publish the full SDK tree.
    inputs = start.get('inputs')
    if type(inputs) is not dict or not inputs:
        raise ValueError('build_inputs_missing')
    for name, digest in inputs.items():
        if sha(Path(name).read_bytes()) != digest:
            raise ValueError('build_input_changed')
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate_wheel_members')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('unsafe_wheel_member')
        records = [name for name in names if name.endswith('.dist-info/RECORD')]
        if len(records) != 1 or any(name.endswith(('/RECORD.jws','/RECORD.p7s')) for name in names):
            raise ValueError('unsupported_wheel_record')
        record_name = records[0]
        members = {name:archive.read(name) for name in names if not name.endswith('/') and name != record_name}
    for name in ('__init__.py','gate.py','platform_capacity.py','active_soc.py','diagnostics_v2.py','loaded_sdk.py'):
        if PACKAGE+name not in members:
            raise ValueError('source_wheel_helper_missing')
    added = (PACKAGE+'_platform_capacity_bridge.so', PACKAGE+'build_result.json')
    if any(name in members for name in added):
        raise ValueError('wheel_already_augmented')
    sdk_libraries = {}
    for name in SDK_OBJECTS:
        path = sdk/'lib64'/name
        if str(path) not in inputs:
            raise ValueError('sdk_library_not_in_build_inputs')
        sdk_libraries[name] = dict(path=str(path),sha256=inputs[str(path)])
    manifest = dict(result, sdk=str(sdk), sdk_libraries=sdk_libraries, source_wheel_sha256=sha(source.read_bytes()),
        build_start_sha256=sha((built/'START.json').read_bytes()),
        build_result_sha256=sha((built/'RESULTS.json').read_bytes()),
        scope='Evaluator-built metadata bridge; not kernel or live-device qualification')
    members[added[0]] = binary
    members[added[1]] = (json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    record = io.StringIO(newline='')
    writer = csv.writer(record,lineterminator='\n')
    for name,data in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        writer.writerow((name,'sha256='+digest,len(data)))
    writer.writerow((record_name,'',''))
    members[record_name] = record.getvalue().encode()
    # Exclusive output preserves a failed/previous build; never mutate source.
    with destination.open('xb') as stream, zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(members.items()):
            archive.writestr(name,data)
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(destination) as after:
        for name in before.namelist():
            if name != record_name and not name.endswith('/') and before.read(name) != after.read(name):
                raise ValueError('original_wheel_payload_changed')
    return dict(wheel_sha256=sha(destination.read_bytes()), binary_sha256=sha(binary),
                original_members_preserved=len(members)-3, submission_qualified=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--built',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(augment(args.source,args.built,args.destination)))
