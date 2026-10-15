"""Linux first-load SDK object compatibility, not symbol-interposition proof."""
import hashlib
from pathlib import Path

SDK_OBJECTS = ('libplatform.so', 'libunified_dlog.so',
               'libascend_protobuf.so.3.13.0.0', 'libmmpa.so', 'libc_sec.so')


def verify_loaded_sdk(expected, *, maps=None):
    if type(expected) is not dict or set(expected) != set(SDK_OBJECTS):
        raise ValueError('sdk_library_manifest_missing')
    records = {}
    for name,row in expected.items():
        if type(row) is not dict or type(row.get('sha256')) is not str:
            raise ValueError('sdk_library_manifest_invalid')
        path = Path(row['path']).resolve(strict=True)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        if digest != row['sha256']:
            raise ValueError('sdk_library_bytes_changed')
        records[name] = path
    text = Path('/proc/self/maps').read_text() if maps is None else maps
    observed = {name:set() for name in SDK_OBJECTS}
    for line in text.splitlines():
        fields = line.split(None,5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        raw = fields[5]
        deleted = raw.endswith(' (deleted)')
        path = Path(raw.removesuffix(' (deleted)'))
        # Only relevant SDK DSOs; no general process/path logging.
        for name in SDK_OBJECTS:
            if path.name == name or path.name == records[name].name:
                if deleted:
                    raise ValueError('sdk_library_mapping_deleted')
                observed[name].add(path.resolve(strict=True))
    for name,path in records.items():
        if observed[name] != {path}:
            raise ValueError('sdk_library_mapping_mismatch')
    return {'checked_sdk_objects':len(records),
            'scope':'Mapped SDK object paths and disk hashes; not PLT interposition or mutable-page proof'}
