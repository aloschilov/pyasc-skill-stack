"""Draft capacity gate for a NEW candidate; does not alter frozen B resources."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading

try:
    from .platform_capacity import MetadataQueries, read_capacity
    from .diagnostics_v2 import preflight_message
    from .loaded_sdk import verify_loaded_sdk
except ImportError:
    from platform_capacity import MetadataQueries, read_capacity
    from diagnostics_v2 import preflight_message
    from loaded_sdk import verify_loaded_sdk

_lock = threading.RLock()
_metadata = None  # Loaded library only, never mutable capacity/resource values.
_build_record = None
_reported = set()  # Diagnostic deduplication only.
INT32_MAX = 2**31 - 1


def validate_sdk_identity(record, launcher_sdk, environment_sdk):
    """Same explicit installed SDK; not proof of dynamic linker isolation."""
    if type(record) is not dict or not record.get('sdk') or not environment_sdk:
        raise ValueError('metadata_sdk_identity_missing')
    paths = [Path(value).resolve(strict=True) for value in
             (record['sdk'], launcher_sdk, environment_sdk)]
    if len(set(paths)) != 1 or not paths[0].is_dir():
        raise ValueError('metadata_sdk_identity_mismatch')


def load_metadata():
    global _metadata, _build_record
    from asc.lib.utils import get_ascend_path
    with _lock:
        # Recheck even for an already-loaded library: launcher paths may be
        # cached while ASCEND_HOME_PATH changes. Never silently cross SDKs.
        record = _build_record
        if record is None:
            record = json.loads((Path(__file__).parent/'build_result.json').read_text())
        validate_sdk_identity(record, get_ascend_path(), os.environ.get('ASCEND_HOME_PATH'))
        if _metadata is None:
            root = Path(__file__).parent
            # A C ABI library must not shadow platform_capacity.py as a Python
            # extension: CPython otherwise expects PyInit_platform_capacity.
            binary = root/'_platform_capacity_bridge.so'
            with binary.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if record.get('passed') is not True or record.get('binary_sha256') != actual:
                raise ValueError('metadata_library_identity_failed')
            library = ctypes.CDLL(str(binary))
            verify_loaded_sdk(record.get('sdk_libraries'))
            _metadata = MetadataQueries(library)
            _build_record = record
        return _metadata


def diagnostic(reason, route, resources, capacity=None, *, stream_limit=None):
    capacity = capacity or {}
    raw = resources.as_dict()
    raw['raw_simt_ub_bytes'] = raw.pop('ub_bytes', None)
    raw['stream_limit'] = stream_limit
    observation = capacity.get('metadata', {})
    if type(observation) is not dict:
        observation = {}
    summary = dict(ascendc_metadata_ub_bytes=capacity.get('allocation_capacity_bytes'),
                   active_device_binding_consistent=capacity.get('qualified') is True,
                   source=capacity.get('source'), soc=capacity.get('soc'),
                   bridge_status=observation.get('bridge_status'), cann_status=observation.get('cann_status'),
                   driver_device=capacity.get('before', {}).get('driver_device'))
    observations = list(resources.observations)
    for key in ('before', 'after'):
        observations.extend(capacity.get(key, {}).get('observations', []))
    return preflight_message(reason, route=route, resources=raw, capacity=summary, observations=observations)


def record_capacity(resources, route, capacity, *, stream_limit=None):
    message = diagnostic('ready', route, resources, capacity, stream_limit=stream_limit)
    with _lock:
        if message not in _reported:
            print(message, flush=True)
            _reported.add(message)


def require_default_device(resources, route):
    # Before ensure_npu_platform/current_device can initialize pinned default0.
    if type(resources.device) is not int or resources.device != 0:
        raise RuntimeError(diagnostic('unsupported_context', route, resources))


def validate_capacity(resources, route, *, runtime=None, metadata=None, soc_queries=None, stream_limit=None):
    """Fresh bound SIMD metadata; raw ACL/SIMT value is NEVER overwritten.

    Caller retains the launch_context guard and launches immediately after this
    check. Does not prove immunity to unrelated concurrent/ABA device changes.
    """
    def fail(reason, capacity=None):
        raise RuntimeError(diagnostic(reason, route, resources, capacity, stream_limit=stream_limit))
    require_default_device(resources, route)
    elements, tile, demand = (route.get(key) for key in ('elements', 'tile', 'compiled_ub_bytes'))
    if type(elements) is not int or type(tile) is not int or elements < 0 or tile < 1 or elements > INT32_MAX-tile+1:
        fail('invalid_launch_domain')
    if type(demand) is not int or not 0 < demand <= 253952:
        fail('unknown_compiled_demand')
    try:
        provider = metadata if metadata is not None else load_metadata()
        capacity = read_capacity(resources.device, metadata=provider, soc_queries=soc_queries)
    except (OSError, ValueError, TypeError, AttributeError):
        fail('unknown_capacity')
    if capacity.get('qualified') is not True:
        fail('unknown_capacity', capacity)
    if runtime is None:
        from asc.lib import runtime
    if runtime.current_device() != resources.device or runtime.current_platform() != capacity['soc']:
        fail('context_mismatch', capacity)
    if capacity['allocation_capacity_bytes'] < demand:
        fail('insufficient_capacity', capacity)
    return capacity
