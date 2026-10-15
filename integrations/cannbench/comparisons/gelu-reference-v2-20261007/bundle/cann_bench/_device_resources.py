"""Read-only CANN 9 resource discovery; no SoC-name/core-count heuristics.

The queried count is a runtime report, NOT measured physical occupancy.
Unknown limits fail conservatively to one launch block. No values are cached:
resource limits may change per thread/device between operator calls.
"""
import ctypes
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class Resources:
    device: Optional[int]
    reported_aiv: Optional[int]
    device_limit: Optional[int]
    thread_limit: Optional[int]
    ub_bytes: Optional[int]
    launch_cap: int
    fallback: bool
    observations: tuple

    def as_dict(self):
        return asdict(self)


class CANNQueries:
    """ABI from CANN 9 acl/acl_rt.h; load only the public runtime library."""
    def __init__(self, library=None):
        self.library = library if library is not None else ctypes.CDLL('libascendcl.so')

    def read(self, name, args, types, output_type):
        function = getattr(self.library, name)
        function.argtypes = [*types, ctypes.POINTER(output_type)]
        function.restype = ctypes.c_int32
        output = output_type()
        status = int(function(*args, ctypes.byref(output)))
        return status, int(output.value)

    def current_device(self):
        return self.read('aclrtGetDevice', [], [], ctypes.c_int32)

    def device_info(self, device, attribute):
        return self.read('aclrtGetDeviceInfo', [device, attribute],
                         [ctypes.c_uint32, ctypes.c_int], ctypes.c_int64)

    def device_resources(self, device):
        return self.read('aclrtGetDeviceResLimit', [device, 1],
                         [ctypes.c_int32, ctypes.c_int], ctypes.c_uint32)

    def thread_resources(self):
        return self.read('aclrtGetResInCurrentThread', [1],
                         [ctypes.c_int], ctypes.c_uint32)


def query_resources(expected_device=None, *, queries=None):
    """Query current device; never select/change a device as a side effect.

Zero resource limits have no established meaning in this integration and are
therefore unknown, not inferred unlimited. The observations retain raw values.
Tests inject a query provider, not a compiler/JIT monkeypatch.
"""
    observations = []
    def read(label, function, *, nonnegative=False):
        try:
            code, value = function()
            valid = code == 0 and type(value) is int and (value >= 0 if nonnegative else value > 0)
            observations.append(dict(api=label, status=code, raw=value, valid=valid))
            return value if valid else None
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            observations.append(dict(api=label, error=type(exc).__name__, detail=str(exc)))
            return None
    try:
        q = queries if queries is not None else CANNQueries()
    except OSError as exc:
        return Resources(None, None, None, None, None, 1, True,
                         (dict(api='load:libascendcl.so', error=str(exc)),))
    device = read('aclrtGetDevice', q.current_device, nonnegative=True)
    if device is None or (expected_device is not None and device != expected_device):
        if device is not None:
            raise RuntimeError(f'active NPU device {device} differs from tensor device {expected_device}')
        return Resources(device, None, None, None, None, 1, True, tuple(observations))
    reported = read('aclrtGetDeviceInfo:VECTOR_CORE_NUM', lambda: q.device_info(device, 201))
    ub = read('aclrtGetDeviceInfo:UBUF_PER_VECTOR_CORE', lambda: q.device_info(device, 204))
    device_limit = read('aclrtGetDeviceResLimit:VECTOR_CORE', lambda: q.device_resources(device))
    thread_limit = read('aclrtGetResInCurrentThread:VECTOR_CORE', q.thread_resources)
    counts = (reported, device_limit, thread_limit)
    fallback = any(value is None for value in counts)
    cap = 1 if fallback else min(counts)
    return Resources(device, reported, device_limit, thread_limit, ub, cap, fallback, tuple(observations))


def launch_blocks(elements, tile, resources):
    if type(elements) is not int or elements < 0 or type(tile) is not int or tile <= 0:
        raise ValueError('elements must be nonnegative and tile positive integers')
    if resources.launch_cap < 1:
        raise ValueError('invalid launch cap')
    return min(resources.launch_cap, (elements + tile - 1) // tile)



