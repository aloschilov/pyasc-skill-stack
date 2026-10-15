"""Draft read-only active-device SoC binding; no compiler/platform overrides.

ACL returns a USER device ID. Map it to the DRIVER logical ID before querying
halGetSocVersion, as the public CANN runtime does for its current Device::Id_().
Do not pass the user ID directly or map to a physical chip ID instead.
Availability on the evaluator still requires qualification. Missing APIs fail
closed; no cached/global/first-device SoC fallback is permitted here.
"""
import ctypes
import re

SOC = re.compile(r'[A-Za-z0-9_]{1,96}\Z', re.ASCII)


class Queries:
    def __init__(self, acl=None, driver=None):
        self.acl = acl if acl is not None else ctypes.CDLL('libascendcl.so')
        self.driver = driver  # Load only after ACL device/mapping checks succeed.

    def _integer(self, name, args):
        fn = getattr(self.acl, name)
        fn.argtypes = [ctypes.c_int32] * len(args) + [ctypes.POINTER(ctypes.c_int32)]
        fn.restype = ctypes.c_int32
        value = ctypes.c_int32(-1)
        code = int(fn(*args, ctypes.byref(value)))
        return code, int(value.value)

    def current(self):
        return self._integer('aclrtGetDevice', [])

    def logical(self, user_device):
        return self._integer('aclrtGetLogicDevIdByUserDevId', [user_device])

    def soc(self, driver_device):
        if self.driver is None:
            self.driver = ctypes.CDLL('libascend_hal.so')
        fn = self.driver.halGetSocVersion
        fn.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
        fn.restype = ctypes.c_int32
        buffer = ctypes.create_string_buffer(128)
        code = int(fn(driver_device, buffer, len(buffer)))
        raw = buffer.raw
        if code != 0:
            return code, None
        if b'\0' not in raw:
            return code, None
        try:
            name = raw.split(b'\0', 1)[0].decode('ascii')
        except UnicodeError:
            return code, None
        return code, name if SOC.fullmatch(name) else None


def read_active_soc(expected_device, *, queries=None):
    """Fresh getters before/after binding. Does not promise concurrent isolation.

    The caller must hold its launch-context lock and recheck context immediately
    before launching. This observation is not a mutable resource-limit query.
    """
    result = dict(bound=False, user_device=None, driver_device=None, soc=None,
                  source='halGetSocVersion(mapped_current_user_device)', observations=[])
    def read(api, fn):
        code, value = fn()
        result['observations'].append(dict(api=api, status=code, raw=value))
        if type(code) is not int or code != 0:
            raise ValueError('query_failed')
        return value
    def device(value):
        if type(value) is not int or not 0 <= value <= 2**31-1:
            raise ValueError('invalid_device_id')
        return value
    try:
        device(expected_device)
        q = queries if queries is not None else Queries()
        user = device(read('aclrtGetDevice', q.current))
        result['user_device'] = user
        if user != expected_device:
            raise ValueError('tensor_device_mismatch')
        logical = device(read('aclrtGetLogicDevIdByUserDevId', lambda: q.logical(user)))
        result['driver_device'] = logical
        soc = read('halGetSocVersion', lambda: q.soc(logical))
        if type(soc) is not str or not SOC.fullmatch(soc):
            raise ValueError('invalid_soc')
        after = device(read('aclrtGetDevice', q.current))
        if after != user:
            raise ValueError('active_device_changed')
        mapped_after = device(read('aclrtGetLogicDevIdByUserDevId', lambda: q.logical(after)))
        if mapped_after != logical:
            raise ValueError('device_mapping_changed')
        result.update(bound=True, soc=soc)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        # Do not include arbitrary dlopen/path/error strings in remote telemetry.
        result['error'] = type(exc).__name__
        known = {'query_failed', 'invalid_device_id', 'tensor_device_mismatch',
                 'invalid_soc', 'active_device_changed', 'device_mapping_changed'}
        result['reason'] = str(exc) if type(exc) is ValueError and str(exc) in known else 'query_unavailable'
    return result
