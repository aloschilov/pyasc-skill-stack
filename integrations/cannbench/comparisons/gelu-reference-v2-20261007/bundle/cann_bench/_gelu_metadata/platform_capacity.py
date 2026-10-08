"""Draft join of fresh mapped-device SoC and installed AscendC metadata.

No capacity value cache, no mutation of resource limits, no submission approval.
Only the loaded metadata library may be retained by a caller. Its identity and
evaluator compatibility must be qualified before wiring this into a candidate.
"""
import ctypes

try:
    from .active_soc import SOC, read_active_soc
except ImportError:
    from active_soc import SOC, read_active_soc


class MetadataQueries:
    def __init__(self, library):
        self.library = library

    def query(self, soc):
        if type(soc) is not str or not SOC.fullmatch(soc):
            raise ValueError('invalid_soc')
        function = self.library.gelu_platform_ub
        function.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]
        function.restype = ctypes.c_int
        ub = ctypes.c_uint64(0)
        status = ctypes.c_uint32(0)
        code = int(function(soc.encode('ascii'), ctypes.byref(ub), ctypes.byref(status)))
        return dict(bridge_status=code, cann_status=int(status.value), ub_bytes=int(ub.value))


def read_capacity(expected_device, *, metadata, soc_queries=None):
    """Bind metadata between two fresh device/SoC observations.

    Covers observable changes during query, not arbitrary ABA/concurrent changes
    after return. Caller must retain its launch-context guard and final recheck.
    Capacity means AscendC platform allocation capacity, never ACL SIMT UB or a
    live mutable UB limit. Raw ACL/SIMT observations remain separately recorded.
    """
    result = dict(qualified=False, allocation_capacity_bytes=None, soc=None,
                  source='installed PlatformAscendC.GetCoreMemSize',
                  scope='active_soc_bound_metadata_not_live_resource_limit')
    first = read_active_soc(expected_device, queries=soc_queries)
    result['before'] = first
    if first['bound'] is not True:
        result['reason'] = 'active_soc_unavailable'
        return result
    try:
        observation = metadata.query(first['soc'])
        result['metadata'] = observation
        if type(observation) is not dict:
            raise ValueError('invalid_metadata_record')
        if any(type(observation.get(key)) is not int or observation[key] != 0
               for key in ('bridge_status', 'cann_status')):
            raise ValueError('metadata_query_failed')
        ub = observation.get('ub_bytes')
        if type(ub) is not int or not 0 < ub < (1 << 64):
            raise ValueError('invalid_metadata_capacity')
        last = read_active_soc(expected_device, queries=soc_queries)
        result['after'] = last
        if last['bound'] is not True or any(first[key] != last[key] for key in ('user_device', 'driver_device', 'soc')):
            raise ValueError('active_binding_changed')
        result.update(qualified=True, allocation_capacity_bytes=ub, soc=first['soc'])
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        result['error'] = type(exc).__name__
        known = {'invalid_metadata_record', 'metadata_query_failed', 'invalid_metadata_capacity', 'active_binding_changed'}
        result['reason'] = str(exc) if type(exc) is ValueError and str(exc) in known else 'metadata_unavailable'
    return result
