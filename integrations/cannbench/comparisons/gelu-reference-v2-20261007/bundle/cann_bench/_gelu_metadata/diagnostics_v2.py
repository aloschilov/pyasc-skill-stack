"""Draft bounded preflight telemetry; no querying, approval or launch decisions.

Frozen B is deliberately not wired to this module. A future repaired candidate
must bind this source hash and requalify its host/package gates. Error messages
are necessary because the evaluator did not preserve successful-path stdout.
"""
import json
import re

SOC = re.compile(r'[A-Za-z0-9_]{1,96}\Z', re.ASCII)

PREFIX = 'GELU_PREFLIGHT '
MAX_BYTES = 4096
REASONS = {'unknown_capacity', 'insufficient_capacity', 'unknown_compiled_demand',
           'context_mismatch', 'unsupported_context', 'invalid_launch_domain', 'ready'}
API_NAMES = {'aclrtGetDevice', 'aclrtGetDeviceInfo:VECTOR_CORE_NUM',
             'aclrtGetDeviceInfo:UBUF_PER_VECTOR_CORE',
             'aclrtGetDeviceResLimit:VECTOR_CORE',
             'aclrtGetResInCurrentThread:VECTOR_CORE',
             'aclrtGetStreamResLimit:VECTOR_CORE', 'load:libascendcl.so', 'aclrtGetLogicDevIdByUserDevId', 'halGetSocVersion'}
ERROR_NAMES = {'AttributeError', 'OSError', 'TypeError', 'ValueError'}


def integer(value):
    # Preserve negative raw statuses/values, but never serialize arbitrary ints
    # or object reprs. bool is not an integer observation.
    return value if type(value) is int and -(1 << 63) <= value < (1 << 64) else None


def preflight_message(reason, *, route, resources, observations=(), capacity=None):
    """Accept plain data snapshots only; redact all unrecognized fields.

    capacity is a distinct metadata observation, never a replacement for raw
    ACL/SIMT UB. Its active-device proof is not inferred here. This formatter
    grants no permission to launch even if both numbers happen to be positive.
    """
    route = route if type(route) is dict else {}
    resources = resources if type(resources) is dict else {}
    capacity = capacity if type(capacity) is dict else {}
    row = {'schema': 1, 'reason': reason if type(reason) is str and reason in REASONS else 'unspecified',
           'route': {key: integer(route.get(key)) for key in
                     ('elements', 'tile', 'unroll', 'compiled_ub_bytes', 'reuse_alloc', 'opt_level')},
           'resources': {key: integer(resources.get(key)) for key in
                         ('device', 'reported_aiv', 'device_limit', 'thread_limit',
                          'stream_limit', 'launch_cap', 'raw_simt_ub_bytes')},
           'capacity': {'ascendc_metadata_ub_bytes': integer(capacity.get('ascendc_metadata_ub_bytes')),
                        'active_device_binding_consistent': capacity.get('active_device_binding_consistent') is True},
           'observations': []}
    for key, choices in (('dtype', {'float16', 'bfloat16', 'float32'}), ('mode', {'none', 'tanh'})):
        value = route.get(key)
        row['route'][key] = value if type(value) is str and value in choices else None
    for key in ('static_alloc', 'vf_fusion', 'insert_sync'):
        value = route.get(key)
        row['route'][key] = value if type(value) is bool else None
    sources = {'installed PlatformAscendC.GetCoreMemSize', 'aclplatformGetDeviceInfo'}
    source = capacity.get('source')
    row['capacity']['source'] = source if type(source) is str and source in sources else None
    row['capacity']['scope'] = 'metadata_only_not_live_limit'
    for key in ('bridge_status', 'cann_status', 'driver_device'):
        row['capacity'][key] = integer(capacity.get(key))
    soc = capacity.get('soc')
    row['capacity']['soc'] = soc if type(soc) is str and SOC.fullmatch(soc) else None
    row['capacity']['binding_scope'] = 'fresh_observation_not_concurrent_isolation'
    items = observations if type(observations) in (list, tuple) else ()
    # Bound repeated before/after API observations as well as hostile
    # payloads and explicitly report omissions without echoing exception details.
    for item in items[:12]:
        if type(item) is not dict:
            continue
        api = item.get('api')
        if type(api) is not str or api not in API_NAMES:
            continue
        error = item.get('error')
        row['observations'].append({'api': api, 'status': integer(item.get('status')),
                                    'raw': (item.get('raw') if api == 'halGetSocVersion' and type(item.get('raw')) is str and SOC.fullmatch(item['raw']) else integer(item.get('raw'))),
                                    'valid': item.get('valid') if type(item.get('valid')) is bool else None,
                                    'error': error if type(error) is str and error in ERROR_NAMES else None})
    row['observations_omitted'] = len(items) - len(row['observations'])
    message = PREFIX + json.dumps(row, ensure_ascii=True, separators=(',', ':'), sort_keys=True)
    # With whitelisted strings and bounded rows the structured record fits;
    # never slice JSON into an unparseable prefix if the schema grows later.
    while len(message.encode('ascii')) > MAX_BYTES and row['observations']:
        row['observations'].pop()
        row['observations_omitted'] += 1
        message = PREFIX + json.dumps(row, ensure_ascii=True, separators=(',', ':'), sort_keys=True)
    if len(message.encode('ascii')) > MAX_BYTES:
        return PREFIX + '{"schema":1,"reason":"diagnostic_overflow"}'
    return message
