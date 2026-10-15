"""Draft host-only launch context for pinned v2; not yet wired/promoted.

Explicit numBlocks launches need the actual stream's resource limit. The
thread getter alone can describe a different stream bound by an aclnn caller.
No limit setter or thread-resource binding is used here. Public PyAsc stream
setup is ordinary launcher setup, not a read-only ACL query: current_device
may lazily select default device0, so other active devices are rejected BEFORE
calling it. This conservative integration does not claim multi-device support.
"""
from contextlib import contextmanager
import ctypes
import json
from dataclasses import dataclass, replace
import threading

_lock = threading.RLock()
_reported = set()  # Deduplicates diagnostics only; resource queries still repeat.


@dataclass(frozen=True)
class LaunchContext:
    resources: object
    stream: object
    stream_limit: object


@contextmanager
def launch_context(resources, *, runtime=None, queries=None):
    # Do not silently let pinned runtime initialize device0 for an active device1.
    if type(resources.device) is not int or resources.device != 0:
        raise RuntimeError('Pinned v2 context qualified only for active device0; no device switch performed')
    with _lock:
        if runtime is None:
            from asc.lib import runtime
        if queries is None:
            from ._device_resources import CANNQueries
            queries = CANNQueries()
        if runtime.is_model():
            raise RuntimeError('NPU launch context cannot use Model backend')
        if runtime.current_device() != resources.device:
            raise RuntimeError('Cached PyAsc device differs from active ACL device')
        if runtime.current_platform() != runtime.get_soc_version().value:
            raise RuntimeError('Cached PyAsc platform differs from runtime report')
        stream = runtime.current_stream()
        # This is exactly the stream to pass through kernel[cores, ctx.stream].
        if not isinstance(stream,ctypes.c_void_p) or stream.value is None:
            raise RuntimeError('PyAsc did not supply a concrete launch stream')
        status, device = queries.current_device()
        if status != 0 or type(device) is not int or device != resources.device:
            raise RuntimeError('Active device changed during context setup')
        try:
            code, value = queries.read('aclrtGetStreamResLimit',[stream,1],
                                      [ctypes.c_void_p,ctypes.c_int],ctypes.c_uint32)
            valid = code == 0 and type(value) is int and value > 0
            observation = dict(api='aclrtGetStreamResLimit:VECTOR_CORE',status=code,raw=value,valid=valid)
        except (AttributeError,OSError,TypeError,ValueError) as exc:
            valid=False;value=None
            observation=dict(api='aclrtGetStreamResLimit:VECTOR_CORE',error=type(exc).__name__,detail=str(exc))
        cap=min(resources.launch_cap,value) if valid else 1
        effective=replace(resources,launch_cap=cap,fallback=resources.fallback or not valid,
                          observations=resources.observations+(observation,))
        yield LaunchContext(effective,stream,value if valid else None)


def record_launch(context, *, elements, tile, unroll, cores, dtype, mode, vf, ub, reuse=1):
    """One host log per distinct configuration; never used for compute dispatch."""
    tiles=(elements+tile-1)//tile
    row=dict(resources=context.resources.as_dict(),stream_limit=context.stream_limit,
             elements=elements,tile_shape=[tile],unroll=unroll,dtype=dtype,mode=mode,
             launched=cores,useful=min(cores,tiles),tiles_per_block=[tiles//cores,(tiles+cores-1)//cores],
             compiled_ub_bytes=ub,requested_jit=dict(reuse_alloc=reuse,static_alloc=None,vf_fusion=vf,
                 insert_sync=True,opt_level=3,vf_vec_len=None,debug=False))
    message=json.dumps(row,sort_keys=True,separators=(',',':'))
    with _lock:
        if message not in _reported:
            print('GELU_LAUNCH_METADATA '+message,flush=True)
            _reported.add(message)
