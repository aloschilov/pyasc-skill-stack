import ctypes
from asc.lib import runtime
from asc.experimental import asctile
def vector_blocks(work_items):
    """Host-side grid for independent AIV work on the configured backend."""
    if work_items <= 0:
        return 0
    device = runtime.current_device()
    available = runtime.device_info(runtime.DeviceModuleType.RT_MODULE_TYPE_VECTOR_CORE,
                                    runtime.DeviceInfoType.INFO_TYPE_CORE_NUM, device)
    if available <= 0:
        raise RuntimeError(f"Invalid vector core count {available} on device {device}")
    if not runtime.is_model():
        # CANN 9 acl_rt.h: ACL_RT_DEV_RES_VECTOR_CORE = 1, uint32_t output.
        acl = ctypes.CDLL("libascendcl.so")
        for query, args in ((acl.aclrtGetDeviceResLimit, (device, 1)),
                            (acl.aclrtGetResInCurrentThread, (1,))):
            query.argtypes = [ctypes.c_int32] * len(args) + [ctypes.POINTER(ctypes.c_uint32)]
            query.restype = ctypes.c_int32
            limit = ctypes.c_uint32()
            status = query(*args, ctypes.byref(limit))
            if status or not limit.value:
                raise RuntimeError(f"{query.__name__} returned {status}, limit {limit.value}")
            available = min(available, limit.value)
    return min(available, work_items)
def ensure_npu_platform():
    asctile.set_platform(asctile.Backend.NPU, runtime.current_platform())
