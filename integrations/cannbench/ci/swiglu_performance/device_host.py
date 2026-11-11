"""Launch the SwiGlu AscendC objects on the current torch NPU stream."""
import ctypes as C
from functools import lru_cache
from pathlib import Path
import struct

import torch


class DeviceBinary(C.Structure):
    _fields_ = [("magic",C.c_uint32),("version",C.c_uint32),
                ("data",C.c_void_p),("length",C.c_uint64)]


@lru_cache(maxsize=1)
def runtime():
    library = C.CDLL("libruntime.so")
    return configure(library)


def configure(library):
    ptr, u32 = C.c_void_p, C.c_uint32
    functions = {
        "rtDevBinaryRegister":[C.POINTER(DeviceBinary), C.POINTER(ptr)],
        "rtFunctionRegister":[ptr, ptr, C.c_char_p, ptr, u32],
        "rtKernelLaunch":[ptr, u32, ptr, u32, ptr, ptr],
        "rtStreamSynchronize":[ptr],
        "rtDevBinaryUnRegister":[ptr],
    }
    for name, parameters in functions.items():
        fn = getattr(library,name)
        fn.argtypes,fn.restype = parameters,C.c_int
    return library


def device_ptr(tensor):
    return tensor.data_ptr()


def stream_ptr(tensor):
    return torch.npu.current_stream(tensor.device).npu_stream


def checked(fn,*args):
    status = fn(*args)
    if status:
        raise RuntimeError(f"{fn.__name__} failed for SwiGlu with status {status}")


def launch(code,name,blocks,packed,stream):
    lib = runtime()
    data = C.create_string_buffer(code)
    symbol = C.create_string_buffer(name.encode("ascii"))
    stub = C.cast(symbol,C.c_void_p)
    binary = DeviceBinary(0x41415246,0,C.addressof(data),len(code))
    handle = C.c_void_p()
    checked(lib.rtDevBinaryRegister,C.byref(binary),C.byref(handle))
    try:
        checked(lib.rtFunctionRegister,handle,stub,C.cast(stub,C.c_char_p),stub,0)
        args=C.create_string_buffer(packed)
        checked(lib.rtKernelLaunch,stub,blocks,C.addressof(args),len(packed),None,stream)
        checked(lib.rtStreamSynchronize,stream)
    finally:
        checked(lib.rtDevBinaryUnRegister,handle)



def core_count(device):
    # acl_rt.h, CANN 9: current device and stream thread resource limits.
    acl = C.CDLL('libascendcl.so')
    acl.aclrtGetDeviceInfo.argtypes = [C.c_uint32, C.c_int, C.POINTER(C.c_int64)]
    acl.aclrtGetDeviceInfo.restype = C.c_int
    physical = C.c_int64()
    checked(acl.aclrtGetDeviceInfo, device, 201, C.byref(physical))
    if physical.value <= 0:
        raise RuntimeError('Missing physical vector core count')
    limits = [physical.value]
    for fn, args in ((acl.aclrtGetDeviceResLimit,(device,1)), (acl.aclrtGetResInCurrentThread,(1,))):
        fn.argtypes = [C.c_int32] * len(args) + [C.POINTER(C.c_uint32)]
        fn.restype = C.c_int
        limit = C.c_uint32()
        checked(fn,*args,C.byref(limit))
        if not limit.value:
            raise RuntimeError('Missing vector core limit')
        limits.append(limit.value)
    return min(limits)


@lru_cache(maxsize=128)
def device_object(key):
    return (Path(__file__).parent/'device'/(key+'.o')).read_bytes()


@lru_cache(maxsize=64)
def gather_indices(half, tile, device):
    lane = torch.arange(tile * half, device=device, dtype=torch.int32)
    gate = lane // half * (2 * half) + lane % half
    return gate, gate + half


def swi_glu(input, dim=-1):
    from .geometry import key, layout
    dtype = str(input.dtype).removeprefix('torch.')
    if not input.is_contiguous():
        raise ValueError('Expected a contiguous input')
    shape, rows, half, route, const, groups = layout(input.shape, dim, dtype)
    out = torch.empty(shape, device=input.device, dtype=input.dtype)
    if not groups:
        return out
    pointers = [device_ptr(input), device_ptr(out)]
    if route == 'slab':
        pointers.extend(device_ptr(t) for t in gather_indices(*const, input.device))
    scalars = [rows] if route == 'slab' else [rows, groups if route == '_swiglu_pair_groups' else half]
    packed = struct.pack('<' + 'Q'*len(pointers) + 'i'*len(scalars), *pointers, *scalars)
    packed = packed.ljust((len(packed)+7)//8*8, bytes(1))
    launch(device_object(key(dtype, route, const)), route, min(groups, core_count(input.device.index)),
           packed, stream_ptr(input))
    return out
