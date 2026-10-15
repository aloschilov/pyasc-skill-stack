"""Host launcher for reference-DAG GeLU at pyasc v2 9069108e.

The finite UB inventory is specific to kernel.py, unroll=1, reuse_alloc=2,
vf_fusion=True and the C310 target. All six routes promote compute to FP32;
their measured allocations happen to match, not because dtype is ignored.
"""
import torch

from .kernel import gelu_kernel
from ._device_resources import query_resources
from ._gelu_metadata.gate import require_default_device, validate_capacity, record_capacity
from ._pyasc_runtime import ensure_npu_platform
from ._runtime_context import launch_context, record_launch


def gelu(x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    if approximate not in ('none', 'tanh'):
        raise ValueError('Unsupported approximation')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Unsupported dtype')
    if x.device.type != 'npu' or not x.is_contiguous():
        raise ValueError('Requires a contiguous NPU tensor')
    if not x.numel():
        return torch.empty_like(x)
    dtype = str(x.dtype).removeprefix('torch.')
    tile, ub = 15872, 190464
    route = dict(elements=x.numel(), tile=tile, unroll=1, dtype=dtype, mode=approximate,
                 compiled_ub_bytes=ub, reuse_alloc=2, static_alloc=None, vf_fusion=True,
                 insert_sync=True, opt_level=3)
    resources = query_resources(x.device.index)
    require_default_device(resources, route)
    ensure_npu_platform()
    y = torch.empty_like(x)
    resources = query_resources(x.device.index)
    with launch_context(resources) as context:
        capacity = validate_capacity(context.resources, route, stream_limit=context.stream_limit)
        tiles = (x.numel() + tile - 1) // tile
        cores = min(context.resources.launch_cap, tiles)
        record_launch(context, elements=x.numel(), tile=tile, unroll=1, cores=cores,
                      dtype=dtype, mode=approximate, vf=True, ub=ub, reuse=2)
        record_capacity(context.resources, route, capacity, stream_limit=context.stream_limit)
        gelu_kernel[cores, context.stream](x.reshape(-1), y.reshape(-1), x.numel(), tile,
                                         approximate == 'tanh',
                                         approximate == 'none' and x.dtype == torch.float32, 1)
    return y
