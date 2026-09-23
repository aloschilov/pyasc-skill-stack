"""High-level GeLU hybrid: public Erf lowp, user-authorized inline Erfc FP32."""
import asctile
import torch

from ._pyasc_runtime import ensure_npu_platform
from ._device_resources import query_resources
from ._gelu_metadata.gate import require_default_device, validate_capacity, record_capacity
from ._runtime_context import launch_context

# PR575 geometry-only hardware diagnostic; device bodies were tested on VM.
UNIFORM_REUSE = 2
# All20 geometries compiled; full numerical Model checks cover cases10 and15.
QUALIFIED_TILES = {('float16', 'none'): {4608: 73728}, ('float32', 'none'): {5440: 87040}, ('bfloat16', 'none'): {4608: 73728}, ('float16', 'tanh'): {5376: 64512}, ('float32', 'tanh'): {8192: 131072}, ('bfloat16', 'tanh'): {5376: 64512}}


# These are logical grid dimensions, not requests to allocate more physical AIVs.
# Exact PR575 partitioning is retained; all six routes use their literal ubFormer.
def choose_geometry(elements: int, available_blocks: int, ub_budget: int, dtype: str, mode: str):
    if elements < 1 or available_blocks < 1:
        raise ValueError('Positive work and known resources required')
    inventory = QUALIFIED_TILES[(dtype, mode)]
    tile = min(inventory)
    ub = inventory[tile]
    if ub > ub_budget:
        raise RuntimeError('TTK tile exceeds bound UB capacity')
    block_length = ((elements + 71) // 72 + 511) // 512 * 512
    logical_blocks = (elements + block_length - 1) // block_length
    return tile, logical_blocks, ub


def geometry(is_fp32: bool, approximate: str):
    if is_fp32:
        return (5440 if approximate == 'none' else 8192), 2, True
    return (4608 if approximate == 'none' else 5376), 2, True


def require_unreduced_resources(reported: int, device_limit: int, thread_limit: int,
                                stream_limit: int, launch_cap: int, fallback: bool):
    if fallback or type(reported) is not int or reported < 1:
        raise RuntimeError('TTK logical grid requires known full device resources')
    if device_limit != reported or thread_limit != reported or stream_limit != reported or launch_cap != reported:
        raise RuntimeError('TTK logical grid refuses reduced device/thread/stream resources')


_TTK_REPORTED = set()


def record_ttk_launch(size: int, dtype: str, mode: str, tile: int, blocks: int, ub: int,
                      reported: int, device_limit: int, thread_limit: int, stream_limit: int):
    key = (size, dtype, mode, reported)
    if key not in _TTK_REPORTED:
        print('GELU_TTK_GRID', dict(elements=size, dtype=dtype, mode=mode,
              core_num=72, block_length=((size+71)//72+511)//512*512,
              tile=tile, logical_blocks=blocks, compiled_ub_bytes=ub,
              reported_aiv=reported,
              device_limit=device_limit,
              thread_limit=thread_limit,
              stream_limit=stream_limit), flush=True)
        _TTK_REPORTED.add(key)


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def gelu_kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
                input_length, tile_length: asctile.ConstExpr,
                unroll_factor: asctile.ConstExpr, approximate: asctile.ConstExpr,
                is_fp32: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [input_length])
    out_gm = asctile.global_tensor(output_ptr, [input_length])
    block_length = asctile.ceildiv(asctile.ceildiv(input_length, 72), 512) * 512
    block_start = asctile.block_idx() * block_length
    block_end = min(block_start + block_length, input_length)
    local_tiles = asctile.ceildiv(max(block_end - block_start, 0), tile_length)
    # Explicit fixed-size inner group avoids a separate dynamic remainder loop.
    for base in range(0, local_tiles, unroll_factor):
        for lane in asctile.range(unroll_factor, unroll_factor=unroll_factor):
            i = base + lane
            if i < local_tiles:
                offset = block_start + i * tile_length
                valid = max(0, min(block_end - offset, tile_length))
                raw = asctile.copy_in(in_gm, [offset], [tile_length], real_shape=[valid], pad_value=0)
                x = raw.to(asctile.float32)
                if approximate:
                    square = x * x
                    cube = square * x
                    exponent = (x + cube * 0.044715) * -1.5957691216057308
                    out = x / (asctile.exp(exponent) + 1.0)
                elif is_fp32:
                    erf_arg = x * -0.7071067811865476
                    complement = asctile.inline_vf("""
                        {
                            AscendC::Erfc<float, false>($0, $1, $1.GetSize());
                        }
                    """, [tile_length], asctile.float32, [erf_arg])
                    out = (x * 0.5) * complement
                else:
                    out = (x * 0.5) * (asctile.erf(x * 0.7071067811865476) + 1.0)
                asctile.copy_out(out.to(raw.dtype), out_gm, [offset], real_shape=[valid])


@asctile.jit(reuse_alloc=2, vf_fusion=True)
def gelu_reference_kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
                input_length, tile_length: asctile.ConstExpr,
                unroll_factor: asctile.ConstExpr, approximate: asctile.ConstExpr,
                is_fp32: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [input_length])
    out_gm = asctile.global_tensor(output_ptr, [input_length])
    block_length = asctile.ceildiv(asctile.ceildiv(input_length, 72), 512) * 512
    block_start = asctile.block_idx() * block_length
    block_end = min(block_start + block_length, input_length)
    local_tiles = asctile.ceildiv(max(block_end - block_start, 0), tile_length)
    for i in asctile.range(local_tiles, unroll_factor=unroll_factor):
        offset = block_start + i * tile_length
        valid = max(0, min(block_end - offset, tile_length))
        raw = asctile.copy_in(in_gm, [offset], [tile_length], real_shape=[valid], pad_value=0)
        x = raw.to(asctile.float32)
        if approximate:
            square = x * x
            cube = square * x
            exponent = (x + cube * 0.044715) * -1.5957691216057308
            out = x / (asctile.exp(exponent) + 1.0)
        elif is_fp32:
            erf_arg = x * -0.7071067811865476
            complement = asctile.inline_vf("""
                {
                    AscendC::Erfc<float, false>($0, $1, $1.GetSize());
                }
            """, [tile_length], asctile.float32, [erf_arg])
            out = (x * 0.5) * complement
        else:
            out = (x * 0.5) * (asctile.erf(x * 0.7071067811865476) + 1.0)
        asctile.copy_out(out.to(raw.dtype), out_gm, [offset], real_shape=[valid])


def gelu(x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    if approximate not in ('none', 'tanh'):
        raise ValueError("approximate must be 'none' or 'tanh'")
    if x.device.type != 'npu':
        raise ValueError("GeLU requires an NPU tensor")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Only float32, float16 and bfloat16 are qualified")
    size = x.numel()
    if not size:
        return torch.empty_like(x)
    tile, unroll, vf = geometry(x.dtype == torch.float32, approximate)
    exact_fp32 = approximate == 'none' and x.dtype == torch.float32
    reuse = UNIFORM_REUSE
    dtype = 'float32' if x.dtype == torch.float32 else ('float16' if x.dtype == torch.float16 else 'bfloat16')
    inventory = QUALIFIED_TILES.get((dtype, approximate), {})
    # An initial qualified demand binds platform metadata before final geometry.
    if not inventory:
        raise RuntimeError("Route has no compiled UB inventory")
    tile = min(inventory)
    ub = inventory[tile]
    route = dict(elements=size, tile=tile, unroll=unroll, dtype=dtype, mode=approximate,
                 compiled_ub_bytes=ub, reuse_alloc=reuse, static_alloc=None, vf_fusion=vf,
                 insert_sync=True, opt_level=3)
    resources = query_resources(x.device.index)
    require_default_device(resources, route)
    ensure_npu_platform()
    x = x.contiguous()
    y = torch.empty_like(x)
    # Allocation/platform setup may touch host context. Observe resources again
    # afterwards; launch_context then reads the actual launch stream's limit.
    resources = query_resources(x.device.index)
    require_default_device(resources, route)
    with launch_context(resources) as context:
        capacity = validate_capacity(context.resources, route, stream_limit=context.stream_limit)
        require_unreduced_resources(context.resources.reported_aiv, context.resources.device_limit,
                                    context.resources.thread_limit, context.stream_limit,
                                    context.resources.launch_cap, context.resources.fallback)
        # Retain exact logical TTK grid; never silently clamp it to physical AIVs.
        tile, cores, ub = choose_geometry(size, context.resources.launch_cap,
                                         min(253952, capacity['allocation_capacity_bytes']), dtype, approximate)
        route.update(tile=tile, compiled_ub_bytes=ub)
        capacity = validate_capacity(context.resources, route, stream_limit=context.stream_limit)
        record_ttk_launch(size, dtype, approximate, tile, cores, ub, context.resources.reported_aiv,
                          context.resources.device_limit, context.resources.thread_limit, context.stream_limit)
        record_capacity(context.resources, route, capacity, stream_limit=context.stream_limit)
        selected_kernel = gelu_kernel if (exact_fp32 or (approximate == 'tanh' and x.dtype != torch.float32)) else gelu_reference_kernel
        selected_kernel[cores, context.stream](x.reshape(-1), y.reshape(-1), size, tile, unroll,
                           approximate == 'tanh', approximate == 'none' and x.dtype == torch.float32,
                           reuse_alloc=reuse, static_alloc=None, vf_fusion=vf,
                           insert_sync=True, opt_level=3, debug=False)
    return y

