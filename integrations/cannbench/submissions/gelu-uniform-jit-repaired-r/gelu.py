"""High-level GeLU hybrid: public Erf lowp, user-authorized inline Erfc FP32."""
import asctile
import torch

from ._pyasc_runtime import ensure_npu_platform
from ._device_resources import query_resources
from ._gelu_metadata.gate import require_default_device, validate_capacity, record_capacity
from ._runtime_context import launch_context, record_launch

# Uniform JIT experiment derived from F; not hardware-qualified.
UNIFORM_REUSE = 2
# Compiler-qualified allocation only. CaModel completion explicitly waived for this hardware diagnostic.
QUALIFIED_TILES = {('float32', 'none'): {4160: 66560, 15872: 253952}, ('float32', 'tanh'): {15872: 253952}, ('float16', 'none'): {8192: 131072, 15872: 253952}, ('float16', 'tanh'): {8192: 98304, 21120: 253440}, ('bfloat16', 'none'): {8192: 131072, 15872: 253952}, ('bfloat16', 'tanh'): {8192: 98304, 21120: 253440}}


def choose_geometry(elements: int, available_blocks: int, ub_budget: int, dtype: str, mode: str):
    """Choose only measured fitting tiles; avoid starving available blocks.

    qualified_tiles maps tile elements to compiler UB bytes for this exact
    dtype/mode/JIT/unroll/kernel. Unknown inventories never permit a launch.
    Minimize maximum padded work per block, then number of tile iterations.
    This is a geometric heuristic, not a performance measurement.
    """
    if elements < 1 or available_blocks < 1 or ub_budget < 1:
        raise ValueError('Positive work, resource cap and UB capacity required')
    qualified_tiles = QUALIFIED_TILES[(dtype, mode)]
    choices = []
    for tile, ub in qualified_tiles.items():
        if type(tile) is not int or type(ub) is not int or tile < 1 or ub < 1:
            raise ValueError('Invalid compiler inventory')
        if ub > ub_budget:
            continue
        tiles = (elements + tile - 1) // tile
        blocks = min(available_blocks, tiles)
        iterations = (tiles + blocks - 1) // blocks
        choices.append(((iterations * tile, iterations, -blocks), tile, blocks, ub))
    if not choices:
        raise RuntimeError('No compiler-qualified tile fits UB')
    _, tile, blocks, ub = min(choices)
    return tile, blocks, ub


def geometry(is_fp32: bool, approximate: str):
    if approximate == 'none' and is_fp32:
        return 4160, 2, True
    if approximate == 'tanh' and is_fp32:
        return 15872, 2, True
    return 8192, 2, True



@asctile.jit(reuse_alloc=2, vf_fusion=True)
def gelu_kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
                input_length, tile_length: asctile.ConstExpr,
                unroll_factor: asctile.ConstExpr, approximate: asctile.ConstExpr,
                is_fp32: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [input_length])
    out_gm = asctile.global_tensor(output_ptr, [input_length])
    total_tiles = asctile.ceildiv(input_length, tile_length)
    quotient = total_tiles // asctile.block_num()
    remainder = total_tiles % asctile.block_num()
    first_tile = asctile.block_idx() * quotient + min(asctile.block_idx(), remainder)
    local_tiles = quotient + min(max(remainder - asctile.block_idx(), 0), 1)
    # Explicit fixed-size inner group avoids a separate dynamic remainder loop.
    for base in range(0, local_tiles, unroll_factor):
        for lane in asctile.range(unroll_factor, unroll_factor=unroll_factor):
            i = base + lane
            if i < local_tiles:
                offset = (first_tile + i) * tile_length
                valid = max(0, min(input_length - offset, tile_length))
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
    total_tiles = asctile.ceildiv(input_length, tile_length)
    quotient = total_tiles // asctile.block_num()
    remainder = total_tiles % asctile.block_num()
    first_tile = asctile.block_idx() * quotient + min(asctile.block_idx(), remainder)
    local_tiles = quotient + min(max(remainder - asctile.block_idx(), 0), 1)
    for i in asctile.range(local_tiles, unroll_factor=unroll_factor):
        offset = (first_tile + i) * tile_length
        valid = max(0, min(input_length - offset, tile_length))
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
        # Same policy for all six routes: live resource cap, useful work, measured UB.
        tile, cores, ub = choose_geometry(size, context.resources.launch_cap,
                                         min(253952, capacity['allocation_capacity_bytes']), dtype, approximate)
        route.update(tile=tile, compiled_ub_bytes=ub)
        capacity = validate_capacity(context.resources, route, stream_limit=context.stream_limit)
        record_launch(context, elements=size, tile=tile, unroll=unroll, cores=cores,
                      dtype=dtype, mode=approximate, vf=vf, ub=ub, reuse=reuse)
        record_capacity(context.resources, route, capacity, stream_limit=context.stream_limit)
        selected_kernel = gelu_kernel if (exact_fp32 or (approximate == 'tanh' and x.dtype != torch.float32)) else gelu_reference_kernel
        selected_kernel[cores, context.stream](x.reshape(-1), y.reshape(-1), size, tile, unroll,
                           approximate == 'tanh', approximate == 'none' and x.dtype == torch.float32,
                           reuse_alloc=reuse, static_alloc=None, vf_fusion=vf,
                           insert_sync=True, opt_level=3, debug=False)
    return y

