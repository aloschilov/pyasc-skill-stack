"""High-level GeLU hybrid: public Erf lowp, user-authorized inline Erfc FP32."""
import asctile
import torch

from ._pyasc_runtime import ensure_npu_platform
from ._device_resources import query_resources, launch_blocks
from ._launch_safety import required_ub, validate_launch
from ._runtime_context import launch_context, record_launch


def geometry(is_fp32: bool, approximate: str):
    if approximate == 'none' and is_fp32:
        return 5120, 1, True
    if approximate == 'tanh' and is_fp32:
        return 15872, 2, False
    return 8192, 2, False


def select_tile(size: int, is_fp32: bool, approximate: str, tile: int):
    if not size or not is_fp32 or approximate != 'none':
        return tile
    wide = 20480
    narrow_cores = min(72, (size + tile - 1) // tile)
    wide_cores = min(72, (size + wide - 1) // wide)
    narrow_span = (((size + narrow_cores - 1) // narrow_cores + tile - 1) // tile) * tile
    wide_span = (((size + wide_cores - 1) // wide_cores + wide - 1) // wide) * wide
    return wide if wide_span <= narrow_span else tile


@asctile.jit(reuse_alloc=1)
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
    ensure_npu_platform()
    size = x.numel()
    if not size:
        return torch.empty_like(x)
    tile, unroll, vf = geometry(x.dtype == torch.float32, approximate)
    tile = select_tile(size, x.dtype == torch.float32, approximate, tile)
    resources = query_resources(x.device.index)
    with launch_context(resources) as context:
        dtype = 'float32' if x.dtype == torch.float32 else ('float16' if x.dtype == torch.float16 else 'bfloat16')
        ub = required_ub(dtype, approximate, tile, unroll, vf=vf)
        validate_launch(size, tile, context.resources, ub)
        cores = launch_blocks(size, tile, context.resources)
        record_launch(context, elements=size, tile=tile, unroll=unroll, cores=cores,
                      dtype=dtype, mode=approximate, vf=vf, ub=ub)
        x = x.contiguous()
        y = torch.empty_like(x)
        gelu_kernel[cores, context.stream](x.reshape(-1), y.reshape(-1), size, tile, unroll,
                           approximate == 'tanh', approximate == 'none' and x.dtype == torch.float32,
                           reuse_alloc=1, static_alloc=None, vf_fusion=vf,
                           insert_sync=True, opt_level=3, debug=False)
    return y

