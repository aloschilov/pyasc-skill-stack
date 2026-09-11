"""High-level GeLU hybrid: public Erf lowp, user-authorized inline Erfc FP32."""
import asctile
import torch

from ._pyasc_runtime import ensure_npu_platform


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
    block_loop_num = asctile.ceildiv(asctile.ceildiv(input_length, asctile.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = block_length * asctile.block_idx()
    # Do not issue zero-length DMA for the unused end of a core's partition.
    local_tiles = min(block_loop_num, asctile.ceildiv(max(0, input_length - block_offset), tile_length))
    for i in asctile.range(local_tiles, unroll_factor=unroll_factor):
        offset = block_offset + i * tile_length
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
    ensure_npu_platform()
    x = x.contiguous()
    y = torch.empty_like(x)
    size = x.numel()
    tile, unroll, vf = geometry(x.dtype == torch.float32, approximate)
    tile = select_tile(size, x.dtype == torch.float32, approximate, tile)
    if size:
        cores = min(72, (size + tile - 1) // tile)
        gelu_kernel[cores](x.reshape(-1), y.reshape(-1), size, tile, unroll,
                           approximate == 'tanh', approximate == 'none' and x.dtype == torch.float32,
                           reuse_alloc=1, static_alloc=None, vf_fusion=vf,
                           insert_sync=True, opt_level=3, debug=False)
    return y
