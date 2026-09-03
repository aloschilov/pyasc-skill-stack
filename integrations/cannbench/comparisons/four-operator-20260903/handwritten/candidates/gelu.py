# Source-derived from compiler-team/pyasc v2; see PROVENANCE.json.
import math
import torch
import asctile

from ._pyasc_runtime import ensure_npu_platform

@asctile.jit(reuse_alloc=1)
def _target_gelu_kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress, input_length,
         tile_length: asctile.ConstExpr, TANH_APPROX_FACTOR: asctile.ConstExpr,
         NEG_SQRT_EIGHT_OVER_PI: asctile.ConstExpr, unroll_factor: asctile.ConstExpr):
    in_gm = asctile.global_tensor(input_ptr, [input_length])
    out_gm = asctile.global_tensor(output_ptr, [input_length])

    block_loop_num = asctile.ceildiv(asctile.ceildiv(input_length, asctile.block_num()), tile_length)
    block_length = tile_length * block_loop_num
    block_offset = block_length * asctile.block_idx()

    for i in asctile.range(block_loop_num, unroll_factor=unroll_factor):
        current_offset = block_offset + i * tile_length
        row = asctile.copy_in(in_gm, [current_offset], [tile_length])
        input_sq = row * row
        input_cub = input_sq * row
        input_cub = row + input_cub * TANH_APPROX_FACTOR
        input_cub = input_cub * NEG_SQRT_EIGHT_OVER_PI
        input_cub = asctile.exp(input_cub)
        input_cub = input_cub + 1
        out = row / input_cub
        asctile.copy_out(out, out_gm, [current_offset])

_BLOCK_NUM = 72
_TILE_LENGTH = 15872
_UNROLL_FACTOR = 2
_TANH_APPROX_FACTOR = 1.0 / 0.044715
_NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715


def gelu(x, approximate="none"):
    ensure_npu_platform()
    original_shape = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    x_flat = x.view(-1)
    input_length = x_flat.numel()
    out = torch.empty_like(x_flat)
    if input_length == 0:
        return out.view(original_shape)
    _target_gelu_kernel[_BLOCK_NUM](
        x_flat,
        out,
        input_length,
        _TILE_LENGTH,
        _TANH_APPROX_FACTOR,
        _NEG_SQRT_EIGHT_OVER_PI,
        _UNROLL_FACTOR,
    )
    return out.view(original_shape)

# ADAPTER_DONE
