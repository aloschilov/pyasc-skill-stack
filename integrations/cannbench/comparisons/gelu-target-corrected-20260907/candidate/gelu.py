"""Public AscTile GeLU, corrected target tanh and stable FP32 exact tail.

Pinned runtime: adadd7d66ed0ee16d33d79487bf584899a26ef1e.
No inline AscendC or compiler modifications. See EXACT_REMEDY.md.
"""
import asctile
import torch

from ._pyasc_runtime import ensure_npu_platform


def geometry(is_fp32: bool, approximate: str):
    if approximate == 'none' and is_fp32:
        return 1024, 1, True
    if approximate == 'tanh' and is_fp32:
        return 15872, 2, False
    return 8192, 2, False


@asctile.jit(reuse_alloc=1)
def gelu_kernel(input_ptr: asctile.GlobalAddress, output_ptr: asctile.GlobalAddress,
                input_length, tile_length: asctile.ConstExpr,
                unroll_factor: asctile.ConstExpr, approximate: asctile.ConstExpr,
                stable_tail: asctile.ConstExpr):
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
        else:
            out = (x * 0.5) * (asctile.erf(x * 0.7071067811865476) + 1.0)
            if stable_tail:
                # Normal-tail Mills ratio (DLMF 7.9.1), six levels for a >= 3.
                a = asctile.maximum(-x, 3.0)
                remainder = asctile.full([tile_length], 0.0, asctile.float32)
                for order in asctile.static_range(6, 0, -1):
                    remainder = order / (a + remainder)
                tail = (x / (a + remainder)) * asctile.exp((-0.5 * a) * a) * 0.3989422804014327
                out = asctile.where(x < -3.0, tail, out)
        asctile.copy_out(out.to(raw.dtype), out_gm, [offset], real_shape=[valid])


def gelu(x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    if approximate not in ('none', 'tanh'):
        raise ValueError("approximate must be 'none' or 'tanh'")
    ensure_npu_platform()
    x = x.contiguous()
    y = torch.empty_like(x)
    size = x.numel()
    tile, unroll, vf = geometry(x.dtype == torch.float32, approximate)
    if size:
        cores = min(72, (size + tile - 1) // tile)
        gelu_kernel[cores](x.reshape(-1), y.reshape(-1), size, tile, unroll,
                           approximate == 'tanh', approximate == 'none' and x.dtype == torch.float32,
                           reuse_alloc=1, static_alloc=None, vf_fusion=vf,
                           insert_sync=True, opt_level=3, debug=False)
    return y
