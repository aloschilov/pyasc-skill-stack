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
    gelu[_BLOCK_NUM](
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
