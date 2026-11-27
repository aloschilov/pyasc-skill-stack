import torch

try:
    from asc.experimental import asctile
except ImportError:
    import asctile

from ._pyasc_runtime import ensure_npu_platform


@asctile.jit(reuse_alloc=2)
def _swiglu_row_kernel(
    x_ptr: asctile.GlobalAddress,
    y_ptr: asctile.GlobalAddress,
    R: int,
    H: int,
    T: asctile.ConstExpr[int],
    W: asctile.ConstExpr[int],
):
    x_gm = asctile.global_tensor(x_ptr, [R, 2 * H])
    y_gm = asctile.global_tensor(y_ptr, [R, H])
    tiles = asctile.ceildiv(R, T)
    for tid in asctile.range(
        asctile.block_idx(), tiles, asctile.block_num(), unroll_factor=1
    ):
        r0 = tid * T
        valid_rows = min(R - r0, T)
        gate = asctile.copy_in(
            x_gm, [r0, 0], [T, W], real_shape=[valid_rows, H], pad_value=0.0
        )
        value = asctile.copy_in(
            x_gm, [r0, H], [T, W], real_shape=[valid_rows, H], pad_value=0.0
        )
        gate_f = gate.to(asctile.float32)
        value_f = value.to(asctile.float32)
        silu = gate_f / (1.0 + asctile.exp(-gate_f))
        out_f = silu * value_f
        out = out_f.to(gate.dtype)
        asctile.copy_out(out, y_gm, [r0, 0], real_shape=[valid_rows, H])


_AIV_COUNT = None
_AIV_FALLBACK = 72
_UB_USABLE_LANES = 15000


def _pow2_ceil(n):
    return 1 << (n - 1).bit_length()


def _vector_core_count():
    global _AIV_COUNT
    if _AIV_COUNT is not None:
        return _AIV_COUNT
    count = 0
    try:
        import asc.lib.runtime as rt

        raw = rt.device_info(
            rt.DeviceModuleType.RT_MODULE_TYPE_VECTOR_CORE,
            rt.DeviceInfoType.INFO_TYPE_CORE_NUM,
        )
        count = int(raw)
    except Exception:
        count = 0
    if count <= 0:
        count = _AIV_FALLBACK
    _AIV_COUNT = count
    return count


def _row_geometry(R, H, aiv):
    W = 32
    while W < H:
        W *= 2
    T = 128
    while T * W > _UB_USABLE_LANES and T > 1:
        T //= 2
    T = min(T, _pow2_ceil(R))
    T = max(T, 1)
    tiles = (R + T - 1) // T
    blocks = max(1, min(aiv, tiles))
    return T, W, blocks


def swi_glu(input, dim=-1):
    ensure_npu_platform()
    if not input.is_contiguous():
        input = input.contiguous()
    dim = int(dim)
    rank = input.dim()
    d = dim if dim >= 0 else dim + rank
    if d < 0 or d >= rank:
        raise ValueError("dim out of range for input rank")
    shape = list(input.shape)
    if shape[d] % 2 != 0:
        raise ValueError("split dimension must have even size")
    R = 1
    for i in range(d):
        R *= shape[i]
    H = shape[d] // 2
    for i in range(d + 1, rank):
        H *= shape[i]
    out_shape = shape[:d] + [shape[d] // 2] + shape[d + 1 :]
    out = input.new_empty(out_shape)
    if R * H == 0:
        return out
    aiv = _vector_core_count()
    T, W, blocks = _row_geometry(R, H, aiv)
    _swiglu_row_kernel[blocks](input, out, R, H, T, W)
    return out
