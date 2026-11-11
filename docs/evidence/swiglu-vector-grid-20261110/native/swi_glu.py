"""SwiGLU with runtime vector grids and UB-sized BF16 pair tiles."""
import torch
from grid_template import vector_blocks

try:
    from asc.experimental import asctile
except ImportError:
    import asctile

from ._pyasc_runtime import ensure_npu_platform

_RT = 16
_MAX_CORES = 8
_MAX_TILE_ELEMS = 1024
_UB_FIXED_BYTES = 32 * 1024


@asctile.jit
def _swiglu_tile(in_ptr: asctile.GlobalAddress,
                 out_ptr: asctile.GlobalAddress,
                 rows: int,
                 cols: int,
                 r0: int,
                 c0: int,
                 rt: asctile.ConstExpr[int],
                 ct: asctile.ConstExpr[int],
                 r_valid: asctile.ConstExpr[int],
                 c_valid: asctile.ConstExpr[int],
                 out_mode: asctile.ConstExpr[int]):
    in_gm = asctile.global_tensor(in_ptr, [rows, 2 * cols])
    out_gm = asctile.global_tensor(out_ptr, [rows, cols])
    x0 = asctile.copy_in(in_gm, [r0, c0], [rt, ct],
                         real_shape=[r_valid, c_valid], pad_value=0.0)
    x1 = asctile.copy_in(in_gm, [r0, cols + c0], [rt, ct],
                         real_shape=[r_valid, c_valid], pad_value=0.0)
    x0f = asctile.cast(x0, asctile.float32)
    x1f = asctile.cast(x1, asctile.float32)
    e = asctile.exp(asctile.mul(x0f, -1.0))
    s = asctile.div(1.0, asctile.add(e, 1.0))
    res = asctile.mul(asctile.mul(x0f, s), x1f)
    if out_mode == 1:
        out_t = asctile.cast(res, asctile.float16)
    elif out_mode == 2:
        out_t = asctile.cast(res, asctile.bfloat16)
    else:
        out_t = res
    asctile.copy_out(out_t, out_gm, [r0, c0], real_shape=[r_valid, c_valid])


@asctile.jit
def _swiglu_impl(in_ptr: asctile.GlobalAddress,
                 out_ptr: asctile.GlobalAddress,
                 rows: int,
                 cols: int,
                 r_base: int,
                 n_rb: int,
                 n_c_full: int,
                 rt: asctile.ConstExpr[int],
                 ct: asctile.ConstExpr[int],
                 r_valid: asctile.ConstExpr[int],
                 c_tail: asctile.ConstExpr[int],
                 out_mode: asctile.ConstExpr[int]):
    for rb in asctile.range(asctile.block_idx(), n_rb, asctile.block_num()):
        r0 = r_base + rb * rt
        for c in asctile.range(0, n_c_full):
            _swiglu_tile(in_ptr, out_ptr, rows, cols, r0, c * ct,
                         rt, ct, r_valid, ct, out_mode)
        if c_tail > 0:
            _swiglu_tile(in_ptr, out_ptr, rows, cols, r0, n_c_full * ct,
                         rt, ct, r_valid, c_tail, out_mode)


@asctile.jit
def _swiglu_wide_row(in_ptr: asctile.GlobalAddress,
                     out_ptr: asctile.GlobalAddress,
                     rows: int,
                     cols: int,
                     n_c_full: int,
                     rt: asctile.ConstExpr[int],
                     ct: asctile.ConstExpr[int],
                     r_valid: asctile.ConstExpr[int],
                     c_tail: asctile.ConstExpr[int],
                     out_mode: asctile.ConstExpr[int]):
    for c in asctile.range(asctile.block_idx(), n_c_full, asctile.block_num()):
        _swiglu_tile(in_ptr, out_ptr, rows, cols, 0, c * ct,
                     rt, ct, r_valid, ct, out_mode)
    if c_tail > 0:
        if asctile.block_idx() == n_c_full % asctile.block_num():
            _swiglu_tile(in_ptr, out_ptr, rows, cols, 0, n_c_full * ct,
                         rt, ct, r_valid, c_tail, out_mode)


def _col_tile(cols):
    if cols <= 16:
        return 16
    if cols <= 32:
        return 32
    return 64


def _wide_col_tile(rows, cols):
    ct = _MAX_TILE_ELEMS // rows
    ct -= ct % 16
    cap = cols // _MAX_CORES
    cap -= cap % 16
    if cap < ct:
        ct = cap
    if ct < 16:
        ct = 16
    return ct


def _query_platform_ub_bytes():
    try:
        import importlib

        from asc.runtime import launcher
        get_soc_version = None
        for module_name in ("asc.lib.runtime", "asc.runtime"):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            get_soc_version = getattr(module, "get_soc_version", None)
            if get_soc_version is not None:
                break
        if get_soc_version is None:
            return None
        return int(launcher.get_platform_info(get_soc_version()).ub_size)
    except Exception:
        return None


def _short_row_height(rows, ct, esize, out_mode):
    bytes_per_row = 80 * ct
    ub_bytes = _query_platform_ub_bytes()
    if ub_bytes is None:
        return _RT
    ceiling = (ub_bytes - _UB_FIXED_BYTES) // bytes_per_row
    if ceiling < _RT:
        raise ValueError(f"UB ceiling {ceiling} < _RT {_RT}")
    if rows <= _RT:
        return _RT
    aiv = vector_blocks(rows)
    limit = min(ceiling, max(1, rows // aiv))
    return max(_RT, 1 << (limit.bit_length() - 1))




@asctile.jit
def _swiglu_pair_groups(xptr: asctile.GlobalAddress, yptr: asctile.GlobalAddress,
                        rows: int, groups: int, tile: asctile.ConstExpr[int]):
    x = asctile.global_tensor(xptr, [2 * rows])
    y = asctile.global_tensor(yptr, [rows])
    for group in asctile.range(asctile.block_idx(), groups, asctile.block_num()):
        first = group * tile
        valid = min(tile, rows - first)
        flat = asctile.copy_in(x, [2 * first], [2 * tile],
                               real_shape=[2 * valid], pad_value=0.0)
        # A pair tile is [16, tile // 16, 2]. Move parity to the leading axis.
        columns = flat.reshape(16, tile // 8).transpose(1, 0)
        planes = columns.reshape(tile // 16, 32).transpose(1, 0).reshape(2, tile)
        xy = asctile.cast(planes, asctile.float32)
        plane_id = asctile.concat(asctile.full([1, tile], 0.0, asctile.float32),
                                  asctile.full([1, tile], 1.0, asctile.float32))
        is_gate = plane_id < 0.5
        gate = asctile.reduce_sum(asctile.where(is_gate, xy, 0.0), 0)
        value = asctile.reduce_sum(asctile.where(is_gate, 0.0, xy), 0)
        sigmoid = asctile.div(1.0, asctile.add(asctile.exp(-gate), 1.0))
        result = asctile.cast(gate * sigmoid * value, asctile.bfloat16)
        asctile.copy_out(result.reshape(tile), y, [first], real_shape=[valid])


def _pair_tile_size(rows, ub_bytes, vector_count):
    """Largest qualified tile that fits UB while keeping the vector grid occupied."""
    if ub_bytes is None or rows <= 0:
        return None
    # CaModel allocations for this kernel and pin, including the runtime tail.
    occupancy_limit = max(256, (rows + vector_count - 1) // vector_count)
    for tile, allocated in ((2048, 184832), (1024, 92416), (512, 46208), (256, 23104)):
        if allocated <= ub_bytes and tile <= occupancy_limit:
            return tile
    return None


@asctile.jit
def _swiglu_register_pairs(xptr: asctile.GlobalAddress, yptr: asctile.GlobalAddress,
                        rows: int, groups: int):
    tile = 2048
    x = asctile.global_tensor(xptr, [2 * rows])
    y = asctile.global_tensor(yptr, [rows])
    for group in asctile.range(asctile.block_idx(), groups, asctile.block_num()):
        first = group * tile
        valid = min(tile, rows - first)
        flat = asctile.copy_in(x, [2 * first], [2 * tile],
                               real_shape=[2 * valid], pad_value=0.0)
        result = asctile.inline_vf('''
            using namespace AscendC::Reg;
            auto* src = reinterpret_cast<__ubuf__ bfloat16_t*>($1.GetPhyAddr());
            auto* dst = reinterpret_cast<__ubuf__ bfloat16_t*>($0.GetPhyAddr());
            auto all = CreateMask<uint8_t>();
            static constexpr CastTrait to0 = {RegLayout::ZERO,SatMode::UNKNOWN,MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
            static constexpr CastTrait to1 = {RegLayout::ONE,SatMode::UNKNOWN,MaskMergeMode::ZEROING,AscendC::RoundMode::UNKNOWN};
            static constexpr CastTrait out0 = {RegLayout::ZERO,SatMode::NO_SAT,MaskMergeMode::ZEROING,AscendC::RoundMode::CAST_RINT};
            static constexpr CastTrait out1 = {RegLayout::ONE,SatMode::NO_SAT,MaskMergeMode::ZEROING,AscendC::RoundMode::CAST_RINT};
            RegTensor<bfloat16_t> a,b,g,v,o0,o1;
            RegTensor<float> gf,vf,tmp,one;
            Duplicate(one,1.0f,all);
            // Two 128-lane loads contain 128 adjacent gate/value pairs.
            for (uint16_t i=0; i<16; ++i) {
                LoadAlign(a,src+256*i);
                LoadAlign(b,src+256*i+128);
                DeInterleave(g,v,a,b);
                // Widen alternating lanes separately, then repack into BF16.
                Cast<float,bfloat16_t,to0>(gf,g,all);
                Cast<float,bfloat16_t,to0>(vf,v,all);
                Muls(tmp,gf,-1.0f,all);
                Exp(tmp,tmp,all);
                Adds(tmp,tmp,1.0f,all);
                Div(tmp,one,tmp,all);
                Mul(tmp,gf,tmp,all);
                Mul(tmp,tmp,vf,all);
                Cast<bfloat16_t,float,out0>(o0,tmp,all);
                Cast<float,bfloat16_t,to1>(gf,g,all);
                Cast<float,bfloat16_t,to1>(vf,v,all);
                Muls(tmp,gf,-1.0f,all);
                Exp(tmp,tmp,all);
                Adds(tmp,tmp,1.0f,all);
                Div(tmp,one,tmp,all);
                Mul(tmp,gf,tmp,all);
                Mul(tmp,tmp,vf,all);
                Cast<bfloat16_t,float,out1>(o1,tmp,all);
                Or((RegTensor<uint16_t>&)o0,(RegTensor<uint16_t>&)o0,(RegTensor<uint16_t>&)o1,all);
                StoreAlign(dst+128*i,o0,all);
            }
            ''', [2048], asctile.bfloat16, [flat])
        asctile.copy_out(result, y, [first], real_shape=[valid])


def _register_pair_tile_size(rows, ub_bytes, vector_count):
    """Use the qualified 2048-pair specialization when it fits and fills the grid."""
    if ub_bytes is None or ub_bytes < 12288 or rows <= 0:
        return None
    if (rows + 2047) // 2048 < vector_count or not hasattr(asctile, "inline_vf"):
        return None
    from asc.lib import runtime
    if getattr(runtime.get_soc_version(), "value", None) == "Ascend950PR_9599":
        return 2048
    return None


def swi_glu(input, dim=-1):
    ensure_npu_platform()
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    dtype = input.dtype
    if dtype == torch.float32:
        out_mode = 0
    elif dtype == torch.float16:
        out_mode = 1
    elif dtype == torch.bfloat16:
        out_mode = 2
    else:
        raise TypeError("swi_glu supports float16, float32 and bfloat16 inputs")
    rank = input.dim()
    if rank == 0:
        raise ValueError("input must have at least one dimension")
    d = int(dim)
    if d < 0:
        d += rank
    if d < 0 or d >= rank:
        raise ValueError("dim is out of range for the input rank")
    shape = list(input.shape)
    size = shape[d]
    if size % 2 != 0:
        raise ValueError("size of the split dimension must be even")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    half = size // 2
    rows = 1
    for i in range(d):
        rows *= shape[i]
    inner = 1
    for i in range(d + 1, rank):
        inner *= shape[i]
    cols = half * inner
    out_shape = shape[:d] + [half] + shape[d + 1:]
    out = torch.empty(out_shape, dtype=dtype, device=input.device)
    if rows == 0 or cols == 0:
        return out
    if dtype == torch.bfloat16 and d == rank - 1 and size == 2:
        # Native-enabled 950PR build: attempt the register kernel ahead of the
        # public packed tile path. The selector receives the effective UB
        # capacity (any caller-supplied runner limit is already folded in at
        # the query boundary) together with the active vector count, and it
        # returns None when the capacity is missing or no qualified tile fits.
        tile = _register_pair_tile_size(
            rows, _query_platform_ub_bytes(), vector_blocks(rows)
        )
        if tile:
            # The qualified specialization fixes its physical tile (2048 pairs)
            # and takes exactly four arguments; groups is ceil(rows / tile).
            groups = (rows + tile - 1) // tile
            _swiglu_register_pairs[vector_blocks(groups)](input, out, rows, groups)
            return out
        # Tile declined: fall through to the public packed tile path below,
        # which in turn falls back to the general row kernel.
    if dtype == torch.bfloat16 and d == rank - 1 and size == 2:
        tile = _pair_tile_size(rows, _query_platform_ub_bytes(), vector_blocks(rows))
        if tile:
            groups = (rows + tile - 1) // tile
            _swiglu_pair_groups[vector_blocks(groups)](input, out, rows, groups, tile)
            return out

    ct = _col_tile(cols)
    n_c_full = cols // ct
    c_tail = cols % ct
    esize = input.element_size()
    rt = _RT
    if cols * esize < 32:
        rt = _short_row_height(rows, ct, esize, out_mode)
    n_rb_full = rows // rt
    r_tail = rows % rt
    if rows <= _RT and n_c_full >= _MAX_CORES:
        w_ct = _wide_col_tile(rows, cols)
        n_w_full = cols // w_ct
        w_tail = cols % w_ct
        _swiglu_wide_row[vector_blocks(n_w_full + bool(w_tail))](input, out, rows, cols, n_w_full,
                                     rows, w_ct, rows, w_tail, out_mode)
        return out
    if n_rb_full > 0:
        cores = vector_blocks(n_rb_full)
        _swiglu_impl[cores](input, out, rows, cols, 0, n_rb_full, n_c_full,
                            rt, ct, rt, c_tail, out_mode)
    if r_tail > 0:
        _swiglu_impl[1](input, out, rows, cols, n_rb_full * rt, 1, n_c_full,
                        rt, ct, r_tail, c_tail, out_mode)
    return out
