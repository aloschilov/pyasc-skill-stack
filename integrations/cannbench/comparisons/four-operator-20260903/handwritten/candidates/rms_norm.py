# Source-derived from compiler-team/pyasc v2; see PROVENANCE.json.
import math
import torch
import asctile

from ._pyasc_runtime import ensure_npu_platform

@asctile.jit
def calculate_square_reduce_sum(x: asctile.LocalTensor):
    return asctile.reduce_sum(x.to(asctile.float32)**2, 1, keep_dims=True)

@asctile.jit
def compute_rstd_newton_raphson(src: asctile.LocalTensor, epsilon, avg_factor, need_max=False, need_avg_factor=True):
    pos_inf = 3.40282366920938E+38
    src = src.to(asctile.float32)
    if need_avg_factor:
        src = src * avg_factor
    var = src + epsilon
    if need_max:
        var = asctile.maximum(var, -99.99)
    y_0 = asctile.sqrt(1.0 / var)
    y_1 = y_0 * (1.5 - 0.5 * var * y_0**2)
    rstd = y_1 + 0.5 * (1.0 - var * y_1**2) * y_1
    rstd = asctile.where(var == pos_inf, asctile.cast(0, asctile.float32), rstd)
    rstd = asctile.where(var == 0.0, asctile.cast(pos_inf, asctile.float32), rstd)
    return rstd

@asctile.jit
def compute_y(x: asctile.LocalTensor, gamma: asctile.LocalTensor, rstd_f32: asctile.LocalTensor):
    mul = x.to(asctile.float32) * rstd_f32 * gamma.to(asctile.float32)
    return mul.to(x.dtype)

@asctile.jit(reuse_alloc=2)
def rms_norm_kernel(x_ptr: asctile.GlobalAddress, gamma_ptr: asctile.GlobalAddress, y_ptr: asctile.GlobalAddress,
                    rstd_ptr: asctile.GlobalAddress, num_row, num_col, num_col_align, block_factor, col_flod_factor,
                    ub_factor, epsilon, avg_factor, last_block_factor):
    x_gm = asctile.global_tensor(x_ptr, [num_row, num_col])
    gamma_gm = asctile.global_tensor(gamma_ptr, [1, num_col])
    y_gm = asctile.global_tensor(y_ptr, [num_row, num_col])
    rstd_gm = asctile.global_tensor(rstd_ptr, [num_row])
    cur_block_factor = last_block_factor if asctile.block_idx() == (asctile.block_num() - 1) else block_factor
    cur_block_loops = asctile.ceildiv(cur_block_factor, ub_factor)
    cur_ub_tails = cur_block_factor - (cur_block_loops - 1) * ub_factor
    base_offset = block_factor * asctile.block_idx()

    gamma = asctile.copy_in(gamma_gm, [0, 0], [1, num_col_align], real_shape=[1, num_col])
    for i in asctile.range(cur_block_loops, unroll_factor=1):  # TODO: fix uf
        cur_ub_factor = cur_ub_tails if i == (cur_block_loops - 1) else ub_factor
        x = asctile.copy_in(x_gm, [base_offset + i * ub_factor, 0], [ub_factor, num_col_align],
                            real_shape=[cur_ub_factor, num_col])
        tmp = calculate_square_reduce_sum(x) / num_col
        rstd_f32 = compute_rstd_newton_raphson(tmp, epsilon, avg_factor)
        y = compute_y(x, gamma, rstd_f32)
        asctile.copy_out(rstd_f32.reshape(ub_factor), rstd_gm, offsets=[base_offset + i * ub_factor],
                         real_shape=[cur_ub_factor])  # TODO: fix sync for scalar store
        asctile.copy_out(y, y_gm, offsets=[base_offset + i * ub_factor, 0], real_shape=[cur_ub_factor, num_col])

def rms_norm(x, gamma, epsilon=1e-6):
    ensure_npu_platform()
    original_shape = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()
    num_col = x.shape[-1]
    num_row = x.numel() // num_col
    x_2d = x.view(num_row, num_col)
    y = torch.empty_like(x_2d)
    if num_row == 0 or num_col == 0:
        return y.view(original_shape)
    rstd = torch.zeros(num_row, dtype=torch.float32, device=x.device)
    items_in_block = 32 // x.element_size()
    num_col_align = ((num_col + items_in_block - 1) // items_in_block) * items_in_block
    target_cores = min(72, num_row)
    block_factor = (num_row + target_cores - 1) // target_cores
    cores = (num_row + block_factor - 1) // block_factor
    last_block_factor = num_row - block_factor * (cores - 1)
    # The kernel already divides the square sum by num_col.  CANNBench's
    # standard RMS contract therefore requires no second scaling here.
    avg_factor = 1.0
    rms_norm_kernel[cores](
        x_2d, gamma, y, rstd,
        num_row, num_col, asctile.ConstExpr(num_col_align),
        asctile.ConstExpr(block_factor), 1, asctile.ConstExpr(1),
        epsilon, avg_factor, last_block_factor,
    )
    return y.view(original_shape)
