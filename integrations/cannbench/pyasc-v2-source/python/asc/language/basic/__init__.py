# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from .block_sync import (
    data_sync_barrier,
    pipe_barrier,
    set_flag,
    wait_flag,
    cross_core_set_flag,
    cross_core_wait_flag,
    ib_set,
    ib_wait,
    sync_all,
)
from .cache import data_cache_clean_and_invalid, get_icache_preload_status, icache_preload
from .common import (
    ascend_is_aic,
    ascend_is_aiv,
    check_local_memory_ia,
    data_cache_preload,
    get_hccl_context,
    get_store_atomic_config,
    get_sys_workspace,
    init_soc_state,
    reset_mask,
    set_aipp_functions,
    set_hccl_context,
    set_hf32_mode,
    set_hf32_trans_mode,
    set_mask_count,
    set_mask_norm,
    set_mm_layout_transform,
    set_store_atomic_config,
    set_sys_workspace,
    set_vector_mask,
)
from .data_copy import copy, data_copy, data_copy_pad, load_image_to_local, set_pad_value
from .determine_compute_sync import notify_next_block, wait_pre_block
from .dump_tensor import (
    dump_acc_chk_point,
    dump_tensor,
    printf,
    print_time_stamp,
    metrics_prof_start,
    metrics_prof_stop,
)
from .fixpipe import (
    fixpipe,
    set_fix_pipe_pre_quant_flag,
)
from .list_tensor import TensorDesc, ListTensorDesc
from .mm import (
    init_const_value,
    load_data,
    load_data_with_sparse,
    load_data_with_transpose,
    mmad,
    mmad_with_sparse,
    set_load_data_boundary,
    set_load_data_padding_value,
    set_load_data_repeat,
)
from .mm import load_data, load_data_with_transpose, mmad
from .scalar import (
    count_bits_cnt_same_as_sign_bit,
    scalar_cast,
    scalar_count_leading_zero,
    scalar_get_count_of_value,
    scalar_get_sff_value,
)
from .set_atomic import (
    set_atomic_add,
    set_atomic_max,
    set_atomic_min,
    set_atomic_none,
    set_atomic_type,
)
from .sys_var import (
    get_arch_version,
    get_block_idx,
    get_block_num,
    get_data_block_size_in_bytes,
    get_program_counter,
    get_sub_block_idx,
    get_sub_block_num,
    get_system_cycle,
    get_task_ratio,
    trap,
)
from .vec_binary import (
    add,
    add_deq_relu,
    add_relu,
    bilinear_interpolation,
    bitwise_and,
    bitwise_or,
    div,
    fused_mul_add,
    fused_mul_add_relu,
    max,
    min,
    mul,
    mul_add_dst,
    mul_cast,
    sub,
    sub_relu,
)
from .vec_binary_scalar import (
    adds,
    leaky_relu,
    maxs,
    mins,
    muls,
    shift_left,
    shift_right,
)
from .vec_brcb import brcb
from .vec_cmpsel import compare, compare_scalar, get_cmp_mask, select, set_cmp_mask
from .vec_duplicate import duplicate
from .vec_gather import (
    gather,
    gatherb,
)
from .vec_gather_mask import gather_mask, get_gather_mask_remain_count
from .proposal import (
    get_mrg_sort_result,
    get_sort_len,
    get_sort_offset,
    mrg_sort,
    mrg_sort4,
    proposal_concat,
    proposal_extract,
    rp_sort16,
    sort32,
    sort,
)
from .vec_reduce import (
    block_reduce_sum,
    block_reduce_max,
    block_reduce_min,
    pair_reduce_sum,
    reduce_max,
    reduce_min,
    reduce_sum,
    repeat_reduce_sum,
    whole_reduce_max,
    whole_reduce_min,
    whole_reduce_sum,
)
from .vec_scatter import scatter
from .vec_ternary_scalar import axpy
from .vec_transpose import transpose, trans_data_to_5hd
from .vec_unary import (
    abs,
    exp,
    ln,
    bitwise_not,
    reciprocal,
    relu,
    rsqrt,
    sqrt,
)
from .vec_vconv import (
    add_relu_cast,
    cast,
    cast_deq,
    set_deq_scale,
    sub_relu_cast,
)

__all__ = [
    # .block_sync
    "data_sync_barrier",
    "pipe_barrier",
    "set_flag",
    "wait_flag",
    "cross_core_set_flag",
    "cross_core_wait_flag",
    "ib_set",
    "ib_wait",
    "sync_all",
    # .common
    "ascend_is_aic",
    "ascend_is_aiv",
    "check_local_memory_ia",
    "data_cache_preload",
    "get_hccl_context",
    "get_store_atomic_config",
    "get_sys_workspace",
    "init_soc_state",
    "reset_mask",
    "set_aipp_functions",
    "set_hccl_context",
    "set_hf32_mode",
    "set_hf32_trans_mode",
    "set_mask_count",
    "set_mask_norm",
    "set_mm_layout_transform",
    "set_store_atomic_config",
    "set_sys_workspace",
    "set_vector_mask",
    # .data_cache
    "data_cache_clean_and_invalid",
    "get_icache_preload_status",
    "icache_preload",
    # .data_conversion
    "transpose",
    "trans_data_to_5hd",
    # .data_copy
    "copy",
    "data_copy",
    "data_copy_pad",
    "load_image_to_local",
    "set_pad_value",
    # .determine_compute_sync
    "notify_next_block",
    "wait_pre_block",
    # .dump_tensor
    "dump_acc_chk_point",
    "dump_tensor",
    "printf",
    "print_time_stamp",
    "metrics_prof_start",
    "metrics_prof_stop",
    # .fixpipe
    "fixpipe",
    "set_fix_pipe_pre_quant_flag",
    # .list_tensor
    "TensorDesc",
    "ListTensorDesc",
    # .mm
    "init_const_value",
    "load_data",
    "load_data_with_sparse",
    "load_data_with_transpose",
    "mmad",
    "mmad_with_sparse",
    "set_load_data_boundary",
    "set_load_data_padding_value",
    "set_load_data_repeat",
    # .proposal
    "get_sort_len",
    "get_sort_offset",
    # .scalar
    "count_bits_cnt_same_as_sign_bit",
    "scalar_cast",
    "scalar_count_leading_zero",
    "scalar_get_count_of_value",
    "scalar_get_sff_value",
    # .set_atomic
    "set_atomic_add",
    "set_atomic_max",
    "set_atomic_min",
    "set_atomic_none",
    "set_atomic_type",
    # .sys_var
    "get_arch_version",
    "get_block_idx",
    "get_block_num",
    "get_data_block_size_in_bytes",
    "get_program_counter",
    "get_sub_block_idx",
    "get_sub_block_num",
    "get_system_cycle",
    "get_task_ratio",
    "trap",
    # .vec_binary
    "add",
    "add_deq_relu",
    "add_relu",
    "add_relu_cast",
    "bilinear_interpolation",
    "bitwise_and",
    "bitwise_or",
    "div",
    "fused_mul_add",
    "fused_mul_add_relu",
    "gather_mask",
    "get_gather_mask_remain_count",
    "max",
    "min",
    "mul",
    "mul_add_dst",
    "mul_cast",
    "sub",
    "sub_relu",
    "sub_relu_cast",
    # .vec_binary_scalar
    "adds",
    "leaky_relu",
    "maxs",
    "mins",
    "muls",
    "shift_left",
    "shift_right",
    # .vec_brcb
    "brcb",
    # .vec_cmpsel
    "compare",
    "compare_scalar",
    "get_cmp_mask",
    "select",
    "set_cmp_mask",
    # .vec_duplicate
    "duplicate",
    # .vec_gather
    "gather",
    "gatherb",
    # .vec_proposal
    "get_mrg_sort_result",
    "mrg_sort",
    "mrg_sort4",
    "proposal_concat",
    "proposal_extract",
    "rp_sort16",
    "sort",
    "sort32",
    # .vec_reduce
    "block_reduce_sum",
    "block_reduce_max",
    "block_reduce_min",
    "pair_reduce_sum",
    "repeat_reduce_sum",
    "whole_reduce_max",
    "whole_reduce_min",
    "whole_reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_sum",
    # .vec_scatter
    "scatter",
    # .vec_ternary_scalar
    "axpy",
    # .vec_unary
    "abs",
    "exp",
    "ln",
    "bitwise_not",
    "reciprocal",
    "relu",
    "rsqrt",
    "sqrt",
    # .vec_vconv
    "cast",
    "cast_deq",
    "set_deq_scale",
]
