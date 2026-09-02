# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from . import adv
from . import basic
from . import core
from . import fwk

from .basic import __all__ as basic_all
from .core import __all__ as core_all
from .fwk import __all__ as fwk_all

# basic
from .basic.block_sync import (
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
from .basic.cache import data_cache_clean_and_invalid, get_icache_preload_status, icache_preload
from .basic.common import (
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
from .basic.data_copy import copy, data_copy, data_copy_pad, load_image_to_local, set_pad_value
from .basic.determine_compute_sync import notify_next_block, wait_pre_block
from .basic.dump_tensor import (
    dump_acc_chk_point,
    dump_tensor,
    printf,
    print_time_stamp,
    metrics_prof_start,
    metrics_prof_stop,
)
from .basic.fixpipe import (
    fixpipe,
    set_fix_pipe_pre_quant_flag,
)
from .basic.list_tensor import TensorDesc, ListTensorDesc
from .basic.mm import (
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
from .basic.scalar import (
    count_bits_cnt_same_as_sign_bit,
    scalar_cast,
    scalar_count_leading_zero,
    scalar_get_count_of_value,
    scalar_get_sff_value,
)
from .basic.set_atomic import (
    set_atomic_add,
    set_atomic_max,
    set_atomic_min,
    set_atomic_none,
    set_atomic_type,
)
from .basic.sys_var import (
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
from .basic.vec_binary import (
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
from .basic.vec_binary_scalar import (
    adds,
    leaky_relu,
    maxs,
    mins,
    muls,
    shift_left,
    shift_right,
)
from .basic.vec_cmpsel import compare, compare_scalar, get_cmp_mask, select, set_cmp_mask
from .basic.vec_duplicate import duplicate
from .basic.vec_brcb import brcb
from .basic.vec_gather import gather, gatherb
from .basic.vec_gather_mask import gather_mask, get_gather_mask_remain_count
from .basic.vec_transpose import transpose, trans_data_to_5hd
from .basic.proposal import (
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
from .basic.vec_reduce import (
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
from .basic.vec_scatter import scatter
from .basic.vec_ternary_scalar import axpy
from .basic.vec_unary import (
    abs,
    exp,
    ln,
    bitwise_not,
    reciprocal,
    relu,
    rsqrt,
    sqrt,
)
from .basic.vec_vconv import (
    add_relu_cast,
    cast,
    cast_deq,
    set_deq_scale,
    sub_relu_cast,
)

# core
from .core.array import array
from .core.constexpr import ConstExpr
from .core.dtype import (
    DataType,
    void,
    int8,
    int16,
    int32,
    int64,
    bfloat16,
    float16,
    float32,
    float64,
    uint8,
    uint16,
    uint32,
    uint64,
    int_,
    half,
    float_,
    double,
)
from .core.enums import (
    AippInputFormat,
    BlockMode,
    CacheLine,
    CacheMode,
    CacheRwMode,
    CMPMODE,
    CO2Layout,
    CubeFormat,
    DataFormat,
    DcciDst,
    DeqScale,
    GatherMaskMode,
    HardEvent,
    Hardware,
    PipeID,
    MemDsbT,
    TPosition,
    pad_t,
    ReduceOrder,
    RoundMode,
    TransposeType,
    BatchMode,
    IterateMode,
    IterateOrder,
    ScheduleType,
    LayoutMode,
    MaskMode,
    QuantModes,
    MatmulConfigMode,
    SelMode,
    AtomicDtype,
    AtomicOp,
)
from .core.ir_value import GlobalAddress
from .core.memory_allocator import LocalMemAllocator
from .core.ops import inline, number
from .core.properties import (
    property,
    DEFAULT_C0_SIZE,
    ONE_BLK_SIZE,
    TOTAL_L0C_SIZE,
    TOTAL_L1_SIZE,
)
from .core.range import range, static_range
from .core.tensor import GlobalTensor, LocalTensor, LocalTensorAuto, MrgSortSrcList
from .core.types import (
    BinaryRepeatParams,
    BrcbRepeatParams,
    CheckLocalMemoryIAParam,
    CopyRepeatParams,
    DataCopyParams,
    DataCopyEnhancedParams,
    DataCopyExtParams,
    DataCopyPadExtParams,
    DataCopyPadParams,
    FixpipeConfig,
    FixpipeParamsV220,
    GatherMaskParams,
    GatherRepeatParams,
    InitConstValueParams,
    LoadData2DParams,
    LoadImageToLocalParams,
    MmadParams,
    MrgSort4Info,
    Nd2NzParams,
    ShapeInfo,
    SliceInfo,
    TransDataTo5HDParams,
    TransposeParamsExt,
    UnaryRepeatParams,
    Nd2NzParams,
    Nz2NdParamsFull,
    DataCopyCO12DstParams,
    LoadData2DParams,
    LoadData2DParamsV2,
    LoadData2dTransposeParams,
    LoadData2dTransposeParamsV2,
    LoadData3DParamsV1,
    LoadData3DParamsV2,
    LoadData3DParamsV2Pro,
    LoadDataRepeatParam,
    get_shape_size,
    VdeqInfo,
)
from .core.aipp_types import (
    AippParams,
    AippPaddingParams,
    AippSwapParams,
    AippSingleLineParams,
    AippDataTypeConvParams,
    AippChannelPaddingParams,
    AippColorSpaceConvParams,
)
from .core.utils import static_assert, ceildiv

# fwk
from .fwk.tpipe import TBuf, TBufPool, TPipe, TQue, TQueBind, get_tpipe_ptr

__all__ = [
    # .language
    "adv",
    "basic",
    "core",
    "fwk",
]

__all__.extend(basic_all)
__all__.extend(core_all)
__all__.extend(fwk_all)
