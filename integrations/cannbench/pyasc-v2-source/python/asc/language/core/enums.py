# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from enum import IntEnum


class AippInputFormat(IntEnum):
    YUV420SP_U8 = 0
    XRGB8888_U8 = 1
    RGB888_U8 = 4
    YUV400_U8 = 9


class BatchMode(IntEnum):
    NONE = 0
    BATCH_LESS_THAN_L1 = 1
    BATCH_LARGE_THAN_L1 = 2
    SINGLE_LARGE_THAN_L1 = 3


class BlockMode(IntEnum):
    BLOCK_MODE_NORMAL = 0
    BLOCK_MODE_MATRIX = 1
    BLOCK_MODE_VECTOR = 2
    BLOCK_MODE_SMALL_CHANNEL = 3
    BLOCK_MODE_DEPTHWISE = 4


class CacheLine(IntEnum):
    SINGLE_CACHE_LINE = 0
    ENTIRE_DATA_CACHE = 1


class CacheMode(IntEnum):
    CACHE_MODE_NORMAL = 0
    CACHE_MODE_DISABLE = 1
    CACHE_MODE_LAST = 2
    CACHE_MODE_PERSISTENT = 4


class CacheRwMode(IntEnum):
    READ = 1
    WRITE = 2
    RW = 3


class CubeFormat(IntEnum):
    ND = 0
    NZ = 1


class DataFormat(IntEnum):
    ND = 0
    NZ = 1
    NCHW = 2
    NC1HWC0 = 3
    NHWC = 4


class DcciDst(IntEnum):
    CACHELINE_ALL = 0
    CACHELINE_UB = 1
    CACHELINE_OUT = 2
    CACHELINE_ATOMIC = 3


class DeqScale(IntEnum):
    DEQ_NONE = 0
    DEQ = 1
    VDEQ = 2
    DEQ8 = 3
    VDEQ8 = 4
    DEQ16 = 5
    VDEQ16 = 6


class GatherMaskMode(IntEnum):
    DEFAULT = 0


class HardEvent(IntEnum):
    MTE2_MTE1 = 0
    MTE1_MTE2 = 1
    MTE1_M = 2
    M_MTE1 = 3
    MTE2_V = 4
    V_MTE2 = 5
    MTE3_V = 6
    V_MTE3 = 7
    M_V = 8
    V_M = 9
    V_V = 10
    MTE3_MTE1 = 11
    MTE1_MTE3 = 12
    MTE1_V = 13
    MTE2_M = 14
    M_MTE2 = 15
    V_MTE1 = 16
    M_FIX = 17
    FIX_M = 18
    MTE3_MTE2 = 19
    MTE2_MTE3 = 20
    S_V = 21
    V_S = 22
    S_MTE2 = 23
    MTE2_S = 24
    S_MTE3 = 25
    MTE3_S = 26
    MTE2_FIX = 27
    FIX_MTE2 = 28
    FIX_S = 29
    M_S = 30
    FIX_MTE3 = 31
    MTE1_FIX = 32
    FIX_MTE1 = 33
    FIX_FIX = 34


class Hardware(IntEnum):
    GM = 0
    UB = 1
    L1 = 2
    L0A = 3
    L0B = 4
    L0C = 5
    BIAS = 6
    FIXBUF = 7
    MAX = 8


class IterateMode(IntEnum):
    ITERATE_MODE_NORMAL = 0b00000001
    ITERATE_MODE_ALL = 0b00000010
    ITERATE_MODE_BATCH = 0b00000100
    ITERATE_MODE_N_BATCH = 0b00001000
    ITERATE_MODE_DEFAULT = 0b11111111


class IterateOrder(IntEnum):
    ORDER_M = 0
    ORDER_N = 1
    UNDEF = 2


class LayoutMode(IntEnum):
    NONE = 0
    NORMAL = 1
    BSNGD = 2
    SBNGD = 3
    BNGS1S2 = 4


class MatmulConfigMode(IntEnum):
    CONFIG_NORM = 1
    CONFIG_MDL = 2
    CONFIG_SPECIALMDL = 3
    CONFIG_IBSHARE = 4


class MatmulPolicy(IntEnum):
    MatmulPolicy = 0
    TrianUpperMatmulPolicy = 1
    TrianLowerMatmulPolicy = 2
    NBuffer33MatmulPolicy = 3


class PipeID(IntEnum):
    PIPE_S = 0
    PIPE_V = 1
    PIPE_M = 2
    PIPE_MTE1 = 3
    PIPE_MTE2 = 4
    PIPE_MTE3 = 5
    PIPE_ALL = 6
    PIPE_MTE4 = 7
    PIPE_MTE5 = 8
    PIPE_V2 = 9
    PIPE_FIX = 10


class MemDsbT(IntEnum):
    ALL = 0
    DDR = 1
    UB = 2
    SEQ = 3


class QuantModes(IntEnum):
    NoQuant = 0
    F322F16 = 1
    F322BF16 = 2
    DEQF16 = 3
    VDEQF16 = 4
    QF322B8_PRE = 5
    VQF322B8_PRE = 6
    REQ8 = 7
    VREQ8 = 8


class MaskMode(IntEnum):
    NORMAL = 0
    COUNTER = 1


class ScheduleType(IntEnum):
    INNER_PRODUCT = 0
    OUTER_PRODUCT = 1


class ReduceOrder(IntEnum):
    ORDER_VALUE_INDEX = 0
    ORDER_INDEX_VALUE = 1
    ORDER_ONLY_VALUE = 2
    ORDER_ONLY_INDEX = 3


class RoundMode(IntEnum):
    CAST_NONE = 0
    CAST_RINT = 1
    CAST_FLOOR = 2
    CAST_CEIL = 3
    CAST_ROUND = 4
    CAST_TRUNC = 5
    CAST_ODD = 6


class TransposeType(IntEnum):
    TRANSPOSE_TYPE_NONE = 0
    TRANSPOSE_NZ2ND_0213 = 1
    TRANSPOSE_NZ2NZ_0213 = 2
    TRANSPOSE_NZ2NZ_012_WITH_N = 3
    TRANSPOSE_NZ2ND_012_WITH_N = 4
    TRANSPOSE_NZ2ND_012_WITHOUT_N = 5
    TRANSPOSE_NZ2NZ_012_WITHOUT_N = 6
    TRANSPOSE_ND2ND_ONLY = 7
    TRANSPOSE_ND_UB_GM = 8
    TRANSPOSE_GRAD_ND_UB_GM = 9
    TRANSPOSE_ND2ND_B16 = 10
    TRANSPOSE_NCHW2NHWC = 11
    TRANSPOSE_NHWC2NCHW = 12


class TPosition(IntEnum):
    GM = 0
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6
    CO1 = 7
    CO2 = 8
    VECIN = 9
    VECOUT = 10
    VECCALC = 11
    MAX = 12


class pad_t(IntEnum):
    PAD_NONE = 0
    PAD_MODE1 = 1
    PAD_MODE2 = 2
    PAD_MODE3 = 3
    PAD_MODE4 = 4
    PAD_MODE5 = 5
    PAD_MODE6 = 6
    PAD_MODE7 = 7
    PAD_MODE8 = 8


class BatchOutMode(IntEnum):
    SINGLE_BATCH = 0,
    MULTI_BATCH = 1,
    DYNAMIC = 2,


class CMPMODE(IntEnum):
    LT = 0
    GT = 1
    EQ = 2
    LE = 3
    GE = 4
    NE = 5


class CO2Layout(IntEnum):
    NZ = 0
    ROW_MAJOR = 1
    COLUMN_MAJOR = 2


class SelMode(IntEnum):
    VSEL_CMPMASK_SPR = 0,
    VSEL_TENSOR_SCALAR_MODE = 1,
    VSEL_TENSOR_TENSOR_MODE = 2,


class AtomicDtype(IntEnum):
    ATOMIC_NONE = 0
    ATOMIC_F32 = 1
    ATOMIC_F16 = 2
    ATOMIC_S16 = 3
    ATOMIC_S32 = 4
    ATOMIC_S8 = 5
    ATOMIC_BF16 = 6


class AtomicOp(IntEnum):
    ATOMIC_SUM = 0
