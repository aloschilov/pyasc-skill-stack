# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from asc.experimental import asctile
import pytest
import torch

from ..helpers import parametrize_is_static
from . import FullLoad, Transpose, run_matmul_v3_test

test_cases = [
    (None, (1500, 1669, 113, 256, 256, 128, 256, 256, 32), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.NONE,
     False, (1, 1, 1, 2, 2)),
    (None, (45000, 92, 32, 336, 96, 32, 336, 96, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B, False,
     (1, 1, 1, 2, 2)),
    (None, (10000, 200, 256, 144, 208, 256, 144, 208, 64), torch.float16, Transpose.NONE, Transpose.NONE, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (46500, 88, 104, 336, 96, 64, 336, 96, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.B, False,
     (1, 1, 1, 2, 2)),
    (None, (39000, 116, 132, 256, 128, 64, 256, 128, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (45000, 124, 124, 256, 128, 64, 256, 128, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (49500, 144, 128, 224, 144, 64, 224, 144, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (192000, 66, 64, 400, 80, 64, 400, 80, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.B, False,
     (1, 1, 1, 2, 2)),
    (None, (250000, 120, 145, 256, 128, 64, 256, 128, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (96000, 104, 64, 288, 112, 64, 288, 112, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B, False,
     (1, 1, 1, 2, 2)),
    (None, (75000, 116, 116, 256, 128, 64, 256, 128, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (150000, 76, 64, 400, 80, 64, 400, 80, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B, False,
     (1, 1, 1, 2, 2)),
    (None, (40960, 280, 256, 112, 288, 64, 112, 288, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (102400, 168, 64, 176, 176, 64, 176, 176, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (180000, 84, 64, 336, 96, 64, 336, 96, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B, False,
     (1, 1, 1, 2, 2)),
    (None, (250000, 145, 120, 192, 160, 64, 192, 160, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (307200, 200, 128, 144, 208, 64, 144, 208, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (375000, 148, 148, 192, 160, 64, 192, 160, 16), torch.float32, Transpose.NONE, Transpose.L0, FullLoad.B,
     False, (1, 1, 1, 2, 2)),
    (None, (4096, 13664, 32, 256, 256, 32, 256, 256, 32), torch.float16, Transpose.L0, Transpose.NONE, FullLoad.NONE,
     False, (1, 1, 1, 2, 2)),
    (None, (4800, 2864, 128, 320, 192, 64, 320, 192, 16), torch.float32, Transpose.NONE, Transpose.NONE, FullLoad.NONE,
     False, (1, 1, 1, 2, 2)),
]


def base_id(tc):
    vals = getattr(tc, "values", tc)
    tiling, dtype = vals[1], vals[2]
    distrib_mode = asctile.DistribMode.SplitByM if dtype == torch.float32 else asctile.DistribMode.FullVec0
    return f"{tiling[0]}_{tiling[1]}_{tiling[2]}_{str(dtype).split('.')[-1]}_{str(distrib_mode).split('.')[-1]}"


def case_ids(cases):
    seen = {}
    ids = []
    for tc in cases:
        base = base_id(tc)
        if base in seen:
            seen[base] += 1
            ids.append(f"{base}_v{seen[base]}")
        else:
            seen[base] = 0
            ids.append(base)
    return ids


@parametrize_is_static()
@pytest.mark.parametrize("core_num, tiling_data, dtype, a_transp, b_transp, full_load_mode, has_bias, double_buffering",
                         test_cases, ids=case_ids(test_cases))
def test_matmul_v3(profiler, runs, is_static, core_num, tiling_data, dtype, a_transp, b_transp, full_load_mode,
                   has_bias, double_buffering):
    distrib_mode = asctile.DistribMode.SplitByM if dtype == torch.float32 else None
    run_matmul_v3_test(profiler, runs, is_static, core_num, tiling_data, dtype, a_transp, b_transp, full_load_mode,
                       has_bias, double_buffering, distrib_mode, l0c2ub=True)
