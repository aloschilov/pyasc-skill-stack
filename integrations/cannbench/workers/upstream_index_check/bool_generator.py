#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations
import torch

def _gen_bool(self, shape: List[int], min_val: Any, max_val: Any,
              generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """生成布尔张量

    bool 既不属于 is_float_dtype 也不属于 is_int_dtype，此前会落到 zeros 分支，
    导致所有 bool 输入被静默生成为全 False、value_range 完全失效（例如
    MoeGatingTopKSoftmax 的 finished 恒为 False，"已完成 token" 分支从未被覆盖）。

    torch.randint 不接受 dtype=torch.bool，故先按 int64 采样再转 bool。
    value_range 被夹到 {0, 1}：[0, 1] → 随机；[0, 0] → 全 False；[1, 1] → 全 True。
    未指定 value_range 时 _parse_range 返回 (0, 100)，夹取后等价于 [0, 1] 随机。

    Args:
        shape: 张量形状
        min_val: 最小值（夹到 0/1）
        max_val: 最大值（夹到 0/1）
        generator: torch.Generator 用于确定性随机数生成（可选）
    """
    lo = min(1, max(0, int(min_val)))
    hi = min(1, max(0, int(max_val)))
    if lo > hi:
        lo, hi = hi, lo

    if lo == hi:
        return torch.full(shape, bool(lo), dtype=torch.bool)
    return torch.randint(
        lo, hi + 1, shape, generator=generator, dtype=torch.int64,
    ).to(torch.bool)
