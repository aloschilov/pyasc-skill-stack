# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Literal, Union
from typing_extensions import TypeAlias

from asc._C import ir

TensorLocation: TypeAlias = ir.asctile_TensorLocation
TensorLocation.__name__ = "TensorLocation"

TensorLocStr: TypeAlias = Literal["BT", "FIX", "L0A", "L0B", "L0C", "L1", "UB"]
TensorLocLike: TypeAlias = Union[TensorLocation, TensorLocStr]
