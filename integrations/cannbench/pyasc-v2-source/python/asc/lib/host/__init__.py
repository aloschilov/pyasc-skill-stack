# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from .wrappers import (
    BatchMatmulTiling,
    CubeFormat,
    DataType,
    DequantType,
    MatmulApiTiling,
    MatmulConfigParams,
    MatrixMadType,
    MatrixTraverse,
    MultiCoreMatmulTiling,
    PlatformAscendC,
    PlatformAscendCManager,
    ScheduleType,
    TPosition,
)

from ..runtime.interface import get_soc_version


def get_ascendc_platform():
    soc_version = get_soc_version()
    return PlatformAscendCManager.get_instance(soc_version.value)


__all__ = [
    "BatchMatmulTiling",
    "CubeFormat",
    "DataType",
    "DequantType",
    "MatmulApiTiling",
    "MatmulConfigParams",
    "MatrixMadType",
    "MatrixTraverse",
    "MultiCoreMatmulTiling",
    "PlatformAscendC",
    "PlatformAscendCManager",
    "ScheduleType",
    "TPosition",
    "get_ascendc_platform",
]
