# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple
from typing_extensions import Self

from .._C import ir
from ..lib.runtime import CoreType, Function


@dataclass(frozen=True)
class KernelMeta:
    core_type: CoreType = CoreType.VectorCore
    kernel_args: Optional[Tuple[ir.KernelArgument]] = None
    enable_debug: bool = False
    memory_consumed: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class CompiledKernel:
    binary: bytes
    meta: KernelMeta


@dataclass(frozen=True)
class LaunchedKernel(CompiledKernel):
    handle: Function

    @classmethod
    def from_compiled(cls, kernel: CompiledKernel, handle: Function) -> Self:
        return cls(**{f.name: getattr(kernel, f.name) for f in fields(kernel)}, handle=handle)
