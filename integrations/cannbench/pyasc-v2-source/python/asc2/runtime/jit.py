# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Callable, Optional, TypeVar, overload
from typing_extensions import ParamSpec

from asc.runtime.jit import JITFunction as JITFunctionBase

from .compiler import Compiler
from .custom_builtins import get_custom_builtins

P = ParamSpec("P")
T = TypeVar("T")


class JITFunction(JITFunctionBase[P, T]):
    compiler = Compiler


@overload
def jit(fn: Callable[P, T]) -> JITFunction[P, T]:
    ...


@overload
def jit(**options) -> Callable[[Callable[P, T]], JITFunction[P, T]]:
    ...


def jit(fn: Optional[Callable[P, T]] = None, **options):
    """
    Instantiate a JIT function using the default options.
    This should be used as a decorator for the kernel function:

    .. code-block:: python

        @asc2.jit
        def kernel(x, y):
            ...

    JIT options may be provided as keyword arguments to be applied to the decorated kernel function.
    See :py:obj:`asc.CodegenOptions`, :py:obj:`asc2.CompileOptions`, :py:obj:`asc.LaunchOptions` for the details.
    """
    options.setdefault("custom_builtins", get_custom_builtins())

    def decorator(fn: Callable[P, T]) -> JITFunction[P, T]:
        return JITFunction(fn, **options)

    if fn is None:
        return decorator
    return decorator(fn)
