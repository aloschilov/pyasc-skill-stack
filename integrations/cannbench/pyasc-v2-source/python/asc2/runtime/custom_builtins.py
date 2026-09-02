# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
import functools
import itertools
import operator
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

from asc.codegen.function_visitor import CustomBuiltins
from asc.language.core.ir_value import PlainValue, RuntimeNumeric
from asc.language.core.utils import static_assert

from ..language.binary_ops import maximum, minimum
from ..language.range import range as custom_range
from ..language.local_tensor import LocalTensor


class SplitAccumulation:

    @dataclass
    class SplitResult:
        guard: Callable[[Any], bool]
        accum: Callable[[Any, Any], Any]
        value: Any

    def __init__(self, *guard_acc_fns: Tuple[Callable[[Any], bool], Callable[[Any, Any], Any]]):
        self.sentinel = object()
        self.results: List[__class__.SplitResult] = []
        for guard, accum in guard_acc_fns:
            self.results.append(self.SplitResult(guard, accum, self.sentinel))

    def append(self, value: Any) -> None:
        for result in self.results:
            if result.guard(value):
                result.value = value if result.value is self.sentinel else result.accum(result.value, value)
                return
        raise ValueError(f"{value!r} doesn't match any guard")

    def reduce(self, **kwargs) -> Any:
        defined_results: List[__class__.SplitResult] = []
        for result in self.results:
            if result.value is not self.sentinel:
                defined_results.append(result)
        if len(defined_results) == 0:
            default = "default"
            if default in kwargs:
                return kwargs[default]
            fn_name = kwargs.get("name", "...")
            raise ValueError(f"{fn_name}() arg is an empty sequence")
        if len(defined_results) == 1:
            return defined_results[0].value
        result = defined_results[-1].value
        for curr in itertools.islice(reversed(defined_results), 1, None, 1):
            result = curr.accum(curr.value, result)
        return result


def custom_accumulator(iterable: Iterable, *, ir_tile_fn: Callable[..., LocalTensor],
                       ir_scalar_fn: Callable[..., PlainValue], builtin_fn: Callable, **kwargs) -> Any:
    split_acc = SplitAccumulation(
        (lambda arg: isinstance(arg, LocalTensor), ir_tile_fn),
        (lambda arg: isinstance(arg, PlainValue), ir_scalar_fn),
        (lambda _: True, builtin_fn),
    )
    key = kwargs.get("key", lambda arg: arg)
    for arg in iterable:
        split_acc.append(key(arg))
    kwargs.setdefault("name", builtin_fn.__name__)
    return split_acc.reduce(**kwargs)


def first_or_all(args: tuple) -> Iterable:
    return args[0] if len(args) == 1 and isinstance(args[0], Iterable) else args


@functools.wraps(max)
def custom_max(*args: Any, **kwargs) -> Any:
    return custom_accumulator(first_or_all(args), ir_tile_fn=maximum, ir_scalar_fn=PlainValue.max, builtin_fn=max,
                              **kwargs)


@functools.wraps(min)
def custom_min(*args: Any, **kwargs) -> Any:
    return custom_accumulator(first_or_all(args), ir_tile_fn=minimum, ir_scalar_fn=PlainValue.min, builtin_fn=min,
                              **kwargs)


@functools.wraps(sum)
def custom_sum(iterable: Iterable, /, start: Union[LocalTensor, RuntimeNumeric] = 0) -> Any:
    return custom_accumulator(itertools.chain((start, ), iterable), ir_tile_fn=operator.add, ir_scalar_fn=operator.add,
                              builtin_fn=operator.add)


def custom_assert(test: bool, message: Optional[str] = None) -> None:
    static_assert(test, message)


def get_custom_builtins() -> CustomBuiltins:
    return CustomBuiltins({
        "assert": custom_assert,
        "max": custom_max,
        "min": custom_min,
        "range": custom_range,
        "sum": custom_sum,
    })
