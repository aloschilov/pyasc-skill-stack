# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from typing import Any, Generic, Optional, Type, TypeVar, Union, get_args, overload
from typing_extensions import TypeAlias

from ...common.compat import isinstance

Numeric: TypeAlias = Union[bool, int, float]
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class ConstExpr(Generic[T_co]):
    value: T_co

    @overload
    def __init__(self, value: T_co):
        ...

    @overload
    def __init__(self, value: ConstExpr):
        ...

    def __init__(self, value):
        self.value = self.unwrap(value)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"ConstExpr[{self.value.__class__.__name__}]({self.value!r})"

    def __format__(self, format_spec: str) -> str:
        return ("{:%s}" % (format_spec, )).format(self.value)

    @overload
    @staticmethod
    def unwrap(obj: ConstExpr[T]) -> T:
        ...

    @overload
    @staticmethod
    def unwrap(obj: T) -> T:
        ...

    @staticmethod
    def unwrap(obj):
        return obj.value if isinstance(obj, ConstExpr) else obj


def require_constexpr(obj: Union[ConstExpr, Any], *constraints: Type, arg_name: Optional[str] = None) -> None:
    if len(constraints) == 0:
        raise RuntimeError("Constraints must be provided for ConstExpr requirement")
    if isinstance(obj, constraints):
        return
    if isinstance(obj, ConstExpr):
        require_constexpr(obj.value, *constraints, arg_name=arg_name)
        return
    constraint_union = constraints[0]
    for constraint in constraints:
        constraint_union = Union[constraint_union, constraint]
    constraint_str = " | ".join(map(lambda t: t.__name__, get_args(constraint_union)))
    note = "" if arg_name is None else f" for '{arg_name}' argument"
    raise RuntimeError(f"ConstExpr[{constraint_str}] is required{note}, {obj.__class__.__name__} was provided: {obj}")
