# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any, Iterable, Optional, Protocol, Tuple, Type, TypeGuard, Union

from asc.common.compat import isinstance
from asc.language.core.dtype import DataType
from asc.language.core.ir_value import PlainValue, RuntimeInt
from asc.language.core.utils import get_type_name

from .tensor_location import TensorLocation


class DataTyped(Protocol):
    dtype: DataType


def check_type(name: str, value: Any, constraint: Union[Type, Tuple[Type, ...]],
               exc_type: Type[Exception] = TypeError) -> None:
    if isinstance(value, constraint):
        return
    raise exc_type(f"'{name}' argument must be {get_type_name(constraint)}, got {value.__class__.__name__}")


def check_dtype(name: str, value: Union[DataType, DataTyped], dtypes: Union[DataType, Tuple[DataType, ...]],
                exc_type: Type[Exception] = RuntimeError, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    dtype = value if isinstance(value, DataType) else value.dtype
    dtypes = (dtypes, ) if isinstance(dtypes, DataType) else dtypes
    if dtype not in dtypes:
        dtypes_str = ", ".join(map(str, dtypes))
        if len(dtypes) > 1:
            dtypes_str = f"one of {dtypes_str}"
        raise exc_type(f"'{name}' dtype must be {dtypes_str}, got {dtype}")


def is_runtime_int(value: Any) -> TypeGuard[RuntimeInt]:
    return isinstance(value, int) or isinstance(value, PlainValue) and value.dtype.is_signed()


def check_runtime_int(name: str, value: Any, exc_type: Type[Exception] = TypeError) -> None:
    if not is_runtime_int(value):
        raise exc_type(f"'{name}' argument must be int or integer PlainValue, got {value.__class__.__name__}")


def check_runtime_float(name: str, value: Any, exc_type: Type[Exception] = TypeError) -> None:
    if isinstance(value, float) or isinstance(value, PlainValue) and value.dtype.is_float():
        return
    raise exc_type(f"'{name}' argument must be float or PlainValue with float dtype, got {value.__class__.__name__}")


def iterable_to_non_empty_tuple(iterable: Any, name: str, size: Optional[int] = None) -> tuple:
    if not isinstance(iterable, Iterable):
        raise TypeError(f"'{name}' must be Iterable")
    values = iterable if isinstance(iterable, tuple) else tuple(iterable)
    act_size = len(values)
    if act_size < 1:
        raise RuntimeError(f"'{name}' must have at least one value")
    if size is not None and act_size != size:
        raise ValueError(f"'{name}' must have {size} values, got {act_size}")
    return values


def verify_runtime_ints(values: Iterable[Any], name: str, size: Optional[int] = None) -> Tuple[RuntimeInt, ...]:
    values = iterable_to_non_empty_tuple(values, name, size)
    if not all(is_runtime_int(value) for value in values):
        raise TypeError(f"All values in '{name}' must be int or integer PlainValue")
    return values


def verify_shape(shape: Iterable[int], name: str = "shape", size: Optional[int] = None) -> Tuple[int, ...]:
    shape = iterable_to_non_empty_tuple(shape, name, size)
    if not all(isinstance(dim, int) for dim in shape):
        raise TypeError(f"All values in '{name}' must be integers")
    if any(dim <= 0 for dim in shape):
        raise RuntimeError(f"All values in '{name}' must be positive")
    return shape


def verify_location(location: Any, name: str = "location",
                    allow: Optional[Union[TensorLocation, Tuple[TensorLocation, ...]]] = None) -> TensorLocation:
    check_type(name, location, (str, TensorLocation))
    location = TensorLocation(location)
    if allow is None:
        return location
    allow = allow if isinstance(allow, tuple) else (allow, )
    if location == TensorLocation.Auto or location in allow:
        return location
    loc_str = " or ".join(loc.name for loc in allow)
    raise RuntimeError(f"'{name}' tensor location must be {loc_str}, got {location.name}")
