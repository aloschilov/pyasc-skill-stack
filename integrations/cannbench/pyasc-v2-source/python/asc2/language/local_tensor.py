# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import functools
import inspect
import math
from typing import Callable, Final, Optional, Tuple, TypeVar, Union, overload
from typing_extensions import Self, TypeAlias

from asc._C import ir
from asc.language.core.dtype import DataType
from asc.language.core.ir_value import IRHandle, IRValue, PlainValue, RuntimeInt, RuntimeNumeric
from asc.language.core.utils import DefaultValued, OverloadDispatcher, allow_jit, jit_allowed

from .tensor_location import TensorLocation, TensorLocLike
from .validation import check_type

T = TypeVar("T")

RoundMode: TypeAlias = ir.asctile_RoundMode


class LocalTensor(IRValue):
    """
    A local tensor is a multi-dimensional array of values in local memory (Unified Buffer, L1 Cache, etc.)

    Each element is of :py:attr:`dtype` type and number of elements is defined by :py:attr:`shape` tuple.

    .. rubric:: Special methods

    .. automethod:: to
    """

    dtype: DataType
    """Tensor element type"""

    shape: Tuple[int, ...]
    """Tensor shape"""

    size: int
    """Number of elements"""

    rank: int
    """Number of dimensions"""

    location: TensorLocation
    """Memory location of a tensor"""

    def __init__(self, handle: IRHandle) -> None:
        """
        This constructor is not called by user.

        Use :py:func:`copy_in`, :py:func:`zeros`, or other functions to create a local tensor.
        """
        super().__init__()
        check_type("handle", handle, IRHandle)
        self.handle: Final = handle
        ir_type = handle.get_type()
        self.dtype: Final = DataType.from_ir(ir.get_element_type(ir_type))
        self.shape: Final = tuple(ir.get_shape(ir_type))
        if len(self.shape) < 1:
            raise RuntimeError("Tensor shape must have at least one dimension")
        self.size: Final = math.prod(self.shape)
        self.rank: Final = len(self.shape)
        self.location: Final = ir.get_tensor_location(ir_type)

    @classmethod
    def from_ir(cls, handle: IRHandle) -> Self:
        return cls(handle=handle)

    def to_ir(self) -> IRHandle:
        return self.handle

    @overload
    def to(self, dtype: DataType, round_mode: RoundMode = RoundMode.Default) -> Self:
        ...

    @overload
    def to(self, location: TensorLocLike) -> Self:
        ...

    def to(self, *args, **kwargs) -> Self:
        """Transforms data type (see :py:func:`cast`) or location (see :py:func:`copy`) of a tensor."""

        dispatcher = OverloadDispatcher("asc2.LocalTensor.to")

        @dispatcher.register(dtype=DataType, round_mode=DefaultValued(RoundMode, RoundMode.Default))
        def to_dtype(dtype, round_mode):
            from .creation_ops import cast
            return cast(self, dtype, round_mode)

        @dispatcher.register(location=Union[TensorLocation, str])
        def to_location(location):
            if location == self.location:
                return self
            from .memory_ops import copy
            return copy(self, location=location)

        return dispatcher(*args, **kwargs)

    @property
    def T(self) -> Self:
        """Transpose a 2D tensor by swapping its dimensions (see :py:func:`transpose`)."""
        from .shape_ops import transpose
        return transpose(self)

    # Binary operations

    def __add__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __sub__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __mul__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __truediv__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __floordiv__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        return self / other

    def __and__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __or__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __xor__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __lshift__(self, other: RuntimeInt) -> Self:
        ...

    def __rshift__(self, other: RuntimeInt) -> Self:
        ...

    def __matmul__(self, other: Self) -> Self:
        ...

    # Binary operations (reversed)

    def __radd__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        from .binary_ops import add
        return add(other, self)

    def __rsub__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        from .binary_ops import sub
        return sub(other, self)

    def __rmul__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        from .binary_ops import mul
        return mul(other, self)

    def __rtruediv__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        from .binary_ops import div
        return div(other, self)

    def __rfloordiv__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        return other / self

    def __rand__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        from .binary_ops import bitwise_and
        return bitwise_and(other, self)

    def __ror__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        from .binary_ops import bitwise_or
        return bitwise_or(other, self)

    def __rxor__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        from .binary_ops import bitwise_xor
        return bitwise_xor(other, self)

    def __rmatmul__(self, other: Self) -> Self:
        from .binary_ops import matmul
        return matmul(other, self)

    # Comparison operations

    def __eq__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __ne__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __gt__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __ge__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __lt__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    def __le__(self, other: Union[Self, RuntimeNumeric]) -> Self:
        ...

    # Reduction operations

    @overload
    def sum(self, *dims: int, keep_dims: bool = False) -> Self:
        ...

    @overload
    def sum(self) -> PlainValue:
        ...

    @overload
    def max(self, *dims: int, keep_dims: bool = False) -> Self:
        ...

    @overload
    def max(self) -> PlainValue:
        ...

    @overload
    def min(self, *dims: int, keep_dims: bool = False) -> Self:
        ...

    @overload
    def min(self) -> PlainValue:
        ...

    def prod(self, *dims: int, keep_dims: bool = False) -> Self:
        ...

    # Unary operations

    def __neg__(self) -> Self:
        ...

    def __pos__(self) -> Self:
        return self

    def __invert__(self) -> Self:
        ...

    def __abs__(self) -> Self:
        return self.abs()

    def sin(self) -> Self:
        ...

    def cos(self) -> Self:
        ...

    def tan(self) -> Self:
        ...

    def sinh(self) -> Self:
        ...

    def cosh(self) -> Self:
        ...

    def tanh(self) -> Self:
        ...

    def exp(self) -> Self:
        ...

    def log(self) -> Self:
        ...

    def log2(self) -> Self:
        ...

    def floor(self) -> Self:
        ...

    def ceil(self) -> Self:
        ...

    def abs(self) -> Self:
        ...

    def erf(self) -> Self:
        ...

    def exp2(self) -> Self:
        ...

    def sqrt(self) -> Self:
        ...

    def rsqrt(self) -> Self:
        ...

    def relu(self) -> Self:
        ...

    # Shape operations

    def broadcast_to(self, *shape: int) -> Self:
        ...

    def reshape(self, *shape: int) -> Self:
        ...

    def ravel(self) -> Self:
        ...

    def expand_dims(self, *axis: int) -> Self:
        ...

    def squeeze(self, *axis: int) -> Self:
        ...

    def transpose(self, *axis: int) -> Self:
        ...


class BinaryOperandTypeError(TypeError):
    """Exception for dunder methods implementing binary operators"""
    pass


class Binder:

    def __init__(self, name: Optional[str] = None, binary_op: Optional[str] = None, unary_op: Optional[str] = None):
        self.name = name
        self.binary_op = binary_op
        self.unary_op = unary_op

    def __call__(self, fn: T) -> T:
        fn_name = fn.__name__
        name = self.name or fn_name
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) < 1:
            raise ValueError("Bound function must have at least one parameter")
        if not fn.__doc__:
            fn.__doc__ = ""
        if self.binary_op:
            call_kind = "via a binary operator"
            call_func = f"{fn_name}(input, other)"
            call_alias = f"input {self.binary_op} other"
        elif self.unary_op:
            call_kind = "via an unary operator"
            call_func = f"{fn_name}(input)"
            call_alias = f"{self.unary_op}input"
        else:
            call_kind = "as a member function"
            call_func = f"{fn_name}(input, ...)"
            call_alias = f"input.{name}(...)"
        fn.__doc__ += f"""
    This function can also be called {call_kind} on :py:class:`LocalTensor`,
    as ``{call_alias}`` instead of ``{call_func}``.
        """
        params[0] = params[0].replace(name="self")
        new_sig = sig.replace(parameters=params)

        if self.binary_op:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except BinaryOperandTypeError:
                    return NotImplemented

        else:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

        wrapper.__signature__ = new_sig
        wrapper.__doc__ = f"Forwards to :py:func:`{fn_name}` function."
        setattr(LocalTensor, name, wrapper)
        if jit_allowed(fn):
            allow_jit(wrapper)
        return fn


@overload
def bind_tensor_method(fn: T) -> T:
    ...


@overload
def bind_tensor_method(name: str, binary_op: Optional[str] = None, unary_op: Optional[str] = None) -> Callable[[T], T]:
    ...


def bind_tensor_method(fn: Optional[T] = None, *, name: Optional[str] = None, binary_op: Optional[str] = None,
                       unary_op: Optional[str] = None):
    binder = Binder(name, binary_op, unary_op)
    if fn is None:
        return binder
    return binder(fn)
