# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import overload

from asc._C import ir
from asc.language.core.range import BaseRange, static_range as asc_static_range
from asc.language.core.utils import global_builder

from .validation import check_type


class range(BaseRange):
    """
    A range loop construct for use in JIT functions.

    This class provides a range-based loop similar to Python's built-in ``range``, with additional support for loop
    unrolling and memory barrier control on NPU.

    In fact, built-in ``range`` automatically becomes ``asc2.range`` when used inside JIT function.

    Args:
        start: Start index of the loop (or stop index if only one argument provided).
        stop: Stop index of the loop (exclusive). If None, start is treated as stop and start is 0.
        step: Step size for the loop iteration.
        unroll_factor: Number of iterations to unroll. Default is 1 (no unrolling).
        gm_barrier: Whether to prevent parallel load/store optimization across iterations. Default is False.
            When True, a global memory barrier is inserted between iterations to prevent overlapping memory
            operations. Must be True if loop iterations depend on previous iterations' memory writes.

    Raises:
        ValueError: If ``unroll_factor`` is less than 1

    Note:
        The ``unroll_factor`` parameter controls loop unrolling during compilation.
        Higher values can improve performance but increase code size.

    Examples:
        Basic loop from 0 to N: ::

            for i in asc2.range(N):
                ...

        Loop with unrolling: ::

            for i in asc2.range(0, N, 1, unroll_factor=4):
                ...

        Loop with a global memory barrier (prevent parallel memory optimizations): ::

            for i in asc2.range(N, gm_barrier=True):
                ...
    """

    @overload
    def __init__(self, stop: int, /, *, unroll_factor: int = 1, gm_barrier: bool = False):
        ...

    @overload
    def __init__(self, start: int, stop: int, /, *, unroll_factor: int = 1, gm_barrier: bool = False):
        ...

    @overload
    def __init__(self, start: int, stop: int, step: int, /, *, unroll_factor: int = 1, gm_barrier: bool = False):
        ...

    def __init__(self, *args, unroll_factor: int = 1, gm_barrier: bool = False):
        check_type("unroll_factor", unroll_factor, int)
        check_type("gm_barrier", gm_barrier, bool)
        if unroll_factor < 1:
            raise ValueError(f"Loop unroll factor must be 1 or greater, got {unroll_factor}")
        super().__init__(*args)
        self.unroll_factor = unroll_factor
        self.gm_barrier = gm_barrier

    def handle_op(self, op: ir.ForOp) -> None:
        builder = global_builder.get_ir_builder()
        op.set_attr(ir.attr.unroll_factor, builder.get_index_attr(self.unroll_factor))
        if self.gm_barrier:
            op.set_attr(ir.attr.gm_barrier, builder.get_unit_attr())


class static_range(asc_static_range):
    """
    A static range loop construct for use in JIT functions.

    Unlike :py:class:`range`, this class requires all loop bounds to be compile-time constants (integers), not runtime
    values. This enables more aggressive compile-time optimizations such as complete loop unrolling.

    Args:
        start: Start index of the loop (or stop index if only one argument provided)
        stop: Stop index of the loop (exclusive). If None, start is treated as stop and start is 0.
        step: Step size for the loop iteration

    Raises:
        ValueError: If number of arguments is not between 1 and 3

    Note:
        All arguments must be integer constants, not runtime values.
        Use :py:class:`range` when loop bounds are runtime-dependent.

    Examples:
        Loop with compile-time constant bounds: ::

            for i in asc2.static_range(0, 128):
                ...
    """
