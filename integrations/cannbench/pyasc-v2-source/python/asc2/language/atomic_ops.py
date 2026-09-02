# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable

from asc._C import ir
from asc.language.core.dtype import KnownTypes as KT
from asc.language.core.ir_value import RuntimeInt, materialize_ir_value as _mat
from asc.language.core.utils import global_builder, require_jit

from .global_tensor import GlobalTensor
from .local_tensor import LocalTensor
from .tensor_location import TensorLocation
from .utils import cast_tensor_location as cast_loc
from .validation import check_dtype, check_type, verify_runtime_ints


def op_atomic_impl(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[RuntimeInt], kind: ir.AtomicKind) -> None:
    check_type("src", src, LocalTensor)
    check_type("dst", dst, GlobalTensor)
    check_dtype("dst", dst, (KT.int16, KT.int32, KT.float16, KT.bfloat16, KT.float32))
    check_dtype("src", src, (dst.dtype, ))
    src = cast_loc(src, TensorLocation.UB)
    offsets = [_mat(v, KT.int32).to_ir() for v in verify_runtime_ints(offsets, "offsets")]
    global_builder.get_ir_builder().create_asctile_AtomicRMWOp(src.to_ir(), dst.to_ir(), offsets, kind)


@require_jit
def atomic_add(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[RuntimeInt]) -> None:
    """
    Atomically add local tensor elements to a global tensor at specified offsets.

    Performs an atomic read-modify-write operation, adding each element of ``src`` to the corresponding element in
    ``dst`` at the given ``offsets``.

    The supported data types for the inputs are: ``int16``, ``int32``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        src: The source local tensor whose elements will be added
        dst: The destination global tensor
        offsets: The offsets into the global tensor for each dimension

    Raises:
        TypeError: If src is not a LocalTensor, dst is not a GlobalTensor, or offsets contains non-integer values
        RuntimeError: If src or dst dtype is not supported, or offsets is empty

    Examples:
        Atomically add local tensor elements to a global tensor: ::

            src = asc2.copy_in(x_gm, [0], [128])
            asc2.atomic_add(src, out_gm, [0])
    """
    return op_atomic_impl(src, dst, offsets, ir.AtomicKind.Add)


@require_jit
def atomic_max(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[RuntimeInt]) -> None:
    """
    Atomically compute the maximum between local tensor elements and global tensor elements at specified offsets.

    Performs an atomic read-modify-write operation, storing the maximum of each element in ``src`` and the
    corresponding element in ``dst`` at the given ``offsets``.

    The supported data types for the inputs are: ``int16``, ``int32``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        src: The source local tensor containing comparison values
        dst: The destination global tensor
        offsets: The offsets into the global tensor for each dimension

    Raises:
        TypeError: If src is not a LocalTensor, dst is not a GlobalTensor, or offsets contains non-integer values
        RuntimeError: If src or dst dtype is not supported, or offsets is empty

    Examples:
        Atomically compute the maximum between local tensor and global tensor elements: ::

            src = asc2.copy_in(x_gm, [0], [128])
            asc2.atomic_max(src, out_gm, [0])
    """
    return op_atomic_impl(src, dst, offsets, ir.AtomicKind.Max)


@require_jit
def atomic_min(src: LocalTensor, dst: GlobalTensor, offsets: Iterable[RuntimeInt]) -> None:
    """
    Atomically compute the minimum between local tensor elements and global tensor elements at specified offsets.

    Performs an atomic read-modify-write operation, storing the minimum of each element in ``src`` and the
    corresponding element in ``dst`` at the given ``offsets``.

    The supported data types for the inputs are: ``int16``, ``int32``, ``float16``, ``bfloat16``, ``float32``.

    Args:
        src: The source local tensor containing comparison values
        dst: The destination global tensor
        offsets: The offsets into the global tensor for each dimension

    Raises:
        TypeError: If src is not a LocalTensor, dst is not a GlobalTensor, or offsets contains non-integer values
        RuntimeError: If src or dst dtype is not supported, or offsets is empty

    Examples:
        Atomically compute the minimum between local tensor and global tensor elements: ::

            src = asc2.copy_in(x_gm, [0], [128])
            asc2.atomic_min(src, out_gm, [0])
    """
    return op_atomic_impl(src, dst, offsets, ir.AtomicKind.Min)
