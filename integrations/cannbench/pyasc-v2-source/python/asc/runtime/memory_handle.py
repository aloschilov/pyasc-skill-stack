# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import abc
import ctypes
from typing import Protocol, Union
from typing_extensions import Self
import numpy

from ..lib import runtime as rt


class TorchTensor(Protocol):
    nbytes: int

    def data_ptr(self) -> int:
        ...

    def ravel(self) -> Self:
        ...


class MemoryHandle(abc.ABC):

    @abc.abstractmethod
    def copy_to_device(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def copy_from_device(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def release_memory(self) -> None:
        raise NotImplementedError


class ByteArrayHandle(MemoryHandle):

    def __init__(self, data: Union[bytearray, bytes]):
        self.data = data if isinstance(data, bytearray) else bytearray(data)
        self.buffer_ptr = ctypes.cast(ctypes.pointer(ctypes.c_char.from_buffer(self.data)), ctypes.c_void_p)

    def copy_to_device(self) -> int:
        self.handle = rt.copy_data_to_device(self.buffer_ptr, len(self.data))
        return int(self.handle.value)

    def copy_from_device(self) -> None:
        rt.copy_data_from_device(self.buffer_ptr, self.handle, len(self.data))

    def release_memory(self) -> None:
        rt.free(self.handle)


class NumpyArrayHandle(MemoryHandle):

    def __init__(self, array: numpy.ndarray):
        self.array = array

    def copy_to_device(self) -> int:
        flat = self.array.ravel(order="C")
        self.handle = rt.copy_data_to_device(flat.ctypes.data_as(ctypes.c_void_p), flat.nbytes)
        return int(self.handle.value)

    def copy_from_device(self) -> None:
        rt.copy_data_from_device(self.array.ctypes.data_as(ctypes.c_void_p), self.handle, self.array.nbytes)

    def release_memory(self) -> None:
        rt.free(self.handle)


class TorchCpuTensorHandle(MemoryHandle):

    def __init__(self, tensor: TorchTensor):
        self.tensor = tensor

    def copy_to_device(self) -> int:
        flat = self.tensor.ravel()
        self.handle = rt.copy_data_to_device(ctypes.c_void_p(flat.data_ptr()), flat.nbytes)
        return self.handle.value

    def copy_from_device(self) -> None:
        rt.copy_data_from_device(ctypes.c_void_p(self.tensor.data_ptr()), self.handle, self.tensor.nbytes)

    def release_memory(self) -> None:
        rt.free(self.handle)


class TorchNpuTensorArgument(MemoryHandle):

    def __init__(self, tensor: TorchTensor):
        self.tensor = tensor

    def copy_to_device(self) -> int:
        return self.tensor.data_ptr()

    def copy_from_device(self) -> None:
        pass

    def release_memory(self) -> None:
        pass


def resolve_memory_handle(obj) -> MemoryHandle:
    if isinstance(obj, MemoryHandle):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return ByteArrayHandle(obj)
    if isinstance(obj, numpy.ndarray):
        return NumpyArrayHandle(obj)
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if getattr(obj, "is_cpu", False):
                return TorchCpuTensorHandle(obj)
            if getattr(obj, "is_npu", False):
                return TorchNpuTensorArgument(obj)
    except ModuleNotFoundError:
        pass
    raise RuntimeError(f"Unsupported memory handle of type {obj.__class__.__name__}")
