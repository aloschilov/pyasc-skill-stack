# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import ctypes
from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from typing_extensions import TypeAlias

import numpy as np

from . import utils
from .._C import ir
from ..language.core.struct import Struct
from ..lib import runtime as rt
from .config import Platform
from .kernel_meta import CompiledKernel, LaunchedKernel
from .memory_handle import MemoryHandle, resolve_memory_handle

KernelCallback: TypeAlias = Callable[[rt.Function], None]


class MsprofLauncher(object):

    def __init__(self, is_model_: bool):
        self.is_model = is_model_
        self.utils = rt.npu_utils()
        self.start_time = 0

    def start(self):
        if self.is_model:
            return

        self.start_time = self.utils.msprof_sys_cycle_time()

    def process(self, kernel_name: str, block_num: int, task_type: int):
        if self.is_model:
            return

        time_stamp = self.utils.msprof_sys_cycle_time()
        self.utils.msprof_report_compact_info(time_stamp, kernel_name, block_num, task_type)

        end_time = self.utils.msprof_sys_cycle_time()
        self.utils.msprof_report_api(self.start_time, end_time, kernel_name)


@dataclass(frozen=True)
class LaunchOptions:
    """
    Kernel launch and device runtime options.

    These options can also be used as positional arguments in ``[`` brackets ``]`` when launch JIT function:

    .. code-block:: python

        @asc.jit
        def kernel(x_ptr, y_ptr):
            ...

        def launch(x, y, core_num = 16):
            kernel[core_num](x, y)
    """

    core_num: Optional[int] = None
    """
    Number of active execution blocks (AI cores) used to launch the kernel.
    By default, all cores available on the current platform will be used.
    """

    stream: Optional[rt.Stream] = None


@dataclass(frozen=True)
class PlatformInfo:
    ub_size: int
    l1_size: int
    l0a_size: int
    l0b_size: int
    l0c_size: int
    bt_size: int


def get_platform_info(platform: Platform) -> PlatformInfo:
    info_910b = PlatformInfo(ub_size=192 * 1024, l1_size=512 * 1024, l0a_size=64 * 1024, l0b_size=64 * 1024,
                             l0c_size=128 * 1024, bt_size=1024)
    if platform in (Platform.Ascend910B1, Platform.Ascend910B2, Platform.Ascend910B2C, Platform.Ascend910B3,
                    Platform.Ascend910B4, Platform.Ascend910B4_1):
        return info_910b
    info_910_93 = dataclass_replace(info_910b, l0c_size=256 * 1024)
    if platform in (Platform.Ascend910_9362, Platform.Ascend910_9372, Platform.Ascend910_9381, Platform.Ascend910_9382,
                    Platform.Ascend910_9391, Platform.Ascend910_9392):
        return info_910_93
    if platform in (Platform.Ascend950PR_950z, Platform.Ascend950PR_9579, Platform.Ascend950PR_957b,
                    Platform.Ascend950PR_957c, Platform.Ascend950PR_957d, Platform.Ascend950PR_9589,
                    Platform.Ascend950PR_958b, Platform.Ascend950PR_9599):
        return dataclass_replace(info_910_93, ub_size=248 * 1024, bt_size=4 * 1024)
    raise ValueError(f"Unknown platform: {platform}")


class Launcher:
    options_cls: Type[LaunchOptions] = LaunchOptions

    def __init__(self, options: LaunchOptions):
        self.options = options
        core_num = self.options.core_num
        if core_num is not None and core_num <= 0:
            raise ValueError("'core_num' must be positive")

    @staticmethod
    def is_torch_scalar(value: Any) -> bool:
        try:
            import torch
            return isinstance(value, torch.Tensor) and value.dim() == 0
        except ModuleNotFoundError:
            return False

    @staticmethod
    def scalar_to_bytes(value: Any) -> Optional[bytes]:
        if isinstance(value, np.generic):
            return value.tobytes()
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return bytes(value.view(value.numel()).view(torch.uint8))
        except ModuleNotFoundError:
            pass
        return None

    @staticmethod
    def get_core_num() -> int:
        return rt.device_info(rt.DeviceModuleType.RT_MODULE_TYPE_AICORE, rt.DeviceInfoType.INFO_TYPE_CORE_NUM)

    @staticmethod
    def check_memory_overflow(memory_consumed: Dict[str, int]) -> None:
        platform_info = get_platform_info(rt.get_soc_version())
        key_to_attr = (
            ("UB", "ub_size"),
            ("L1", "l1_size"),
            ("L0A", "l0a_size"),
            ("L0B", "l0b_size"),
            ("L0C", "l0c_size"),
            ("BT", "bt_size"),
        )
        for key, attr in key_to_attr:
            consumed = memory_consumed.get(key, 0)
            capacity = getattr(platform_info, attr)
            if consumed > capacity:
                raise RuntimeError(f"{key} overflow: {capacity} bytes are available, {consumed} bytes are used.")

    @classmethod
    def expand_kernel_args(cls, args: Iterable[Any]) -> List[Union[np.generic, MemoryHandle]]:
        kernel_args = []
        for arg in args:
            if isinstance(arg, int):
                kernel_args.append(np.int32(arg))
            elif isinstance(arg, float):
                kernel_args.append(np.float32(arg))
            elif isinstance(arg, bool):
                kernel_args.append(np.int8(int(arg)))
            elif isinstance(arg, np.generic) or cls.is_torch_scalar(arg):
                kernel_args.append(arg)
            elif isinstance(arg, Struct):
                kernel_args.append(resolve_memory_handle(arg.pack()))
            else:
                kernel_args.append(resolve_memory_handle(arg))
        return kernel_args

    def launch_kernel(self, function: rt.Function, kernel_args: List[Union[np.generic, MemoryHandle]],
                      enable_debug: bool, func_name: str) -> None:

        def blobs_size(input_blobs: List[bytes]) -> int:
            return sum(len(x) for x in input_blobs)

        input_blobs: List[bytes] = []
        memory_args: List[MemoryHandle] = []
        for arg in kernel_args:
            scalar_bytes = self.scalar_to_bytes(arg)
            if scalar_bytes is not None:
                input_blobs.append(scalar_bytes)
                item_size = len(scalar_bytes)
                if item_size < 4:
                    input_blobs.append(b"\0" * (4 - item_size))
                elif item_size > 4 and item_size < 8:
                    input_blobs.append(b"\0" * (8 - item_size))
            elif isinstance(arg, MemoryHandle):
                if blobs_size(input_blobs) % 8 != 0:
                    input_blobs.append(b"\0" * 4)
                handle = arg.copy_to_device()
                input_blobs.append(np.uint64(handle).tobytes())
                memory_args.append(arg)
            else:
                raise TypeError(f"Unsupported kernel argument of type {type(arg)}")
        aligned_len = int(np.ceil(blobs_size(input_blobs) / 8)) * 8
        combined_inputs = bytes().join(input_blobs).ljust(aligned_len, b"\0")
        chunks = [combined_inputs[i:i + 8] for i in range(0, len(combined_inputs), 8)]
        inputs = [ctypes.c_uint64(int.from_bytes(x, "little")) for x in chunks]
        core_num = self.options.core_num or self.get_core_num()
        stream = self.options.stream or rt.current_stream()
        rt.launch_kernel(function, core_num, inputs, stream_handle=stream)
        rt.synchronize()
        for index, arg in enumerate(memory_args):
            try:
                if enable_debug and index == len(memory_args) - 1:
                    rt.call_print_interface(inputs[-1], utils.TOTAL_DUMP_SIZE, stream, func_name)
                else:
                    arg.copy_from_device()
            finally:
                arg.release_memory()

    def run(self, kernel: CompiledKernel, function_name: str, user_args: Tuple[Any], discard_handles: bool = True,
            kernel_callback: Optional[KernelCallback] = None) -> None:
        is_launched = isinstance(kernel, LaunchedKernel)
        if not is_launched and kernel.meta.memory_consumed is not None:
            self.check_memory_overflow(kernel.meta.memory_consumed)
        if os.environ.get("PYASC_DRY_RUN"):
            return
        if not isinstance(kernel.binary, bytes):
            raise RuntimeError("Compiled binary is required to launch the kernel")
        explicit_arg = iter(user_args)
        kernel_args = []
        for kind in kernel.meta.kernel_args:
            if kind == ir.KernelArgument.Explicit:
                kernel_args.append(next(explicit_arg))
            elif kind == ir.KernelArgument.FftsAddr:
                ffts_addr = np.array([rt.c2c_ctrl_addr()], dtype=np.uint64)
                kernel_args.append(ffts_addr)
            else:
                raise ValueError(f"Unexpected KernelArgument value: {kind}")
        if kernel.meta.enable_debug:
            kernel_args.append(np.zeros(utils.TOTAL_DUMP_SIZE, dtype=np.int8))
        kernel_args = self.expand_kernel_args(tuple(kernel_args))
        if is_launched:
            kernel_handle = None
            function = kernel.handle
        else:
            kernel_handle = rt.register_device_binary_kernel(kernel.binary, rt.magic_elf_value(kernel.meta.core_type))
            function = rt.register_function(kernel_handle, function_name, mode=0)
            if kernel_callback is not None:
                kernel_callback(function)
        self.launch_kernel(function, kernel_args, kernel.meta.enable_debug, function_name)
        if discard_handles and kernel_handle is not None:
            rt.unregister_device_binary_kernel(kernel_handle)
