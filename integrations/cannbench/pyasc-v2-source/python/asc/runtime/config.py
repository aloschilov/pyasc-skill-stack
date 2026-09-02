# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from enum import Enum
from typing import Optional, Union

from ..lib import runtime as rt


class Backend(Enum):
    """Execution backend for kernel compilation and execution."""

    Model = "Model"
    NPU = "NPU"


class Platform(Enum):
    """Ascend NPU platform types.

    This enum defines the supported Ascend chip variants for kernel compilation and execution. Each platform has
    specific hardware characteristics that affect code generation and optimization.
    """

    Ascend910B1 = "Ascend910B1"
    Ascend910B2 = "Ascend910B2"
    Ascend910B2C = "Ascend910B2C"
    Ascend910B3 = "Ascend910B3"
    Ascend910B4 = "Ascend910B4"
    Ascend910B4_1 = "Ascend910B4-1"
    Ascend910_9362 = "Ascend910_9362"
    Ascend910_9372 = "Ascend910_9372"
    Ascend910_9381 = "Ascend910_9381"
    Ascend910_9382 = "Ascend910_9382"
    Ascend910_9391 = "Ascend910_9391"
    Ascend910_9392 = "Ascend910_9392"
    Ascend950PR_950z = "Ascend950PR_950z"
    Ascend950PR_9579 = "Ascend950PR_9579"
    Ascend950PR_957b = "Ascend950PR_957b"
    Ascend950PR_957c = "Ascend950PR_957c"
    Ascend950PR_957d = "Ascend950PR_957d"
    Ascend950PR_9589 = "Ascend950PR_9589"
    Ascend950PR_958b = "Ascend950PR_958b"
    Ascend950PR_9599 = "Ascend950PR_9599"


class CompilationArch(Enum):
    C220 = "c220"
    C310 = "c310"

    def __str__(self) -> str:
        return self.value


class KernelType(Enum):
    AIV_ONLY = 0
    AIC_ONLY = 1
    MIX_AIV_HARD_SYNC = 2
    MIX_AIC_HARD_SYNC = 3
    MIX_AIV_1_0 = 4
    MIX_AIC_1_0 = 5
    MIX_AIC_1_1 = 6
    MIX_AIC_1_2 = 7


def set_platform(
    backend: Union[Backend, str],
    soc_version: Optional[Union[Platform, str]] = None,
    device_id: Optional[int] = None,
    check=True,
) -> None:
    """Configure the execution platform and backend for kernel execution.

    Sets up the runtime environment by specifying the backend (Model simulator or NPU hardware) and optionally the SoC
    version and device ID. For the Model backend, a default platform is used if none is specified. For the NPU backend,
    the actual hardware platform must match the specified soc_version.

    Args:
        backend: Execution backend type, either Backend.Model for simulator or Backend.NPU for hardware execution.
            Can be specified as a Backend enum or string ("Model" or "NPU").
        soc_version: Target SoC platform version. Can be specified as a Platform enum or string.
            Required for Model backend; must match actual hardware for NPU backend.
            Defaults to Ascend910B1 for Model backend if not specified.
        device_id: Device ID to use for execution. If specified, sets the device for subsequent kernel executions.
            Uses 0-th device otherwise.
        check: Whether to verify runtime library availability. If True, raises an error if the library is not available.

    Raises:
        ValueError: If the backend type is unknown, or if the specified soc_version does not match the actual hardware
            platform when using NPU backend.
        RuntimeError: If check is True and the runtime library is not available.

    Example:
        >>> set_platform(Backend.Model, Platform.Ascend910B1)
        >>> set_platform("NPU", device_id=0)
        >>> set_platform(Backend.Model, "Ascend910B3", check=False)
    """

    backend = Backend(backend)
    if soc_version is not None:
        soc_version = Platform(soc_version)
    if backend == Backend.Model:
        if soc_version is None:
            soc_version = Platform.Ascend910B1
        rt.use_model()
    elif backend == Backend.NPU:
        soc_ver = Platform(rt.current_platform())
        if soc_version is not None and soc_version != soc_ver:
            raise ValueError(f"Input soc version: {soc_version} is different from actual: {soc_ver}")
        soc_version = soc_ver
        rt.use_npu()
    else:
        raise ValueError(f"Unknown execution backend: {backend}")
    rt.set_soc_version(soc_version)
    if device_id is not None:
        rt.set_device(device_id)
    if check and not rt.is_available():
        error_msg = "Runtime library is not available! "
        if backend == Backend.Model:
            error_msg += ("Please export "
                          f"LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/{soc_version.value}/lib:$LD_LIBRARY_PATH")
        raise RuntimeError(error_msg)


def platform_to_arch(platform: Union[Platform, str]) -> CompilationArch:
    platform_name = Platform(platform).value
    if platform_name.startswith("Ascend910B") or platform_name.startswith("Ascend910_93"):
        return CompilationArch.C220
    if platform_name.startswith("Ascend950PR_95"):
        return CompilationArch.C310
    raise ValueError(f"There is no compilation arch for '{platform.value}' platform")
