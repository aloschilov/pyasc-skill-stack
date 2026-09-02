# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
from enum import Enum
from typing import Any, Iterable, Optional
from typing_extensions import TypeAlias

ProfilerConfig: TypeAlias = ctypes.c_void_p


class AicoreMetrics(Enum):
    ARITHMETIC_UTILIZATION = 0
    PIPE_UTILIZATION = 1
    MEMORY_BANDWIDTH = 2
    L0B_AND_WIDTH = 3
    RESOURCE_CONFLICT_RATIO = 4
    MEMORY_UB = 5
    L2_CACHE = 6
    PIPE_EXECUTE_UTILIZATION = 7
    MEMORY_ACCESS = 8
    NONE = 0xFF


class ProfileType(Enum):
    ACL_API = 0x0001
    TASK_TIME = 0x0002
    AICORE_METRICS = 0x0004
    AICPU = 0x0008
    L2CACHE = 0x0010
    HCCL_TRACE = 0x0020
    TRAINING_TRACE = 0x0040
    MSPROFTX = 0x0080
    RUNTIME_API = 0x0100
    TASK_TIME_L0 = 0x0800
    TASK_MEMORY = 0x1000
    OP_ATTR = 0x4000


class MsprofInterface:

    def __init__(self):
        self.lib = ctypes.CDLL("libmsprofiler.so")

    def call(self, fn_name: str, *args, ret_type=None) -> Optional[Any]:
        fn = getattr(self.lib, fn_name)
        if ret_type is not None:
            fn.restype = ret_type
        result = fn(*args)
        if ret_type is None and result != 0:
            raise RuntimeError(f"Function {fn_name} returned {result}")
        return result

    def init(self, result_path: str) -> None:
        c_path = result_path.encode("utf-8")
        c_size = ctypes.c_size_t(len(result_path))
        self.call("aclprofInit", c_path, c_size)

    def create_config(self, device_ids: Iterable[int], metric: AicoreMetrics,
                      types: Optional[Iterable[ProfileType]] = None) -> ProfilerConfig:
        type_ids = tuple() if types is None else (t.value for t in types)
        type_config = 0
        for type_id in type_ids:
            type_config |= type_id
        device_ids = tuple(device_ids)
        c_device_id_list = (ctypes.c_uint32 * 1)(*device_ids)
        c_device_nums = ctypes.c_uint32(len(device_ids))
        c_aicore_metrics = ctypes.c_int(metric.value)
        c_data_type_config = ctypes.c_uint64(type_config)
        config = self.call("aclprofCreateConfig", c_device_id_list, c_device_nums, c_aicore_metrics, None,
                           c_data_type_config, ret_type=ctypes.c_void_p)
        if config is None:
            raise RuntimeError("Function aclprofCreateConfig returned nullptr")
        return ProfilerConfig(config)

    def start(self, config: ProfilerConfig) -> None:
        self.call("aclprofStart", config)

    def stop(self, config: ProfilerConfig) -> None:
        self.call("aclprofStop", config)

    def destroy_config(self, config: ProfilerConfig) -> None:
        self.call("aclprofDestroyConfig", config)

    def finalize(self) -> None:
        self.call("aclprofFinalize")
