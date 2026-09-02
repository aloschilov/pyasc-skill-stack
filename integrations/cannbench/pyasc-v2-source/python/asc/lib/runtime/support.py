# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
import enum
from dataclasses import dataclass

from typing_extensions import TypeAlias

Stream: TypeAlias = ctypes.c_void_p
Memory: TypeAlias = ctypes.c_void_p
Kernel: TypeAlias = ctypes.c_void_p
Function: TypeAlias = ctypes.c_void_p


class CoreType(enum.Enum):
    AiCore = "AiCore"
    VectorCore = "VectorCore"
    CubeCore = "CubeCore"
    AiCpu = "AiCpu"


class DeviceModuleType(enum.Enum):
    RT_MODULE_TYPE_SYSTEM = 0
    RT_MODULE_TYPE_AICPU = 1
    RT_MODULE_TYPE_CCPU = 2
    RT_MODULE_TYPE_DCPU = 3
    RT_MODULE_TYPE_AICORE = 4
    RT_MODULE_TYPE_TSCPU = 5
    RT_MODULE_TYPE_PCIE = 6
    RT_MODULE_TYPE_VECTOR_CORE = 7
    RT_MODULE_TYPE_HOST_AICPU = 8


class DeviceInfoType(enum.Enum):
    INFO_TYPE_ENV = 0
    INFO_TYPE_VERSION = 1
    INFO_TYPE_MASTERID = 2
    INFO_TYPE_CORE_NUM = 3
    INFO_TYPE_OS_SCHED = 4
    INFO_TYPE_IN_USED = 5
    INFO_TYPE_ERROR_MAP = 6
    INFO_TYPE_OCCUPY = 7
    INFO_TYPE_ID = 8
    INFO_TYPE_IP = 9
    INFO_TYPE_ENDIAN = 10
    INFO_TYPE_CUBE_NUM = 0x775A5A5A


class DevBinary(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data", ctypes.c_char_p),
        ("length", ctypes.c_uint64),
    ]


class MagicElf(enum.Enum):
    RT_DEV_BINARY_MAGIC_ELF = 0x43554245
    RT_DEV_BINARY_MAGIC_ELF_AIVEC = 0x41415246
    RT_DEV_BINARY_MAGIC_ELF_AICUBE = 0x41494343
    RT_DEV_BINARY_MAGIC_ELF_AICPU = 0x41415243


class MemoryType(enum.Enum):
    RT_MEMORY_DEFAULT = 0
    RT_MEMORY_HBM = 2
    RT_MEMORY_DDR = 4
    RT_MEMORY_SPM = 8
    RT_MEMORY_P2P_HBM = 0x10
    RT_MEMORY_P2P_DDR = 0x11
    RT_MEMORY_DDR_NC = 0x20
    RT_MEMORY_TS_4G = 0x40
    RT_MEMORY_TS = 0x80
    RT_MEMORY_RESERVED = 0x100
    RT_MEMORY_L1 = 0x10000
    RT_MEMORY_L2 = 0x20000


class MallocPolicy(enum.Enum):
    RT_MEMORY_POLICY_NONE = 0x0
    RT_MEMORY_POLICY_HUGE_PAGE_FIRST = 0x400
    RT_MEMORY_POLICY_HUGE_PAGE_ONLY = 0x800
    RT_MEMORY_POLICY_DEFAULT_PAGE_ONLY = 0x1000
    RT_MEMORY_POLICY_HUGE_PAGE_FIRST_P2P = 0x2000
    RT_MEMORY_POLICY_HUGE_PAGE_ONLY_P2P = 0x4000
    RT_MEMORY_POLICY_DEFAULT_PAGE_ONLY_P2P = 0x8000


class MemcpyKind(enum.Enum):
    RT_MEMCPY_HOST_TO_HOST = 0
    RT_MEMCPY_HOST_TO_DEVICE = 1
    RT_MEMCPY_DEVICE_TO_HOST = 2
    RT_MEMCPY_DEVICE_TO_DEVICE = 3
    RT_MEMCPY_MANAGED = 4
    RT_MEMCPY_ADDR_DEVICE_TO_DEVICE = 5
    RT_MEMCPY_HOST_TO_DEVICE_EX = 6
    RT_MEMCPY_DEVICE_TO_HOST_EX = 7
    RT_MEMCPY_RESERVED = 8


@dataclass(frozen=True)
class MemorySizes:
    l0a_size: int
    l0b_size: int
    l0c_size: int
    l1_size: int
    ub_size: int
    l2_size: int
    l2_page_num: int
    block_size: int
    bank_size: int
    bank_num: int
    burst_in_one_block: int
    bank_group_num: int

    class Handle(ctypes.Structure):
        fields = [
            ("l0a_size", ctypes.c_uint32),
            ("l0b_size", ctypes.c_uint32),
            ("l0c_size", ctypes.c_uint32),
            ("l1_size", ctypes.c_uint32),
            ("ub_size", ctypes.c_uint32),
            ("l2_size", ctypes.c_uint32),
            ("l2_page_num", ctypes.c_uint32),
            ("block_size", ctypes.c_uint32),
            ("bank_size", ctypes.c_uint64),
            ("bank_num", ctypes.c_uint64),
            ("burst_in_one_block", ctypes.c_uint64),
            ("bank_group_num", ctypes.c_uint64),
        ]


class MsprofTaskType(enum.Enum):
    MSPROF_TASK_TYPE_AI_CORE = 0
    MSPROF_TASK_TYPE_AI_CPU = 1
    MSPROF_TASK_TYPE_AIV = 2
    MSPROF_TASK_TYPE_MIX_AIC = 4
    MSPROF_TASK_TYPE_MIX_AIV = 5
    MSPROF_TASK_TYPE_INVALID = 11


class ProfilingValues:
    RT_PROF_MAX_DEV_NUM = 64
    RT_PROF_PATH_LEN_MAX = 1023
    RT_PROF_PARAM_LEN_MAX = 4095


class ProfilingCommandHandle(ctypes.Structure):

    class Type(enum.Enum):
        INIT = 0
        START = 1
        STOP = 2
        FINALIZE = 3
        MODEL_SUBSCRIBE = 4
        MODEL_UNSUBSCRIBE = 5

    class Params(ctypes.Structure):
        _fields_ = [
            ("pathLen", ctypes.c_uint32),
            ("storageLimit", ctypes.c_uint32),
            ("profDataLen", ctypes.c_uint32),
            ("path", ctypes.c_char * (ProfilingValues.RT_PROF_PATH_LEN_MAX + 1)),
            ("profData", ctypes.c_char * (ProfilingValues.RT_PROF_PARAM_LEN_MAX + 1)),
        ]

    _fields_ = [
        ("profSwitch", ctypes.c_uint64),
        ("profSwitchHi", ctypes.c_uint64),
        ("devNums", ctypes.c_uint32),
        ("devIdList", ctypes.c_uint32 * ProfilingValues.RT_PROF_MAX_DEV_NUM),
        ("modelId", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("cacheFlag", ctypes.c_uint32),
        ("commandHandleParams", Params),
    ]
