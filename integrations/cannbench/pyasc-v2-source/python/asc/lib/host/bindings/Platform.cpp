/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling/platform/platform_ascendc.h"
#include <cstdint>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pybind11 {
namespace asc {
void pyasc_init_platform(py::module& m)
{
    using ret = py::return_value_policy;
    using namespace platform_ascendc;

    py::class_<PlatformAscendC>(m, "PlatformAscendC", py::module_local());

    py::class_<PlatformAscendCManager, std::unique_ptr<PlatformAscendCManager, py::nodelete>>(
        m, "PlatformAscendCManager", py::module_local())
        .def_static(
            "get_instance", []() { return PlatformAscendCManager::GetInstance(); }, ret::reference)
        .def_static(
            "get_instance",
            [](const std::string& socVersion) { return PlatformAscendCManager::GetInstance(socVersion.c_str()); },
            ret::reference, "soc_version"_a);
}
} // namespace asc
} // namespace pybind11
