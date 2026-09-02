/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "InitFuncDef.h"

#include <pybind11/cast.h>
#include <pybind11/functional.h>

namespace py = pybind11;

namespace {
PYBIND11_MODULE(libpyasc, m)
{
    m.doc() = "Python bindings to the C++ AscIR API";
    py::asc::initIRModule(m.def_submodule("ir"));
    py::asc::initPassesModule(m.def_submodule("passes"));
    py::asc::initTranslationModule(m.def_submodule("translation"));
}
} // namespace
