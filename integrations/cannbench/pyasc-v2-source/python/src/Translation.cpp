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

#include "ascir/Target/Asc/Translation.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // automatic casts between containers and python types

#include <stdexcept>

namespace py = pybind11;
using namespace mlir;

namespace pybind11 {
namespace asc {
void initTranslationModule(py::module&& m)
{
    m.def("ir_to_ascendc", [](ModuleOp& mod) -> std::string {
        std::string result;
        llvm::raw_string_ostream os(result);
        if (translateToAscendC(mod.getOperation(), os).failed())
            throw std::runtime_error("Failed to translate IR to Ascend C");
        os.flush();
        return result;
    });
}
} // namespace asc
} // namespace pybind11
