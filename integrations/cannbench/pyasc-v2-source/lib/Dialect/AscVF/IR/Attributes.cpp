/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/AscVF/IR/AscVF.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ascir/Dialect/AscVF/IR/AscVFEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ascir/Dialect/AscVF/IR/AscVFAttributes.cpp.inc"

using namespace mlir;
using namespace mlir::ascvf;

//===----------------------------------------------------------------------===//
// AscVFDialect
//===----------------------------------------------------------------------===//

void AscVFDialect::registerAttributes()
{
    addAttributes<
#define GET_ATTRDEF_LIST
#include "ascir/Dialect/AscVF/IR/AscVFAttributes.cpp.inc"
        >();
}
