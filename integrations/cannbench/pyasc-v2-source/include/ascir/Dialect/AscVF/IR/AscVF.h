/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_DIALECT_ASCVF_IR_ASCVF_H
#define ASCIR_DIALECT_ASCVF_IR_ASCVF_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "ascir/Dialect/AscVF/IR/AscVFDialect.h.inc"

#include "ascir/Dialect/AscVF/IR/AscVFEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ascir/Dialect/AscVF/IR/AscVFAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ascir/Dialect/AscVF/IR/AscVFTypes.h.inc"

#define GET_OP_CLASSES
#include "ascir/Dialect/AscVF/IR/AscVFOps.h.inc"

namespace mlir {
namespace ascvf {

void registerExternalModels(DialectRegistry& registry);

} // namespace ascvf
} // namespace mlir

#endif // ASCIR_DIALECT_ASCVF_IR_ASCVF_H
