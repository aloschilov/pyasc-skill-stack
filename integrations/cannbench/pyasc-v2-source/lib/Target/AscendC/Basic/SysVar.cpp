/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/SysVar.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// System Variable operations
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GetBlockIdxOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = static_cast<";
    if (failed(emitter.emitType(op.getLoc(), op.getType()))) {
        return failure();
    }
    os << ">(" << ascNamespace << "::" << op.getAPIName() << "())";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::GetBlockNumOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    auto& os = emitter.ostream();
    os << " = static_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << ">(" << ascNamespace << "::" << op.getAPIName() << "())";
    return success();
}
