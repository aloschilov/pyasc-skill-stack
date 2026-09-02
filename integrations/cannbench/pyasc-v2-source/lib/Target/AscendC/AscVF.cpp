/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/AscVF.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/AscVF/IR/AscVF.h"
#include "ascir/Target/Asc/External/Scf.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::ascvf;

LogicalResult mlir::ascvf::printOperation(CodeEmitter& emitter, ascvf::VFForOp forOp)
{
    auto& os = emitter.ostream();
    os << "for (";
    FAIL_OR(emitter.emitType(forOp.getLoc(), forOp.getUnderlyingType()));
    os << " " << emitter.getOrCreateName(forOp.getInductionVar());
    os << " = " << forOp.getLowerBoundAsInt();
    os << "; " << emitter.getOrCreateName(forOp.getInductionVar());
    os << " < static_cast<";
    FAIL_OR(emitter.emitType(forOp.getLoc(), forOp.getUnderlyingType()));
    os << ">(" << emitter.getOrCreateName(forOp.getUpperBound()) << ")";
    os << "; " << emitter.getOrCreateName(forOp.getInductionVar());
    os << " += " << forOp.getStepAsInt() << ") {\n";
    os.indent();
    Region& forRegion = forOp.getRegion();
    auto regionOps = forRegion.getOps();
    for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
        Operation& op = *it;
        if (failed(emitOperation(emitter, op, needsSemicolon(op)))) {
            return failure();
        }
    }
    os.unindent() << "}";
    return success();
}

LogicalResult mlir::ascvf::printOperation(CodeEmitter& emitter, ascvf::VecScopeOp vecScopeOp)
{
    auto& os = emitter.ostream();
    os << "__VEC_SCOPE__\n";
    os << "{\n";
    os.indent();
    FAIL_OR(emitBlock(emitter, *vecScopeOp.getBody()));
    os.unindent() << "}";
    return success();
}

LogicalResult mlir::ascvf::printOperation(CodeEmitter& emitter, ascvf::VFGroupOp op)
{
    auto& os = emitter.ostream();
    os << "{\n";
    os.indent();
    FAIL_OR(emitBlock(emitter, *op.getBody()));
    os.unindent() << "}";
    return success();
}
