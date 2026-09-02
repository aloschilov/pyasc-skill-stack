/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/AscVF/IR/AscVF.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "ascir/Dialect/AscVF/IR/AscVFOps.cpp.inc"

using namespace mlir;
using namespace mlir::ascvf;

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify()
{
    if (!getOperation()->getParentOfType<ascvf::VecScopeOp>()) {
        return emitOpError("The operation must belong to vec_scope");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify()
{
    if (!getOperation()->getParentOfType<ascvf::VecScopeOp>()) {
        return emitOpError("The operation must belong to vec_scope");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// VFGroupOp
//===----------------------------------------------------------------------===//

Type VFGroupOp::getGroupType()
{
    Value tensor{};
    if (auto dstList = getDstList(); !dstList.empty()) {
        tensor = dstList.back();
    } else if (auto srcList = getSrcList(); !srcList.empty()) {
        tensor = srcList.back();
    } else {
        return Type{};
    }
    auto tensorType = dyn_cast<ascendc::LocalTensorType>(tensor.getType());
    assert(tensorType && "expected local tensor");
    return tensorType.getElementType();
}

//===----------------------------------------------------------------------===//
// VFForOp
//===----------------------------------------------------------------------===//

void VFForOp::build(OpBuilder& builder, OperationState& state, Value upperBound)
{
    OpBuilder::InsertionGuard guard(builder);
    state.addOperands(upperBound);
    Type type = builder.getIndexType();
    Region* bodyRegion = state.addRegion();
    Block* bodyBlock = builder.createBlock(bodyRegion);
    bodyBlock->addArgument(type, state.location);
    ensureTerminator(*bodyRegion, builder, state.location);
}

LogicalResult VFForOp::canonicalize(VFForOp op, PatternRewriter& rewriter)
{
    Block& block = op.getRegion().front();
    if (block.without_terminator().empty()) {
        rewriter.eraseOp(op);
        return success();
    }
    return failure();
}

LogicalResult VFForOp::verify()
{
    if (getBody()->getArguments().size() != 1)
        return emitOpError("block must have one argument");
    return success();
}

//===----------------------------------------------------------------------===//
// AscVFDialect
//===----------------------------------------------------------------------===//

void AscVFDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "ascir/Dialect/AscVF/IR/AscVFOps.cpp.inc"
        >();
}
